"""
Web Audio & NLU Coordinator — Menghubungkan mic hardware Jetson Nano,
Sherpa-ONNX RNN-T streaming recognizer, NLU pipeline, dan WebSocket clients.
"""

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Callable, Optional, Dict, Any, List

import numpy as np
import sounddevice as sd

from audio import (
    get_supported_sample_rate,
    get_optimal_channels,
    resample_to_16k,
    list_audio_devices,
)
from audio import stt as _stt
from audio.tts import speak, is_tts_available
from nlu.pipeline import NLUPipeline
from nlu.ros2_adapter import (
    SocketROS2Adapter,
    MockROS2Adapter,
    load_ros2_config,
    save_ros2_config,
)
from nlu.dialog import get_confirmation_message, merge_dialog_context
import responses

logger = logging.getLogger("web.listener")


class WebVoiceCoordinator:
    """
    Koordinator mikrofon hardware Jetson & WebSocket broadcasting.
    Mengatur state (IDLE, LISTENING, PROCESSING, SPEAKING) dan menyebarkan
    live partial text RNN-T ke seluruh web client yang terhubung.
    """

    def __init__(self, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
        self.lock = threading.Lock()
        self.state = "IDLE"  # "IDLE" | "LISTENING" | "PROCESSING" | "SPEAKING"
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Inisialisasi ROS2 Adapter & Pipeline (Default Mode Mock untuk development offline)
        self.use_mock_ros2 = True
        self.ros2_host = ros2_host
        self.ros2_port = ros2_port
        self.ros2_adapter = None
        self._init_ros2_adapter()
        self.pipeline = NLUPipeline(ros2_adapter=self.ros2_adapter)

        # Background listen thread handles
        self.listen_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.current_partial = ""
        self.last_result: Optional[Dict[str, Any]] = None

        # Konfigurasi audio
        self.device_index: Optional[int] = None
        self.default_duration: int = 8
        self.tts_enabled: bool = True

    def _init_ros2_adapter(self):
        """Inisialisasi ROS2 adapter (Mock atau Socket)."""
        if self.use_mock_ros2:
            self.ros2_adapter = MockROS2Adapter()
        else:
            self.ros2_adapter = SocketROS2Adapter(host=self.ros2_host, port=self.ros2_port)

    def set_mock_mode(self, enabled: bool):
        """Aktifkan/nonaktifkan mode mock ROS 2 (simulasi)."""
        with self.lock:
            self.use_mock_ros2 = enabled
            self._init_ros2_adapter()
            self.pipeline.ros2_adapter = self.ros2_adapter
            logger.info(f"[WebCoordinator] Mode ROS2 diganti ke: {'MOCK' if enabled else 'SOCKET'}")
            self.broadcast({
                "type": "ros2_mode_changed",
                "mock": self.use_mock_ros2,
                "target": "MOCK_ROS2" if self.use_mock_ros2 else f"{self.ros2_adapter.host}:{self.ros2_adapter.port}"
            })

    def update_ros2_target(self, host: str, port: int) -> bool:
        """Update host dan port koneksi ROS 2."""
        with self.lock:
            self.ros2_host = host
            self.ros2_port = port
            save_ros2_config(host, port)
            if not self.use_mock_ros2 and isinstance(self.ros2_adapter, SocketROS2Adapter):
                self.ros2_adapter.set_target(host, port)
            self.broadcast({
                "type": "ros2_target_updated",
                "host": host,
                "port": port
            })
            return True

    def ping_ros2(self) -> Dict[str, Any]:
        """Uji koneksi ke target ROS 2 saat ini."""
        if self.ros2_adapter:
            ok, msg = self.ros2_adapter.ping(timeout=1.5)
            return {"connected": ok, "message": msg}
        return {"connected": False, "message": "Adapter belum siap"}

    def add_subscriber(self, callback: Callable[[Dict[str, Any]], None]):
        """Daftarkan WebSocket callback untuk menerima event broadcast."""
        with self.lock:
            if callback not in self.subscribers:
                self.subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[Dict[str, Any]], None]):
        """Hapus WebSocket callback saat koneksi terputus."""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def broadcast(self, payload: Dict[str, Any]):
        """Kirim event ke seluruh WebSocket client."""
        with self.lock:
            listeners = list(self.subscribers)
        for cb in listeners:
            try:
                cb(payload)
            except Exception as e:
                logger.debug(f"[WebCoordinator] Gagal kirim broadcast: {e}")

    def set_state(self, new_state: str, details: Optional[str] = None):
        """Ubah state sistem dan umumkan ke client."""
        self.state = new_state
        self.broadcast({
            "type": "state_changed",
            "state": new_state,
            "details": details or ""
        })

    def start_listening(self, duration: Optional[int] = None, device: Optional[int] = None) -> bool:
        """
        Mulai mendengarkan microphone hardware Jetson secara asinkron.
        Mengalirkan teks kata-per-kata via RNN-T ke WebSocket.
        """
        with self.lock:
            if self.state in ("LISTENING", "PROCESSING"):
                logger.warning("[WebCoordinator] Sedang aktif mendengarkan/memproses.")
                return False
            self.stop_event.clear()
            dur = duration or self.default_duration
            dev = device if device is not None else self.device_index

            self.listen_thread = threading.Thread(
                target=self._listen_worker,
                args=(dur, dev),
                name="Web-Mic-Worker",
                daemon=True,
            )
            self.listen_thread.start()
            return True

    def stop_listening(self):
        """Hentikan perekaman mikrofon secara paksa."""
        self.stop_event.set()

    def _listen_worker(self, duration: int, device: Optional[int]):
        """Worker thread untuk menangkap audio dan men-decode secara streaming."""
        self.set_state("LISTENING", "Mendengarkan suara dari mikrofon Jetson...")
        self.current_partial = ""

        try:
            recognizer = _stt.get_recognizer()
            create_stream = _stt.create_stream
            capture_rate = get_supported_sample_rate(device)
            channels = get_optimal_channels(device, capture_rate)
            stt_rate = 16000
            stream = create_stream()

            audio_q = queue.Queue()
            last_text = ""

            def audio_callback(indata, frames, time_info, status_flags):
                if self.stop_event.is_set():
                    return
                audio_q.put(indata.copy())

            decode_stop = threading.Event()
            final_text_holder = [""]

            def decode_loop():
                nonlocal last_text
                while not self.stop_event.is_set() and not decode_stop.is_set():
                    try:
                        indata = audio_q.get(timeout=0.08)
                    except queue.Empty:
                        continue

                    try:
                        if indata.shape[1] > 1:
                            samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
                        else:
                            samples = np.ascontiguousarray(indata.flatten(), dtype=np.float32)

                        if capture_rate != stt_rate:
                            samples = resample_to_16k(samples, capture_rate)

                        # Hitung volume audio (RMS) untuk visualisasi gelombang spectrum
                        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
                        norm_level = float(min(1.0, rms * 18.0))
                        self.broadcast({
                            "type": "audio_level",
                            "level": round(norm_level, 3)
                        })

                        stream.accept_waveform(stt_rate, samples)
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                        is_endpoint = recognizer.is_endpoint(stream)
                        partial = recognizer.get_result(stream).strip()

                        if partial and partial != last_text:
                            last_text = partial
                            self.current_partial = partial
                            self.broadcast({
                                "type": "stt_partial",
                                "text": partial
                            })

                        if is_endpoint:
                            if partial:
                                final_text_holder[0] = partial
                                decode_stop.set()
                                return
                            else:
                                recognizer.reset(stream)

                    except Exception as err:
                        logger.error(f"[WebCoordinator] Decode error: {err}")

            decoder_thread = threading.Thread(target=decode_loop, name="Web-STT-Decoder", daemon=True)
            decoder_thread.start()

            chunk_size = int(capture_rate * 0.08)  # 80ms chunks
            with sd.InputStream(
                samplerate=capture_rate,
                channels=channels,
                dtype="float32",
                blocksize=chunk_size,
                device=device,
                callback=audio_callback,
            ):
                start_time = time.time()
                while (time.time() - start_time < duration) and not self.stop_event.is_set() and not decode_stop.is_set():
                    time.sleep(0.05)

            decode_stop.set()
            decoder_thread.join(timeout=1.0)

            final_text = final_text_holder[0] or last_text or recognizer.get_result(stream).strip()

            if final_text:
                self.broadcast({
                    "type": "stt_final",
                    "text": final_text
                })
                # Lanjutkan ke NLU pipeline
                self.process_text(final_text, source="mic")
            else:
                self.broadcast({
                    "type": "stt_final",
                    "text": "",
                    "message": "Tidak ada suara yang terdeteksi."
                })
                self.set_state("IDLE", "Tidak ada suara.")

        except Exception as e:
            logger.error(f"[WebCoordinator] Mic listen error: {e}", exc_info=True)
            self.broadcast({
                "type": "error",
                "message": f"Error mikrofon: {str(e)}"
            })
            self.set_state("IDLE", "Error mikrofon")

    def process_text(self, text: str, source: str = "text") -> Dict[str, Any]:
        """
        Proses teks (baik dari transkripsi mic atau ketikan manual) lewat NLU pipeline,
        kirim ke ROS 2 / Mock, dan bunyikan TTS jika diaktifkan.
        """
        clean_text = text.strip()
        if not clean_text:
            return {"status": "EMPTY", "error": "Teks perintah kosong"}

        self.set_state("PROCESSING", f"Menganalisis: \"{clean_text}\"")

        try:
            # 1. Pipeline NLU
            t0 = time.time()
            result = self.pipeline.process(clean_text)
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            result["latency_ms"] = elapsed_ms
            result["source"] = source
            result["input_text"] = clean_text

            # 2. Ambil respons natural dialog robot
            label_text, speech_msg = get_confirmation_message(result)
            result["robot_response"] = speech_msg or label_text
            self.last_result = result

            # 3. Broadcast hasil NLU ke semua web clients
            self.broadcast({
                "type": "nlu_result",
                "data": result,
                "response": result["robot_response"]
            })

            # 4. Mainkan TTS di speaker Jetson jika tersedia & diaktifkan
            if self.tts_enabled and result["robot_response"]:
                self.set_state("SPEAKING", result["robot_response"])
                try:
                    speak(result["robot_response"], blocking=True)
                except Exception as tts_err:
                    logger.warning(f"[WebCoordinator] TTS error: {tts_err}")

            self.set_state("IDLE", "Siap menerima perintah")
            return result

        except Exception as e:
            logger.error(f"[WebCoordinator] Error processing NLU: {e}", exc_info=True)
            self.set_state("IDLE", f"Error: {e}")
            return {"status": "ERROR", "error": str(e)}


    # =========================================================================
    # Browser / HP Mic Session (audio di-stream via WebSocket binary frames)
    # Decode dijalankan di worker thread terpisah agar event loop FastAPI
    # tidak pernah terblokir (penyebab tombol mic "tidak bisa ditekan").
    # =========================================================================

    def start_browser_mic_session(self, owner: Any = None) -> bool:
        """Mulai sesi streaming dari mikrofon browser (HP/laptop)."""
        with self.lock:
            if self.state in ("LISTENING", "PROCESSING"):
                return False
            self.stop_event.clear()
            self.current_partial = ""
            self._browser_owner = owner
            self._browser_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=200)
            self._browser_cancel = False
            self._browser_thread = threading.Thread(
                target=self._browser_worker, name="Web-BrowserMic-Worker", daemon=True
            )
        self.set_state("LISTENING", "Mendengarkan dari mikrofon perangkat Anda...")
        self._browser_thread.start()
        return True

    def feed_browser_audio(self, pcm_bytes: bytes):
        """Masukkan chunk PCM float32 16kHz mono ke antrian (non-blocking)."""
        q = getattr(self, "_browser_q", None)
        if q is None or self.state != "LISTENING":
            return
        try:
            q.put_nowait(pcm_bytes)
        except queue.Full:
            pass  # buang chunk jika decoder tertinggal

    def stop_browser_mic_session(self, cancel: bool = False):
        """Akhiri sesi: worker akan memproses hasil akhir (atau batal jika cancel)."""
        q = getattr(self, "_browser_q", None)
        if q is None:
            return
        self._browser_cancel = cancel
        try:
            q.put_nowait(None)  # sentinel
        except queue.Full:
            self._browser_cancel = True
            self.stop_event.set()

    def owns_browser_session(self, owner: Any) -> bool:
        return getattr(self, "_browser_q", None) is not None and getattr(self, "_browser_owner", None) is owner

    def _browser_worker(self):
        q = self._browser_q
        final_text = ""
        try:
            recognizer = _stt.get_recognizer()
            stream = _stt.create_stream()
            last_text = ""
            started = time.time()
            last_level_ts = 0.0
            n_chunks, n_samples, peak = 0, 0, 0.0
            while True:
                if self.stop_event.is_set() or time.time() - started > 20:
                    break
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                samples = np.frombuffer(chunk, dtype=np.float32)
                if samples.size == 0:
                    continue
                # Lindungi recognizer dari NaN/Inf (satu NaN merusak seluruh decoding)
                if not np.isfinite(samples).all():
                    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
                samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
                n_chunks += 1
                n_samples += samples.size
                peak = max(peak, float(np.abs(samples).max()))
                if n_chunks == 1:
                    print(f"[BrowserMic] chunk pertama diterima ({samples.size} sampel)", flush=True)

                now = time.time()
                if now - last_level_ts > 0.06:
                    last_level_ts = now
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    self.broadcast({"type": "audio_level", "level": round(min(1.0, rms * 18.0), 3)})

                stream.accept_waveform(16000, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                partial = recognizer.get_result(stream).strip()
                if partial and partial != last_text:
                    last_text = partial
                    self.current_partial = partial
                    self.broadcast({"type": "stt_partial", "text": partial})

                if partial and recognizer.is_endpoint(stream):
                    break

            print(f"[BrowserMic] sesi selesai: {n_chunks} chunk, {n_samples/16000:.1f} detik audio, "
                  f"peak={peak:.4f}, teks='{last_text}'", flush=True)
            if not self._browser_cancel:
                # Flush sisa audio
                try:
                    stream.input_finished()
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                except Exception:
                    pass
                final_text = recognizer.get_result(stream).strip() or last_text
        except Exception as e:
            logger.error(f"[WebCoordinator] Browser mic worker error: {e}", exc_info=True)
            self.broadcast({"type": "error", "message": f"Error STT: {e}"})
        finally:
            self._browser_q = None
            self._browser_owner = None

        if self._browser_cancel:
            self.broadcast({"type": "stt_final", "text": ""})
            self.set_state("IDLE", "Sentuh untuk bicara")
            return

        self.broadcast({
            "type": "stt_final",
            "text": final_text,
            **({} if final_text else {"message": "Tidak ada suara yang terdeteksi."}),
        })
        if final_text:
            self.process_text(final_text, source="browser_mic")
        else:
            self.set_state("IDLE", "Tidak ada suara terdeteksi")

    def force_reset(self):
        """Paksa kembali ke IDLE (pemulihan jika state macet)."""
        self.stop_event.set()
        if getattr(self, "_browser_q", None) is not None:
            self.stop_browser_mic_session(cancel=True)
        else:
            self.set_state("IDLE", "Sentuh untuk bicara")


# Global instance
coordinator: Optional[WebVoiceCoordinator] = None

def get_coordinator(ros2_host: Optional[str] = None, ros2_port: Optional[int] = None) -> WebVoiceCoordinator:
    global coordinator
    if coordinator is None:
        coordinator = WebVoiceCoordinator(ros2_host=ros2_host, ros2_port=ros2_port)
    return coordinator
