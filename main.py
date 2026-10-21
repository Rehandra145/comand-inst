"""
Robot Voice Command — Main Entry Point.

Usage:
    # Text mode (tanpa microphone/STT) — untuk testing NLU:
    python main.py --text "tolong antar saya ke ruang ICU"

    # GUI mode (microphone + real-time streaming STT + NLU):
    python main.py
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np

# Pastikan output terminal mendukung UTF-8 di Windows maupun Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)

from config import DEFAULT_MIC_DEVICE
import responses

try:
    from scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None


# =========================
# CLI / TERMINAL HELPERS
# =========================

def get_effective_input_device(device: int = None) -> int:
    """
    Menentukan index perangkat mikrofon yang aktif:
    1. Jika ditentukan via argumen/flag (--device <ID>), prioritaskan itu.
    2. Jika diset di config.py (DEFAULT_MIC_DEVICE), gunakan itu.
    3. Jika default OS memiliki channel input > 0, gunakan itu.
    4. Jika default OS tidak punya input channel (kasus umum di Ubuntu Server / Raspi),
       otomatis cari dan gunakan perangkat input pertama yang aktif (misal USB Mic).
    """
    if device is not None:
        return device

    if DEFAULT_MIC_DEVICE is not None:
        return DEFAULT_MIC_DEVICE

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else -1
        if 0 <= default_in < len(devices) and devices[default_in].get("max_input_channels", 0) > 0:
            return default_in

        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                logger.info(f"[AUDIO] Otomatis memilih input device ID {i}: {dev['name']}")
                return i
    except Exception:
        pass

    return None


def get_supported_sample_rate(device: int = None) -> int:
    """
    Deteksi sample rate yang didukung langsung oleh hardware mikrofon.
    Prioritaskan 16000 Hz (native STT). Jika hardware menolak 16000 Hz
    (misal USB mic yang hanya mendukung 48000 Hz / 44100 Hz), otomatis
    gunakan sample rate asli hardware tersebut.
    """
    import sounddevice as sd
    candidates = [16000, 48000, 44100, 32000, 22050, 8000]
    for rate in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            return rate
        except Exception:
            continue

    try:
        dev_info = sd.query_devices(device, "input")
        return int(dev_info.get("default_samplerate", 16000))
    except Exception:
        return 16000


def get_optimal_channels(device: int = None, rate: int = 16000) -> int:
    """Deteksi jumlah channel minimum yang didukung hardware (1 atau max_channels)."""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
        return 1
    except Exception:
        try:
            dev_info = sd.query_devices(device, "input")
            max_ch = int(dev_info.get("max_input_channels", 2))
            return max_ch if max_ch > 0 else 1
        except Exception:
            return 1


def resample_to_16k(samples, orig_sr: int):
    """Resample array audio float32 dari orig_sr ke 16000 Hz untuk Sherpa-ONNX."""
    import numpy as np
    if orig_sr == 16000 or len(samples) == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)

    if orig_sr == 48000:
        # Decimation 3:1 (48000 Hz / 3 = 16000 Hz) — C-contiguous array wajib untuk C++ pybind11!
        return np.ascontiguousarray(samples[::3], dtype=np.float32)

    if orig_sr == 32000:
        return np.ascontiguousarray(samples[::2], dtype=np.float32)

    if orig_sr == 44100:
        try:
            from scipy.signal import resample_poly
            return np.ascontiguousarray(resample_poly(samples, 160, 441), dtype=np.float32)
        except ImportError:
            pass

    # Interpolasi cepat untuk sample rate sembarang (misal 44100 Hz tanpa scipy)
    target_length = int(len(samples) * 16000 / orig_sr)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0, 1, len(samples), endpoint=False)
    x_new = np.linspace(0, 1, target_length, endpoint=False)
    return np.ascontiguousarray(np.interp(x_new, x_old, samples), dtype=np.float32)


def list_audio_devices():
    """Tampilkan daftar perangkat input audio (microphone) yang tersedia."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n=== DAFTAR PERANGKAT INPUT AUDIO ===")
        found = False
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else -1
        effective_in = get_effective_input_device()
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                found = True
                marker = "*" if i == effective_in else " "
                native_sr = get_supported_sample_rate(i)
                print(f" [{marker}] Device ID {i:2d}: {dev['name']} (Channels: {dev['max_input_channels']}, Rate: {native_sr} Hz)")
        if not found:
            print("  (Tidak ada perangkat microphone yang terdeteksi)")
        else:
            print(f"\n  Keterangan: [*] = Perangkat input aktif yang akan digunakan (ID: {effective_in}).")
            print("  Gunakan flag --device <ID> untuk memilih mikrofon tertentu.\n")
    except Exception as e:
        print(f"Gagal mendeteksi perangkat audio: {e}\n")


def test_mic_hardware(device: int = None, duration: int = 3):
    """Diagnosa dan uji coba mikrofon, periksa level sinyal audio & volume."""
    device = get_effective_input_device(device)
    print("\n" + "=" * 60)
    print("  [DIAGNOSA] Driver Audio & Perangkat Mikrofon")
    print("=" * 60)
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        print(f"[ERROR] Library belum lengkap: {e}")
        print("   Jalankan: sudo apt install -y portaudio19-dev libasound2-dev")
        print("   Lalu: uv pip install sounddevice numpy\n")
        return

    try:
        dev_info = sd.query_devices(device, "input")
    except Exception as e:
        print(f"[ERROR] Gagal mengakses perangkat audio input (ID: {device}): {e}")
        print("   Gunakan flag --list-devices untuk melihat ID perangkat yang tersedia.\n")
        return

    dev_name = dev_info.get("name", "Unknown")
    channels = dev_info.get("max_input_channels", 0)
    capture_sr = get_supported_sample_rate(device)
    optimal_channels = get_optimal_channels(device, capture_sr)

    print(f"\n1. Perangkat Terpilih : [{device if device is not None else 'Default'}] {dev_name}")
    print(f"   Jumlah Channel     : {channels} (Menggunakan: {optimal_channels} channel)")
    print(f"   Hardware Sample Rate: {capture_sr} Hz {'(akan di-resample otomatis ke 16000 Hz untuk STT)' if capture_sr != 16000 else ''}")

    print(f"\n2. Merekam suara sampel selama {duration} detik...")
    print("   >> SILAKAN BERBICARA ATAU BUAT SUARA SEKARANG...")

    try:
        recording = sd.rec(
            int(duration * capture_sr),
            samplerate=capture_sr,
            channels=optimal_channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as e:
        print(f"\n[ERROR] Gagal merekam audio dari driver: {e}")
        print("   Tips Ubuntu Server:")
        print("   1. Pastikan user masuk ke grup audio: 'sudo usermod -aG audio $USER'")
        print("   2. Cek apakah ALSA mengenali mic USB: 'arecord -l'\n")
        return

    # Hitung RMS & Peak Amplitude
    samples_flat = recording.flatten()
    rms = float(np.sqrt(np.mean(samples_flat ** 2)))
    peak = float(np.max(np.abs(samples_flat)))

    print(f"\n3. Hasil Pengukuran Level Sinyal Audio:")
    print(f"   - Root Mean Square (RMS) Level : {rms:.5f}")
    print(f"   - Peak Amplitude (0.0 s/d 1.0) : {peak:.5f}")

    if peak < 0.01:
        print("\n[PERINGATAN] Sinyal suara SANGAT LEMAH atau HENING (MUTE)!")
        print("   Driver hardware terdeteksi, tetapi mikrofon tidak menangkap suara.")
        print("   Penyebab & Solusi Umum di Ubuntu Server:")
        print("   - Volume Capture ALSA di-mute atau bernilai 0%.")
        print("   - Jalankan 'alsamixer' di terminal -> Tekan F6 (pilih USB Mic) -> Tekan F4 (Capture) -> Naikkan volume tombol Panah Atas.")
    elif peak < 0.05:
        print("\n[CATATAN] Sinyal suara terdeteksi tetapi volumenya agak pelan.")
        print("   Disarankan menaikkan gain mikrofon lewat 'alsamixer'.")
    else:
        print("\n[BERHASIL] Driver & Mikrofon berfungsi NORMAL! Sinyal audio masuk dengan jelas.")

    # Simpan sampel ke file test_mic.wav
    try:
        import soundfile as sf
        sf.write("test_mic.wav", recording, capture_sr)
        print("   (File rekaman sampel berhasil disimpan ke 'test_mic.wav' untuk verifikasi)")
    except Exception:
        pass

    print("=" * 60 + "\n")



def print_result_box(text: str, result: dict):
    """Cetak hasil transkripsi dan NLU dalam format terminal yang rapi."""
    status = result.get("status", "UNKNOWN")
    intent = result.get("intent", "UNKNOWN")
    conf = result.get("confidence", 0.0)
    slots = result.get("slots", {})
    payload = result.get("command") or (
        {
            "action": intent.lower(),
            "params": slots,
            "confidence": round(conf, 2),
        } if status == "VALID" else {}
    )

    # ANSI Styling
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    if status == "VALID":
        status_colored = f"{GREEN}{status}{RESET}"
    elif status == "INVALID_LOCATION":
        status_colored = f"{RED}{status}{RESET}"
    else:
        status_colored = f"{YELLOW}{status}{RESET}"

    print(f"\n{BOLD}=========================== HASIL NLU ==========================={RESET}")
    print(f" {BOLD}Audio Transkripsi :{RESET} {CYAN}\"{text}\"{RESET}")
    print(f" {BOLD}Intent            :{RESET} {BOLD}{intent}{RESET} (Confidence: {conf:.2f})")
    print(f" {BOLD}Extracted Slots   :{RESET} {json.dumps(slots, ensure_ascii=False) if slots else '-'}")
    print(f" {BOLD}Validation Status :{RESET} {status_colored}")
    if "invalid_room" in result:
        print(f" {BOLD}Invalid Room      :{RESET} {RED}{result['invalid_room']}{RESET}")
    
    ros2_sent = result.get("ros2_sent", False)
    ros2_target = result.get("ros2_target", "")
    ros2_err = result.get("ros2_error", "")
    if ros2_sent:
        print(f" {BOLD}ROS 2 Status      :{RESET} {GREEN}✓ TERKIRIM{RESET} ke {ros2_target}")
    elif status == "VALID":
        print(f" {BOLD}ROS 2 Status      :{RESET} {RED}✕ GAGAL KIRIM{RESET} ke {ros2_target} ({ros2_err})")
    else:
        print(f" {BOLD}ROS 2 Status      :{RESET} {YELLOW}⏸ DITAHAN{RESET} (status {status})")

    print(f" {BOLD}ROS 2 Payload     :{RESET}")
    print(f"{DIM}{json.dumps(payload, indent=2, ensure_ascii=False)}{RESET}")
    print(f"{BOLD}================================================================={RESET}\n")


# =========================
# DIALOG & CONTEXT HELPERS
# =========================

def merge_dialog_context(command_text: str, answer_text: str, missing_slots: list) -> str:
    """
    Gabungkan perintah awal dan jawaban klarifikasi secara cerdas berdasarkan slot yang kurang.
    
    Contoh:
      - cmd: "antarkan obat ini", ans: "204"           -> "antarkan obat ini ke ruang 204"
      - cmd: "antarkan obat ini", ans: "ke ruang 204"  -> "antarkan obat ini ke ruang 204" (bukan double "ke")
      - cmd: "antarkan obat ini", ans: "laboratorium"  -> "antarkan obat ini ke laboratorium"
      - cmd: "tolong gerak",      ans: "kanan"         -> "tolong gerak ke kanan"
      - cmd: "tolong ikuti",      ans: "dokter budi"   -> "tolong ikuti dokter budi"
    """
    import re
    cmd = command_text.strip()
    ans = answer_text.strip().rstrip('.?!')
    if not ans:
        return cmd

    if "location" in missing_slots:
        # Jika user hanya menjawab angka (misal "204"), tambahkan prefix "ruang"
        if re.match(r'^\d+$', ans):
            ans = f"ruang {ans}"
        if ans.lower().startswith("ke "):
            return f"{cmd} {ans}"
        elif ans.lower().startswith("ruang "):
            return f"{cmd} ke {ans}"
        else:
            return f"{cmd} ke {ans}"

    elif "direction" in missing_slots:
        if ans.lower().startswith("ke "):
            return f"{cmd} {ans}"
        else:
            return f"{cmd} ke {ans}"

    elif "person" in missing_slots:
        return f"{cmd} {ans}"

    else:
        return f"{cmd} {ans}"


# Mapping nama slot internal → label Indonesia untuk TTS/display
_SLOT_DISPLAY_NAMES = {
    "location": "lokasi",
    "direction": "arah",
    "person": "orang",
    "distance": "jarak",
    "unit": "satuan",
    "object": "objek",
    "duration": "durasi",
}

def _translate_slot_names(slot_names: list) -> str:
    """Terjemahkan daftar nama slot ke bahasa Indonesia untuk TTS."""
    return ", ".join(_SLOT_DISPLAY_NAMES.get(s, s) for s in slot_names)


def get_confirmation_message(result: dict) -> tuple:
    """
    Mengembalikan tuple (robot_label_text, tts_speech_message) dari hasil validasi NLU.
    """
    vstatus = result["status"]
    intent = result.get("intent", "UNKNOWN")
    slots = result.get("slots", {})

    if vstatus == "VALID":
        if intent == "NAVIGATE":
            loc = slots.get('location', '')
            obj = slots.get('object', '')
            if obj and loc:
                return f"$ robot: mengantarkan {obj} ke {loc}", f"Baik, akan saya antarkan {obj} ke {loc}."
            elif obj:
                return f"$ robot: mengantarkan {obj}", f"Baik, akan saya antarkan {obj}."
            elif loc:
                return f"$ robot: menuju {loc}", responses.navigate_confirm(loc)
            else:
                return "$ robot: menuju lokasi", "Baik, saya akan menuju ke lokasi."
        elif intent == "MOVE":
            direction = slots.get("direction", "?")
            distance = slots.get("distance", "")
            unit = slots.get("unit", "")
            dist_str = f" {distance} {unit}" if distance else ""
            return f"$ robot: bergerak {direction}{dist_str}", f"Baik, saya bergerak ke {direction}{dist_str}."
        elif intent == "STOP":
            return "$ robot: berhenti", "Baik, saya berhenti."
        elif intent == "GO_BACK":
            return "$ robot: kembali", "Baik, saya kembali."
        elif intent == "WAIT":
            return "$ robot: menunggu", "Baik, saya akan menunggu."
        elif intent == "FOLLOW":
            person = slots.get('person', 'user')
            return f"$ robot: mengikuti {person}", f"Baik, saya akan mengikuti {person}."
        elif intent == "EMERGENCY":
            loc = slots.get('location', '')
            if loc:
                return f"$ robot: [DARURAT] menuju {loc}", f"Peringatan kondisi darurat. Segera menuju {loc} sekarang juga."
            return "$ robot: [DARURAT] mode darurat aktif", "Peringatan kondisi darurat. Mode prioritas darurat diaktifkan."
        elif intent == "STATUS":
            return "$ robot: status normal (baterai 85%)", "Status robot normal. Daya baterai delapan puluh lima persen dan siap beroperasi."
        elif intent == "GREETING":
            return "$ robot: menyapa", "Halo, ada yang bisa saya bantu?"
        elif intent == "THANK":
            return "$ robot: sama-sama", "Sama-sama, senang bisa membantu."
        else:
            return "$ robot: stand by", "Perintah diterima."
    elif vstatus == "INCOMPLETE":
        missing_raw = ", ".join(result.get("missing_slots", []))
        missing_id = _translate_slot_names(result.get("missing_slots", []))
        return f"$ robot: perintah tidak lengkap (kurang: {missing_raw})", f"Maaf, perintah belum lengkap. Informasi {missing_id} belum ada."
    elif vstatus == "INVALID_LOCATION":
        loc = slots.get('location', '?')
        return f'$ robot: lokasi "{loc}" tidak ditemukan', f"Maaf, lokasi {loc} tidak terdaftar di sistem."
    elif vstatus == "UNKNOWN_INTENT":
        msg = responses.unknown_intent()
        return f"$ robot: {msg.lower()}", msg
    else:
        return "$ robot: stand by", "Perintah diproses."


# =========================
# CLI AUDIO & TASK RUNNERS
# =========================

def listen_microphone_cli(duration: int = 5, device: int = None, prompt: str = "   Live Audio Stream: ") -> str:
    """
    Mendengarkan mikrofon satu kali (single-shot non-streaming / blocking) untuk CLI.
    Terintegrasi dengan scheduler (enter_stt, exit_stt, task pinning).
    """
    device = get_effective_input_device(device)
    import queue
    import sounddevice as sd
    import numpy as np
    import threading
    import stt as _stt

    if not _stt.is_ready():
        sys.stdout.write("   [STT] Menyiapkan model STT, mohon tunggu...")
        sys.stdout.flush()
        _t0 = time.time()
        _stt.get_recognizer()
        sys.stdout.write(f"\r   [STT] Model siap ({time.time() - _t0:.1f}s).            \n")
        sys.stdout.flush()
    recognizer = _stt.get_recognizer()
    create_stream = _stt.create_stream

    capture_rate = get_supported_sample_rate(device)
    channels = get_optimal_channels(device, capture_rate)
    stt_rate = 16000
    stream = create_stream()
    audio_q = queue.Queue()
    stop_event = threading.Event()
    local_stop = threading.Event()

    last_text = ""
    last_change_time = None

    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    stream_start_time = time.time()

    def audio_callback(indata, frames, time_info, status_flags):
        if local_stop.is_set() or stop_event.is_set():
            return
        audio_q.put(indata.copy())

    def decode_worker():
        nonlocal last_text, last_change_time
        set_thread_high_priority()
        while not stop_event.is_set() and not local_stop.is_set():
            try:
                indata = audio_q.get(timeout=0.1)
            except queue.Empty:
                if local_stop.is_set():
                    break
                continue

            if local_stop.is_set():
                break

            try:
                if indata.shape[1] > 1:
                    samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
                else:
                    samples = np.ascontiguousarray(indata.flatten(), dtype=np.float32)

                if capture_rate != stt_rate:
                    samples = resample_to_16k(samples, capture_rate)

                stream.accept_waveform(stt_rate, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                is_endpoint = recognizer.is_endpoint(stream)
                partial = recognizer.get_result(stream).strip()

                if partial:
                    if partial != last_text:
                        last_text = partial
                        last_change_time = time.time()
                    sys.stdout.write(f"\r   Live Audio Stream: \"\033[36m{partial}\033[0m...\" ")
                    sys.stdout.flush()

                if is_endpoint:
                    if partial:
                        last_text = partial
                        local_stop.set()
                        return  # Berhenti langsung saat endpoint terdeteksi (seperti mode streaming)
                    else:
                        recognizer.reset(stream)

            except Exception as err:
                logger.debug(f"[AUDIO] CLI decode error: {err}")

    if scheduler:
        scheduler.enter_stt()

    worker = threading.Thread(target=decode_worker, name="CLI-STT-Worker", daemon=True)
    worker.start()

    chunk_size = int(capture_rate * 0.1)
    try:
        with sd.InputStream(
            samplerate=capture_rate,
            channels=channels,
            dtype="float32",
            blocksize=chunk_size,
            device=device,
            callback=audio_callback,
        ):
            start = time.time()
            while time.time() - start < duration and not stop_event.is_set() and not local_stop.is_set():
                time.sleep(0.05)
                # Auto-stop jika pengguna sudah hening 2.2 detik setelah berucap (jeda berpikir alami)
                if last_text and last_change_time and (time.time() - last_change_time > 2.2):
                    local_stop.set()
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR Microphone]: {e}")
        return ""
    finally:
        local_stop.set()
        stop_event.set()
        worker.join(timeout=2)
        if scheduler:
            scheduler.exit_stt()

    text = (last_text or recognizer.get_result(stream)).strip()
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    return text


def record_and_process_cli(pipeline, duration: int = 5, device: int = None):
    """Rekam audio dari mikrofon dan proses real-time streaming di terminal."""
    print(f"\n[LISTENING] Silakan berbicara (maks {duration} detik)... Selesai bicara otomatis diproses.")
    text = listen_microphone_cli(duration=duration, device=device, prompt="   Live Audio Stream: ")
    print(f"\r   Transkripsi Akhir: \"\033[32m{text}\033[0m\"                                  \n")
    if not text:
        print("[WARN] Tidak ada ucapan/suara yang terdeteksi.")
        return

    result = pipeline.process(text)
    print_result_box(text, result)


def run_task_cli(duration: int = 5, device: int = None, ros2_host: str = None, ros2_port: int = None):
    """
    Mode Tugas Dialog Interaktif Non-Streaming Sequential di Terminal (CLI).

    Arsitektur Sequential & Penjadwalan Resource:
    - Mutual exclusion penuh: Audio Output (TTS) dan Audio Input (STT) tidak berebut hardware
    - Settling delay 150ms agar mikrofon tidak menangkap gema speaker
    - Audio caching: kalimat umum di-sintesis dengan 0 ms latency
    - Task pinning & scheduling priority diterapkan otomatis
    """
    from stt import preload_async as stt_preload_async
    from nlu.pipeline import NLUPipeline
    from tts import speak_and_wait, is_tts_available, warmup_async

    # Load model STT di background selagi greeting TTS diputar
    stt_preload_async()
    warmup_async()
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    tts_ok = is_tts_available()

    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    MAGENTA = "\033[1;35m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    print("\n" + "=" * 64)
    print("  MODE TUGAS DIALOG INTERAKTIF (SEQUENTIAL TTS ↔ STT)")
    print("  Resource-Optimized: Task Pinning & Audio Mutual Exclusion")
    print("=" * 64)

    try:
        # === STEP 1: Greeting TTS ===
        greeting = responses.greeting()
        print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{greeting}\"")
        if tts_ok:
            speak_and_wait(greeting)
        else:
            print("[WARN] Model TTS offline belum siap, melanjutkan tanpa suara robot...")

        # Mainkan nada lembut penanda mikrofon siap merekam
        try:
            from tts import play_mic_cue
            play_mic_cue()
        except Exception:
            pass

        # === STEP 2: Dengarkan Perintah Pengguna ===
        print(f"\n{CYAN}[🎤 ANDA]{RESET}  Silakan berbicara perintah Anda (maks {duration} detik)...")
        command_text = listen_microphone_cli(duration=duration, device=device, prompt="   Perintah: ")

        if not command_text:
            print(f"\n{RED}[WARN] Tidak ada suara terdeteksi.{RESET}")
            no_sound_msg = "Maaf, saya tidak mendengar perintah. Silakan coba lagi."
            print(f"{MAGENTA}[🤖 ROBOT]{RESET} \"{no_sound_msg}\"")
            if tts_ok:
                speak_and_wait(no_sound_msg)
            return

        print(f"\n{GREEN}[✓ ANDA]{RESET}  \"{command_text}\"")

        # === STEP 3: NLU Processing ===
        print(f"\n{YELLOW}[⚙ NLU]{RESET}   Memproses intent dan ekstraksi slot...")
        result = pipeline.process(command_text)

        # === STEP 4: Cek jika slot kurang (INCOMPLETE) ===
        if result["status"] == "INCOMPLETE" and result.get("missing_slots"):
            missing = result["missing_slots"]
            if "location" in missing:
                clarify_question = responses.ask_location()
            elif "direction" in missing:
                clarify_question = "Ke arah mana?"
            elif "person" in missing:
                clarify_question = "Siapa yang harus saya ikuti?"
            else:
                clarify_question = f"Mohon informasikan {_translate_slot_names(missing)}."

            print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{clarify_question}\"")
            if tts_ok:
                speak_and_wait(clarify_question)

            try:
                from tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            print(f"\n{CYAN}[🎤 ANDA]{RESET}  Silakan sebutkan jawaban Anda...")
            answer_text = listen_microphone_cli(duration=duration, device=device, prompt="   Jawaban: ")

            if answer_text:
                print(f"\n{GREEN}[✓ ANDA]{RESET}  \"{answer_text}\"")
                combined_text = merge_dialog_context(command_text, answer_text, missing)
                print(f"{YELLOW}[⚙ NLU]{RESET}   Konteks dialog digabungkan: \"{combined_text}\"")
                result = pipeline.process(combined_text)
                final_text = combined_text
            else:
                print(f"\n{RED}[WARN] Tidak ada jawaban diterima. Menggunakan hasil parsial.{RESET}")
                final_text = command_text
        else:
            final_text = command_text

        # === STEP 5: Tampilkan Hasil Akhir & TTS Konfirmasi ===
        print_result_box(final_text, result)
        robot_text, confirm_msg = get_confirmation_message(result)

        # Jika perintah VALID dan membawa objek penting, lakukan konfirmasi dua arah
        if result["status"] == "VALID" and result.get("intent") == "NAVIGATE" and result.get("slots", {}).get("object"):
            obj = result["slots"]["object"]
            loc = result["slots"].get("location", "")
            loc_str = f" ke {loc}" if loc else ""
            ask_confirm = f"Saya akan mengantarkan {obj}{loc_str}. Apakah sudah benar?"
            guide_confirm = "Ucapkan 'benar' untuk lanjut, atau 'tidak' untuk batal."
            print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{ask_confirm}\"")
            print(f"{YELLOW}[STATUS]{RESET}  ● Menunggu Konfirmasi: ({guide_confirm})")
            if tts_ok:
                speak_and_wait(ask_confirm)

            print(f"\n{CYAN}[🎤 ANDA]{RESET}  {guide_confirm}")
            yes_no_ans = listen_microphone_cli(duration=duration, device=device, prompt="   Konfirmasi (benar/tidak): ")
            ans_clean = (yes_no_ans or "").lower().strip()
            user_spoken = yes_no_ans.strip() if yes_no_ans else "(tidak ada suara terdeteksi)"

            print(f"\r   Ucapan Konfirmasi: \"\033[36m{user_spoken}\033[0m\"                                  \n")

            if any(w in ans_clean for w in ["tidak", "bukan", "batal", "jangan", "salah", "no"]):
                cancel_msg = "Baik, pengantaran dibatalkan."
                print(f"{RED}[✕ KONFIRMASI DIBATALKAN]{RESET} Kata terdeteksi: \"{user_spoken}\"")
                print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{cancel_msg}\"\n")
                if tts_ok:
                    speak_and_wait(cancel_msg)
                return
            else:
                # User menjawab ya / benar / setuju
                confirmed_word = user_spoken if yes_no_ans else "benar"
                print(f"{GREEN}[✓ KONFIRMASI DISETUJUI]{RESET} Kata terdeteksi: \"{confirmed_word}\"")

        print(f"{MAGENTA}[🤖 ROBOT]{RESET} \"{confirm_msg}\"\n")
        if tts_ok and confirm_msg:
            speak_and_wait(confirm_msg)

        # Follow-up kontekstual jika berhasil dieksekusi
        if result["status"] == "VALID" and result.get("intent") == "NAVIGATE":
            target_loc = result.get("slots", {}).get("location", "tujuan")
            arrival_msg = f"Robot telah tiba di {target_loc}. Ada yang bisa saya bantu lagi?"
            print(f"{CYAN}[INFO]{RESET} Robot dalam perjalanan... (simulasi navigasi)")
            time.sleep(1.0)  # Jeda natural simulasi perjalanan robot
            print(f"{MAGENTA}[🤖 ROBOT]{RESET} \"{arrival_msg}\"\n")
            if tts_ok:
                speak_and_wait(arrival_msg)

    except KeyboardInterrupt:
        print("\n[INFO] Mode tugas dibatalkan oleh pengguna.\n")


def run_cli_interactive(duration: int = 5, device: int = None, ros2_host: str = None, ros2_port: int = None):
    """Menu CLI interaktif untuk terminal headless / Ubuntu Server."""
    from stt import preload_async as stt_preload_async
    from nlu.pipeline import NLUPipeline
    from tts import warmup_async

    # Load model STT di background selagi menu ditampilkan (agar siap saat mode dipilih)
    stt_preload_async()
    # Prewarm audio cache & engine di background
    warmup_async()
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)

    print("\n" + "=" * 58)
    print("  ROBOT VOICE COMMAND (CLI / HEADLESS MODE)")
    print("  Optimal untuk Ubuntu Server & Raspberry Pi 4")
    print("=" * 58)

    while True:
        print("\nPilih Mode Operasi:")
        print("  [1] Mode Tugas Dialog Interaktif (TTS ↔ STT Sequential)")
        print("  [2] Bicara Langsung via Mikrofon (Streaming STT + NLU)")
        print("  [3] Ketik Perintah Teks Manual")
        print("  [4] List Perangkat Input Audio")
        print("  [5] Tes Diagnosa & Level Volume Mikrofon")
        print("  [q] Keluar")

        try:
            choice = input("\nPilihan [1/2/3/4/5/q]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar...")
            break

        if choice == "1":
            run_task_cli(duration=duration, device=device, ros2_host=ros2_host, ros2_port=ros2_port)
        elif choice == "2":
            record_and_process_cli(pipeline, duration=duration, device=device)
        elif choice == "3":
            try:
                txt = input("\nMasukkan perintah (misal: 'tolong antar saya ke ruang ICU'): ").strip()
                if txt:
                    result = pipeline.process(txt)
                    print_result_box(txt, result)
            except (KeyboardInterrupt, EOFError):
                pass
        elif choice == "4":
            list_audio_devices()
        elif choice == "5":
            test_mic_hardware(device=device, duration=duration)
        elif choice in ["q", "quit", "exit"]:
            print("Keluar dari program. Sampai jumpa!\n")
            break
        else:
            print("Pilihan tidak valid.")


# =========================
# TEXT MODE
# =========================

def run_text_mode(text: str, ros2_host: str = None, ros2_port: int = None):
    """Process text langsung melalui NLU pipeline (tanpa STT)."""

    from nlu.pipeline import NLUPipeline

    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    result = pipeline.process(text)
    print_result_box(text, result)


# =========================
# GUI MODE
# =========================

def run_gui_mode(duration: int = 5, device: int = None, ros2_host: str = None, ros2_port: int = None):
    """Jalankan GUI Tkinter dengan microphone + real-time streaming STT + NLU pipeline."""

    try:
        import tkinter as tk
        from tkinter import messagebox
        import tkinter.font as tkfont
        root = tk.Tk()
    except Exception as e:
        logger.warning(f"\n[WARN] Gagal membuka GUI Tkinter (tidak ada Display Server / $DISPLAY): {e}")
        logger.info("[INFO] Beralih otomatis ke Mode CLI Interaktif (Terminal)...\n")
        run_cli_interactive(duration=duration, device=device, ros2_host=ros2_host, ros2_port=ros2_port)
        return

    device = get_effective_input_device(device)
    import sounddevice as sd
    import threading
    import queue as _queue

    # `recognizer` adalah proxy lazy: import instan, model dimuat di background
    from stt import recognizer, create_stream, preload_async as stt_preload_async
    from stt import is_ready as stt_is_ready
    stt_preload_async()
    from nlu.pipeline import NLUPipeline
    from nlu.ros2_adapter import save_ros2_config, SocketROS2Adapter

    # Audio Config
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_SECONDS = duration  # Max duration (dapat dihentikan lebih awal via tombol)

    # NLU Pipeline instance
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    ros2_adapter = pipeline.ros2_adapter

    # Prewarm model TTS & cache audio di background
    try:
        from tts import warmup_async
        warmup_async()
    except Exception:
        pass

    # =========================
    # GUI INITIALIZATION & FONT
    # =========================
    root.title("Robot Voice Assistant • Control Center")
    root.geometry("960x860")
    root.minsize(900, 780)

    # Deteksi modern fonts di Windows / Linux
    installed_fonts = set(tkfont.families())
    FONT_FAMILY_UI = "Segoe UI" if "Segoe UI" in installed_fonts else ("Arial" if "Arial" in installed_fonts else "Helvetica")
    FONT_FAMILY_CODE = "Cascadia Code" if "Cascadia Code" in installed_fonts else (
        "JetBrains Mono" if "JetBrains Mono" in installed_fonts else "Consolas"
    )

    F_TITLE = (FONT_FAMILY_UI, 13, "bold")
    F_SUBTITLE = (FONT_FAMILY_UI, 9)
    F_SECTION = (FONT_FAMILY_UI, 9, "bold")
    F_BTN_TITLE = (FONT_FAMILY_UI, 10, "bold")
    F_CODE = (FONT_FAMILY_CODE, 9)
    F_CODE_BOLD = (FONT_FAMILY_CODE, 9, "bold")
    F_ROBOT = (FONT_FAMILY_UI, 10, "bold")

    # =========================
    # COLORS (Modern Clean Studio / Calm Slate Palette)
    # Sangat nyaman di mata, matte modern, bukan neon cyberpunk
    # =========================
    BG_BASE = "#111317"          # Deep warm charcoal slate
    BG_CARD = "#191c23"          # Elevated matte card surface
    BG_CARD_ALT = "#212530"      # Inner container & inputs
    BG_CARD_GLOW = "#1d222d"     # Robot execution banner container
    BORDER = "#2b3140"           # Subtle hairline separator
    BORDER_HOVER = "#40495d"     # Subtle border hover
    FG_WHITE = "#f1f5f9"         # Crisp readable soft white
    FG_SUB = "#94a3b8"           # Slate secondary text
    FG_MUTED = "#64748b"         # Subtle hint / muted text

    # Status & Accent Colors (Calm, desaturated, gentle on eyes)
    COLOR_PRIMARY = "#6366f1"    # Modern Indigo
    COLOR_PRIMARY_BG = "#1e2238"
    COLOR_PRIMARY_BORDER = "#3730a3"

    COLOR_SKY = "#38bdf8"        # Soft Sky Blue
    COLOR_SKY_BG = "#142533"
    COLOR_SKY_BORDER = "#0369a1"

    COLOR_GREEN = "#10b981"      # Calm Emerald Green
    COLOR_GREEN_BG = "#132d24"
    COLOR_GREEN_BORDER = "#047857"

    COLOR_AMBER = "#f59e0b"      # Soft Honey Amber
    COLOR_AMBER_BG = "#2e2415"
    COLOR_AMBER_BORDER = "#b45309"

    COLOR_RED = "#f43f5e"        # Soft Rose Red
    COLOR_RED_BG = "#311820"
    COLOR_RED_BORDER = "#be123c"

    COLOR_PURPLE = "#a78bfa"     # Soft Lilac / Violet
    COLOR_PURPLE_BG = "#262138"
    COLOR_PURPLE_BORDER = "#6d28d9"

    root.configure(bg=BG_BASE)

    # State tracking & View switching
    is_recording = False
    is_task_running = False
    stop_event = threading.Event()
    record_thread = None
    task_thread = None
    current_view = "user"  # Default: Tampilan Pengguna (Kiosk)
    current_quick_mode = "arah"  # Opsi mode tujuan cepat: "arah" (Penunjuk Arah) atau "obat" (Antar Obat)

    # End-User UI Widget References
    user_face_label = None
    user_status_title = None
    user_status_sub = None
    user_action_pill = None
    user_voice_btn = None
    user_chat_box = None
    user_live_speech_label = None
    user_chat_status_label = None

    # Admin UI Widget References
    dialog_history_box = None
    ros2_status_badge = None
    ros2_detail_label = None
    ros2_title_label = None

    # Thread-safe UI update helper
    def ui_call(fn):
        root.after(0, fn)

    def update_ros2_ui_status(result: dict):
        """Update indikator visual dan log transmisi payload ROS 2 di Mode Teknisi."""
        def _apply():
            if not ros2_status_badge or not ros2_detail_label:
                return

            sent = result.get("ros2_sent", False)
            cur_h = getattr(pipeline.ros2_adapter, "host", "10.107.245.221")
            cur_p = getattr(pipeline.ros2_adapter, "port", 5000)
            target = result.get("ros2_target") or f"{cur_h}:{cur_p}"
            err = result.get("ros2_error")
            status = result.get("status", "UNKNOWN")
            intent = result.get("intent", "-")
            payload = result.get("ros2_payload", "")

            if sent:
                ros2_status_badge.config(
                    text="● TERKIRIM ✓",
                    fg=COLOR_GREEN,
                    bg=COLOR_GREEN_BG,
                    highlightbackground=COLOR_GREEN_BORDER,
                )
                ros2_detail_label.config(
                    text=f"✓ Payload berhasil dikirim ke Node ROS 2 [{target}] (Intent: {intent})",
                    fg=COLOR_GREEN,
                )
                log_msg = f"✓ Payload BERHASIL dikirim ke {target} | Intent: {intent}"
                if payload:
                    log_msg += f"\n   Payload: {payload}"
                append_dialog_history("🌐 ROS 2", log_msg, COLOR_GREEN)
            elif status == "VALID":
                ros2_status_badge.config(
                    text="✕ GAGAL KIRIM",
                    fg=COLOR_RED,
                    bg=COLOR_RED_BG,
                    highlightbackground=COLOR_RED_BORDER,
                )
                err_text = f" ({err})" if err else ""
                ros2_detail_label.config(
                    text=f"✕ Gagal terhubung ke Node ROS 2 [{target}]{err_text}",
                    fg=COLOR_RED,
                )
                log_msg = f"✕ GAGAL kirim payload ke {target}{err_text}"
                if payload:
                    log_msg += f"\n   Payload tertahan: {payload}"
                append_dialog_history("🌐 ROS 2", log_msg, COLOR_RED)
            else:
                ros2_status_badge.config(
                    text="⏸ DITAHAN",
                    fg=COLOR_AMBER,
                    bg=COLOR_AMBER_BG,
                    highlightbackground=COLOR_AMBER_BORDER,
                )
                ros2_detail_label.config(
                    text=f"⏸ Payload ditahan (status validasi: {status})",
                    fg=COLOR_AMBER,
                )
                append_dialog_history("🌐 ROS 2", f"⏸ Payload DITAHAN (status={status}, belum valid dieksekusi)", COLOR_AMBER)

        ui_call(_apply)

    def set_user_ui_state(face: str, title: str, subtitle: str, color: str = COLOR_GREEN):
        """Update tampilan persona, status, dan badge pada antarmuka End-User."""
        def _apply():
            if user_face_label:
                user_face_label.config(text=face, fg=color)
            if user_status_title:
                user_status_title.config(text=title)
            if user_status_sub:
                user_status_sub.config(text=subtitle)
            if user_action_pill:
                user_action_pill.config(text=f"● {title}", fg=color)
            if user_live_speech_label:
                user_live_speech_label.config(text=f"🎙️ Status: {title}", fg=color)
            if user_chat_status_label:
                user_chat_status_label.config(text=f"● {title}", fg=color)
        ui_call(_apply)

    # Helper untuk hover effect pada button
    def add_hover_effect(widget, normal_bg, hover_bg, normal_border=None, hover_border=None):
        def on_enter(e):
            if str(widget.cget("state")) != "disabled":
                widget.config(bg=hover_bg)
                if hover_border:
                    widget.config(highlightbackground=hover_border, highlightcolor=hover_border)
        def on_leave(e):
            if str(widget.cget("state")) != "disabled":
                widget.config(bg=normal_bg)
                if normal_border:
                    widget.config(highlightbackground=normal_border, highlightcolor=normal_border)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # =========================
    # NON-STREAMING STT (single-shot recording untuk interactive mode)
    # =========================

    def stt_listen_once(
        listen_duration: int = 8,
        stage_title: str = "MENDENGARKAN PERINTAH",
        context_hint: str = "Silakan berbicara langsung ke mikrofon...",
    ) -> str:
        """
        Rekam audio dan transkripsi secara non-streaming (blocking).
        Mengembalikan teks hasil STT.

        Arsitektur non-streaming menghindari race condition:
        - Tidak ada concurrent consumer audio output
        - TTS selesai dulu → baru STT mulai merekam
        - STT selesai dulu → baru proses NLU
        """
        if not stt_is_ready():
            ui_call(lambda: status_label.config(text="● Menyiapkan model STT...", fg=COLOR_AMBER))
            ui_call(lambda: stt_label.config(text="(menyiapkan model...)", fg=COLOR_AMBER))
            create_stream()
            ui_call(lambda: stt_label.config(text="(mendengarkan...)", fg=COLOR_SKY))
        stream = create_stream()
        audio_q = _queue.Queue()
        local_stop = threading.Event()

        capture_rate = get_supported_sample_rate(device)
        optimal_channels = get_optimal_channels(device, capture_rate)

        last_text = ""
        last_change_time = None

        def audio_cb(indata, frames, time_info, status):
            if local_stop.is_set() or stop_event.is_set():
                return
            audio_q.put(indata.copy())

        def decode_worker():
            nonlocal last_text, last_change_time
            set_thread_high_priority()
            while not stop_event.is_set() and not local_stop.is_set():
                try:
                    indata = audio_q.get(timeout=0.1)
                except _queue.Empty:
                    if local_stop.is_set():
                        break
                    continue

                if local_stop.is_set():
                    break

                try:
                    if indata.shape[1] > 1:
                        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
                    else:
                        samples = np.ascontiguousarray(indata.flatten(), dtype=np.float32)

                    if capture_rate != SAMPLE_RATE:
                        samples = resample_to_16k(samples, capture_rate)

                    stream.accept_waveform(SAMPLE_RATE, samples)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    is_endpoint = recognizer.is_endpoint(stream)
                    partial = recognizer.get_result(stream).strip()

                    if partial:
                        if partial != last_text:
                            last_text = partial
                            last_change_time = time.time()

                        # 1. Log Real-Time Terminal (Stdout ANSI realtime ticker)
                        sys.stdout.write(f"\r   [STT REAL-TIME] \"\033[36m{partial}\033[0m...\" ")
                        sys.stdout.flush()

                        # 2. Update Live UI di Mode Teknisi dan Mode Pengguna
                        def _update_live_ui(p=partial):
                            # Mode Teknisi
                            stt_label.config(text=f'“{p}...”', fg=COLOR_SKY)
                            status_label.config(text=f'● Mendeteksi: “{p}...”', fg=COLOR_SKY)

                            # Mode Pengguna (Indikator visual real-time langsung saat pengguna berucap)
                            if user_voice_btn:
                                user_voice_btn.config(
                                    text=f"🎙️   SUARA TERDETEKSI: “{p}...”\n({context_hint})",
                                    fg=COLOR_GREEN,
                                    bg=COLOR_GREEN_BG,
                                    highlightbackground=COLOR_GREEN_BORDER,
                                )
                            if user_action_pill:
                                user_action_pill.config(
                                    text=f"● Suara Terdeteksi: “{p}...”",
                                    fg=COLOR_GREEN,
                                )
                            if user_live_speech_label:
                                user_live_speech_label.config(
                                    text=f"🎙️ Terdeteksi: “{p}...”",
                                    fg=COLOR_GREEN,
                                )
                            if user_face_label:
                                user_face_label.config(text="(👂◕‿◕)", fg=COLOR_GREEN)
                            if user_status_sub:
                                user_status_sub.config(text=f'Robot mendengar: “{p}...”', fg=COLOR_SKY)
                            if user_chat_status_label:
                                user_chat_status_label.config(text=f'● Mendengar: “{p}...”', fg=COLOR_GREEN)
                        ui_call(_update_live_ui)

                    if is_endpoint:
                        if partial:
                            last_text = partial
                            local_stop.set()
                            return  # Langsung berhenti saat endpoint terdeteksi (identik dengan mode streaming)
                        else:
                            recognizer.reset(stream)

                except Exception as err:
                    logger.debug(f"[AUDIO] listen_once decode error: {err}")

        worker = threading.Thread(target=decode_worker, name="GUI-STT-Worker", daemon=True)
        worker.start()

        if scheduler:
            scheduler.enter_stt()

        chunk_size = int(capture_rate * 0.1)
        try:
            with sd.InputStream(
                samplerate=capture_rate,
                channels=optimal_channels,
                dtype="float32",
                blocksize=chunk_size,
                device=device,
                callback=audio_cb,
            ):
                # Sinyal visual instan saat mikrofon hardware telah aktif
                def _on_mic_ready():
                    # Mode Teknisi
                    stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY)
                    status_label.config(text=f"● {stage_title}: Mikrofon aktif...", fg=COLOR_SKY)

                    # Mode Pengguna
                    if user_voice_btn:
                        user_voice_btn.config(
                            text=f"🎙️   MIKROFON AKTIF — MENDENGARKAN...\n{context_hint}",
                            fg=COLOR_SKY,
                            bg=COLOR_SKY_BG,
                            highlightbackground=COLOR_SKY_BORDER,
                        )
                    if user_action_pill:
                        user_action_pill.config(
                            text="● Mikrofon Terbuka — Silakan Berbicara",
                            fg=COLOR_SKY,
                        )
                    if user_live_speech_label:
                        user_live_speech_label.config(
                            text=f"🎙️ [MIKROFON TERBUKA] {context_hint}",
                            fg=COLOR_SKY,
                        )
                    if user_face_label:
                        user_face_label.config(text="(👂‿👂)", fg=COLOR_SKY)
                    if user_chat_status_label:
                        user_chat_status_label.config(text="● Mendengarkan...", fg=COLOR_SKY)
                ui_call(_on_mic_ready)

                sys.stdout.write(f"\n   [🎤 MIKROFON AKTIF] {stage_title} ({context_hint})\n")
                sys.stdout.flush()

                start_t = time.time()
                while (time.time() - start_t < listen_duration
                       and not local_stop.is_set()
                       and not stop_event.is_set()):
                    time.sleep(0.05)
                    # Beri waktu hening 2.2 detik agar jeda berpikir manusia tidak memotong kalimat di tengah
                    if last_text and last_change_time and (time.time() - last_change_time > 2.2):
                        local_stop.set()
                        break
        except Exception as e:
            logger.error(f"[STT] listen_once InputStream error: {e}")
            return ""
        finally:
            local_stop.set()
            worker.join(timeout=2)
            if scheduler:
                scheduler.exit_stt()

        text = (last_text or recognizer.get_result(stream)).strip()
        if text:
            sys.stdout.write(f"\r   [✓ STT SELESAI] \"\033[32m{text}\033[0m\"" + " " * 30 + "\n")
        else:
            sys.stdout.write("\r   [✕ STT SELESAI] (tidak ada suara terdeteksi)" + " " * 30 + "\n")
        sys.stdout.flush()
        return text

    # =========================
    # DIALOG HISTORY APPEND HELPER (SYNC DUAL-VIEW)
    # =========================
    def append_dialog_history(speaker: str, message: str, color: str = None):
        """Tambahkan baris percakapan ke riwayat Admin Console dan Chat Balon Pengguna."""
        def _append():
            import datetime
            now_t = datetime.datetime.now().strftime("%H:%M:%S")

            # 1. Update Admin Console (jika sudah di-render)
            if dialog_history_box:
                try:
                    dialog_history_box.config(state="normal")
                    spk_lower = speaker.lower()
                    if "robot" in spk_lower:
                        spk_tag = "tag_robot"
                    elif "anda" in spk_lower:
                        spk_tag = "tag_user"
                    elif "manual" in spk_lower:
                        spk_tag = "tag_manual"
                    elif "ros" in spk_lower:
                        spk_tag = "tag_ros2"
                    else:
                        spk_tag = "tag_info"

                    msg_tag = f"tag_msg_{color.replace('#', '')}" if color else "tag_info"
                    if color:
                        dialog_history_box.tag_configure(msg_tag, foreground=color)

                    dialog_history_box.insert(tk.END, f"[{now_t}] ", "tag_time")
                    dialog_history_box.insert(tk.END, f"{speaker}: ", spk_tag)
                    dialog_history_box.insert(tk.END, f"{message}\n", msg_tag)
                    dialog_history_box.see(tk.END)
                    dialog_history_box.config(state="disabled")
                except Exception:
                    pass

            # 2. Update End-User Chat Display (jika sudah di-render)
            if user_chat_box:
                # Sembunyikan log teknis transmisi ROS 2 dari kiosk pengguna
                if "ros" in speaker.lower():
                    return

                try:
                    user_chat_box.config(state="normal")
                    spk_lower = speaker.lower()
                    if "robot" in spk_lower:
                        role_label = "🤖 Robot: "
                        role_tag = "user_tag_robot"
                    elif "anda" in spk_lower or "manual" in spk_lower:
                        role_label = "👤 Anda: "
                        role_tag = "user_tag_user"
                    else:
                        role_label = "ℹ️ "
                        role_tag = "user_tag_info"

                    user_chat_box.insert(tk.END, f"[{now_t}] ", "user_tag_time")
                    user_chat_box.insert(tk.END, role_label, role_tag)
                    user_chat_box.insert(tk.END, f"{message}\n\n", "user_tag_body")
                    user_chat_box.see(tk.END)
                    user_chat_box.config(state="disabled")
                except Exception:
                    pass

        ui_call(_append)

    def execute_quick_destination(dest_name: str, task_mode: str = None):
        """Eksekusi tujuan cepat sekali sentuh oleh end-user (misal: Ruang ICU) sesuai mode objek terpilih."""
        if is_recording or is_task_running:
            return

        mode = task_mode or current_quick_mode
        if mode == "obat":
            cmd = f"antar obat ke {dest_name}"
            title = f"Mengantar Obat ke {dest_name.title()}..."
            sub = f"Robot membawa pasokan obat dan bersiap menuju {dest_name}."
            face = "(📦‿📦)"
            color = COLOR_GREEN
        else:
            cmd = f"antar saya ke {dest_name}"
            title = f"Menuju {dest_name.title()}..."
            sub = f"Robot sedang bersiap memandu dan mengantar Anda ke {dest_name}."
            face = "(★‿★)"
            color = COLOR_SKY

        set_user_ui_state(face, title, sub, color)
        append_dialog_history("👤 Anda", cmd, COLOR_SKY)
        ui_call(lambda: status_label.config(text=f"● MEMPROSES TUJUAN CEPAT: {cmd}...", fg=COLOR_AMBER))

        def _worker():
            res = pipeline.process(cmd)
            try:
                from tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            _show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    def execute_quick_action(action_cmd: str):
        """Eksekusi aksi cepat sekali sentuh (misal: Berhenti atau Kembali)."""
        if is_recording or is_task_running:
            stop_event.set()
        append_dialog_history("👤 Anda", action_cmd, COLOR_SKY)
        def _worker():
            res = pipeline.process(action_cmd)
            try:
                from tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            _show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    # =========================
    # INTERACTIVE TASK MODE (sequential TTS↔STT)
    # =========================

    def interactive_task_flow():
        nonlocal is_task_running

        try:
            from tts import speak_and_wait, is_tts_available
        except ImportError:
            logger.error("[TASK] Module tts.py tidak ditemukan")
            ui_call(lambda: status_label.config(text="● Modul TTS tidak ditemukan", fg=COLOR_RED))
            is_task_running = False
            _reset_task_button()
            return

        tts_ok = is_tts_available()

        def update_conv_log(speaker: str, text: str, color: str = FG_WHITE):
            ui_call(lambda: conv_log_label.config(
                text=f"{speaker}: {text}", fg=color
            ))
            append_dialog_history(speaker, text, color)

        try:
            # === STEP 1: Greeting TTS ===
            greeting = responses.greeting()
            set_user_ui_state("(＾▽＾)", "Robot menyapa Anda...", greeting, COLOR_PURPLE)
            ui_call(lambda: status_label.config(text="● Robot menyapa...", fg=COLOR_AMBER))
            ui_call(lambda: status_badge.config(
                text="● BERBICARA", fg=COLOR_AMBER, bg=COLOR_AMBER_BG, highlightbackground=COLOR_AMBER_BORDER
            ))
            update_conv_log("🤖 Robot", greeting, COLOR_AMBER)
            ui_call(lambda: stt_label.config(text='(menunggu giliran bicara...)', fg=FG_MUTED))
            ui_call(lambda: intent_label.config(text="🎯 INTENT: -", fg=FG_MUTED))
            ui_call(lambda: slots_label.config(text="📦 PARAMETER: -", fg=FG_MUTED))
            ui_call(lambda: validation_label.config(text="🛡️ VALIDASI: -", fg=FG_MUTED))
            ui_call(lambda: robot_label.config(text="Robot: Berbicara ke pengguna...", fg=COLOR_AMBER))

            if tts_ok:
                speak_and_wait(greeting)
            else:
                logger.warning("[TASK] TTS tidak tersedia, skip greeting audio")
                time.sleep(1)

            if stop_event.is_set():
                return

            # Mainkan nada beep penanda mikrofon siap merekam
            try:
                from tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            # === STEP 2: Listen for command ===
            set_user_ui_state("(👂‿👂)", "Silakan Bicara Sekarang 🎙️", "Mikrofon aktif — sebutkan tujuan atau instruksi Anda.", COLOR_SKY)
            ui_call(lambda: status_label.config(text="● Mikrofon aktif — silakan berbicara...", fg=COLOR_SKY))
            ui_call(lambda: status_badge.config(
                text="● MENDENGARKAN", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
            ))
            ui_call(lambda: stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY))
            update_conv_log("🎤 Anda", "(sedang mendengarkan...)", COLOR_SKY)
            ui_call(lambda: robot_label.config(text="Robot: Mendengarkan perintah suara Anda", fg=COLOR_SKY))

            command_text = stt_listen_once(
                listen_duration=RECORD_SECONDS + 3,
                stage_title="MENDENGARKAN PERINTAH",
                context_hint="Katakan tujuan Anda, misal: 'Antar saya ke ruang ICU'",
            )

            if stop_event.is_set():
                return

            if not command_text:
                set_user_ui_state("(•_•)", "Tidak ada suara terdeteksi", "Silakan coba lagi atau pilih tujuan di bawah.", COLOR_RED)
                ui_call(lambda: status_label.config(text="● Tidak ada suara terdeteksi", fg=COLOR_RED))
                ui_call(lambda: status_badge.config(
                    text="● TIDAK ADA SUARA", fg=COLOR_RED, bg=COLOR_RED_BG, highlightbackground=COLOR_RED_BORDER
                ))
                ui_call(lambda: stt_label.config(text='(tidak ada suara terdeteksi)', fg=COLOR_RED))
                update_conv_log("🎤 Anda", "(tidak ada suara terdeteksi)", COLOR_RED)
                ui_call(lambda: robot_label.config(text="Robot: Tidak ada instruksi yang terdengar", fg=COLOR_RED))

                no_sound_msg = "Maaf, saya tidak mendengar perintah. Silakan coba lagi."
                update_conv_log("🤖 Robot", no_sound_msg, COLOR_RED)
                if tts_ok:
                    speak_and_wait(no_sound_msg)
                return

            logger.info(f"[TASK] Command text: \"{command_text}\"")
            ui_call(lambda: stt_label.config(text=f'“{command_text}”', fg=FG_WHITE))
            update_conv_log("🎤 Anda", f'"{command_text}"', COLOR_GREEN)

            # === STEP 3: NLU Processing ===
            set_user_ui_state("(・_・)", "Memproses...", f"Robot sedang memahami: \"{command_text}\"", COLOR_AMBER)
            ui_call(lambda: status_label.config(text="● Memproses pemahaman bahasa (NLU)...", fg=COLOR_AMBER))
            result = pipeline.process(command_text)

            if stop_event.is_set():
                return

            # === STEP 4: Check if INCOMPLETE (missing slots) ===
            if result["status"] == "INCOMPLETE" and result.get("missing_slots"):
                missing = result["missing_slots"]
                logger.info(f"[TASK] Slot belum lengkap: {missing}")

                if "location" in missing:
                    clarify_question = responses.ask_location()
                elif "direction" in missing:
                    clarify_question = "Ke arah mana?"
                elif "person" in missing:
                    clarify_question = "Siapa yang harus saya ikuti?"
                else:
                    clarify_question = f"Mohon informasikan {_translate_slot_names(missing)}."

                _show_partial_result(result)

                set_user_ui_state("(・_・)", "Mohon informasikan tujuan", clarify_question, COLOR_AMBER)
                ui_call(lambda: status_label.config(text="● Robot meminta klarifikasi...", fg=COLOR_AMBER))
                ui_call(lambda: status_badge.config(
                    text="● KLARIFIKASI", fg=COLOR_AMBER, bg=COLOR_AMBER_BG, highlightbackground=COLOR_AMBER_BORDER
                ))
                update_conv_log("🤖 Robot", clarify_question, COLOR_AMBER)
                ui_call(lambda: robot_label.config(text=f"Robot: \"{clarify_question}\"", fg=COLOR_AMBER))

                if tts_ok:
                    speak_and_wait(clarify_question)
                else:
                    time.sleep(0.5)

                if stop_event.is_set():
                    return

                # Mainkan nada beep penanda mikrofon siap untuk klarifikasi
                try:
                    from tts import play_mic_cue
                    play_mic_cue()
                except Exception:
                    pass

                # STT dengarkan jawaban
                set_user_ui_state("(👂‿👂)", "Silakan Jawab Sekarang 🎙️", clarify_question, COLOR_SKY)
                ui_call(lambda: status_label.config(text="● Mendengarkan jawaban klarifikasi...", fg=COLOR_SKY))
                ui_call(lambda: status_badge.config(
                    text="● MENDENGARKAN", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
                ))
                ui_call(lambda: stt_label.config(text='(mendengarkan jawaban klarifikasi...)', fg=COLOR_SKY))
                update_conv_log("🎤 Anda", "(mendengarkan jawaban...)", COLOR_SKY)

                answer_text = stt_listen_once(
                    listen_duration=RECORD_SECONDS + 3,
                    stage_title="KLARIFIKASI TUJUAN",
                    context_hint=clarify_question,
                )

                if stop_event.is_set():
                    return

                if answer_text:
                    logger.info(f"[TASK] Answer text: \"{answer_text}\"")
                    ui_call(lambda: stt_label.config(text=f'“{answer_text}”', fg=FG_WHITE))
                    update_conv_log("🎤 Anda", f'"{answer_text}"', COLOR_GREEN)

                    combined_text = merge_dialog_context(command_text, answer_text, missing)
                    logger.info(f"[TASK] Combined text: \"{combined_text}\"")

                    ui_call(lambda: status_label.config(text="● Memproses ulang NLU...", fg=COLOR_AMBER))
                    result = pipeline.process(combined_text)
                else:
                    logger.info("[TASK] Tidak ada jawaban — gunakan hasil parsial")
                    update_conv_log("🎤 Anda", "(tidak ada jawaban)", COLOR_RED)

            # === STEP 5: Display final result ===
            _show_final_result(result, tts_ok)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[TASK ERROR] {err_msg}")
            ui_call(lambda: status_label.config(text="● Terjadi kesalahan pada sistem", fg=COLOR_RED))
            ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            is_task_running = False
            stop_event.clear()
            _reset_task_button()

    def _show_partial_result(result):
        """Tampilkan hasil NLU parsial (saat masih ada slot yang kurang)."""
        intent_text = f"INTENT: {result['intent']} ({result['confidence']:.2f})"
        slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(kosong)"
        missing_id = _translate_slot_names(result.get("missing_slots", []))

        def apply_partial():
            intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
            slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
            validation_label.config(text=f"⚠ KURANG: {missing_id}", fg=COLOR_AMBER)
        ui_call(apply_partial)

    def _show_final_result(result, tts_ok=False):
        """Tampilkan hasil NLU final di GUI, mainkan TTS respons dan kedatangan."""
        intent_text = f"INTENT: {result['intent']} ({result['confidence']:.2f})"
        slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(kosong)"
        vstatus = result["status"]

        if vstatus == "VALID":
            vcolor = COLOR_GREEN
            vicon = "✓"
        elif vstatus == "INVALID_LOCATION":
            vcolor = COLOR_RED
            vicon = "✕"
        else:
            vcolor = COLOR_AMBER
            vicon = "⚠"

        confirm_msg = ""
        if vstatus == "VALID":
            rcolor = COLOR_GREEN
            intent = result["intent"]
            slots = result["slots"]
            if intent == "NAVIGATE":
                loc = slots.get('location', '')
                obj = slots.get('object', '')
                if obj and loc:
                    rtext = f"Robot: Mengantarkan {obj} ke {loc}"
                    confirm_msg = f"Baik, akan saya antarkan {obj} ke {loc}."
                elif obj:
                    rtext = f"Robot: Mengantarkan {obj}"
                    confirm_msg = f"Baik, akan saya antarkan {obj}."
                elif loc:
                    rtext = f"Robot: Menuju {loc}"
                    confirm_msg = responses.navigate_confirm(loc)
                else:
                    rtext = "Robot: Menuju lokasi"
                    confirm_msg = "Baik, saya akan menuju ke lokasi."
                set_user_ui_state("(★‿★)", f"Mengantar ke {loc or 'tujuan'}...", confirm_msg, COLOR_GREEN)
            elif intent == "MOVE":
                direction = slots.get("direction", "?")
                distance = slots.get("distance", "")
                unit = slots.get("unit", "")
                dist_str = f" {distance} {unit}" if distance else ""
                rtext = f"Robot: Bergerak {direction}{dist_str}"
                confirm_msg = f"Baik, saya bergerak ke {direction}{dist_str}."
                set_user_ui_state("(🚗)", f"Bergerak {direction}", confirm_msg, COLOR_GREEN)
            elif intent == "STOP":
                rtext = "Robot: Berhenti"
                confirm_msg = "Baik, saya berhenti."
                set_user_ui_state("(🛑)", "Robot Berhenti", confirm_msg, COLOR_RED)
            elif intent == "GO_BACK":
                rtext = "Robot: Kembali ke posisi awal"
                confirm_msg = "Baik, saya kembali."
                set_user_ui_state("(🏠)", "Kembali ke Pos", confirm_msg, COLOR_GREEN)
            elif intent == "WAIT":
                rtext = "Robot: Menunggu"
                confirm_msg = "Baik, saya akan menunggu."
                set_user_ui_state("(⏳)", "Menunggu", confirm_msg, COLOR_AMBER)
            elif intent == "FOLLOW":
                person = slots.get('person', 'user')
                rtext = f"Robot: Mengikuti {person}"
                confirm_msg = f"Baik, saya akan mengikuti {person}."
                set_user_ui_state("(🚶)", f"Mengikuti {person}", confirm_msg, COLOR_GREEN)
            elif intent == "EMERGENCY":
                loc = slots.get('location', '')
                if loc:
                    rtext = f"Robot: [DARURAT] Menuju {loc}"
                    confirm_msg = f"Peringatan kondisi darurat. Segera menuju {loc} sekarang juga."
                else:
                    rtext = "Robot: [DARURAT] Mode darurat diaktifkan"
                    confirm_msg = "Peringatan kondisi darurat. Mode prioritas darurat diaktifkan."
                set_user_ui_state("(🚨)", "MODE DARURAT", confirm_msg, COLOR_RED)
            elif intent == "STATUS":
                rtext = "Robot: Status normal (baterai 85%)"
                confirm_msg = "Status robot normal. Daya baterai delapan puluh lima persen dan siap beroperasi."
                set_user_ui_state("(🔋)", "Status Normal (85%)", confirm_msg, COLOR_GREEN)
            elif intent == "GREETING":
                rtext = "Robot: Menyapa pengguna"
                confirm_msg = "Halo, ada yang bisa saya bantu?"
                set_user_ui_state("(＾▽＾)", "Halo!", confirm_msg, COLOR_PURPLE)
            elif intent == "THANK":
                rtext = "Robot: Merespons terima kasih"
                confirm_msg = "Sama-sama, senang bisa membantu."
                set_user_ui_state("(◕‿◕)", "Sama-sama!", confirm_msg, COLOR_GREEN)
            else:
                rtext = "Robot: Siap siaga (stand by)"
                confirm_msg = "Perintah diterima."
                set_user_ui_state("(◕‿◕)", "Siap Melayani", confirm_msg, COLOR_GREEN)
        elif vstatus == "INCOMPLETE":
            rcolor = COLOR_AMBER
            missing_raw = ", ".join(result.get("missing_slots", []))
            missing_id = _translate_slot_names(result.get("missing_slots", []))
            rtext = f"Robot: Perintah belum lengkap (kurang: {missing_raw})"
            confirm_msg = f"Maaf, perintah belum lengkap. Informasi {missing_id} belum ada."
            set_user_ui_state("(・_・)", "Perintah Belum Lengkap", confirm_msg, COLOR_AMBER)
        elif vstatus == "INVALID_LOCATION":
            rcolor = COLOR_RED
            loc = result['slots'].get('location', '?')
            rtext = f'Robot: Ruangan "{loc}" tidak ditemukan'
            confirm_msg = f"Maaf, lokasi {loc} tidak ditemukan."
            set_user_ui_state("(×_×)", "Lokasi Tidak Ditemukan", confirm_msg, COLOR_RED)
        elif vstatus == "UNKNOWN_INTENT":
            rcolor = COLOR_RED
            confirm_msg = responses.unknown_intent()
            rtext = f"Robot: {confirm_msg}"
            set_user_ui_state("(•_•)", "Perintah Tidak Dikenali", confirm_msg, COLOR_RED)
        else:
            rcolor = COLOR_RED
            rtext = "Robot: Perintah tidak dikenali"
            confirm_msg = "Maaf, saya tidak mengenali perintah tersebut."
            set_user_ui_state("(•_•)", "Perintah Tidak Dikenali", confirm_msg, COLOR_RED)

        def apply_final():
            intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
            slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
            validation_label.config(text=f"{vicon} VALIDASI: {vstatus}", fg=vcolor)
            robot_label.config(text=rtext, fg=rcolor)
            status_label.config(text="● Status: Siap beroperasi", fg=COLOR_GREEN)
            status_badge.config(
                text="● SIAP", fg=COLOR_GREEN, bg=COLOR_GREEN_BG, highlightbackground=COLOR_GREEN_BORDER
            )
            conv_log_label.config(text=f"🤖 Robot: {confirm_msg}", fg=rcolor)
        ui_call(apply_final)
        update_ros2_ui_status(result)

        # Dua arah konfirmasi jika NAVIGATE membawa objek penting
        if vstatus == "VALID" and result.get("intent") == "NAVIGATE" and result.get("slots", {}).get("object"):
            obj = result["slots"]["object"]
            loc = result["slots"].get("location", "")
            loc_str = f" ke {loc}" if loc else ""
            ask_confirm = f"Saya akan mengantarkan {obj}{loc_str}. Apakah sudah benar?"
            guide_confirm = "Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal."

            ui_call(lambda: conv_log_label.config(text=f"🤖 Robot: {ask_confirm} ({guide_confirm})", fg=COLOR_AMBER))
            append_dialog_history("🤖 Robot", f"{ask_confirm}\n   💡 ({guide_confirm})", COLOR_AMBER)
            set_user_ui_state(
                "(・_・)",
                "Menunggu Konfirmasi Anda",
                f"{ask_confirm}\n({guide_confirm})",
                COLOR_AMBER
            )
            if user_action_pill:
                ui_call(lambda: user_action_pill.config(
                    text="● Menunggu Konfirmasi (Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal)",
                    fg=COLOR_AMBER
                ))
            if user_voice_btn:
                ui_call(lambda: user_voice_btn.config(
                    text="🎙️   MENUNGGU KONFIRMASI ANDA\nUcapkan \"Benar\" untuk lanjut, atau \"Tidak\" untuk batal",
                    fg=COLOR_AMBER,
                    bg=COLOR_AMBER_BG,
                    highlightbackground=COLOR_AMBER_BORDER,
                ))

            if tts_ok:
                try:
                    from tts import speak_and_wait
                    speak_and_wait(ask_confirm)
                except Exception:
                    pass

            if stop_event.is_set():
                if user_voice_btn:
                    ui_call(lambda: user_voice_btn.config(
                        text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
                        fg=COLOR_PRIMARY,
                        bg=COLOR_PRIMARY_BG,
                        highlightbackground=COLOR_PRIMARY_BORDER,
                    ))
                return

            try:
                from tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            ui_call(lambda: status_label.config(
                text="● Menunggu konfirmasi: (Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal)...",
                fg=COLOR_PURPLE
            ))
            ui_call(lambda: status_badge.config(
                text="● MENUNGGU KONFIRMASI",
                fg=COLOR_PURPLE,
                bg=COLOR_PURPLE_BG,
                highlightbackground=COLOR_PURPLE_BORDER
            ))
            ans_confirm = stt_listen_once(
                listen_duration=RECORD_SECONDS + 2,
                stage_title="MENUNGGU KONFIRMASI",
                context_hint="Ucapkan 'Benar' untuk lanjut, atau 'Tidak' untuk batal",
            )
            ans_clean = (ans_confirm or "").lower().strip()
            user_spoken = ans_confirm.strip() if ans_confirm else "(tidak ada respon)"

            # Kembalikan tombol suara ke status normal
            if user_voice_btn:
                ui_call(lambda: user_voice_btn.config(
                    text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
                    fg=COLOR_PRIMARY,
                    bg=COLOR_PRIMARY_BG,
                    highlightbackground=COLOR_PRIMARY_BORDER,
                ))

            if any(w in ans_clean for w in ["tidak", "bukan", "batal", "jangan", "salah", "no"]):
                cancel_msg = "Baik, pengantaran dibatalkan."
                ui_call(lambda: stt_label.config(text=f'“{user_spoken}” [konfirmasi: batal]', fg=COLOR_RED))
                append_dialog_history("👤 Anda", f'"{user_spoken}" (Konfirmasi: Batal)', COLOR_RED)
                ui_call(lambda: conv_log_label.config(text=f'👤 Anda: "{user_spoken}" → 🤖 Robot: {cancel_msg}', fg=COLOR_RED))
                ui_call(lambda: robot_label.config(text=f'Robot: Tugas dibatalkan (kata: "{user_spoken}")', fg=COLOR_RED))
                ui_call(lambda: status_label.config(text=f'● Status: Tugas dibatalkan (respon: "{user_spoken}")', fg=COLOR_RED))
                ui_call(lambda: status_badge.config(
                    text="● DIBATALKAN", fg=COLOR_RED, bg=COLOR_RED_BG, highlightbackground=COLOR_RED_BORDER
                ))
                append_dialog_history("🤖 Robot", cancel_msg, COLOR_RED)
                set_user_ui_state("(•_•)", "Tugas Dibatalkan", f'Anda mengatakan: "{user_spoken}". Robot kembali siaga.', COLOR_RED)
                if tts_ok:
                    try:
                        from tts import speak_and_wait
                        speak_and_wait(cancel_msg)
                    except Exception:
                        pass
                return
            else:
                confirmed_word = user_spoken if ans_confirm else "benar"
                ui_call(lambda: stt_label.config(text=f'“{confirmed_word}” [konfirmasi: lanjut]', fg=COLOR_GREEN))
                append_dialog_history("👤 Anda", f'"{confirmed_word}" (Konfirmasi: Lanjut)', COLOR_GREEN)
                ui_call(lambda: conv_log_label.config(text=f'✓ Konfirmasi disetujui: "{confirmed_word}"', fg=COLOR_GREEN))

        # TTS konfirmasi akhir
        if tts_ok and confirm_msg:
            try:
                from tts import speak_and_wait
                speak_and_wait(confirm_msg)
            except Exception as e:
                logger.error(f"[TTS] Gagal konfirmasi: {e}")
        append_dialog_history("🤖 Robot", confirm_msg, rcolor)

        # Follow-up simulasi kedatangan jika NAVIGATE berhasil
        if vstatus == "VALID" and result.get("intent") == "NAVIGATE":
            target_loc = result.get("slots", {}).get("location", "tujuan")
            arrival_msg = f"Robot telah tiba di {target_loc}. Ada yang bisa saya bantu lagi?"
            time.sleep(1.0)
            ui_call(lambda: conv_log_label.config(text=f"🤖 Robot: {arrival_msg}", fg=COLOR_GREEN))
            ui_call(lambda: robot_label.config(text=f"Robot: Tiba di {target_loc}", fg=COLOR_GREEN))
            append_dialog_history("🤖 Robot", arrival_msg, COLOR_GREEN)
            set_user_ui_state("(◕‿◕)✓", f"Tiba di {target_loc}!", arrival_msg, COLOR_GREEN)
            if tts_ok:
                try:
                    from tts import speak_and_wait
                    speak_and_wait(arrival_msg)
                except Exception:
                    pass

    def _reset_task_button():
        """Reset tombol TUGAS dan Tombol Suara End-User ke state awal."""
        ui_call(lambda: task_button.config(
            text="💬  DIALOG TUGAS\nMode interaktif dua arah (TTS ↔ STT)",
            bg=BG_CARD,
            fg=COLOR_PURPLE,
            highlightbackground=BORDER,
            state="normal",
        ))
        if user_voice_btn:
            ui_call(lambda: user_voice_btn.config(
                text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Poli Jantung\"",
                bg=COLOR_PRIMARY_BG,
                fg=COLOR_PRIMARY,
                highlightbackground=COLOR_PRIMARY_BORDER,
            ))
        if user_live_speech_label:
            ui_call(lambda: user_live_speech_label.config(
                text="🎙️ Audio: Standby (Sentuh tombol suara untuk berbicara)",
                fg=FG_MUTED,
            ))
        if user_chat_status_label:
            ui_call(lambda: user_chat_status_label.config(
                text="● Siaga",
                fg=COLOR_GREEN,
            ))
        ui_call(lambda: talk_button.config(state="normal"))

    def on_task_button_clicked():
        nonlocal is_task_running, task_thread
        if is_task_running:
            stop_event.set()
            return

        if is_recording:
            return

        is_task_running = True
        stop_event.clear()

        ui_call(lambda: talk_button.config(state="disabled"))
        ui_call(lambda: task_button.config(
            text="⏹  BATALKAN DIALOG\nKlik untuk menghentikan percakapan",
            bg=COLOR_RED_BG,
            fg=COLOR_RED,
            highlightbackground=COLOR_RED_BORDER,
        ))
        if user_voice_btn:
            ui_call(lambda: user_voice_btn.config(
                text="⏹️   SELESAI / BERHENTI BICARA\n● Sedang mendengarkan ucapan Anda... (Ketuk jika selesai)",
                bg=COLOR_SKY_BG,
                fg=COLOR_SKY,
                highlightbackground=COLOR_SKY_BORDER,
            ))

        task_thread = threading.Thread(
            target=interactive_task_flow,
            daemon=True,
        )
        task_thread.start()

    # =========================
    # REAL-TIME STREAMING RECORD + PROCESS
    # =========================

    def record_and_process_stream():
        nonlocal is_recording

        stop_event.clear()
        if not stt_is_ready():
            ui_call(lambda: status_label.config(text="> status: menyiapkan model STT...", fg=YELLOW))
            ui_call(lambda: stt_label.config(text="$ stt: (menyiapkan model...)", fg=YELLOW))
        stream = create_stream()
        audio_q = _queue.Queue()

        ui_call(lambda: talk_button.config(
            text="⏹  SELESAI BICARA\nKlik untuk memproses perintah suara",
            bg=COLOR_SKY_BG,
            fg=COLOR_SKY,
            highlightbackground=COLOR_SKY_BORDER,
        ))
        ui_call(lambda: status_label.config(text="● Mendengarkan secara langsung (streaming)...", fg=COLOR_SKY))
        ui_call(lambda: status_badge.config(
            text="● STREAMING", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
        ))
        ui_call(lambda: stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY))
        ui_call(lambda: intent_label.config(text="🎯 INTENT: -", fg=FG_MUTED))
        ui_call(lambda: slots_label.config(text="📦 PARAMETER: -", fg=FG_MUTED))
        ui_call(lambda: validation_label.config(text="🛡️ VALIDASI: -", fg=FG_MUTED))
        ui_call(lambda: robot_label.config(text="Robot: Mendengarkan audio real-time...", fg=FG_MUTED))

        capture_rate = get_supported_sample_rate(device)
        optimal_channels = get_optimal_channels(device, capture_rate)

        def audio_callback(indata, frames, time_info, status):
            if stop_event.is_set():
                return
            audio_q.put(indata.copy())

        def gui_decode_worker():
            while not stop_event.is_set():
                try:
                    indata = audio_q.get(timeout=0.1)
                except _queue.Empty:
                    continue

                try:
                    if indata.shape[1] > 1:
                        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
                    else:
                        samples = np.ascontiguousarray(indata.flatten(), dtype=np.float32)

                    if capture_rate != SAMPLE_RATE:
                        samples = resample_to_16k(samples, capture_rate)
                    stream.accept_waveform(SAMPLE_RATE, samples)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    is_endpoint = recognizer.is_endpoint(stream)
                    partial = recognizer.get_result(stream).strip()

                    if partial:
                        ui_call(lambda p=partial: stt_label.config(text=f'“{p}...”', fg=COLOR_SKY))
                        sys.stdout.write(f"\r   [STREAM TALK] \"\033[36m{partial}\033[0m...\" ")
                        sys.stdout.flush()

                    if is_endpoint:
                        if partial:
                            stop_event.set()
                            return
                        else:
                            recognizer.reset(stream)

                except Exception as err:
                    logger.error(f"[AUDIO] GUI decode worker error: {err}")

        decode_thread = threading.Thread(target=gui_decode_worker, daemon=True)
        decode_thread.start()

        try:
            chunk_size = int(capture_rate * 0.1)
            with sd.InputStream(
                samplerate=capture_rate,
                channels=optimal_channels,
                dtype="float32",
                blocksize=chunk_size,
                device=device,
                callback=audio_callback,
            ):
                start_time = time.time()
                while time.time() - start_time < RECORD_SECONDS and not stop_event.is_set():
                    time.sleep(0.05)

            stop_event.set()
            decode_thread.join(timeout=2)

            t_decode_start = time.time()
            text = recognizer.get_result(stream).strip()
            decode_latency_ms = (time.time() - t_decode_start) * 1000

            if text:
                sys.stdout.write(f"\r   [✓ STREAM TALK] \"\033[32m{text}\033[0m\"" + " " * 25 + "\n")
            else:
                sys.stdout.write("\r   [✕ STREAM TALK] (tidak ada suara terdeteksi)" + " " * 25 + "\n")
            sys.stdout.flush()

            logger.info(f"[STT] text = \"{text}\" (final latency: {decode_latency_ms:.2f}ms)")
            ui_call(lambda: stt_label.config(text=f'“{text}”', fg=FG_WHITE))
            append_dialog_history("🎤 Anda", text or "(kosong)", COLOR_SKY)

            # NLU Pipeline
            ui_call(lambda: status_label.config(text="● Memproses NLU...", fg=COLOR_AMBER))
            result = pipeline.process(text)

            intent_text = f"INTENT: {result['intent']} ({result['confidence']:.2f})"
            slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(kosong)"
            vstatus = result["status"]

            if vstatus == "VALID":
                vcolor = COLOR_GREEN
                vicon = "✓"
            elif vstatus == "INVALID_LOCATION":
                vcolor = COLOR_RED
                vicon = "✕"
            else:
                vcolor = COLOR_AMBER
                vicon = "⚠"

            if vstatus == "VALID":
                rcolor = COLOR_GREEN
                intent = result["intent"]
                slots = result["slots"]
                if intent == "NAVIGATE":
                    loc = slots.get('location', '')
                    obj = slots.get('object', '')
                    if obj and loc:
                        rtext = f"Robot: Mengantarkan {obj} ke {loc}"
                    elif obj:
                        rtext = f"Robot: Mengantarkan {obj}"
                    elif loc:
                        rtext = f"Robot: Menuju {loc}"
                elif intent == "MOVE":
                    direction = slots.get("direction", "?")
                    distance = slots.get("distance", "")
                    unit = slots.get("unit", "")
                    dist_str = f" {distance} {unit}" if distance else ""
                    rtext = f"Robot: Bergerak {direction}{dist_str}"
                elif intent == "STOP":
                    rtext = "Robot: Berhenti"
                elif intent == "GO_BACK":
                    rtext = "Robot: Kembali ke posisi awal"
                elif intent == "WAIT":
                    rtext = "Robot: Menunggu"
                elif intent == "FOLLOW":
                    rtext = f"Robot: Mengikuti {slots.get('person', 'user')}"
                elif intent == "EMERGENCY":
                    loc = slots.get('location', '')
                    rtext = f"Robot: [DARURAT] Menuju {loc}" if loc else "Robot: [DARURAT] Mode darurat aktif"
                elif intent == "STATUS":
                    rtext = "Robot: Status normal (baterai 85%)"
                elif intent == "GREETING":
                    rtext = "Robot: Halo, siap melayani"
                elif intent == "THANK":
                    rtext = "Robot: Sama-sama, senang membantu"
                else:
                    rtext = "Robot: Siap siaga (stand by)"
            elif vstatus == "INCOMPLETE":
                rcolor = COLOR_AMBER
                missing_raw = ", ".join(result.get("missing_slots", []))
                rtext = f"Robot: Perintah belum lengkap (kurang: {missing_raw})"
            elif vstatus == "INVALID_LOCATION":
                rcolor = COLOR_RED
                rtext = f'Robot: Ruangan "{result["slots"].get("location", "?")}" tidak ditemukan'
            elif vstatus == "UNKNOWN_INTENT":
                rcolor = COLOR_RED
                rtext = f"Robot: {responses.unknown_intent()}"
            else:
                rcolor = COLOR_RED
                rtext = "Robot: Perintah tidak dikenali"

            def apply_results():
                intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
                slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
                validation_label.config(text=f"{vicon} VALIDASI: {vstatus}", fg=vcolor)
                robot_label.config(text=rtext, fg=rcolor)
                status_label.config(text="● Status: Siap beroperasi", fg=COLOR_GREEN)
                status_badge.config(
                    text="● SIAP", fg=COLOR_GREEN, bg=COLOR_GREEN_BG, highlightbackground=COLOR_GREEN_BORDER
                )
                conv_log_label.config(text=f"🤖 {rtext}", fg=rcolor)

            ui_call(apply_results)
            append_dialog_history("🤖 Robot", rtext, rcolor)
            update_ros2_ui_status(result)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[GUI ERROR] {err_msg}")
            ui_call(lambda: status_label.config(text="● Terjadi kesalahan perekaman audio", fg=COLOR_RED))
            ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            is_recording = False
            stop_event.clear()
            ui_call(lambda: talk_button.config(
                text="🎙️  STREAM TALK\nBicara langsung (real-time continuous STT)",
                bg=BG_CARD,
                fg=COLOR_SKY,
                highlightbackground=BORDER,
            ))
            ui_call(lambda: task_button.config(state="normal"))
            ui_call(lambda: task_button.config(state="normal"))

    def on_talk_button_clicked():
        nonlocal is_recording, record_thread
        if is_recording:
            stop_event.set()
        elif is_task_running:
            return
        else:
            is_recording = True
            stop_event.clear()
            ui_call(lambda: task_button.config(state="disabled"))
            record_thread = threading.Thread(
                target=record_and_process_stream,
                daemon=True,
            )
            record_thread.start()

    # =========================================================================
    # MODERN DUAL-VIEW WIDGET HIERARCHY (END-USER KIOSK + ADMIN DASHBOARD)
    # =========================================================================

    # 1. GLOBAL HEADER BAR (Always visible at top)
    header_frame = tk.Frame(
        root,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=18,
        pady=10,
    )
    header_frame.pack(fill="x", padx=20, pady=(16, 12))

    header_left = tk.Frame(header_frame, bg=BG_CARD)
    header_left.pack(side="left", fill="y")

    # Robot Avatar Logo Icon
    avatar_box = tk.Label(
        header_left,
        text="🤖",
        font=(FONT_FAMILY_UI, 20),
        bg=BG_CARD_ALT,
        fg=COLOR_PRIMARY,
        padx=8,
        pady=2,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    avatar_box.pack(side="left", padx=(0, 12))

    title_box = tk.Frame(header_left, bg=BG_CARD)
    title_box.pack(side="left")

    title_label = tk.Label(
        title_box,
        text="ASISTEN ROBOT PANDUAN RUMAH SAKIT",
        font=F_TITLE,
        fg=FG_WHITE,
        bg=BG_CARD,
    )
    title_label.pack(anchor="w")

    sub_label = tk.Label(
        title_box,
        text="Layanan Mandiri Pasien & Pengunjung • Sentuh Layar atau Bicara Langsung",
        font=F_SUBTITLE,
        fg=FG_SUB,
        bg=BG_CARD,
    )
    sub_label.pack(anchor="w")

    # Header Right (View Switcher & Status Badges)
    header_right = tk.Frame(header_frame, bg=BG_CARD)
    header_right.pack(side="right", fill="y")

    view_toggle_btn = tk.Button(
        header_right,
        text="🛠️  Mode Teknisi",
        font=(FONT_FAMILY_UI, 9, "bold"),
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=BORDER,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=12,
        pady=4,
        cursor="hand2",
    )
    view_toggle_btn.pack(side="right", padx=(10, 0))
    add_hover_effect(view_toggle_btn, BG_CARD_ALT, BORDER, BORDER, BORDER_HOVER)

    status_badge = tk.Label(
        header_right,
        text="● SIAP",
        font=F_CODE_BOLD,
        fg=COLOR_GREEN,
        bg=COLOR_GREEN_BG,
        padx=10,
        pady=4,
        relief="flat",
        highlightthickness=1,
        highlightbackground=COLOR_GREEN_BORDER,
    )
    status_badge.pack(side="right")

    # Mic Badge info
    try:
        cur_dev = sd.query_devices(device, "input")
        dev_short_name = cur_dev.get("name", "Default")
        if len(dev_short_name) > 16:
            dev_short_name = dev_short_name[:16] + ".."
    except Exception:
        dev_short_name = "Default"

    mic_badge_label = tk.Label(
        header_right,
        text=f"🎙️ {dev_short_name}",
        font=F_CODE,
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        padx=8,
        pady=4,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    mic_badge_label.pack(side="right", padx=(6, 0))

    # =========================================================================
    # MASTER CONTAINER FOR DUAL-VIEW
    # =========================================================================
    container = tk.Frame(root, bg=BG_BASE)
    container.pack(fill="both", expand=True, padx=20, pady=(0, 16))

    user_view_frame = tk.Frame(container, bg=BG_BASE)
    admin_view_frame = tk.Frame(container, bg=BG_BASE)

    # =========================================================================
    # VIEW A: END-USER KIOSK INTERFACE (HUMAN-CENTRIC & INTUITIVE)
    # =========================================================================

    # 1. Persona & Expression Hero Card
    user_hero_card = tk.Frame(
        user_view_frame,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=20,
        pady=14,
    )
    user_hero_card.pack(fill="x", pady=(0, 10))

    hero_card_left = tk.Frame(user_hero_card, bg=BG_CARD)
    hero_card_left.pack(side="left")

    user_face_label = tk.Label(
        hero_card_left,
        text="(◕‿◕)",
        font=(FONT_FAMILY_CODE, 28, "bold"),
        fg=COLOR_GREEN,
        bg=BG_CARD_ALT,
        padx=16,
        pady=8,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    user_face_label.pack(side="left", padx=(0, 16))

    hero_card_text = tk.Frame(user_hero_card, bg=BG_CARD)
    hero_card_text.pack(side="left", fill="both", expand=True)

    user_status_title = tk.Label(
        hero_card_text,
        text="Halo! Saya asisten robot pemandu rumah sakit.",
        font=(FONT_FAMILY_UI, 13, "bold"),
        fg=FG_WHITE,
        bg=BG_CARD,
    )
    user_status_title.pack(anchor="w")

    user_status_sub = tk.Label(
        hero_card_text,
        text="Katakan tujuan Anda, sentuh tombol suara di bawah, atau pilih salah satu tujuan cepat.",
        font=(FONT_FAMILY_UI, 9),
        fg=FG_SUB,
        bg=BG_CARD,
    )
    user_status_sub.pack(anchor="w", pady=(2, 4))

    pill_and_speech_row = tk.Frame(hero_card_text, bg=BG_CARD)
    pill_and_speech_row.pack(fill="x", pady=(2, 0))

    user_action_pill = tk.Label(
        pill_and_speech_row,
        text="● Siap Melayani",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=COLOR_GREEN,
        bg=COLOR_GREEN_BG,
        padx=8,
        pady=2,
        highlightthickness=1,
        highlightbackground=COLOR_GREEN_BORDER,
    )
    user_action_pill.pack(side="left")

    user_live_speech_label = tk.Label(
        pill_and_speech_row,
        text="🎙️ Audio: Standby (Sentuh tombol suara untuk berbicara)",
        font=(FONT_FAMILY_CODE, 9),
        fg=FG_MUTED,
        bg=BG_CARD_ALT,
        padx=10,
        pady=2,
        highlightthickness=1,
        highlightbackground=BORDER,
        anchor="w",
    )
    user_live_speech_label.pack(side="left", padx=(8, 0), fill="x", expand=True)

    # 2. Giant Voice Interaction Touchpad (1-Tap to Speak)
    user_voice_btn = tk.Button(
        user_view_frame,
        text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
        font=(FONT_FAMILY_UI, 11, "bold"),
        fg=COLOR_PRIMARY,
        bg=COLOR_PRIMARY_BG,
        activeforeground=FG_WHITE,
        activebackground=COLOR_PRIMARY_BORDER,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=COLOR_PRIMARY_BORDER,
        pady=12,
        cursor="hand2",
        command=on_task_button_clicked,
    )
    user_voice_btn.pack(fill="x", pady=(0, 10))
    add_hover_effect(user_voice_btn, COLOR_PRIMARY_BG, COLOR_PRIMARY_BORDER, COLOR_PRIMARY_BORDER, COLOR_PRIMARY)

    # 3. Quick Destination Tiles (1-Touch Instant Kiosk)
    tiles_card = tk.Frame(
        user_view_frame,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=16,
        pady=10,
    )
    tiles_card.pack(fill="x", pady=(0, 10))

    tiles_header = tk.Frame(tiles_card, bg=BG_CARD)
    tiles_header.pack(fill="x", pady=(0, 6))

    tk.Label(
        tiles_header,
        text="📍 PILIHAN TUJUAN CEPAT:",
        font=(FONT_FAMILY_UI, 9, "bold"),
        fg=FG_SUB,
        bg=BG_CARD,
    ).pack(side="left")

    mode_switch_frame = tk.Frame(tiles_header, bg=BG_CARD)
    mode_switch_frame.pack(side="right")

    tk.Label(
        mode_switch_frame,
        text="Opsi Tugas:",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD,
    ).pack(side="left", padx=(0, 6))

    btn_mode_arah = None
    btn_mode_obat = None
    quick_mode_hint = None

    def select_quick_mode(mode: str):
        nonlocal current_quick_mode
        current_quick_mode = mode
        if mode == "arah":
            btn_mode_arah.config(
                bg=COLOR_SKY_BG,
                fg=COLOR_SKY,
                highlightbackground=COLOR_SKY_BORDER,
            )
            btn_mode_obat.config(
                bg=BG_CARD_ALT,
                fg=FG_MUTED,
                highlightbackground=BORDER,
            )
            if quick_mode_hint:
                quick_mode_hint.config(
                    text="● Mode Aktif: 🧭 Penunjuk Arah (Robot memandu dan mengantar Anda langsung ke lokasi)",
                    fg=COLOR_SKY,
                )
        elif mode == "obat":
            btn_mode_obat.config(
                bg=COLOR_GREEN_BG,
                fg=COLOR_GREEN,
                highlightbackground=COLOR_GREEN_BORDER,
            )
            btn_mode_arah.config(
                bg=BG_CARD_ALT,
                fg=FG_MUTED,
                highlightbackground=BORDER,
            )
            if quick_mode_hint:
                quick_mode_hint.config(
                    text="● Mode Aktif: 💊 Antar Obat (Robot membawa dan mengantarkan obat medis ke lokasi)",
                    fg=COLOR_GREEN,
                )

    btn_mode_arah = tk.Button(
        mode_switch_frame,
        text="🧭 Penunjuk Arah",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=COLOR_SKY,
        bg=COLOR_SKY_BG,
        activeforeground=FG_WHITE,
        activebackground=COLOR_SKY_BORDER,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=COLOR_SKY_BORDER,
        padx=10,
        pady=3,
        cursor="hand2",
        command=lambda: select_quick_mode("arah"),
    )
    btn_mode_arah.pack(side="left", padx=(0, 4))

    btn_mode_obat = tk.Button(
        mode_switch_frame,
        text="💊 Antar Obat",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=COLOR_GREEN_BORDER,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=3,
        cursor="hand2",
        command=lambda: select_quick_mode("obat"),
    )
    btn_mode_obat.pack(side="left")

    quick_mode_hint = tk.Label(
        tiles_card,
        text="● Mode Aktif: 🧭 Penunjuk Arah (Robot memandu dan mengantar Anda langsung ke lokasi)",
        font=(FONT_FAMILY_UI, 8),
        fg=COLOR_SKY,
        bg=BG_CARD,
        anchor="w",
    )
    quick_mode_hint.pack(fill="x", pady=(0, 6))

    tiles_grid = tk.Frame(tiles_card, bg=BG_CARD)
    tiles_grid.pack(fill="x")

    quick_destinations = [
        ("🏥 Ruang ICU", "ruang ICU", COLOR_SKY, COLOR_SKY_BG, COLOR_SKY_BORDER),
        ("🚨 Ruang IGD", "ruang IGD", COLOR_RED, COLOR_RED_BG, COLOR_RED_BORDER),
        ("💊 Farmasi / Apotek", "ruang farmasi", COLOR_GREEN, COLOR_GREEN_BG, COLOR_GREEN_BORDER),
        ("🩺 Poli Jantung", "poli jantung", COLOR_PURPLE, COLOR_PURPLE_BG, COLOR_PURPLE_BORDER),
        ("👶 Poli Anak", "poli anak", COLOR_AMBER, COLOR_AMBER_BG, COLOR_AMBER_BORDER),
        ("🚪 Lobi Utama", "lobi utama", FG_WHITE, BG_CARD_ALT, BORDER),
        ("🛑 Berhenti", "__STOP__", COLOR_RED, COLOR_RED_BG, COLOR_RED_BORDER),
        ("🏠 Kembali ke Pos", "__HOME__", COLOR_GREEN, COLOR_GREEN_BG, COLOR_GREEN_BORDER),
    ]

    for idx, (label_txt, dest_key, btn_fg, btn_bg, btn_bd) in enumerate(quick_destinations):
        row = idx // 4
        col = idx % 4

        def _make_cmd(target=dest_key):
            if target == "__STOP__":
                return lambda: execute_quick_action("berhenti")
            elif target == "__HOME__":
                return lambda: execute_quick_action("kembali")
            else:
                return lambda: execute_quick_destination(target)

        qbtn = tk.Button(
            tiles_grid,
            text=label_txt,
            font=(FONT_FAMILY_UI, 9, "bold"),
            fg=btn_fg,
            bg=btn_bg,
            activeforeground=FG_WHITE,
            activebackground=btn_bd,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=btn_bd,
            padx=10,
            pady=7,
            cursor="hand2",
            command=_make_cmd(),
        )
        qbtn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        add_hover_effect(qbtn, btn_bg, btn_bd, btn_bd, btn_fg)

    for c in range(4):
        tiles_grid.grid_columnconfigure(c, weight=1)

    # 4. Interactive Conversation Card & Touch Search Bar
    user_chat_card = tk.Frame(
        user_view_frame,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=16,
        pady=10,
    )
    user_chat_card.pack(fill="both", expand=True)

    user_chat_header = tk.Frame(user_chat_card, bg=BG_CARD)
    user_chat_header.pack(fill="x", pady=(0, 4))

    tk.Label(
        user_chat_header,
        text="💬 PERCAKAPAN DENGAN ROBOT",
        font=(FONT_FAMILY_UI, 9, "bold"),
        fg=FG_SUB,
        bg=BG_CARD,
    ).pack(side="left")

    user_chat_status_label = tk.Label(
        user_chat_header,
        text="● Siaga",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=COLOR_GREEN,
        bg=BG_CARD,
    )
    user_chat_status_label.pack(side="right")

    user_chat_scroll = tk.Frame(user_chat_card, bg=BG_BASE, highlightthickness=1, highlightbackground=BORDER)
    user_chat_scroll.pack(fill="both", expand=True, pady=(2, 8))

    user_chat_box = tk.Text(
        user_chat_scroll,
        font=(FONT_FAMILY_UI, 10),
        bg=BG_BASE,
        fg=FG_WHITE,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        wrap="word",
        padx=12,
        pady=8,
        height=6,
        state="disabled",
    )
    user_scrollbar = tk.Scrollbar(user_chat_scroll, command=user_chat_box.yview, bg=BG_CARD, troughcolor=BG_BASE, width=12)
    user_chat_box.configure(yscrollcommand=user_scrollbar.set)
    user_scrollbar.pack(side="right", fill="y")
    user_chat_box.pack(side="left", fill="both", expand=True)

    user_chat_box.tag_configure("user_tag_time", foreground=FG_MUTED, font=(FONT_FAMILY_CODE, 8))
    user_chat_box.tag_configure("user_tag_robot", foreground=COLOR_PURPLE, font=(FONT_FAMILY_UI, 10, "bold"))
    user_chat_box.tag_configure("user_tag_user", foreground=COLOR_SKY, font=(FONT_FAMILY_UI, 10, "bold"))
    user_chat_box.tag_configure("user_tag_info", foreground=FG_SUB, font=(FONT_FAMILY_UI, 9, "italic"))
    user_chat_box.tag_configure("user_tag_body", foreground=FG_WHITE)

    # Touch Search / Type Bar
    user_input_box = tk.Frame(user_chat_card, bg=BG_CARD_ALT, highlightthickness=1, highlightbackground=BORDER, padx=8, pady=4)
    user_input_box.pack(fill="x")

    tk.Label(user_input_box, text="⌨️", font=(FONT_FAMILY_UI, 11), bg=BG_CARD_ALT, fg=COLOR_PRIMARY).pack(side="left", padx=(2, 6))

    user_text_entry = tk.Entry(
        user_input_box,
        font=(FONT_FAMILY_UI, 10),
        bg=BG_CARD_ALT,
        fg=FG_WHITE,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        borderwidth=0,
    )
    user_text_entry.pack(side="left", fill="x", expand=True, ipady=4)

    def execute_user_text(event=None):
        txt = user_text_entry.get().strip()
        if not txt:
            return
        user_text_entry.delete(0, tk.END)
        set_user_ui_state("(・_・)", "Memproses...", f"Robot menerima: \"{txt}\"", COLOR_AMBER)
        append_dialog_history("👤 Anda", txt, COLOR_SKY)
        def _worker():
            res = pipeline.process(txt)
            try:
                from tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            _show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    user_text_entry.bind("<Return>", execute_user_text)

    user_send_btn = tk.Button(
        user_input_box,
        text="KIRIM ↵",
        font=(FONT_FAMILY_UI, 9, "bold"),
        fg=FG_WHITE,
        bg=COLOR_PRIMARY,
        activeforeground=FG_WHITE,
        activebackground=COLOR_PRIMARY_BORDER,
        relief="flat",
        borderwidth=0,
        padx=12,
        pady=2,
        cursor="hand2",
        command=execute_user_text,
    )
    user_send_btn.pack(side="right")
    add_hover_effect(user_send_btn, COLOR_PRIMARY, COLOR_PRIMARY_BORDER)

    # =========================================================================
    # VIEW B: ADMIN & TROUBLESHOOTING DASHBOARD
    # =========================================================================

    # 1. Action Buttons (Talk & Task)
    hero_frame = tk.Frame(admin_view_frame, bg=BG_BASE)
    hero_frame.pack(fill="x", pady=(0, 10))

    talk_button = tk.Button(
        hero_frame,
        text="🎙️  STREAM TALK\nBicara langsung (real-time continuous STT)",
        font=F_BTN_TITLE,
        fg=COLOR_SKY,
        bg=BG_CARD,
        activeforeground=COLOR_SKY,
        activebackground=COLOR_SKY_BG,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        height=2,
        cursor="hand2",
        command=on_talk_button_clicked,
    )
    talk_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
    add_hover_effect(talk_button, BG_CARD, BG_CARD_ALT, BORDER, COLOR_SKY_BORDER)

    task_button = tk.Button(
        hero_frame,
        text="💬  DIALOG TUGAS\nMode interaktif dua arah (TTS ↔ STT)",
        font=F_BTN_TITLE,
        fg=COLOR_PURPLE,
        bg=BG_CARD,
        activeforeground=COLOR_PURPLE,
        activebackground=COLOR_PURPLE_BG,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        height=2,
        cursor="hand2",
        command=on_task_button_clicked,
    )
    task_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
    add_hover_effect(task_button, BG_CARD, BG_CARD_ALT, BORDER, COLOR_PURPLE_BORDER)

    # 2. Command Input Bar & Diagnostics Toolbar
    toolbar_frame = tk.Frame(admin_view_frame, bg=BG_BASE)
    toolbar_frame.pack(fill="x", pady=(0, 10))

    input_box = tk.Frame(
        toolbar_frame,
        bg=BG_CARD_ALT,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=4,
    )
    input_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

    tk.Label(
        input_box,
        text="⌨️",
        font=(FONT_FAMILY_UI, 11),
        bg=BG_CARD_ALT,
        fg=COLOR_PRIMARY,
    ).pack(side="left", padx=(2, 6))

    text_entry = tk.Entry(
        input_box,
        font=F_CODE,
        bg=BG_CARD_ALT,
        fg=FG_WHITE,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        borderwidth=0,
    )
    text_entry.pack(side="left", fill="x", expand=True, ipady=4)

    def execute_manual_text(event=None):
        txt = text_entry.get().strip()
        if not txt:
            return
        text_entry.delete(0, tk.END)
        ui_call(lambda: status_label.config(text="● Memproses instruksi teks...", fg=COLOR_AMBER))
        ui_call(lambda: stt_label.config(text=f'“{txt}” [manual]', fg=COLOR_PRIMARY))
        append_dialog_history("⌨️ Manual", txt, COLOR_PRIMARY)

        def run_manual():
            res = pipeline.process(txt)
            try:
                from tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            _show_final_result(res, tts_ok)
        threading.Thread(target=run_manual, daemon=True).start()

    text_entry.bind("<Return>", execute_manual_text)

    send_btn = tk.Button(
        input_box,
        text="KIRIM ↵",
        font=(FONT_FAMILY_UI, 9, "bold"),
        fg=FG_WHITE,
        bg=COLOR_PRIMARY,
        activeforeground=FG_WHITE,
        activebackground=COLOR_PRIMARY_BORDER,
        relief="flat",
        borderwidth=0,
        padx=12,
        pady=2,
        cursor="hand2",
        command=execute_manual_text,
    )
    send_btn.pack(side="right")
    add_hover_effect(send_btn, COLOR_PRIMARY, COLOR_PRIMARY_BORDER)

    # Diagnostic Tool Modals & Commands
    def open_devices_modal():
        modal = tk.Toplevel(root)
        modal.title("Pilih Perangkat Input Audio")
        modal.geometry("560x380")
        modal.configure(bg=BG_BASE)
        modal.transient(root)
        modal.grab_set()

        tk.Label(
            modal,
            text="DAFTAR PERANGKAT INPUT AUDIO (MIKROFON)",
            font=(FONT_FAMILY_UI, 11, "bold"),
            fg=FG_WHITE,
            bg=BG_BASE,
        ).pack(pady=(16, 8))

        dev_listbox = tk.Listbox(
            modal,
            font=F_CODE,
            bg=BG_CARD_ALT,
            fg=FG_WHITE,
            selectbackground=COLOR_PRIMARY_BG,
            selectforeground=COLOR_PRIMARY,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        dev_listbox.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        try:
            import sounddevice as sd
            devs = sd.query_devices()
            in_devs = []
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    in_devs.append((i, d))
                    marker = " [*] " if i == device else "     "
                    dev_listbox.insert(tk.END, f"{marker}[{i}] {d['name']} ({d['max_input_channels']} ch)")
        except Exception as e:
            dev_listbox.insert(tk.END, f"Gagal membaca perangkat: {e}")

        def select_device():
            nonlocal device
            sel = dev_listbox.curselection()
            if sel and in_devs:
                idx = sel[0]
                device = in_devs[idx][0]
                dev_name_chosen = in_devs[idx][1]['name']
                short_n = dev_name_chosen[:16] + ".." if len(dev_name_chosen) > 16 else dev_name_chosen
                mic_badge_label.config(text=f"🎙️ {short_n}")
                status_label.config(text=f"● Mikrofon dialihkan ke ID {device}", fg=COLOR_GREEN)
                modal.destroy()

        btn_select = tk.Button(
            modal,
            text="Gunakan Perangkat Terpilih",
            font=F_BTN_TITLE,
            fg=FG_WHITE,
            bg=COLOR_PRIMARY,
            activeforeground=FG_WHITE,
            activebackground=COLOR_PRIMARY_BORDER,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=select_device,
        )
        btn_select.pack(pady=(0, 16))

    def open_ros2_config_modal():
        """Buka modal modern untuk mengatur Custom IP & Port Node ROS 2."""
        modal = tk.Toplevel(root)
        modal.title("⚙️ Pengaturan Endpoint Node ROS 2")
        modal.configure(bg=BG_CARD)
        modal.resizable(False, False)
        modal.transient(root)
        modal.grab_set()

        mw, mh = 480, 440
        rx = root.winfo_x() + (root.winfo_width() // 2) - (mw // 2)
        ry = root.winfo_y() + (root.winfo_height() // 2) - (mh // 2)
        modal.geometry(f"{mw}x{mh}+{max(0, rx)}+{max(0, ry)}")

        container = tk.Frame(modal, bg=BG_CARD, padx=24, pady=20)
        container.pack(fill="both", expand=True)

        # Header
        tk.Label(
            container,
            text="⚙️ KONFIGURASI ENDPOINT ROS 2",
            font=(FONT_FAMILY_UI, 12, "bold"),
            fg=FG_WHITE,
            bg=BG_CARD,
        ).pack(anchor="w")

        tk.Label(
            container,
            text="Atur IP address & port TCP node ROS 2 di robot tujuan.",
            font=(FONT_FAMILY_UI, 9),
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(anchor="w", pady=(2, 14))

        # Current Host & Port values
        cur_host = getattr(pipeline.ros2_adapter, "host", "10.107.245.221")
        cur_port = getattr(pipeline.ros2_adapter, "port", 5000)

        # Field IP Host
        tk.Label(
            container,
            text="ALAMAT IP / HOSTNAME:",
            font=(FONT_FAMILY_UI, 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD,
        ).pack(anchor="w")

        host_entry = tk.Entry(
            container,
            font=(FONT_FAMILY_CODE, 10),
            bg=BG_CARD_ALT,
            fg=FG_WHITE,
            insertbackground=COLOR_PRIMARY,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        host_entry.pack(fill="x", ipady=4, pady=(3, 10))
        host_entry.insert(0, str(cur_host))

        # Field Port
        tk.Label(
            container,
            text="PORT TCP:",
            font=(FONT_FAMILY_UI, 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD,
        ).pack(anchor="w")

        port_entry = tk.Entry(
            container,
            font=(FONT_FAMILY_CODE, 10),
            bg=BG_CARD_ALT,
            fg=FG_WHITE,
            insertbackground=COLOR_PRIMARY,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        port_entry.pack(fill="x", ipady=4, pady=(3, 10))
        port_entry.insert(0, str(cur_port))

        # Preset Buttons
        preset_frame = tk.Frame(container, bg=BG_CARD)
        preset_frame.pack(fill="x", pady=(0, 12))

        tk.Label(
            preset_frame,
            text="Preset Cepat:",
            font=(FONT_FAMILY_UI, 8),
            fg=FG_MUTED,
            bg=BG_CARD,
        ).pack(side="left", padx=(0, 6))

        def set_preset(h, p):
            host_entry.delete(0, tk.END)
            host_entry.insert(0, h)
            port_entry.delete(0, tk.END)
            port_entry.insert(0, str(p))

        btn_pre_robot = tk.Button(
            preset_frame,
            text="🤖 Robot (10.107.245.221)",
            font=(FONT_FAMILY_UI, 8),
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            activeforeground=FG_WHITE,
            activebackground=BORDER,
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda: set_preset("10.107.245.221", 5000),
        )
        btn_pre_robot.pack(side="left", padx=(0, 4))

        btn_pre_local = tk.Button(
            preset_frame,
            text="💻 Localhost (127.0.0.1)",
            font=(FONT_FAMILY_UI, 8),
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            activeforeground=FG_WHITE,
            activebackground=BORDER,
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda: set_preset("127.0.0.1", 5000),
        )
        btn_pre_local.pack(side="left")

        # Feedback test label
        test_feedback = tk.Label(
            container,
            text="💡 Tekan 'Uji Ping' untuk menguji port terbuka sebelum menyimpan.",
            font=(FONT_FAMILY_UI, 8),
            fg=FG_MUTED,
            bg=BG_CARD,
            anchor="w",
        )
        test_feedback.pack(fill="x", pady=(0, 14))

        def run_test_ping():
            h = host_entry.get().strip()
            p_str = port_entry.get().strip()
            if not h or not p_str:
                test_feedback.config(text="⚠ Alamat IP dan Port tidak boleh kosong", fg=COLOR_AMBER)
                return
            try:
                p = int(p_str)
            except ValueError:
                test_feedback.config(text="⚠ Port harus berupa angka bulat (1-65535)", fg=COLOR_RED)
                return

            test_feedback.config(text=f"Menghubungi {h}:{p}...", fg=COLOR_SKY)

            def _worker():
                import socket
                import time
                t0 = time.time()
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.5)
                        s.connect((h, p))
                    latency_ms = int((time.time() - t0) * 1000)
                    msg = f"✓ Berhasil terhubung ke {h}:{p} ({latency_ms} ms)"
                    col = COLOR_GREEN
                except Exception as e:
                    msg = f"✕ Gagal terhubung ke {h}:{p} ({e})"
                    col = COLOR_RED

                def _apply():
                    if modal.winfo_exists():
                        test_feedback.config(text=msg, fg=col)
                ui_call(_apply)

            threading.Thread(target=_worker, daemon=True).start()

        def save_and_apply():
            h = host_entry.get().strip()
            p_str = port_entry.get().strip()
            if not h or not p_str:
                test_feedback.config(text="⚠ Alamat IP dan Port tidak boleh kosong", fg=COLOR_AMBER)
                return
            try:
                p = int(p_str)
            except ValueError:
                test_feedback.config(text="⚠ Port harus berupa angka bulat", fg=COLOR_RED)
                return

            save_ros2_config(h, p)

            # Update adapter target
            if hasattr(pipeline.ros2_adapter, "set_target"):
                pipeline.ros2_adapter.set_target(h, p)
            else:
                pipeline.ros2_adapter = SocketROS2Adapter(host=h, port=p)

            new_target = f"{h}:{p}"
            if ros2_title_label:
                ros2_title_label.config(text=f"🌐 TRANSMISI PAYLOAD NODE ROS 2 [{new_target}]:")
            if ros2_status_badge:
                ros2_status_badge.config(
                    text="● MENUNGGU",
                    fg=FG_MUTED,
                    bg=BG_CARD,
                    highlightbackground=BORDER,
                )
            if ros2_detail_label:
                ros2_detail_label.config(
                    text=f"Target diperbarui ke [{new_target}]. Belum ada payload dikirim.",
                    fg=FG_SUB,
                )

            append_dialog_history("🌐 ROS 2", f"Konfigurasi endpoint diperbarui ke {new_target} (disimpan)", COLOR_SKY)
            modal.destroy()

        # Action Buttons
        btn_box = tk.Frame(container, bg=BG_CARD)
        btn_box.pack(fill="x")

        btn_test = tk.Button(
            btn_box,
            text="🔍 UJI PING",
            font=(FONT_FAMILY_UI, 8, "bold"),
            fg=COLOR_SKY,
            bg=BG_CARD_ALT,
            activeforeground=FG_WHITE,
            activebackground=BORDER,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=run_test_ping,
        )
        btn_test.pack(side="left", padx=(0, 8))

        btn_save = tk.Button(
            btn_box,
            text="💾 SIMPAN & TERAPKAN",
            font=(FONT_FAMILY_UI, 8, "bold"),
            fg=FG_WHITE,
            bg=COLOR_PRIMARY,
            activeforeground=FG_WHITE,
            activebackground=COLOR_PRIMARY_BORDER,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=save_and_apply,
        )
        btn_save.pack(side="left")

        btn_cancel = tk.Button(
            btn_box,
            text="BATAL",
            font=(FONT_FAMILY_UI, 8),
            fg=FG_MUTED,
            bg=BG_CARD,
            activeforeground=FG_WHITE,
            activebackground=BG_CARD_ALT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=6,
            cursor="hand2",
            command=modal.destroy,
        )
        btn_cancel.pack(side="right")


    def run_mic_diagnostics():
        status_label.config(text="● Menguji level volume audio mikrofon...", fg=COLOR_AMBER)
        robot_label.config(text="Robot: Merekam sampel audio 3 detik untuk pengujian hardware...", fg=COLOR_AMBER)
        conv_log_label.config(text="[UJI MIKROFON] Silakan berbicara sekarang...", fg=COLOR_SKY)
        append_dialog_history("ℹ️ Diagnosa", "Merekam suara 3 detik untuk uji hardware mic...", COLOR_SKY)

        def test_worker():
            try:
                import sounddevice as sd
                import numpy as np
                sr = get_supported_sample_rate(device)
                ch = get_optimal_channels(device, sr)
                recording = sd.rec(int(3 * sr), samplerate=sr, channels=ch, dtype="float32", device=device)
                sd.wait()
                samples_flat = recording.flatten()
                rms = float(np.sqrt(np.mean(samples_flat ** 2)))
                peak = float(np.max(np.abs(samples_flat)))

                if peak < 0.01:
                    diag_res = "Sinyal sangat lemah / mute (Periksa koneksi & volume input)"
                    col = COLOR_RED
                elif peak < 0.05:
                    diag_res = "Sinyal audio terdeteksi (Volume agak pelan)"
                    col = COLOR_AMBER
                else:
                    diag_res = "Sinyal normal & jernih (Mikrofon siap digunakan)"
                    col = COLOR_GREEN

                res_detail = f"Peak Amplitude: {peak:.4f} | RMS: {rms:.4f} → {diag_res}"
                ui_call(lambda: status_label.config(text="● Pengujian mikrofon selesai", fg=COLOR_GREEN))
                ui_call(lambda: robot_label.config(text=f"Hasil Tes Mic: {diag_res}", fg=col))
                ui_call(lambda: conv_log_label.config(text=f"Hasil Tes: {diag_res}", fg=col))
                append_dialog_history("📊 Hardware Mic", res_detail, col)
            except Exception as e:
                ui_call(lambda: status_label.config(text=f"● Gagal menguji mikrofon ({e})", fg=COLOR_RED))

        threading.Thread(target=test_worker, daemon=True).start()

    def clear_dialog_history():
        dialog_history_box.config(state="normal")
        dialog_history_box.delete("1.0", tk.END)
        dialog_history_box.config(state="disabled")
        conv_log_label.config(text="Riwayat percakapan telah dibersihkan", fg=FG_MUTED)
        if ros2_status_badge:
            ros2_status_badge.config(
                text="● MENUNGGU",
                fg=FG_MUTED,
                bg=BG_CARD,
                highlightbackground=BORDER,
            )
        if ros2_detail_label:
            ros2_detail_label.config(
                text="Belum ada payload dikirim. Kirim instruksi untuk memicu transmisi TCP ke ROS 2.",
                fg=FG_SUB,
            )

    btn_mic = tk.Button(
        toolbar_frame,
        text="🎧 PILIH MIC",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_SUB,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=5,
        cursor="hand2",
        command=open_devices_modal,
    )
    btn_mic.pack(side="left", padx=(0, 6))
    add_hover_effect(btn_mic, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

    btn_diag = tk.Button(
        toolbar_frame,
        text="📊 TES MIC",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_SUB,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=5,
        cursor="hand2",
        command=run_mic_diagnostics,
    )
    btn_diag.pack(side="left", padx=(0, 6))
    add_hover_effect(btn_diag, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

    btn_clear = tk.Button(
        toolbar_frame,
        text="🧹 BERSIHKAN",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=5,
        cursor="hand2",
        command=clear_dialog_history,
    )
    btn_clear.pack(side="left")
    add_hover_effect(btn_clear, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

    def check_ros2_ping():
        def _bg_ping():
            target_str = getattr(pipeline.ros2_adapter, "host", "Node ROS 2")
            if hasattr(pipeline.ros2_adapter, "port"):
                target_str = f"{target_str}:{pipeline.ros2_adapter.port}"
            append_dialog_history("🌐 ROS 2", f"Memeriksa koneksi TCP ke {target_str}...", COLOR_SKY)
            ok, msg = pipeline.ros2_adapter.ping(timeout=1.5)
            def _ui():
                if ok:
                    if ros2_status_badge:
                        ros2_status_badge.config(
                            text="● TERHUBUNG ✓",
                            fg=COLOR_GREEN,
                            bg=COLOR_GREEN_BG,
                            highlightbackground=COLOR_GREEN_BORDER,
                        )
                    if ros2_detail_label:
                        ros2_detail_label.config(
                            text=f"✓ Node ROS 2 terjangkau ({msg})",
                            fg=COLOR_GREEN,
                        )
                    append_dialog_history("🌐 ROS 2", f"✓ {msg}", COLOR_GREEN)
                else:
                    if ros2_status_badge:
                        ros2_status_badge.config(
                            text="✕ TERPUTUS",
                            fg=COLOR_RED,
                            bg=COLOR_RED_BG,
                            highlightbackground=COLOR_RED_BORDER,
                        )
                    if ros2_detail_label:
                        ros2_detail_label.config(
                            text=f"✕ Node ROS 2 tidak terjangkau ({msg})",
                            fg=COLOR_RED,
                        )
                    append_dialog_history("🌐 ROS 2", f"✕ Gagal ping: {msg}", COLOR_RED)
            ui_call(_ui)
        threading.Thread(target=_bg_ping, daemon=True).start()

    btn_ros2 = tk.Button(
        toolbar_frame,
        text="🌐 CEK ROS 2",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=COLOR_SKY,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=5,
        cursor="hand2",
        command=check_ros2_ping,
    )
    btn_ros2.pack(side="left", padx=(6, 0))
    add_hover_effect(btn_ros2, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

    btn_ros2_cfg = tk.Button(
        toolbar_frame,
        text="⚙️ IP & PORT",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_SUB,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=10,
        pady=5,
        cursor="hand2",
        command=open_ros2_config_modal,
    )
    btn_ros2_cfg.pack(side="left", padx=(6, 0))
    add_hover_effect(btn_ros2_cfg, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

    # 3. Telemetry & NLU Status Card
    telemetry_card = tk.Frame(
        admin_view_frame,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=18,
        pady=10,
    )
    telemetry_card.pack(fill="x", pady=(0, 10))

    card1_header = tk.Frame(telemetry_card, bg=BG_CARD)
    card1_header.pack(fill="x", pady=(0, 6))

    tk.Label(
        card1_header,
        text="STATUS & TELEMETRI NLU",
        font=F_SECTION,
        fg=FG_SUB,
        bg=BG_CARD,
    ).pack(side="left")

    status_label = tk.Label(
        card1_header,
        text="● Status: Siap",
        font=F_CODE_BOLD,
        fg=COLOR_GREEN,
        bg=BG_CARD,
    )
    status_label.pack(side="right")

    speech_frame = tk.Frame(
        telemetry_card,
        bg=BG_CARD_ALT,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=12,
        pady=5,
    )
    speech_frame.pack(fill="x", pady=(0, 6))

    tk.Label(
        speech_frame,
        text="🎙️ SUARA TERDETEKSI:",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD_ALT,
    ).pack(anchor="w")

    stt_label = tk.Label(
        speech_frame,
        text='“(menunggu perintah suara dari mikrofon atau ketikan manual...)”',
        font=(FONT_FAMILY_CODE, 10, "italic"),
        fg=FG_MUTED,
        bg=BG_CARD_ALT,
        anchor="w",
        justify="left",
    )
    stt_label.pack(fill="x", pady=(2, 2))

    tags_bar = tk.Frame(telemetry_card, bg=BG_CARD)
    tags_bar.pack(fill="x", pady=(0, 6))

    intent_label = tk.Label(
        tags_bar,
        text="🎯 INTENT: -",
        font=F_CODE_BOLD,
        fg=COLOR_PRIMARY,
        bg=BG_CARD_ALT,
        padx=10,
        pady=3,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    intent_label.pack(side="left", padx=(0, 6))

    validation_label = tk.Label(
        tags_bar,
        text="🛡️ VALIDASI: -",
        font=F_CODE_BOLD,
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        padx=10,
        pady=3,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    validation_label.pack(side="left", padx=(0, 6))

    slots_label = tk.Label(
        tags_bar,
        text="📦 PARAMETER: -",
        font=F_CODE,
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        padx=10,
        pady=3,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    slots_label.pack(side="left", fill="x", expand=True)

    robot_banner = tk.Frame(
        telemetry_card,
        bg=BG_CARD_GLOW,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=14,
        pady=6,
    )
    robot_banner.pack(fill="x")

    tk.Label(
        robot_banner,
        text="🤖 TINDAKAN ROBOT:",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_SUB,
        bg=BG_CARD_GLOW,
    ).pack(anchor="w")

    robot_label = tk.Label(
        robot_banner,
        text="Robot siap menerima instruksi.",
        font=F_ROBOT,
        fg=COLOR_GREEN,
        bg=BG_CARD_GLOW,
        anchor="w",
    )
    robot_label.pack(fill="x", pady=(2, 2))

    # 3b. ROS 2 Node Payload Transmission Banner
    ros2_banner = tk.Frame(
        telemetry_card,
        bg=BG_CARD_ALT,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=14,
        pady=6,
    )
    ros2_banner.pack(fill="x", pady=(6, 0))

    ros2_banner_top = tk.Frame(ros2_banner, bg=BG_CARD_ALT)
    ros2_banner_top.pack(fill="x")

    target_endpoint = getattr(pipeline.ros2_adapter, "host", "MOCK")
    if hasattr(pipeline.ros2_adapter, "port"):
        target_endpoint = f"{target_endpoint}:{pipeline.ros2_adapter.port}"

    ros2_title_label = tk.Label(
        ros2_banner_top,
        text=f"🌐 TRANSMISI PAYLOAD NODE ROS 2 [{target_endpoint}]:",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD_ALT,
    )
    ros2_title_label.pack(side="left")

    btn_cfg_inline = tk.Button(
        ros2_banner_top,
        text="⚙️ Ubah IP/Port",
        font=(FONT_FAMILY_UI, 8),
        fg=COLOR_SKY,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD,
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        padx=6,
        command=open_ros2_config_modal,
    )
    btn_cfg_inline.pack(side="left", padx=(6, 0))

    ros2_status_badge = tk.Label(
        ros2_banner_top,
        text="● MENUNGGU",
        font=(FONT_FAMILY_UI, 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD,
        padx=8,
        pady=2,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    ros2_status_badge.pack(side="right")

    ros2_detail_label = tk.Label(
        ros2_banner,
        text="Belum ada payload dikirim. Kirim instruksi untuk memicu transmisi TCP ke ROS 2.",
        font=(FONT_FAMILY_CODE, 9),
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        anchor="w",
        justify="left",
    )
    ros2_detail_label.pack(fill="x", pady=(2, 0))

    # 4. Detailed Conversation Debug Log
    dialog_card = tk.Frame(
        admin_view_frame,
        bg=BG_CARD,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=18,
        pady=10,
    )
    dialog_card.pack(fill="both", expand=True)

    card2_header = tk.Frame(dialog_card, bg=BG_CARD)
    card2_header.pack(fill="x", pady=(0, 4))

    tk.Label(
        card2_header,
        text="RIWAYAT PERCAKAPAN (LOG LENGKAP)",
        font=F_SECTION,
        fg=FG_SUB,
        bg=BG_CARD,
    ).pack(side="left")

    conv_log_label = tk.Label(
        card2_header,
        text="Tekan [STREAM TALK] atau [DIALOG TUGAS] untuk memulai",
        font=(FONT_FAMILY_UI, 9),
        fg=FG_MUTED,
        bg=BG_CARD,
    )
    conv_log_label.pack(side="right")

    log_scroll_frame = tk.Frame(
        dialog_card,
        bg=BG_BASE,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    log_scroll_frame.pack(fill="both", expand=True, pady=(4, 2))

    dialog_history_box = tk.Text(
        log_scroll_frame,
        font=F_CODE,
        bg=BG_BASE,
        fg=FG_SUB,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        wrap="word",
        padx=12,
        pady=8,
        height=7,
        state="disabled",
    )
    scrollbar = tk.Scrollbar(
        log_scroll_frame,
        command=dialog_history_box.yview,
        bg=BG_CARD,
        troughcolor=BG_BASE,
        width=12,
    )
    dialog_history_box.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    dialog_history_box.pack(side="left", fill="both", expand=True)

    dialog_history_box.tag_configure("tag_time", foreground=FG_MUTED)
    dialog_history_box.tag_configure("tag_robot", foreground=COLOR_PURPLE, font=F_CODE_BOLD)
    dialog_history_box.tag_configure("tag_user", foreground=COLOR_SKY, font=F_CODE_BOLD)
    dialog_history_box.tag_configure("tag_manual", foreground=COLOR_PRIMARY, font=F_CODE_BOLD)
    dialog_history_box.tag_configure("tag_ros2", foreground=COLOR_SKY, font=F_CODE_BOLD)
    dialog_history_box.tag_configure("tag_info", foreground=FG_SUB)

    # =========================================================================
    # VIEW SWITCHING MECHANISM
    # =========================================================================
    def show_user_view():
        nonlocal current_view
        current_view = "user"
        admin_view_frame.pack_forget()
        user_view_frame.pack(fill="both", expand=True)
        view_toggle_btn.config(
            text="🛠️  Mode Teknisi",
            bg=BG_CARD_ALT,
            fg=FG_SUB,
        )

    def show_admin_view():
        nonlocal current_view
        current_view = "admin"
        user_view_frame.pack_forget()
        admin_view_frame.pack(fill="both", expand=True)
        view_toggle_btn.config(
            text="👤  Mode Pengguna",
            bg=COLOR_PRIMARY_BG,
            fg=COLOR_PRIMARY,
        )

    def toggle_view_mode():
        if current_view == "user":
            show_admin_view()
        else:
            show_user_view()

    view_toggle_btn.config(command=toggle_view_mode)

    # Launch directly into intuitive End-User mode by default
    show_user_view()

    # Initial greeting in both chat feeds
    append_dialog_history("🤖 Robot", "Halo! Saya asisten robot panduan rumah sakit. Silakan bicara atau sentuh tombol tujuan.", COLOR_GREEN)

    # Start GUI main loop
    root.mainloop()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Robot Voice Command NLU Pipeline (GUI & Headless CLI)"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Proses teks langsung melalui NLU pipeline (tanpa microphone/STT)."
             " Contoh: --text \"antar saya ke ruang ICU\"",
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Jalankan menu interaktif di terminal (cocok untuk Ubuntu Server / SSH).",
    )

    parser.add_argument(
        "--mic",
        action="store_true",
        help="Rekam langsung dari microphone 1 kali via terminal tanpa GUI.",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Tampilkan daftar perangkat input audio (microphone) yang terdeteksi.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Index device audio microphone (default: None / default sistem).",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Durasi maksimal mendengarkan mikrofon dalam detik (default: 5).",
    )

    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Uji coba driver mikrofon, rekam sampel 3 detik, dan periksa level volume/sinyal.",
    )

    parser.add_argument(
        "--task",
        "--tugas",
        action="store_true",
        help="Jalankan mode tugas dialog interaktif non-streaming sequential (TTS ↔ STT) via terminal.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Paksa menjalankan mode antarmuka GUI Tkinter.",
    )

    parser.add_argument(
        "--ros2-host",
        type=str,
        default=None,
        help="Alamat IP / hostname target node ROS 2 robot (default: 10.107.245.221 atau tersimpan di ros2_config.json)",
    )

    parser.add_argument(
        "--ros2-port",
        type=int,
        default=None,
        help="Nomor port TCP target node ROS 2 robot (default: 5000)",
    )

    args = parser.parse_args()

    # 0. Task dialog mode via terminal
    if args.task or getattr(args, "tugas", False):
        run_task_cli(duration=args.duration, device=args.device, ros2_host=args.ros2_host, ros2_port=args.ros2_port)
        sys.exit(0)

    # 1. List audio devices
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # 2. Test mic hardware & driver
    if args.test_mic:
        test_mic_hardware(device=args.device, duration=args.duration)
        sys.exit(0)

    # 3. Text mode
    if args.text:
        run_text_mode(args.text, ros2_host=args.ros2_host, ros2_port=args.ros2_port)
        sys.exit(0)

    # 4. Direct terminal mic mode
    if args.mic:
        from stt import preload_async as stt_preload_async
        stt_preload_async()
        from nlu.pipeline import NLUPipeline
        pipeline = NLUPipeline(ros2_host=args.ros2_host, ros2_port=args.ros2_port)
        record_and_process_cli(pipeline, duration=args.duration, device=args.device)
        sys.exit(0)

    # 5. Interactive CLI mode
    if args.cli:
        run_cli_interactive(duration=args.duration, device=args.device, ros2_host=args.ros2_host, ros2_port=args.ros2_port)
        sys.exit(0)

    # 6. Default mode:
    # Cek apakah sistem memiliki display server (X11 / Wayland) atau dipaksa GUI
    has_display = bool(os.environ.get("DISPLAY")) or (sys.platform == "win32")

    if args.gui or has_display:
        run_gui_mode(duration=args.duration, device=args.device, ros2_host=args.ros2_host, ros2_port=args.ros2_port)
    else:
        # Otomatis CLI jika di Ubuntu Server / Headless tanpa display
        logger.info("\n[INFO] Menjalankan di environment Headless / Server (tanpa display).")
        logger.info("[INFO] Membuka Mode CLI Interaktif...\n")
        run_cli_interactive(duration=args.duration, device=args.device, ros2_host=args.ros2_host, ros2_port=args.ros2_port)