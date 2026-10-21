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
    if "missing_slots" in result:
        print(f" {BOLD}Missing Slots     :{RESET} {RED}{result['missing_slots']}{RESET}")
    if "invalid_room" in result:
        print(f" {BOLD}Invalid Room      :{RESET} {RED}{result['invalid_room']}{RESET}")
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


def get_confirmation_message(result: dict) -> tuple:
    """
    Mengembalikan tuple (robot_label_text, tts_speech_message) dari hasil validasi NLU.
    """
    vstatus = result["status"]
    intent = result.get("intent", "UNKNOWN")
    slots = result.get("slots", {})

    if vstatus == "VALID":
        if intent == "NAVIGATE":
            loc = slots.get('location', '?')
            return f"$ robot: menuju {loc}", responses.navigate_confirm(loc)
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
        else:
            return "$ robot: stand by", "Perintah diterima."
    elif vstatus == "INCOMPLETE":
        missing = ", ".join(result.get("missing_slots", []))
        return f"$ robot: perintah tidak lengkap (kurang: {missing})", f"Maaf, perintah belum lengkap. Informasi {missing} belum ada."
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

    last_text = ""
    last_change_time = None

    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    def audio_callback(indata, frames, time_info, status_flags):
        if stop_event.is_set():
            return
        audio_q.put(indata.copy())

    def decode_worker():
        nonlocal last_text, last_change_time
        set_thread_high_priority()
        while not stop_event.is_set():
            try:
                indata = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

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
                        stop_event.set()
                        return
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
            while time.time() - start < duration and not stop_event.is_set():
                time.sleep(0.05)
                # Auto-stop jika pengguna sudah hening 1.2 detik setelah berucap
                if last_text and last_change_time and (time.time() - last_change_time > 1.2):
                    stop_event.set()
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR Microphone]: {e}")
        return ""
    finally:
        stop_event.set()
        worker.join(timeout=2)
        if scheduler:
            scheduler.exit_stt()

    text = recognizer.get_result(stream).strip() or last_text
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


def run_task_cli(duration: int = 5, device: int = None):
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
    pipeline = NLUPipeline()
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
                clarify_question = f"Mohon informasikan {', '.join(missing)}."

            print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{clarify_question}\"")
            if tts_ok:
                speak_and_wait(clarify_question)

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
        print(f"{MAGENTA}[🤖 ROBOT]{RESET} \"{confirm_msg}\"\n")
        if tts_ok and confirm_msg:
            speak_and_wait(confirm_msg)

    except KeyboardInterrupt:
        print("\n[INFO] Mode tugas dibatalkan oleh pengguna.\n")


def run_cli_interactive(duration: int = 5, device: int = None):
    """Menu CLI interaktif untuk terminal headless / Ubuntu Server."""
    from stt import preload_async as stt_preload_async
    from nlu.pipeline import NLUPipeline
    from tts import warmup_async

    # Load model STT di background selagi menu ditampilkan (agar siap saat mode dipilih)
    stt_preload_async()
    # Prewarm audio cache & engine di background
    warmup_async()
    pipeline = NLUPipeline()

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
            run_task_cli(duration=duration, device=device)
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

def run_text_mode(text: str):
    """Process text langsung melalui NLU pipeline (tanpa STT)."""

    from nlu.pipeline import NLUPipeline

    pipeline = NLUPipeline()
    result = pipeline.process(text)
    print_result_box(text, result)


# =========================
# GUI MODE
# =========================

def run_gui_mode(duration: int = 5, device: int = None):
    """Jalankan GUI Tkinter dengan microphone + real-time streaming STT + NLU pipeline."""

    try:
        import tkinter as tk
        from tkinter import messagebox
        import tkinter.font as tkfont
        root = tk.Tk()
    except Exception as e:
        logger.warning(f"\n[WARN] Gagal membuka GUI Tkinter (tidak ada Display Server / $DISPLAY): {e}")
        logger.info("[INFO] Beralih otomatis ke Mode CLI Interaktif (Terminal)...\n")
        run_cli_interactive(duration=duration, device=device)
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

    # Audio Config
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_SECONDS = duration  # Max duration (dapat dihentikan lebih awal via tombol)

    # NLU Pipeline instance
    pipeline = NLUPipeline()

    # Prewarm model TTS & cache audio di background
    try:
        from tts import warmup_async
        warmup_async()
    except Exception:
        pass

    # =========================
    # GUI INITIALIZATION & FONT
    # =========================
    root.title("Robot Voice Command")
    root.geometry("780x680")
    root.resizable(False, False)

    # Deteksi font JetBrains Mono (fallback ke Consolas jika belum terpasang)
    installed_fonts = set(tkfont.families())
    FONT = "JetBrains Mono" if "JetBrains Mono" in installed_fonts else "Consolas"

    # =========================
    # COLORS (Matte Charcoal Theme)
    # =========================
    BG = "#18191c"            # Deep charcoal background
    BG_PANEL = "#202226"      # Elevated charcoal card surface
    BORDER = "#2f3238"        # Subtle charcoal separator & border
    FG = "#e2e4e9"            # Clean light gray/white text
    FG_DIM = "#828792"        # Muted charcoal secondary text

    # Action / Status Accents
    GREEN = "#45d686"         # Terminal emerald / success
    YELLOW = "#fbbf24"        # Terminal amber / listening
    RED = "#f87171"           # Terminal coral / error
    CYAN = "#38bdf8"          # Terminal sky cyan / streaming STT
    ORANGE = "#fb923c"        # Terminal orange / interactive task mode

    # Buttons
    BTN_BG = "#2a2d32"
    BTN_ACTIVE = "#35383f"
    BTN_LISTEN_BG = "#42262b"
    BTN_LISTEN_ACTIVE = "#522e35"
    BTN_TASK_BG = "#2b2a1e"
    BTN_TASK_ACTIVE = "#3a381f"

    root.configure(bg=BG)

    # State tracking
    is_recording = False
    is_task_running = False
    stop_event = threading.Event()
    record_thread = None
    task_thread = None

    # Thread-safe UI update helper
    def ui_call(fn):
        root.after(0, fn)

    # =========================
    # NON-STREAMING STT (single-shot recording untuk interactive mode)
    # =========================

    def stt_listen_once(listen_duration: int = 8) -> str:
        """
        Rekam audio dan transkripsi secara non-streaming (blocking).
        Mengembalikan teks hasil STT.

        Arsitektur non-streaming menghindari race condition:
        - Tidak ada concurrent consumer audio output
        - TTS selesai dulu → baru STT mulai merekam
        - STT selesai dulu → baru proses NLU
        """
        if not stt_is_ready():
            ui_call(lambda: status_label.config(text="> status: menyiapkan model STT...", fg=YELLOW))
            ui_call(lambda: stt_label.config(text="$ stt: (menyiapkan model...)", fg=YELLOW))
            create_stream()
            ui_call(lambda: stt_label.config(text="$ stt: (mendengarkan...)", fg=CYAN))
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
            while not local_stop.is_set() and not stop_event.is_set():
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
                        if partial != last_text:
                            last_text = partial
                            last_change_time = time.time()
                        # Update UI live partial
                        ui_call(lambda p=partial: stt_label.config(
                            text=f'$ stt: "{p}..."', fg=CYAN
                        ))

                    if is_endpoint:
                        if partial:
                            last_text = partial
                            local_stop.set()
                            return
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
                start_t = time.time()
                while (time.time() - start_t < listen_duration
                       and not local_stop.is_set()
                       and not stop_event.is_set()):
                    time.sleep(0.05)
                    # Auto-stop jika pengguna sudah hening 1.5s setelah berucap
                    if last_text and last_change_time and (time.time() - last_change_time > 1.5):
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

        text = recognizer.get_result(stream).strip() or last_text
        return text

    # =========================
    # INTERACTIVE TASK MODE (non-streaming, sequential TTS↔STT)
    # =========================

    def interactive_task_flow():
        """
        Mode interaktif non-streaming dengan penjadwalan sequential:

        Alur:
          1. TTS: "Halo, ada yang bisa saya bantu hari ini?"
          2. STT: Dengarkan perintah pengguna (misal "antarkan obat ini")
          3. NLU: Proses intent + slot extraction
          4. Jika slot belum lengkap (misal ruangan belum disebutkan):
             a. TTS: "Ke ruangan berapa?"
             b. STT: Dengarkan jawaban ruangan
             c. NLU: Proses ulang dengan context gabungan
          5. Tampilkan hasil akhir + kirim ke ROS 2

        Arsitektur sequential ini meminimalisir race condition:
        - TTS dan STT TIDAK pernah berjalan bersamaan
        - Hanya satu consumer audio output pada satu waktu
        - Resource audio (speaker + mic) digunakan secara bergantian
        """
        nonlocal is_task_running

        try:
            from tts import speak_and_wait, is_tts_available
        except ImportError:
            logger.error("[TASK] Module tts.py tidak ditemukan")
            ui_call(lambda: status_label.config(text="> status: error (tts module missing)", fg=RED))
            is_task_running = False
            _reset_task_button()
            return

        tts_ok = is_tts_available()

        def update_conv_log(speaker: str, text: str, color: str = FG):
            """Tambahkan entry ke conversation log di GUI."""
            ui_call(lambda: conv_log_label.config(
                text=f"  {speaker}: {text}", fg=color
            ))

        try:
            # === STEP 1: Greeting TTS ===
            greeting = responses.greeting()
            ui_call(lambda: status_label.config(text="> status: robot berbicara...", fg=ORANGE))
            update_conv_log("🤖 Robot", greeting, ORANGE)
            ui_call(lambda: stt_label.config(text="$ stt: (menunggu giliran bicara...)", fg=FG_DIM))
            ui_call(lambda: intent_label.config(text="$ intent: -", fg=FG_DIM))
            ui_call(lambda: slots_label.config(text="$ slots: -", fg=FG_DIM))
            ui_call(lambda: validation_label.config(text="$ validation: -", fg=FG_DIM))
            ui_call(lambda: robot_label.config(text="$ robot: berbicara...", fg=ORANGE))

            if tts_ok:
                speak_and_wait(greeting)
            else:
                logger.warning("[TASK] TTS tidak tersedia, skip greeting audio")
                time.sleep(1)  # Beri jeda visual

            if stop_event.is_set():
                return

            # === STEP 2: Listen for command ===
            ui_call(lambda: status_label.config(text="> status: mendengarkan perintah...", fg=YELLOW))
            ui_call(lambda: stt_label.config(text='$ stt: (mendengarkan...)', fg=CYAN))
            update_conv_log("🎤 Anda", "(sedang mendengarkan...)", YELLOW)
            ui_call(lambda: robot_label.config(text="$ robot: mendengarkan perintah", fg=YELLOW))

            command_text = stt_listen_once(listen_duration=RECORD_SECONDS + 3)

            if stop_event.is_set():
                return

            if not command_text:
                ui_call(lambda: status_label.config(text="> status: tidak ada suara terdeteksi", fg=RED))
                ui_call(lambda: stt_label.config(text='$ stt: (tidak ada suara)', fg=RED))
                update_conv_log("🎤 Anda", "(tidak ada suara terdeteksi)", RED)
                ui_call(lambda: robot_label.config(text="$ robot: tidak ada perintah", fg=RED))

                # TTS konfirmasi tidak ada suara
                no_sound_msg = "Maaf, saya tidak mendengar perintah. Silakan coba lagi."
                update_conv_log("🤖 Robot", no_sound_msg, RED)
                if tts_ok:
                    speak_and_wait(no_sound_msg)
                return

            logger.info(f"[TASK] Command text: \"{command_text}\"")
            ui_call(lambda: stt_label.config(text=f'$ stt: "{command_text}"', fg=FG))
            update_conv_log("🎤 Anda", f'"{command_text}"', GREEN)

            # === STEP 3: NLU Processing ===
            ui_call(lambda: status_label.config(text="> status: memproses perintah...", fg=YELLOW))
            result = pipeline.process(command_text)

            if stop_event.is_set():
                return

            # === STEP 4: Check if INCOMPLETE (missing slots) ===
            if result["status"] == "INCOMPLETE" and result.get("missing_slots"):
                missing = result["missing_slots"]
                logger.info(f"[TASK] Slot belum lengkap: {missing}")

                # Tentukan pertanyaan klarifikasi berdasarkan slot yang kurang
                if "location" in missing:
                    clarify_question = responses.ask_location()
                elif "direction" in missing:
                    clarify_question = "Ke arah mana?"
                elif "person" in missing:
                    clarify_question = "Siapa yang harus saya ikuti?"
                else:
                    slot_names = ", ".join(missing)
                    clarify_question = f"Mohon informasikan {slot_names}."

                # Show intermediate result
                _show_partial_result(result)

                # TTS tanya klarifikasi
                ui_call(lambda: status_label.config(text="> status: robot bertanya...", fg=ORANGE))
                update_conv_log("🤖 Robot", clarify_question, ORANGE)
                ui_call(lambda: robot_label.config(text=f"$ robot: \"{clarify_question}\"", fg=ORANGE))

                if tts_ok:
                    speak_and_wait(clarify_question)
                else:
                    time.sleep(0.5)

                if stop_event.is_set():
                    return

                # STT dengarkan jawaban
                ui_call(lambda: status_label.config(text="> status: mendengarkan jawaban...", fg=YELLOW))
                ui_call(lambda: stt_label.config(text='$ stt: (mendengarkan jawaban...)', fg=CYAN))
                update_conv_log("🎤 Anda", "(mendengarkan jawaban...)", YELLOW)

                answer_text = stt_listen_once(listen_duration=RECORD_SECONDS + 3)

                if stop_event.is_set():
                    return

                if answer_text:
                    logger.info(f"[TASK] Answer text: \"{answer_text}\"")
                    ui_call(lambda: stt_label.config(text=f'$ stt: "{answer_text}"', fg=FG))
                    update_conv_log("🎤 Anda", f'"{answer_text}"', GREEN)

                    # Gabungkan context: perintah awal + jawaban secara cerdas
                    combined_text = merge_dialog_context(command_text, answer_text, missing)
                    logger.info(f"[TASK] Combined text: \"{combined_text}\"")

                    # Re-process NLU dengan context gabungan
                    ui_call(lambda: status_label.config(text="> status: memproses ulang...", fg=YELLOW))
                    result = pipeline.process(combined_text)
                else:
                    logger.info("[TASK] Tidak ada jawaban — gunakan hasil parsial")
                    update_conv_log("🎤 Anda", "(tidak ada jawaban)", RED)

            # === STEP 5: Display final result ===
            _show_final_result(result, tts_ok)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[TASK ERROR] {err_msg}")
            ui_call(lambda: status_label.config(text="> status: error", fg=RED))
            ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            is_task_running = False
            stop_event.clear()
            _reset_task_button()

    def _show_partial_result(result):
        """Tampilkan hasil NLU parsial (saat masih ada slot yang kurang)."""
        intent_text = f"$ intent: {result['intent']}  (confidence: {result['confidence']:.2f})"
        slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(none)"
        missing_str = ", ".join(result.get("missing_slots", []))

        def apply():
            intent_label.config(text=intent_text, fg=CYAN)
            slots_label.config(text=f"$ slots: {slots_str}", fg=FG)
            validation_label.config(text=f"$ validation: INCOMPLETE (kurang: {missing_str})", fg=YELLOW)
        ui_call(apply)

    def _show_final_result(result, tts_ok=False):
        """Tampilkan hasil NLU final dan ucapkan konfirmasi via TTS."""
        intent_text = f"$ intent: {result['intent']}  (confidence: {result['confidence']:.2f})"
        slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(none)"
        vstatus = result["status"]

        if vstatus == "VALID":
            vcolor = GREEN
        elif vstatus == "INVALID_LOCATION":
            vcolor = RED
        else:
            vcolor = YELLOW

        # Robot text & color + TTS confirmation
        confirm_msg = ""
        if vstatus == "VALID":
            rcolor = GREEN
            intent = result["intent"]
            slots = result["slots"]
            if intent == "NAVIGATE":
                loc = slots.get('location', '?')
                rtext = f"$ robot: menuju {loc}"
                confirm_msg = responses.navigate_confirm(loc)
            elif intent == "MOVE":
                direction = slots.get("direction", "?")
                distance = slots.get("distance", "")
                unit = slots.get("unit", "")
                dist_str = f" {distance} {unit}" if distance else ""
                rtext = f"$ robot: bergerak {direction}{dist_str}"
                confirm_msg = f"Baik, saya bergerak ke {direction}{dist_str}."
            elif intent == "STOP":
                rtext = "$ robot: berhenti"
                confirm_msg = "Baik, saya berhenti."
            elif intent == "GO_BACK":
                rtext = "$ robot: kembali"
                confirm_msg = "Baik, saya kembali."
            elif intent == "WAIT":
                rtext = "$ robot: menunggu"
                confirm_msg = "Baik, saya akan menunggu."
            elif intent == "FOLLOW":
                person = slots.get('person', 'user')
                rtext = f"$ robot: mengikuti {person}"
                confirm_msg = f"Baik, saya akan mengikuti {person}."
            else:
                rtext = "$ robot: stand by"
                confirm_msg = "Perintah diterima."
        elif vstatus == "INCOMPLETE":
            rcolor = YELLOW
            missing = ", ".join(result.get("missing_slots", []))
            rtext = f"$ robot: perintah tidak lengkap (kurang: {missing})"
            confirm_msg = f"Maaf, perintah belum lengkap. Informasi {missing} belum ada."
        elif vstatus == "INVALID_LOCATION":
            rcolor = RED
            loc = result['slots'].get('location', '?')
            rtext = f'$ robot: lokasi "{loc}" tidak ditemukan'
            confirm_msg = f"Maaf, lokasi {loc} tidak ditemukan."
        elif vstatus == "UNKNOWN_INTENT":
            rcolor = RED
            confirm_msg = responses.unknown_intent()
            rtext = f"$ robot: {confirm_msg.lower()}"
        else:
            rcolor = RED
            rtext = "$ robot: perintah tidak dikenali"
            confirm_msg = "Maaf, saya tidak mengenali perintah tersebut."

        def apply_final():
            intent_label.config(text=intent_text, fg=CYAN)
            slots_label.config(text=f"$ slots: {slots_str}", fg=FG)
            validation_label.config(text=f"$ validation: {vstatus}", fg=vcolor)
            robot_label.config(text=rtext, fg=rcolor)
            status_label.config(text="> status: selesai", fg=GREEN)
            conv_log_label.config(text=f"  🤖 Robot: {confirm_msg}", fg=rcolor)
        ui_call(apply_final)

        # TTS konfirmasi akhir
        if tts_ok and confirm_msg:
            try:
                from tts import speak_and_wait
                speak_and_wait(confirm_msg)
            except Exception as e:
                logger.error(f"[TTS] Gagal konfirmasi: {e}")

    def _reset_task_button():
        """Reset tombol TUGAS ke state awal."""
        ui_call(lambda: task_button.config(
            text="[ TUGAS ]",
            bg=BTN_TASK_BG,
            activebackground=BTN_TASK_ACTIVE,
            fg=ORANGE,
            activeforeground=ORANGE,
            state="normal",
        ))
        ui_call(lambda: talk_button.config(state="normal"))

    def on_task_button_clicked():
        """Handler untuk tombol TUGAS (interactive non-streaming mode)."""
        nonlocal is_task_running, task_thread
        if is_task_running:
            # Tombol ditekan saat sedang berjalan: batalkan
            stop_event.set()
            return

        if is_recording:
            # Sedang streaming TALK, jangan mulai TUGAS
            return

        is_task_running = True
        stop_event.clear()

        # Disable TALK button saat TUGAS berjalan
        ui_call(lambda: talk_button.config(state="disabled"))
        ui_call(lambda: task_button.config(
            text="[ BATALKAN ]",
            bg=BTN_LISTEN_BG,
            activebackground=BTN_LISTEN_ACTIVE,
            fg=RED,
            activeforeground=RED,
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
        """Rekam dan proses audio streaming (queue-based, aman untuk Raspi)."""
        nonlocal is_recording

        stop_event.clear()
        if not stt_is_ready():
            ui_call(lambda: status_label.config(text="> status: menyiapkan model STT...", fg=YELLOW))
            ui_call(lambda: stt_label.config(text="$ stt: (menyiapkan model...)", fg=YELLOW))
        stream = create_stream()
        audio_q = _queue.Queue()

        ui_call(lambda: talk_button.config(
            text="[ STOP LISTENING ]",
            bg=BTN_LISTEN_BG,
            activebackground=BTN_LISTEN_ACTIVE,
            fg=RED,
            activeforeground=RED,
        ))
        ui_call(lambda: status_label.config(text="> status: listening (streaming real-time)...", fg=YELLOW))
        ui_call(lambda: stt_label.config(text="$ stt: (listening...)", fg=CYAN))
        ui_call(lambda: intent_label.config(text="$ intent: -", fg=FG_DIM))
        ui_call(lambda: slots_label.config(text="$ slots: -", fg=FG_DIM))
        ui_call(lambda: validation_label.config(text="$ validation: -", fg=FG_DIM))
        ui_call(lambda: robot_label.config(text="$ robot: waiting", fg=FG_DIM))

        capture_rate = get_supported_sample_rate(device)
        optimal_channels = get_optimal_channels(device, capture_rate)

        def audio_callback(indata, frames, time_info, status):
            """Callback audio — HANYA salin data ke queue, JANGAN decode di sini."""
            if stop_event.is_set():
                return
            audio_q.put(indata.copy())

        def gui_decode_worker():
            """Worker thread: konsumsi audio dari queue dan jalankan decode STT."""
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
                        ui_call(lambda: stt_label.config(text=f"$ stt: \"{partial}...\"", fg=CYAN))

                    if is_endpoint:
                        if partial:
                            stop_event.set()
                            return
                        else:
                            recognizer.reset(stream)

                except Exception as err:
                    logger.error(f"[AUDIO] GUI decode worker error: {err}")

        # Mulai decode worker thread
        decode_thread = threading.Thread(target=gui_decode_worker, daemon=True)
        decode_thread.start()

        try:
            # Chunk size: 100 ms
            chunk_size = int(capture_rate * 0.1)

            # Stream audio real-time: callback hanya copy ke queue, decode di worker thread
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

            # Pastikan worker selesai sebelum ambil hasil
            stop_event.set()
            decode_thread.join(timeout=2)

            # Selesai merekam — hasil sudah terdecode secara konkruen
            t_decode_start = time.time()
            text = recognizer.get_result(stream).strip()
            decode_latency_ms = (time.time() - t_decode_start) * 1000

            logger.info(f"[STT] text = \"{text}\" (final latency: {decode_latency_ms:.2f}ms)")
            ui_call(lambda: stt_label.config(text=f"$ stt: \"{text}\"", fg=FG))

            # NLU Pipeline
            ui_call(lambda: status_label.config(text="> status: processing nlu...", fg=YELLOW))
            result = pipeline.process(text)

            # Prepare UI updates
            intent_text = f"$ intent: {result['intent']}  (confidence: {result['confidence']:.2f})"
            slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(none)"
            vstatus = result["status"]

            if vstatus == "VALID":
                vcolor = GREEN
            elif vstatus == "INVALID_LOCATION":
                vcolor = RED
            else:
                vcolor = YELLOW

            # Robot text & color
            if vstatus == "VALID":
                rcolor = GREEN
                intent = result["intent"]
                slots = result["slots"]
                if intent == "NAVIGATE":
                    rtext = f"$ robot: menuju {slots.get('location', '?')}"
                elif intent == "MOVE":
                    direction = slots.get("direction", "?")
                    distance = slots.get("distance", "")
                    unit = slots.get("unit", "")
                    dist_str = f" {distance} {unit}" if distance else ""
                    rtext = f"$ robot: bergerak {direction}{dist_str}"
                elif intent == "STOP":
                    rtext = "$ robot: berhenti"
                elif intent == "GO_BACK":
                    rtext = "$ robot: kembali"
                elif intent == "WAIT":
                    rtext = "$ robot: menunggu"
                elif intent == "FOLLOW":
                    rtext = f"$ robot: mengikuti {slots.get('person', 'user')}"
                else:
                    rtext = "$ robot: stand by"
            elif vstatus == "INCOMPLETE":
                rcolor = YELLOW
                missing = ", ".join(result.get("missing_slots", []))
                rtext = f"$ robot: command tidak lengkap (kurang: {missing})"
            elif vstatus == "INVALID_LOCATION":
                rcolor = RED
                rtext = f"$ robot: lokasi \"{result['slots'].get('location', '?')}\" tidak ditemukan"
            elif vstatus == "UNKNOWN_INTENT":
                rcolor = RED
                rtext = f"$ robot: {responses.unknown_intent().lower()}"
            else:
                rcolor = RED
                rtext = "$ robot: perintah tidak dikenali"

            # Apply UI updates
            def apply_results():
                intent_label.config(text=intent_text, fg=CYAN)
                slots_label.config(text=f"$ slots: {slots_str}", fg=FG)
                validation_label.config(text=f"$ validation: {vstatus}", fg=vcolor)
                robot_label.config(text=rtext, fg=rcolor)
                status_label.config(text="> status: ready", fg=GREEN)

            ui_call(apply_results)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[GUI ERROR] {err_msg}")
            ui_call(lambda: status_label.config(text="> status: error", fg=RED))
            ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            is_recording = False
            stop_event.clear()
            ui_call(lambda: talk_button.config(
                text="[ TALK ]",
                bg=BTN_BG,
                activebackground=BTN_ACTIVE,
                fg=GREEN,
                activeforeground=GREEN,
            ))
            ui_call(lambda: task_button.config(state="normal"))

    def on_talk_button_clicked():
        nonlocal is_recording, record_thread
        if is_recording:
            # Tombol ditekan saat sedang listening: hentikan lebih awal
            stop_event.set()
        elif is_task_running:
            # Sedang dalam mode TUGAS, jangan mulai TALK
            return
        else:
            # Memulai listening
            is_recording = True
            stop_event.clear()
            # Disable TUGAS button saat TALK berjalan
            ui_call(lambda: task_button.config(state="disabled"))
            record_thread = threading.Thread(
                target=record_and_process_stream,
                daemon=True,
            )
            record_thread.start()

    # =========================
    # WIDGETS
    # =========================

    # Title
    title_label = tk.Label(
        root,
        text="ROBOT VOICE COMMAND",
        font=(FONT, 18, "bold"),
        fg=GREEN,
        bg=BG,
    )
    title_label.pack(pady=(24, 4))

    # Subtitle
    sub_label = tk.Label(
        root,
        text=f"voice-to-command pipeline v1.1 [streaming + interactive • {FONT}]",
        font=(FONT, 9),
        fg=FG_DIM,
        bg=BG,
    )
    sub_label.pack(pady=(0, 14))

    # Separator
    sep1 = tk.Frame(root, height=1, bg=BORDER)
    sep1.pack(fill="x", padx=32, pady=(0, 14))

    # Button frame (TALK + TUGAS side by side)
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=(0, 14))

    # Talk button (streaming mode)
    talk_button = tk.Button(
        btn_frame,
        text="[ TALK ]",
        font=(FONT, 13, "bold"),
        fg=GREEN,
        bg=BTN_BG,
        activeforeground=GREEN,
        activebackground=BTN_ACTIVE,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=GREEN,
        width=16,
        height=2,
        cursor="hand2",
        command=on_talk_button_clicked,
    )
    talk_button.pack(side="left", padx=(0, 8))

    # Task button (interactive non-streaming mode)
    task_button = tk.Button(
        btn_frame,
        text="[ TUGAS ]",
        font=(FONT, 13, "bold"),
        fg=ORANGE,
        bg=BTN_TASK_BG,
        activeforeground=ORANGE,
        activebackground=BTN_TASK_ACTIVE,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ORANGE,
        width=16,
        height=2,
        cursor="hand2",
        command=on_task_button_clicked,
    )
    task_button.pack(side="left", padx=(8, 0))

    # Mode labels
    mode_frame = tk.Frame(root, bg=BG)
    mode_frame.pack(pady=(0, 4))
    tk.Label(
        mode_frame,
        text="streaming",
        font=(FONT, 8),
        fg=FG_DIM,
        bg=BG,
    ).pack(side="left", padx=(28, 60))
    tk.Label(
        mode_frame,
        text="interactive",
        font=(FONT, 8),
        fg=FG_DIM,
        bg=BG,
    ).pack(side="left", padx=(60, 28))

    # Separator
    sep2 = tk.Frame(root, height=1, bg=BORDER)
    sep2.pack(fill="x", padx=32, pady=(0, 12))

    # Output frame (matte charcoal terminal panel)
    output_frame = tk.Frame(
        root,
        bg=BG_PANEL,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
        padx=18,
        pady=14,
    )
    output_frame.pack(fill="both", expand=True, padx=32, pady=(0, 22))

    # Status
    status_label = tk.Label(
        output_frame,
        text="> status: ready",
        font=(FONT, 10, "bold"),
        fg=GREEN,
        bg=BG_PANEL,
        anchor="w",
    )
    status_label.pack(fill="x", pady=2)

    # STT result
    stt_label = tk.Label(
        output_frame,
        text="$ stt: -",
        font=(FONT, 10),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
        wraplength=620,
        justify="left",
    )
    stt_label.pack(fill="x", pady=2)

    # Intent
    intent_label = tk.Label(
        output_frame,
        text="$ intent: -",
        font=(FONT, 10),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
    )
    intent_label.pack(fill="x", pady=2)

    # Slots
    slots_label = tk.Label(
        output_frame,
        text="$ slots: -",
        font=(FONT, 10),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
        wraplength=620,
        justify="left",
    )
    slots_label.pack(fill="x", pady=2)

    # Validation
    validation_label = tk.Label(
        output_frame,
        text="$ validation: -",
        font=(FONT, 10),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
    )
    validation_label.pack(fill="x", pady=2)

    # Separator inside panel
    sep3 = tk.Frame(output_frame, height=1, bg=BORDER)
    sep3.pack(fill="x", pady=8)

    # Robot status
    robot_label = tk.Label(
        output_frame,
        text="$ robot: waiting",
        font=(FONT, 11, "bold"),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
    )
    robot_label.pack(fill="x", pady=2)

    # Separator inside panel (before conversation log)
    sep4 = tk.Frame(output_frame, height=1, bg=BORDER)
    sep4.pack(fill="x", pady=6)

    # Conversation log (interactive mode dialog)
    conv_log_label = tk.Label(
        output_frame,
        text="  dialog: (tekan TUGAS untuk mode interaktif)",
        font=(FONT, 9),
        fg=FG_DIM,
        bg=BG_PANEL,
        anchor="w",
        wraplength=680,
        justify="left",
    )
    conv_log_label.pack(fill="x", pady=2)

    # Start GUI
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

    args = parser.parse_args()

    # 0. Task dialog mode via terminal
    if args.task or getattr(args, "tugas", False):
        run_task_cli(duration=args.duration, device=args.device)
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
        run_text_mode(args.text)
        sys.exit(0)

    # 4. Direct terminal mic mode
    if args.mic:
        from stt import preload_async as stt_preload_async
        stt_preload_async()
        from nlu.pipeline import NLUPipeline
        pipeline = NLUPipeline()
        record_and_process_cli(pipeline, duration=args.duration, device=args.device)
        sys.exit(0)

    # 5. Interactive CLI mode
    if args.cli:
        run_cli_interactive(duration=args.duration, device=args.device)
        sys.exit(0)

    # 5. Default mode:
    # Cek apakah sistem memiliki display server (X11 / Wayland) atau dipaksa GUI
    has_display = bool(os.environ.get("DISPLAY")) or (sys.platform == "win32")

    if args.gui or has_display:
        run_gui_mode(duration=args.duration, device=args.device)
    else:
        # Otomatis CLI jika di Ubuntu Server / Headless tanpa display
        logger.info("\n[INFO] Menjalankan di environment Headless / Server (tanpa display).")
        logger.info("[INFO] Membuka Mode CLI Interaktif...\n")
        run_cli_interactive(duration=args.duration, device=args.device)