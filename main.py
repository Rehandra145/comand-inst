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


def resample_to_16k(samples, orig_sr: int):
    """Resample array audio float32 dari orig_sr ke 16000 Hz untuk Sherpa-ONNX."""
    import numpy as np
    if orig_sr == 16000 or len(samples) == 0:
        return samples

    if orig_sr == 48000:
        # Decimation 3:1 (48000 Hz / 3 = 16000 Hz) — sangat cepat, tanpa latency
        return samples[::3]

    if orig_sr == 32000:
        return samples[::2]

    # Interpolasi cepat untuk sample rate sembarang (misal 44100 Hz)
    target_length = int(len(samples) * 16000 / orig_sr)
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    x_old = np.linspace(0, 1, len(samples), endpoint=False)
    x_new = np.linspace(0, 1, target_length, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


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

    print(f"\n1. Perangkat Terpilih : [{device if device is not None else 'Default'}] {dev_name}")
    print(f"   Jumlah Channel     : {channels}")
    print(f"   Hardware Sample Rate: {capture_sr} Hz {'(akan di-resample otomatis ke 16000 Hz untuk STT)' if capture_sr != 16000 else ''}")

    print(f"\n2. Merekam suara sampel selama {duration} detik...")
    print("   >> SILAKAN BERBICARA ATAU BUAT SUARA SEKARANG...")

    try:
        recording = sd.rec(
            int(duration * capture_sr),
            samplerate=capture_sr,
            channels=1,
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
        sf.write("test_mic.wav", recording, sr)
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
    payload = result.get("command", {})

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


def record_and_process_cli(pipeline, duration: int = 5, device: int = None):
    """Rekam audio dari mikrofon dan proses real-time streaming di terminal."""
    device = get_effective_input_device(device)
    import sounddevice as sd
    from stt import recognizer, create_stream

    capture_rate = get_supported_sample_rate(device)
    stt_rate = 16000
    channels = 1
    stream = create_stream()

    if capture_rate != stt_rate:
        logger.info(f"[AUDIO] Hardware mic berjalan di {capture_rate} Hz (auto-resample ke {stt_rate} Hz untuk STT)")

    print(f"\n🎤 [LISTENING] Silakan berbicara (maks {duration} detik)... Tekan Ctrl+C untuk berhenti lebih awal.")
    print("   Live Audio Stream: ", end="", flush=True)

    def audio_callback(indata, frames, time_info, status_flags):
        samples = indata.flatten()
        if capture_rate != stt_rate:
            samples = resample_to_16k(samples, capture_rate)
        stream.accept_waveform(stt_rate, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        partial = recognizer.get_result(stream).strip()
        if partial:
            sys.stdout.write(f"\r   Live Audio Stream: \"\033[36m{partial}\033[0m...\" ")
            sys.stdout.flush()

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
            while time.time() - start < duration:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR Microphone]: {e}")
        return

    text = recognizer.get_result(stream).strip()
    print(f"\r   Transkripsi Akhir: \"\033[32m{text}\033[0m\"                                  \n")
    if not text:
        print("⚠️  Tidak ada ucapan/kata yang terdeteksi.")
        return

    result = pipeline.process(text)
    print_result_box(text, result)


def run_cli_interactive(duration: int = 5, device: int = None):
    """Menu CLI interaktif untuk terminal headless / Ubuntu Server."""
    from nlu.pipeline import NLUPipeline
    pipeline = NLUPipeline()

    print("\n" + "=" * 55)
    print("  🤖 ROBOT VOICE COMMAND (CLI / HEADLESS MODE)")
    print("     Optimal untuk Ubuntu Server & Raspberry Pi 4")
    print("=" * 55)

    while True:
        print("\nPilih Mode Operasi:")
        print("  [1] 🎤 Bicara via Mikrofon (Streaming STT + NLU)")
        print("  [2] ⌨️  Ketik Perintah Teks Manual")
        print("  [3] 🔍 List Perangkat Input Audio")
        print("  [4] 🧪 Tes Diagnosa & Level Volume Mikrofon")
        print("  [q] ❌ Keluar")

        try:
            choice = input("\nPilihan [1/2/3/4/q]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar...")
            break

        if choice == "1":
            record_and_process_cli(pipeline, duration=duration, device=device)
        elif choice == "2":
            try:
                txt = input("\nMasukkan perintah (misal: 'tolong antar saya ke ruang ICU'): ").strip()
                if txt:
                    result = pipeline.process(txt)
                    print_result_box(txt, result)
            except (KeyboardInterrupt, EOFError):
                pass
        elif choice == "3":
            list_audio_devices()
        elif choice == "4":
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
        logger.warning(f"\n⚠️  Gagal membuka GUI Tkinter (tidak ada Display Server / $DISPLAY): {e}")
        logger.info("👉 Beralih otomatis ke Mode CLI Interaktif (Terminal)...\n")
        run_cli_interactive(duration=duration, device=device)
        return

    device = get_effective_input_device(device)
    import sounddevice as sd
    import threading

    from stt import recognizer, create_stream
    from nlu.pipeline import NLUPipeline

    # Audio Config
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_SECONDS = duration  # Max duration (dapat dihentikan lebih awal via tombol)

    # NLU Pipeline instance
    pipeline = NLUPipeline()

    # =========================
    # GUI INITIALIZATION & FONT
    # =========================
    root.title("Robot Voice Command")
    root.geometry("720x540")
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

    # Buttons
    BTN_BG = "#2a2d32"
    BTN_ACTIVE = "#35383f"
    BTN_LISTEN_BG = "#42262b"
    BTN_LISTEN_ACTIVE = "#522e35"

    root.configure(bg=BG)

    # State tracking
    is_recording = False
    stop_event = threading.Event()
    record_thread = None

    # Thread-safe UI update helper
    def ui_call(fn):
        root.after(0, fn)

    # =========================
    # REAL-TIME STREAMING RECORD + PROCESS
    # =========================

    def record_and_process_stream():
        nonlocal is_recording
        stream = create_stream()

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

        def audio_callback(indata, frames, time_info, status):
            if stop_event.is_set():
                return
            samples = indata.flatten()
            if capture_rate != SAMPLE_RATE:
                samples = resample_to_16k(samples, capture_rate)
            stream.accept_waveform(SAMPLE_RATE, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # Live partial text update ke GUI
            partial = recognizer.get_result(stream).strip()
            if partial:
                ui_call(lambda: stt_label.config(text=f"$ stt: \"{partial}...\"", fg=CYAN))

        try:
            # Chunk size: 100 ms
            chunk_size = int(capture_rate * 0.1)

            # Stream audio real-time: proses decode berjalan paralel saat berbicara
            with sd.InputStream(
                samplerate=capture_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=chunk_size,
                device=device,
                callback=audio_callback,
            ):
                start_time = time.time()
                while time.time() - start_time < RECORD_SECONDS and not stop_event.is_set():
                    time.sleep(0.05)

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
            ui_call(lambda: status_label.config(text="> status: error", fg=RED))
            ui_call(lambda: messagebox.showerror("Error", str(e)))

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

    def on_talk_button_clicked():
        nonlocal is_recording, record_thread
        if is_recording:
            # Tombol ditekan saat sedang listening: hentikan lebih awal
            stop_event.set()
        else:
            # Memulai listening
            is_recording = True
            stop_event.clear()
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
        text=f"voice-to-command pipeline v1.0 [real-time streaming • {FONT}]",
        font=(FONT, 9),
        fg=FG_DIM,
        bg=BG,
    )
    sub_label.pack(pady=(0, 14))

    # Separator
    sep1 = tk.Frame(root, height=1, bg=BORDER)
    sep1.pack(fill="x", padx=32, pady=(0, 14))

    # Talk button
    talk_button = tk.Button(
        root,
        text="[ TALK ]",
        font=(FONT, 15, "bold"),
        fg=GREEN,
        bg=BTN_BG,
        activeforeground=GREEN,
        activebackground=BTN_ACTIVE,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=GREEN,
        width=24,
        height=2,
        cursor="hand2",
        command=on_talk_button_clicked,
    )
    talk_button.pack(pady=(0, 14))

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
        "--gui",
        action="store_true",
        help="Paksa menjalankan mode antarmuka GUI Tkinter.",
    )

    args = parser.parse_args()

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