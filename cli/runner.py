"""
CLI Runner — Eksekusi command line, interactive terminal menu, dialog task, dan streaming audio CLI.
"""

import json
import logging
import queue
import sys
import textwrap
import threading
import time
from typing import Optional
import numpy as np

import responses
from audio import (
    get_effective_input_device,
    get_supported_sample_rate,
    get_optimal_channels,
    resample_to_16k,
    list_audio_devices,
    test_mic_hardware,
)
from nlu.dialog import (
    translate_slot_names,
    merge_dialog_context,
    get_confirmation_message,
)

# Pastikan output terminal aman di Windows (UTF-8 & ANSI VT100 processing)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hStdOut = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode))
        kernel32.SetConsoleMode(hStdOut, mode.value | 0x0004)
    except Exception:
        pass

try:
    from audio.scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None

logger = logging.getLogger(__name__)

# =========================
# CLI DESIGN TOKENS
# =========================
CLR_CYAN = "\033[1;36m"
CLR_BLUE = "\033[1;34m"
CLR_GREEN = "\033[1;32m"
CLR_YELLOW = "\033[1;33m"
CLR_MAGENTA = "\033[1;35m"
CLR_RED = "\033[1;31m"
CLR_WHITE = "\033[1;37m"
CLR_MUTED = "\033[38;5;244m"
CLR_DIM = "\033[2m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"


def print_banner(ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Cetak banner modern identitas robot di terminal."""
    target = f"{ros2_host or '10.107.245.221'}:{ros2_port or 5000}"
    try:
        from hw_monitor import hw_monitor
        snap = hw_monitor.get_snapshot()
        accel_str = f"GPU (CUDA {snap['cuda_version'] or '12.6'})" if snap.get("is_cuda") else "CPU (Standard)"
        raw_model = snap.get("device_model", "NVIDIA Jetson")
        if "Orin Nano" in raw_model:
            chip_short = "Jetson Orin Nano"
        elif "Orin" in raw_model:
            chip_short = "Jetson Orin"
        else:
            chip_short = raw_model[:18]
    except Exception:
        accel_str = "GPU (CUDA 12.6)"
        chip_short = "Jetson Orin Nano"

    print(f"""
{CLR_CYAN}╭──────────────────────────────────────────────────────────────────────╮
│   {CLR_WHITE}🤖  R O B O T   V O I C E   A S S I S T A N T{CLR_CYAN}                      │
│   {CLR_MUTED}Hospital Autonomous Guidance & Medicine Delivery System{CLR_CYAN}            │
│  ──────────────────────────────────────────────────────────────────  │
│   {CLR_GREEN}● STATUS : ONLINE{CLR_CYAN}           {CLR_WHITE}● CORE : Zipformer RNN-T (INT8){CLR_CYAN}       │
│   {CLR_YELLOW}● TTS    : Piper VITS{CLR_CYAN}       {CLR_WHITE}● ROS2 : {target:<20}{CLR_CYAN} │
│   {CLR_GREEN}● ACCEL  : {accel_str:<20}{CLR_CYAN} {CLR_WHITE}● CHIP : {chip_short:<20}{CLR_CYAN} │
╰──────────────────────────────────────────────────────────────────────╯{CLR_RESET}""")


def print_menu_card():
    """Cetak kartu menu navigasi opsi operasi terminal."""
    print(f"""{CLR_CYAN}╭─── 🧭 PILIH MODE OPERASI ────────────────────────────────────────────╮
│                                                                      │
│   {CLR_YELLOW}[1]{CLR_RESET}  {CLR_WHITE}💬  Mode Tugas Dialog Interaktif{CLR_RESET} {CLR_MUTED}(TTS ↔ STT Sequential){CLR_CYAN}       │
│        {CLR_MUTED}Alur percakapan 2-arah terpandu dengan suara robot{CLR_CYAN}            │
│                                                                      │
│   {CLR_YELLOW}[2]{CLR_RESET}  {CLR_WHITE}🎙️   Bicara Langsung via Mikrofon{CLR_RESET} {CLR_MUTED}(Continuous Streaming){CLR_CYAN}      │
│        {CLR_MUTED}Streaming audio real-time & eksekusi NLU instan{CLR_CYAN}               │
│                                                                      │
│   {CLR_YELLOW}[3]{CLR_RESET}  {CLR_WHITE}⌨️   Ketik Perintah Teks Manual{CLR_RESET} {CLR_MUTED}(Pengujian Cepat){CLR_CYAN}             │
│        {CLR_MUTED}Uji intent & slot extraction tanpa mikrofon{CLR_CYAN}                    │
│                                                                      │
│   {CLR_YELLOW}[4]{CLR_RESET}  {CLR_WHITE}🎧  Daftar Perangkat Input Audio{CLR_RESET} {CLR_MUTED}(Hardware Info){CLR_CYAN}           │
│        {CLR_MUTED}Lihat seluruh mikrofon yang terdeteksi di sistem{CLR_CYAN}              │
│                                                                      │
│   {CLR_YELLOW}[5]{CLR_RESET}  {CLR_WHITE}⚡  Tes Diagnosa & Volume Mikrofon{CLR_RESET} {CLR_MUTED}(Mic Check){CLR_CYAN}              │
│        {CLR_MUTED}Periksa sample rate & level sensitivitas suara{CLR_CYAN}                 │
│                                                                      │
│   {CLR_YELLOW}[6]{CLR_RESET}  {CLR_WHITE}📊  Monitor Hardware (GPU vs CPU Live){CLR_RESET} {CLR_MUTED}(Tegra/CUDA){CLR_CYAN}        │
│        {CLR_MUTED}Pantau utilisasi real-time GPU (GR3D), CPU, dan suhu{CLR_CYAN}             │
│                                                                      │
│   {CLR_RED}[q]{CLR_RESET}  {CLR_WHITE}🚪  Keluar dari Aplikasi{CLR_RESET} {CLR_MUTED}(Exit){CLR_CYAN}                               │
│                                                                      │
╰──────────────────────────────────────────────────────────────────────╯{CLR_RESET}""")


def print_bubble(speaker: str, text: str, color=CLR_MAGENTA, icon="🤖", subtitle: Optional[str] = None):
    """Cetak bubble percakapan dialog dengan border rounded yang rapi."""
    print(f"\n{color}╭─ {icon} {CLR_WHITE}{speaker}{CLR_RESET}{color} " + "─" * 48)
    formatted = f'"{text.strip()}"'
    for line in textwrap.wrap(formatted, width=64) or [formatted]:
        print(f"{color}│{CLR_RESET}  {CLR_BOLD}{line}{CLR_RESET}")
    if subtitle:
        for sub in textwrap.wrap(subtitle, width=64) or [subtitle]:
            print(f"{color}│{CLR_RESET}  {CLR_YELLOW}● {sub}{CLR_RESET}")
    print(f"{color}╰" + "─" * 58 + f"{CLR_RESET}")


def wait_models_ready(verbose: bool = True):
    """
    Tunggu sampai seluruh engine model (STT Sherpa-ONNX dan TTS Piper VITS)
    selesai dimuat ke memori dan di-warmup sebelum pengguna memasukkan input di CLI.
    """
    from audio.stt import get_recognizer, is_ready
    from audio.tts import warmup, is_tts_available

    if is_ready() and is_tts_available():
        return

    if verbose:
        print(f"\n{CLR_CYAN}╭─── ⏳ INISIALISASI & WARMUP MODEL ────────────────────────────────────╮")
        print(f"│  {CLR_MUTED}Memuat model kecerdasan buatan ke memori (0ms run latency)...{CLR_CYAN}        │")
        print(f"│                                                                      │{CLR_RESET}")

    stt_logger = logging.getLogger("audio.stt")
    tts_logger = logging.getLogger("audio.tts")
    prev_stt_level = stt_logger.level
    prev_tts_level = tts_logger.level

    try:
        stt_logger.setLevel(logging.WARNING)
        tts_logger.setLevel(logging.WARNING)

        # 1. Load Model STT
        if not is_ready():
            if verbose:
                sys.stdout.write(f"{CLR_CYAN}│{CLR_RESET}  [1/2] Engine STT (Zipformer RNN-T Streaming) : Memuat... ")
                sys.stdout.flush()
            t0 = time.time()
            get_recognizer()
            if verbose:
                sys.stdout.write(f"{CLR_GREEN}[SIAP]{CLR_RESET} ({time.time() - t0:.1f}s)\n")
                sys.stdout.flush()

        # 2. Warmup Engine TTS & Cache
        if verbose:
            sys.stdout.write(f"{CLR_CYAN}│{CLR_RESET}  [2/2] Engine TTS (Piper VITS Offline & Cache): Melakukan warmup... ")
            sys.stdout.flush()
        t0 = time.time()
        warmup()
        if verbose:
            sys.stdout.write(f"{CLR_GREEN}[SIAP]{CLR_RESET} ({time.time() - t0:.1f}s)\n")
            sys.stdout.flush()

        if verbose:
            print(f"{CLR_CYAN}│                                                                      │")
            print(f"│  {CLR_GREEN}✓ Seluruh model siap! Sistem siap menerima perintah.{CLR_CYAN}               │")
            print(f"╰──────────────────────────────────────────────────────────────────────╯{CLR_RESET}\n")
    finally:
        stt_logger.setLevel(prev_stt_level)
        tts_logger.setLevel(prev_tts_level)


def print_result_box(text: str, result: dict):
    """Cetak hasil transkripsi dan NLU dalam format terminal card yang modern."""
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

    if status == "VALID":
        status_badge = f"{CLR_GREEN}● VALID{CLR_RESET}"
        border_col = CLR_GREEN
    elif status == "INVALID_LOCATION":
        status_badge = f"{CLR_RED}✕ INVALID LOCATION{CLR_RESET}"
        border_col = CLR_RED
    elif status == "INCOMPLETE":
        status_badge = f"{CLR_YELLOW}▲ INCOMPLETE (Slot Kurang){CLR_RESET}"
        border_col = CLR_YELLOW
    else:
        status_badge = f"{CLR_YELLOW}? {status}{CLR_RESET}"
        border_col = CLR_CYAN

    ros2_sent = result.get("ros2_sent", False)
    ros2_target = result.get("ros2_target", "")
    ros2_err = result.get("ros2_error", "")

    if ros2_sent:
        ros2_status = f"{CLR_GREEN}✓ TERKIRIM{CLR_RESET} ke {ros2_target}"
    elif status == "VALID":
        ros2_status = f"{CLR_RED}✕ GAGAL KIRIM{CLR_RESET} ke {ros2_target} ({ros2_err})"
    else:
        ros2_status = f"{CLR_MUTED}⏸ DITAHAN{CLR_RESET} (status {status})"

    print(f"\n{border_col}╭─── 🧠 HASIL ANALISIS NLU & TELEMETRI ──────────────────────────────╮{CLR_RESET}")
    print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Audio Transkripsi :{CLR_RESET} {CLR_CYAN}\"{text}\"{CLR_RESET}")
    print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Intent Terdeteksi :{CLR_RESET} {CLR_BOLD}{intent}{CLR_RESET}  {CLR_MUTED}[Confidence: {conf*100:.0f}%]{CLR_RESET}")
    slots_str = json.dumps(slots, ensure_ascii=False) if slots else "-"
    print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Ekstraksi Slot    :{CLR_RESET} {CLR_WHITE}{slots_str}{CLR_RESET}")
    print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Status Validasi   :{CLR_RESET} {status_badge}")
    if "invalid_room" in result:
        print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Ruangan Salah     :{CLR_RESET} {CLR_RED}{result['invalid_room']}{CLR_RESET}")
    print(f"{border_col}│{CLR_RESET}  {CLR_BOLD}Koneksi ROS 2     :{CLR_RESET} {ros2_status}")
    print(f"{border_col}│{CLR_RESET}")
    print(f"{border_col}│{CLR_RESET}  {CLR_MUTED}── ROS 2 Structured Payload ────────────────────────────────────{CLR_RESET}")

    payload_lines = json.dumps(payload, indent=2, ensure_ascii=False).split("\n")
    for pl in payload_lines:
        print(f"{border_col}│{CLR_RESET}    {CLR_CYAN}{pl}{CLR_RESET}")
    print(f"{border_col}╰────────────────────────────────────────────────────────────────────╯{CLR_RESET}\n")


def listen_microphone_cli(duration: int = 5, device: Optional[int] = None, prompt: str = "   ● Mendengar: ") -> str:
    """
    Mendengarkan mikrofon satu kali (single-shot non-streaming / blocking) untuk CLI.
    Terintegrasi dengan scheduler (enter_stt, exit_stt, task pinning).
    """
    device = get_effective_input_device(device)
    import sounddevice as sd
    import audio.stt as _stt

    if not _stt.is_ready():
        sys.stdout.write(f"  {CLR_CYAN}│{CLR_RESET}  [STT] Menyiapkan model STT...")
        sys.stdout.flush()
        _t0 = time.time()
        _stt.get_recognizer()
        sys.stdout.write(f"\r  {CLR_CYAN}│{CLR_RESET}  [STT] Model siap ({time.time() - _t0:.1f}s).            \n")
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
                    sys.stdout.write(f"\r  {CLR_CYAN}│{CLR_RESET}  ● Mendengar: \"{CLR_WHITE}{partial}{CLR_RESET}...\" ")
                    sys.stdout.flush()

                if is_endpoint:
                    if partial:
                        last_text = partial
                        local_stop.set()
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
            while time.time() - start < duration and not stop_event.is_set() and not local_stop.is_set():
                time.sleep(0.05)
                if last_text and last_change_time and (time.time() - last_change_time > 2.2):
                    local_stop.set()
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n  {CLR_RED}[ERROR Microphone]: {e}{CLR_RESET}")
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


def record_and_process_cli(pipeline, duration: int = 5, device: Optional[int] = None):
    """Rekam audio dari mikrofon dan proses real-time streaming di terminal."""
    wait_models_ready(verbose=True)
    print(f"\n{CLR_CYAN}╭─ 🎙️  STREAM TALK (Maks {duration} detik) ──────────────────────────────────{CLR_RESET}")
    print(f"{CLR_CYAN}│{CLR_RESET}  Silakan berbicara... (Berhenti bicara otomatis diproses)")
    text = listen_microphone_cli(duration=duration, device=device, prompt=f"{CLR_CYAN}│{CLR_RESET}  ● Mendengar: ")
    print(f"{CLR_CYAN}╰─────────────────────────────────────────────────────────────────────{CLR_RESET}")

    if not text:
        print(f"  {CLR_YELLOW}[WARN] Tidak ada ucapan/suara yang terdeteksi.{CLR_RESET}\n")
        return

    result = pipeline.process(text)
    print_result_box(text, result)


def run_task_cli(duration: int = 5, device: Optional[int] = None, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """
    Mode Tugas Dialog Interaktif Non-Streaming Sequential di Terminal (CLI).

    Arsitektur Sequential & Penjadwalan Resource:
    - Mutual exclusion penuh: Audio Output (TTS) dan Audio Input (STT) tidak berebut hardware
    - Settling delay 150ms agar mikrofon tidak menangkap gema speaker
    - Audio caching: kalimat umum di-sintesis dengan 0 ms latency
    - Task pinning & scheduling priority diterapkan otomatis
    """
    print_banner(ros2_host=ros2_host, ros2_port=ros2_port)

    # Pastikan seluruh engine model selesai dimuat dan di-warmup sebelum interaksi
    wait_models_ready(verbose=True)

    from nlu.pipeline import NLUPipeline
    from audio.tts import speak_and_wait, is_tts_available

    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    tts_ok = is_tts_available()

    try:
        # === STEP 1: Greeting TTS ===
        greeting = responses.greeting()
        print_bubble("ROBOT", greeting, CLR_MAGENTA, "🤖")
        if tts_ok:
            speak_and_wait(greeting)

        # Mainkan nada lembut penanda mikrofon siap merekam
        try:
            from audio.tts import play_mic_cue
            play_mic_cue()
        except Exception:
            pass

        # === STEP 2: Dengarkan Perintah Pengguna ===
        print(f"\n{CLR_CYAN}╭─ 🎙️  MIKROFON AKTIF (Maks {duration} detik) ─────────────────────────────────{CLR_RESET}")
        print(f"{CLR_CYAN}│{CLR_RESET}  Silakan berbicara perintah Anda...")
        command_text = listen_microphone_cli(duration=duration, device=device, prompt=f"{CLR_CYAN}│{CLR_RESET}  ● Mendengar: ")
        print(f"{CLR_CYAN}╰─────────────────────────────────────────────────────────────────────{CLR_RESET}")

        if not command_text:
            print(f"\n  {CLR_RED}[WARN] Tidak ada suara terdeteksi.{CLR_RESET}")
            no_sound_msg = "Maaf, saya tidak mendengar perintah. Silakan coba lagi."
            print_bubble("ROBOT", no_sound_msg, CLR_RED, "🤖")
            if tts_ok:
                speak_and_wait(no_sound_msg)
            return

        print_bubble("PENGGUNA", command_text, CLR_GREEN, "👤")

        # === STEP 3: NLU Processing ===
        print(f"  {CLR_YELLOW}⚙  Memproses intent dan ekstraksi slot NLU...{CLR_RESET}")
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
                clarify_question = f"Mohon informasikan {translate_slot_names(missing)}."

            print_bubble("ROBOT (Klarifikasi Ruangan)", clarify_question, CLR_YELLOW, "🤖")
            if tts_ok:
                speak_and_wait(clarify_question)

            try:
                from audio.tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            print(f"\n{CLR_CYAN}╭─ 🎙️  MIKROFON AKTIF (Jawaban) ───────────────────────────────────────{CLR_RESET}")
            print(f"{CLR_CYAN}│{CLR_RESET}  Silakan sebutkan nama ruangan tujuan...")
            answer_text = listen_microphone_cli(duration=duration, device=device, prompt=f"{CLR_CYAN}│{CLR_RESET}  ● Mendengar: ")
            print(f"{CLR_CYAN}╰─────────────────────────────────────────────────────────────────────{CLR_RESET}")

            if answer_text:
                print_bubble("PENGGUNA", answer_text, CLR_GREEN, "👤")
                combined_text = merge_dialog_context(command_text, answer_text, missing)
                print(f"  {CLR_YELLOW}⚙  Konteks dialog digabungkan: \"{combined_text}\"{CLR_RESET}")
                result = pipeline.process(combined_text)
                final_text = combined_text
            else:
                print(f"\n  {CLR_RED}[WARN] Tidak ada jawaban diterima. Menggunakan hasil parsial.{CLR_RESET}")
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
            
            print_bubble("ROBOT (Konfirmasi Tugas)", ask_confirm, CLR_YELLOW, "⚠️", guide_confirm)
            if tts_ok:
                speak_and_wait(ask_confirm)

            try:
                from audio.tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            print(f"\n{CLR_CYAN}╭─ 🎙️  MIKROFON AKTIF (Konfirmasi Benar / Tidak) ──────────────────────{CLR_RESET}")
            print(f"{CLR_CYAN}│{CLR_RESET}  {guide_confirm}")
            yes_no_ans = listen_microphone_cli(duration=duration, device=device, prompt=f"{CLR_CYAN}│{CLR_RESET}  ● Mendengar: ")
            print(f"{CLR_CYAN}╰─────────────────────────────────────────────────────────────────────{CLR_RESET}")

            ans_clean = (yes_no_ans or "").lower().strip()
            user_spoken = yes_no_ans.strip() if yes_no_ans else "(tidak ada suara terdeteksi)"

            if any(w in ans_clean for w in ["tidak", "bukan", "batal", "jangan", "salah", "no"]):
                cancel_msg = "Baik, pengantaran dibatalkan."
                print(f"\n  {CLR_RED}[✕ KONFIRMASI DIBATALKAN]{CLR_RESET} Kata terdeteksi: \"{user_spoken}\"")
                print_bubble("ROBOT", cancel_msg, CLR_RED, "🤖")
                if tts_ok:
                    speak_and_wait(cancel_msg)
                return
            else:
                confirmed_word = user_spoken if yes_no_ans else "benar"
                print(f"\n  {CLR_GREEN}[✓ KONFIRMASI DISETUJUI]{CLR_RESET} Kata terdeteksi: \"{confirmed_word}\"")

        print_bubble("ROBOT", confirm_msg, CLR_MAGENTA, "🤖")
        if tts_ok and confirm_msg:
            speak_and_wait(confirm_msg)

        # Follow-up kontekstual jika berhasil dieksekusi
        if result["status"] == "VALID" and result.get("intent") == "NAVIGATE":
            target_loc = result.get("slots", {}).get("location", "tujuan")
            arrival_msg = f"Robot telah tiba di {target_loc}. Ada yang bisa saya bantu lagi?"
            print(f"\n  {CLR_CYAN}🚀 Robot dalam perjalanan ke {target_loc}... (simulasi navigasi){CLR_RESET}")
            time.sleep(1.0)
            print_bubble("ROBOT", arrival_msg, CLR_GREEN, "🤖")
            if tts_ok:
                speak_and_wait(arrival_msg)

    except KeyboardInterrupt:
        print(f"\n{CLR_MUTED}[INFO] Mode tugas dibatalkan oleh pengguna.{CLR_RESET}\n")


def run_cli_interactive(duration: int = 5, device: Optional[int] = None, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Menu CLI interaktif untuk terminal headless / Ubuntu Server."""
    print_banner(ros2_host=ros2_host, ros2_port=ros2_port)

    # Tunggu seluruh model dimuat & di-warmup sebelum menu interaktif ditampilkan
    wait_models_ready(verbose=True)

    from nlu.pipeline import NLUPipeline
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)

    while True:
        print_menu_card()
        try:
            choice = input(f"   {CLR_CYAN}❯{CLR_RESET} {CLR_BOLD}Pilih menu [1/2/3/4/5/6/q]:{CLR_RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CLR_MUTED}Keluar dari program...{CLR_RESET}\n")
            break

        if choice == "1":
            run_task_cli(duration=duration, device=device, ros2_host=ros2_host, ros2_port=ros2_port)
        elif choice == "2":
            record_and_process_cli(pipeline, duration=duration, device=device)
        elif choice == "3":
            try:
                print(f"\n  {CLR_CYAN}⌨️  Mode Ketik Manual{CLR_RESET}")
                txt = input(f"  {CLR_CYAN}❯{CLR_RESET} {CLR_BOLD}Masukkan perintah (misal: 'tolong antar saya ke ruang ICU'):{CLR_RESET} ").strip()
                if txt:
                    result = pipeline.process(txt)
                    print_result_box(txt, result)
            except (KeyboardInterrupt, EOFError):
                pass
        elif choice == "4":
            list_audio_devices()
        elif choice == "5":
            test_mic_hardware(device=device, duration=duration)
        elif choice == "6":
            from hw_monitor import run_live_monitor_cli
            run_live_monitor_cli()
        elif choice in ["q", "quit", "exit"]:
            print(f"\n  {CLR_GREEN}✓ Keluar dari program. Sampai jumpa!{CLR_RESET}\n")
            break
        else:
            print(f"\n  {CLR_RED}[!] Pilihan tidak valid, silakan ketik nomor 1-6 atau q.{CLR_RESET}")


def run_text_mode(text: str, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Process text langsung melalui NLU pipeline (tanpa STT)."""
    from nlu.pipeline import NLUPipeline
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    result = pipeline.process(text)
    print_result_box(text, result)
