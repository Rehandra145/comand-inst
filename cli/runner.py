"""
CLI Runner — Eksekusi command line, interactive terminal menu, dialog task, dan streaming audio CLI.
"""

import json
import logging
import queue
import sys
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

try:
    from audio.scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None

logger = logging.getLogger(__name__)


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


def listen_microphone_cli(duration: int = 5, device: Optional[int] = None, prompt: str = "   Live Audio Stream: ") -> str:
    """
    Mendengarkan mikrofon satu kali (single-shot non-streaming / blocking) untuk CLI.
    Terintegrasi dengan scheduler (enter_stt, exit_stt, task pinning).
    """
    device = get_effective_input_device(device)
    import sounddevice as sd
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


def record_and_process_cli(pipeline, duration: int = 5, device: Optional[int] = None):
    """Rekam audio dari mikrofon dan proses real-time streaming di terminal."""
    print(f"\n[LISTENING] Silakan berbicara (maks {duration} detik)... Selesai bicara otomatis diproses.")
    text = listen_microphone_cli(duration=duration, device=device, prompt="   Live Audio Stream: ")
    print(f"\r   Transkripsi Akhir: \"\033[32m{text}\033[0m\"                                  \n")
    if not text:
        print("[WARN] Tidak ada ucapan/suara yang terdeteksi.")
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
    from audio.stt import preload_async as stt_preload_async
    from nlu.pipeline import NLUPipeline
    from audio.tts import speak_and_wait, is_tts_available, warmup_async

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
            from audio.tts import play_mic_cue
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
                clarify_question = f"Mohon informasikan {translate_slot_names(missing)}."

            print(f"\n{MAGENTA}[🤖 ROBOT]{RESET} \"{clarify_question}\"")
            if tts_ok:
                speak_and_wait(clarify_question)

            try:
                from audio.tts import play_mic_cue
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
            time.sleep(1.0)
            print(f"{MAGENTA}[🤖 ROBOT]{RESET} \"{arrival_msg}\"\n")
            if tts_ok:
                speak_and_wait(arrival_msg)

    except KeyboardInterrupt:
        print("\n[INFO] Mode tugas dibatalkan oleh pengguna.\n")


def run_cli_interactive(duration: int = 5, device: Optional[int] = None, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Menu CLI interaktif untuk terminal headless / Ubuntu Server."""
    from audio.stt import preload_async as stt_preload_async
    from nlu.pipeline import NLUPipeline
    from audio.tts import warmup_async

    stt_preload_async()
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


def run_text_mode(text: str, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Process text langsung melalui NLU pipeline (tanpa STT)."""
    from nlu.pipeline import NLUPipeline
    pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)
    result = pipeline.process(text)
    print_result_box(text, result)
