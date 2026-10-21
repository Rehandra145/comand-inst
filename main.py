"""
Robot Voice Command — Main Entry Point.

Usage:
    # Text mode (tanpa microphone/STT) — untuk testing NLU:
    python main.py --text "tolong antar saya ke ruang ICU"

    # GUI mode (microphone + real-time streaming STT + NLU):
    python main.py
    python main.py --gui

    # CLI Task mode (dialog interaktif sequential TTS ↔ STT):
    python main.py --task

    # CLI Interactive menu (cocok untuk SSH / Ubuntu Server):
    python main.py --cli

    # Diagnosa mikrofon:
    python main.py --test-mic
    python main.py --list-devices
"""

import argparse
import logging
import os
import sys

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

# =========================
# MODULAR IMPORTS & RE-EXPORTS
# =========================
from audio_utils import (
    get_effective_input_device,
    get_supported_sample_rate,
    get_optimal_channels,
    resample_to_16k,
    list_audio_devices,
    test_mic_hardware,
)
from dialog_utils import (
    translate_slot_names,
    merge_dialog_context,
    get_confirmation_message,
)
from cli_runner import (
    print_result_box,
    listen_microphone_cli,
    record_and_process_cli,
    run_task_cli,
    run_cli_interactive,
    run_text_mode,
)
from gui import run_gui_mode


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