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
import sys
import time


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


# =========================
# TEXT MODE
# =========================

def run_text_mode(text: str):
    """Process text langsung melalui NLU pipeline (tanpa STT)."""

    from nlu.pipeline import NLUPipeline

    pipeline = NLUPipeline()
    result = pipeline.process(text)

    print()
    print("=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


# =========================
# GUI MODE
# =========================

def run_gui_mode():
    """Jalankan GUI Tkinter dengan microphone + real-time streaming STT + NLU pipeline."""

    import tkinter as tk
    from tkinter import messagebox
    import tkinter.font as tkfont
    import sounddevice as sd
    import threading

    from stt import recognizer, create_stream
    from nlu.pipeline import NLUPipeline

    # Audio Config
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_SECONDS = 5  # Max duration (dapat dihentikan lebih awal via tombol)

    # NLU Pipeline instance
    pipeline = NLUPipeline()

    # =========================
    # GUI INITIALIZATION & FONT
    # =========================
    root = tk.Tk()
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

        def audio_callback(indata, frames, time_info, status):
            if stop_event.is_set():
                return
            samples = indata.flatten()
            stream.accept_waveform(SAMPLE_RATE, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # Live partial text update ke GUI
            partial = recognizer.get_result(stream).strip()
            if partial:
                ui_call(lambda: stt_label.config(text=f"$ stt: \"{partial}...\"", fg=CYAN))

        try:
            # Chunk size: 100 ms (1600 samples @ 16kHz)
            chunk_size = int(SAMPLE_RATE * 0.1)

            # Stream audio real-time: proses decode berjalan paralel saat berbicara
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=chunk_size,
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
        description="Robot Voice Command MVP"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Process text langsung (tanpa microphone/STT)."
             " Contoh: --text \"antar saya ke ruang ICU\"",
    )

    args = parser.parse_args()

    if args.text:
        run_text_mode(args.text)
    else:
        run_gui_mode()