"""
GUI App Controller — Orkestrator utama aplikasi GUI Robot Voice Assistant (Tkinter).
"""

import json
import logging
import queue as _queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkfont
from typing import Optional
import numpy as np

import responses
from audio import (
    get_effective_input_device,
    get_supported_sample_rate,
    get_optimal_channels,
    resample_to_16k,
)
from nlu.dialog import translate_slot_names, merge_dialog_context
from gui.theme import (
    BG_BASE,
    BG_CARD,
    BG_CARD_ALT,
    BORDER,
    BORDER_HOVER,
    FG_WHITE,
    FG_SUB,
    FG_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BG,
    COLOR_PRIMARY_BORDER,
    COLOR_GREEN,
    COLOR_GREEN_BG,
    COLOR_GREEN_BORDER,
    COLOR_AMBER,
    COLOR_AMBER_BG,
    COLOR_AMBER_BORDER,
    COLOR_RED,
    COLOR_RED_BG,
    COLOR_RED_BORDER,
    COLOR_SKY,
    COLOR_SKY_BG,
    COLOR_SKY_BORDER,
    COLOR_PURPLE,
    COLOR_PURPLE_BG,
    COLOR_PURPLE_BORDER,
    get_theme_fonts,
    add_hover_effect,
)
from gui.modals import open_devices_modal, open_ros2_config_modal, open_restart_confirm_modal
from gui.user_view import UserView
from gui.admin_view import AdminView

try:
    from audio.scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None

logger = logging.getLogger(__name__)


class RobotGUIApp:
    """Kontroler Aplikasi Antarmuka Grafis Dual-View Robot."""

    def __init__(self, root: tk.Tk, duration: int = 5, device: Optional[int] = None, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
        self.root = root
        self.duration = duration
        self.device = get_effective_input_device(device)
        self.sample_rate = 16000
        self.record_seconds = duration

        # Prewarm & background loaders (sekuensial STT -> TTS untuk stabilitas C++ runtime)
        from audio.stt import recognizer, create_stream, is_ready as stt_is_ready, get_recognizer
        self.stt_recognizer = recognizer
        self.stt_create_stream = create_stream
        self.stt_is_ready = stt_is_ready

        def _bg_prewarm():
            try:
                get_recognizer()
            except Exception as e:
                logger.error(f"[STT] Preload model error: {e}")
            try:
                from audio.tts import warmup, precache_room_phrases
                warmup()
                precache_room_phrases()
            except Exception as e:
                logger.error(f"[TTS] Warmup model error: {e}")

        threading.Thread(target=_bg_prewarm, daemon=True, name="Model-Prewarm").start()

        from nlu.pipeline import NLUPipeline
        self.pipeline = NLUPipeline(ros2_host=ros2_host, ros2_port=ros2_port)

        # State tracking
        self.is_recording = False
        self.is_task_running = False
        self.stop_event = threading.Event()
        self.record_thread = None
        self.task_thread = None
        self.current_view = "user"

        # Theme & Fonts
        self.fonts = get_theme_fonts(tkfont)
        self.root.title("Robot Voice Assistant • Control Center")
        self.root.geometry("960x860")
        self.root.minsize(900, 780)
        self.root.configure(bg=BG_BASE)

        # Build Global Header
        self._build_header()

        # Build Container & Dual-Views
        self.container = tk.Frame(self.root, bg=BG_BASE)
        self.container.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.user_view = UserView(self.container, self, self.fonts)
        self.admin_view = AdminView(self.container, self, self.fonts)

        # Default view: User mode (Kiosk)
        self.show_user_view()

        # Update ROS 2 Title in Admin View
        self._refresh_ros2_header()

        # Initial greeting in chat feeds
        self.append_dialog_history("🤖 Robot", "Halo! Saya asisten robot panduan rumah sakit. Silakan bicara atau sentuh tombol tujuan.", COLOR_GREEN)

        # Mulai loop telemetri monitor hardware GPU vs CPU
        self._tick_hw_monitor()

    def _build_header(self):
        header_frame = tk.Frame(
            self.root,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=10,
        )
        header_frame.pack(fill="x", padx=20, pady=(16, 12))

        header_left = tk.Frame(header_frame, bg=BG_CARD)
        header_left.pack(side="left", fill="y")

        avatar_box = tk.Label(
            header_left,
            text="🤖",
            font=(self.fonts["FONT_FAMILY_UI"], 20),
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
            font=self.fonts["F_TITLE"],
            fg=FG_WHITE,
            bg=BG_CARD,
        )
        title_label.pack(anchor="w")

        sub_label = tk.Label(
            title_box,
            text="Layanan Mandiri Pasien & Pengunjung • Sentuh Layar atau Bicara Langsung",
            font=self.fonts["F_SUBTITLE"],
            fg=FG_SUB,
            bg=BG_CARD,
        )
        sub_label.pack(anchor="w")

        header_right = tk.Frame(header_frame, bg=BG_CARD)
        header_right.pack(side="right", fill="y")

        self.view_toggle_btn = tk.Button(
            header_right,
            text="🛠️  Mode Teknisi",
            font=(self.fonts["FONT_FAMILY_UI"], 9, "bold"),
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
            command=self.toggle_view_mode,
        )
        self.view_toggle_btn.pack(side="right", padx=(10, 0))
        add_hover_effect(self.view_toggle_btn, BG_CARD_ALT, BORDER, BORDER, BORDER_HOVER)

        self.status_badge = tk.Label(
            header_right,
            text="● SIAP",
            font=self.fonts["F_CODE_BOLD"],
            fg=COLOR_GREEN,
            bg=COLOR_GREEN_BG,
            padx=10,
            pady=4,
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLOR_GREEN_BORDER,
        )
        self.status_badge.pack(side="right")

        self.compute_badge_label = tk.Label(
            header_right,
            text="⚡ GPU: CUDA (0%)",
            font=self.fonts["F_CODE_BOLD"],
            fg=COLOR_GREEN,
            bg=COLOR_GREEN_BG,
            padx=10,
            pady=4,
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLOR_GREEN_BORDER,
        )
        self.compute_badge_label.pack(side="right", padx=(6, 0))

        dev_short = "Default"
        try:
            import sounddevice as sd
            cur_dev = sd.query_devices(self.device, "input")
            name = cur_dev.get("name", "Default")
            dev_short = name[:16] + ".." if len(name) > 16 else name
        except Exception:
            pass

        self.mic_badge_label = tk.Label(
            header_right,
            text=f"🎙️ {dev_short}",
            font=self.fonts["F_CODE"],
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            padx=8,
            pady=4,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.mic_badge_label.pack(side="right", padx=(6, 0))

    def _refresh_ros2_header(self):
        target_endpoint = getattr(self.pipeline.ros2_adapter, "host", "MOCK")
        if hasattr(self.pipeline.ros2_adapter, "port"):
            target_endpoint = f"{target_endpoint}:{self.pipeline.ros2_adapter.port}"
        self.admin_view.ros2_title_label.config(text=f"🌐 TRANSMISI PAYLOAD NODE ROS 2 [{target_endpoint}]:")

    def _tick_hw_monitor(self):
        """Update telemetri hardware GPU vs CPU secara periodik non-blocking."""
        try:
            from hw_monitor import hw_monitor
            snap = hw_monitor.get_snapshot()
            gpu_load = snap.get("gpu_load", 0.0)
            is_cuda = snap.get("is_cuda", True)

            if is_cuda:
                badge_text = f"⚡ GPU: CUDA ({gpu_load:.0f}%)"
                if gpu_load > 2.0:
                    badge_fg = COLOR_GREEN
                    badge_bg = COLOR_GREEN_BG
                    badge_border = COLOR_GREEN_BORDER
                else:
                    badge_fg = COLOR_SKY
                    badge_bg = COLOR_SKY_BG
                    badge_border = COLOR_SKY_BORDER
            else:
                cpu_load = snap.get("cpu_load", 0.0)
                badge_text = f"💻 CPU ({cpu_load:.0f}%)"
                badge_fg = COLOR_AMBER
                badge_bg = COLOR_AMBER_BG
                badge_border = COLOR_AMBER_BORDER

            if hasattr(self, "compute_badge_label") and self.compute_badge_label.winfo_exists():
                self.compute_badge_label.config(
                    text=badge_text,
                    fg=badge_fg,
                    bg=badge_bg,
                    highlightbackground=badge_border,
                )

            if hasattr(self, "admin_view") and hasattr(self.admin_view, "update_hw_telemetry"):
                self.admin_view.update_hw_telemetry(snap)
        except Exception as e:
            logger.debug(f"[HW-MONITOR] Tick error: {e}")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(1000, self._tick_hw_monitor)
            except Exception:
                pass

    def ui_call(self, fn):
        """Thread-safe UI invocation via root.after."""
        try:
            self.root.after(0, fn)
        except Exception:
            pass

    def show_user_view(self):
        self.current_view = "user"
        self.admin_view.frame.pack_forget()
        self.user_view.frame.pack(fill="both", expand=True)
        self.view_toggle_btn.config(
            text="🛠️  Mode Teknisi",
            bg=BG_CARD_ALT,
            fg=FG_SUB,
        )

    def show_admin_view(self):
        self.current_view = "admin"
        self.user_view.frame.pack_forget()
        self.admin_view.frame.pack(fill="both", expand=True)
        self.view_toggle_btn.config(
            text="👤  Mode Pengguna",
            bg=COLOR_PRIMARY_BG,
            fg=COLOR_PRIMARY,
        )

    def toggle_view_mode(self):
        if self.current_view == "user":
            self.show_admin_view()
        else:
            self.show_user_view()

    def open_devices_dialog(self):
        open_devices_modal(self.root, self.device, self.fonts, self.on_device_selected)

    def on_device_selected(self, dev_id: int, dev_name: str):
        self.device = dev_id
        short_n = dev_name[:16] + ".." if len(dev_name) > 16 else dev_name
        self.mic_badge_label.config(text=f"🎙️ {short_n}")
        self.admin_view.status_label.config(text=f"● Mikrofon dialihkan ke ID {dev_id}", fg=COLOR_GREEN)

    def open_ros2_dialog(self):
        open_ros2_config_modal(self.root, self.pipeline, self.fonts, self.ui_call, self.on_ros2_endpoint_updated)

    def set_compute_mode(self, mode: str):
        """Ubah preferensi mode akselerasi komputasi (GPU vs CPU)."""
        from audio.provider import save_compute_mode, is_cuda_available
        if mode == "gpu" and not is_cuda_available():
            self.append_dialog_history("⚙️ Sistem", "⚠️ Hardware / pustaka CUDA tidak terdeteksi pada perangkat ini.", COLOR_AMBER)
            self.admin_view.update_compute_mode_ui("cpu", status_msg="⚠️ GPU tidak tersedia, tetap CPU")
            return

        save_compute_mode(mode)
        mode_label = "GPU (CUDA)" if mode == "gpu" else "CPU (Standard)"
        self.admin_view.update_compute_mode_ui(mode, status_msg=f"Mode diatur ke {mode_label}. Klik Restart untuk terapkan.")
        self.append_dialog_history(
            "⚙️ Sistem",
            f"Mode akselerasi diatur ke {mode_label}. Klik tombol [🔄 Restart Aplikasi] untuk menerapkan perubahan.",
            COLOR_AMBER,
        )

    def on_restart_app_clicked(self):
        """Konfirmasi dan restart aplikasi untuk menerapkan mode baru menggunakan modal kustom."""
        from audio.provider import load_compute_mode
        cur_mode = load_compute_mode().upper()
        open_restart_confirm_modal(
            root=self.root,
            fonts=self.fonts,
            target_mode=cur_mode,
            on_confirm=self.restart_application,
        )

    def restart_application(self):
        """Restart proses aplikasi secara bersih."""
        try:
            if hasattr(self, "audio_worker") and self.audio_worker is not None:
                self.audio_worker.stop()
        except Exception:
            pass

        from audio.provider import restart_current_process
        restart_current_process(self.root)

    def on_ros2_endpoint_updated(self, host: str, port: int):
        new_target = f"{host}:{port}"
        self.admin_view.ros2_title_label.config(text=f"🌐 TRANSMISI PAYLOAD NODE ROS 2 [{new_target}]:")
        self.admin_view.ros2_status_badge.config(
            text="● MENUNGGU",
            fg=FG_MUTED,
            bg=BG_CARD,
            highlightbackground=BORDER,
        )
        self.admin_view.ros2_detail_label.config(
            text=f"Target diperbarui ke [{new_target}]. Belum ada payload dikirim.",
            fg=FG_SUB,
        )
        self.append_dialog_history("🌐 ROS 2", f"Konfigurasi endpoint diperbarui ke {new_target} (disimpan)", COLOR_SKY)

    def set_user_ui_state(self, face: str, title: str, subtitle: str, color: str = COLOR_GREEN):
        """Update tampilan persona, status, dan badge pada antarmuka End-User."""
        def _apply():
            if self.user_view.user_face_label:
                self.user_view.user_face_label.config(text=face, fg=color)
            if self.user_view.user_status_title:
                self.user_view.user_status_title.config(text=title)
            if self.user_view.user_status_sub:
                self.user_view.user_status_sub.config(text=subtitle)
            if self.user_view.user_action_pill:
                self.user_view.user_action_pill.config(text=f"● {title}", fg=color)
            if self.user_view.user_live_speech_label:
                self.user_view.user_live_speech_label.config(text=f"🎙️ Status: {title}", fg=color)
            if self.user_view.user_chat_status_label:
                self.user_view.user_chat_status_label.config(text=f"● {title}", fg=color)
        self.ui_call(_apply)

    def update_ros2_ui_status(self, result: dict):
        """Update indikator visual dan log transmisi payload ROS 2 di Mode Teknisi."""
        def _apply():
            badge = self.admin_view.ros2_status_badge
            detail = self.admin_view.ros2_detail_label
            if not badge or not detail:
                return

            sent = result.get("ros2_sent", False)
            cur_h = getattr(self.pipeline.ros2_adapter, "host", "10.107.245.221")
            cur_p = getattr(self.pipeline.ros2_adapter, "port", 5000)
            target = result.get("ros2_target") or f"{cur_h}:{cur_p}"
            err = result.get("ros2_error")
            status = result.get("status", "UNKNOWN")
            intent = result.get("intent", "-")
            payload = result.get("ros2_payload", "")

            if sent:
                badge.config(
                    text="● TERKIRIM ✓",
                    fg=COLOR_GREEN,
                    bg=COLOR_GREEN_BG,
                    highlightbackground=COLOR_GREEN_BORDER,
                )
                detail.config(
                    text=f"✓ Payload berhasil dikirim ke Node ROS 2 [{target}] (Intent: {intent})",
                    fg=COLOR_GREEN,
                )
                log_msg = f"✓ Payload BERHASIL dikirim ke {target} | Intent: {intent}"
                if payload:
                    log_msg += f"\n   Payload: {payload}"
                self.append_dialog_history("🌐 ROS 2", log_msg, COLOR_GREEN)
            elif status == "VALID":
                badge.config(
                    text="✕ GAGAL KIRIM",
                    fg=COLOR_RED,
                    bg=COLOR_RED_BG,
                    highlightbackground=COLOR_RED_BORDER,
                )
                err_text = f" ({err})" if err else ""
                detail.config(
                    text=f"✕ Gagal terhubung ke Node ROS 2 [{target}]{err_text}",
                    fg=COLOR_RED,
                )
                log_msg = f"✕ GAGAL kirim payload ke {target}{err_text}"
                if payload:
                    log_msg += f"\n   Payload tertahan: {payload}"
                self.append_dialog_history("🌐 ROS 2", log_msg, COLOR_RED)
            else:
                badge.config(
                    text="⏸ DITAHAN",
                    fg=COLOR_AMBER,
                    bg=COLOR_AMBER_BG,
                    highlightbackground=COLOR_AMBER_BORDER,
                )
                detail.config(
                    text=f"⏸ Payload ditahan (status validasi: {status})",
                    fg=COLOR_AMBER,
                )
                self.append_dialog_history("🌐 ROS 2", f"⏸ Payload DITAHAN (status={status}, belum valid dieksekusi)", COLOR_AMBER)

        self.ui_call(_apply)

    def append_dialog_history(self, speaker: str, message: str, color: Optional[str] = None):
        """Tambahkan baris percakapan ke riwayat Admin Console dan Chat Balon Pengguna."""
        def _append():
            import datetime
            now_t = datetime.datetime.now().strftime("%H:%M:%S")

            # 1. Update Admin Console
            admin_box = self.admin_view.dialog_history_box
            if admin_box:
                try:
                    admin_box.config(state="normal")
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
                        admin_box.tag_configure(msg_tag, foreground=color)

                    admin_box.insert(tk.END, f"[{now_t}] ", "tag_time")
                    admin_box.insert(tk.END, f"{speaker}: ", spk_tag)
                    admin_box.insert(tk.END, f"{message}\n", msg_tag)
                    admin_box.see(tk.END)
                    admin_box.config(state="disabled")
                except Exception:
                    pass

            # 2. Update End-User Chat Display
            user_box = self.user_view.user_chat_box
            if user_box:
                if "ros" in speaker.lower():
                    return

                try:
                    user_box.config(state="normal")
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

                    user_box.insert(tk.END, f"[{now_t}] ", "user_tag_time")
                    user_box.insert(tk.END, role_label, role_tag)
                    user_box.insert(tk.END, f"{message}\n\n", "user_tag_body")
                    user_box.see(tk.END)
                    user_box.config(state="disabled")
                except Exception:
                    pass

        self.ui_call(_append)

    def stt_listen_once(
        self,
        listen_duration: int = 8,
        stage_title: str = "MENDENGARKAN PERINTAH",
        context_hint: str = "Silakan berbicara langsung ke mikrofon...",
    ) -> str:
        """Rekam audio dan transkripsi secara non-streaming (blocking)."""
        import sounddevice as sd
        if not self.stt_is_ready():
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Menyiapkan model STT...", fg=COLOR_AMBER))
            self.ui_call(lambda: self.admin_view.stt_label.config(text="(menyiapkan model...)", fg=COLOR_AMBER))
            self.stt_create_stream()
            self.ui_call(lambda: self.admin_view.stt_label.config(text="(mendengarkan...)", fg=COLOR_SKY))

        stream = self.stt_create_stream()
        audio_q = _queue.Queue()
        local_stop = threading.Event()

        capture_rate = get_supported_sample_rate(self.device)
        optimal_channels = get_optimal_channels(self.device, capture_rate)

        last_text = ""
        last_change_time = None

        def audio_cb(indata, frames, time_info, status):
            if local_stop.is_set() or self.stop_event.is_set():
                return
            audio_q.put(indata.copy())

        def decode_worker():
            nonlocal last_text, last_change_time
            set_thread_high_priority()
            while not self.stop_event.is_set() and not local_stop.is_set():
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

                    if capture_rate != self.sample_rate:
                        samples = resample_to_16k(samples, capture_rate)

                    stream.accept_waveform(self.sample_rate, samples)
                    while self.stt_recognizer.is_ready(stream):
                        self.stt_recognizer.decode_stream(stream)

                    is_endpoint = self.stt_recognizer.is_endpoint(stream)
                    partial = self.stt_recognizer.get_result(stream).strip()

                    if partial:
                        if partial != last_text:
                            last_text = partial
                            last_change_time = time.time()

                        sys.stdout.write(f"\r   [STT REAL-TIME] \"\033[36m{partial}\033[0m...\" ")
                        sys.stdout.flush()

                        def _update_live_ui(p=partial):
                            self.admin_view.stt_label.config(text=f'“{p}...”', fg=COLOR_SKY)
                            self.admin_view.status_label.config(text=f'● Mendeteksi: “{p}...”', fg=COLOR_SKY)
                            if self.user_view.user_voice_btn:
                                self.user_view.user_voice_btn.config(
                                    text=f"🎙️   SUARA TERDETEKSI: “{p}...”\n({context_hint})",
                                    fg=COLOR_GREEN,
                                    bg=COLOR_GREEN_BG,
                                    highlightbackground=COLOR_GREEN_BORDER,
                                )
                            if self.user_view.user_action_pill:
                                self.user_view.user_action_pill.config(
                                    text=f"● Suara Terdeteksi: “{p}...”",
                                    fg=COLOR_GREEN,
                                )
                            if self.user_view.user_live_speech_label:
                                self.user_view.user_live_speech_label.config(
                                    text=f"🎙️ Terdeteksi: “{p}...”",
                                    fg=COLOR_GREEN,
                                )
                            if self.user_view.user_face_label:
                                self.user_view.user_face_label.config(text="(👂◕‿◕)", fg=COLOR_GREEN)
                            if self.user_view.user_status_sub:
                                self.user_view.user_status_sub.config(text=f'Robot mendengar: “{p}...”', fg=COLOR_SKY)
                            if self.user_view.user_chat_status_label:
                                self.user_view.user_chat_status_label.config(text=f'● Mendengar: “{p}...”', fg=COLOR_GREEN)

                        self.ui_call(_update_live_ui)

                    if is_endpoint:
                        if partial:
                            last_text = partial
                            local_stop.set()
                            return
                        else:
                            self.stt_recognizer.reset(stream)

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
                device=self.device,
                callback=audio_cb,
            ):
                def _on_mic_ready():
                    self.admin_view.stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY)
                    self.admin_view.status_label.config(text=f"● {stage_title}: Mikrofon aktif...", fg=COLOR_SKY)
                    if self.user_view.user_voice_btn:
                        self.user_view.user_voice_btn.config(
                            text=f"🎙️   MIKROFON AKTIF — MENDENGARKAN...\n{context_hint}",
                            fg=COLOR_SKY,
                            bg=COLOR_SKY_BG,
                            highlightbackground=COLOR_SKY_BORDER,
                        )
                    if self.user_view.user_action_pill:
                        self.user_view.user_action_pill.config(
                            text="● Mikrofon Terbuka — Silakan Berbicara",
                            fg=COLOR_SKY,
                        )
                    if self.user_view.user_live_speech_label:
                        self.user_view.user_live_speech_label.config(
                            text=f"🎙️ [MIKROFON TERBUKA] {context_hint}",
                            fg=COLOR_SKY,
                        )
                    if self.user_view.user_face_label:
                        self.user_view.user_face_label.config(text="(👂‿👂)", fg=COLOR_SKY)
                    if self.user_view.user_chat_status_label:
                        self.user_view.user_chat_status_label.config(text="● Mendengarkan...", fg=COLOR_SKY)

                self.ui_call(_on_mic_ready)

                sys.stdout.write(f"\n   [🎤 MIKROFON AKTIF] {stage_title} ({context_hint})\n")
                sys.stdout.flush()

                start_t = time.time()
                while (time.time() - start_t < listen_duration
                       and not local_stop.is_set()
                       and not self.stop_event.is_set()):
                    time.sleep(0.05)
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

        text = (last_text or self.stt_recognizer.get_result(stream)).strip()
        if text:
            sys.stdout.write(f"\r   [✓ STT SELESAI] \"\033[32m{text}\033[0m\"" + " " * 30 + "\n")
        else:
            sys.stdout.write("\r   [✕ STT SELESAI] (tidak ada suara terdeteksi)" + " " * 30 + "\n")
        sys.stdout.flush()
        return text

    def interactive_task_flow(self):
        """Alur percakapan interaktif dua arah (sequential TTS ↔ STT)."""
        try:
            from audio.tts import speak_and_wait, is_tts_available
        except ImportError:
            logger.error("[TASK] Module tts.py tidak ditemukan")
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Modul TTS tidak ditemukan", fg=COLOR_RED))
            self.is_task_running = False
            self._reset_task_button()
            return

        tts_ok = is_tts_available()

        def update_conv_log(speaker: str, text: str, color: str = FG_WHITE):
            self.ui_call(lambda: self.admin_view.conv_log_label.config(
                text=f"{speaker}: {text}", fg=color
            ))
            self.append_dialog_history(speaker, text, color)

        try:
            # === STEP 1: Greeting TTS ===
            greeting = responses.greeting()
            self.set_user_ui_state("(＾▽＾)", "Robot menyapa Anda...", greeting, COLOR_PURPLE)
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Robot menyapa...", fg=COLOR_AMBER))
            self.ui_call(lambda: self.status_badge.config(
                text="● BERBICARA", fg=COLOR_AMBER, bg=COLOR_AMBER_BG, highlightbackground=COLOR_AMBER_BORDER
            ))
            update_conv_log("🤖 Robot", greeting, COLOR_AMBER)
            self.ui_call(lambda: self.admin_view.stt_label.config(text='(menunggu giliran bicara...)', fg=FG_MUTED))
            self.ui_call(lambda: self.admin_view.intent_label.config(text="🎯 INTENT: -", fg=FG_MUTED))
            self.ui_call(lambda: self.admin_view.slots_label.config(text="📦 PARAMETER: -", fg=FG_MUTED))
            self.ui_call(lambda: self.admin_view.validation_label.config(text="🛡️ VALIDASI: -", fg=FG_MUTED))
            self.ui_call(lambda: self.admin_view.robot_label.config(text="Robot: Berbicara ke pengguna...", fg=COLOR_AMBER))

            if tts_ok:
                speak_and_wait(greeting)
            else:
                logger.warning("[TASK] TTS tidak tersedia, skip greeting audio")
                time.sleep(1)

            if self.stop_event.is_set():
                return

            try:
                from audio.tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            # === STEP 2: Listen for command ===
            self.set_user_ui_state("(👂‿👂)", "Silakan Bicara Sekarang 🎙️", "Mikrofon aktif — sebutkan tujuan atau instruksi Anda.", COLOR_SKY)
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Mikrofon aktif — silakan berbicara...", fg=COLOR_SKY))
            self.ui_call(lambda: self.status_badge.config(
                text="● MENDENGARKAN", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
            ))
            self.ui_call(lambda: self.admin_view.stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY))
            update_conv_log("🎤 Anda", "(sedang mendengarkan...)", COLOR_SKY)
            self.ui_call(lambda: self.admin_view.robot_label.config(text="Robot: Mendengarkan perintah suara Anda", fg=COLOR_SKY))

            command_text = self.stt_listen_once(
                listen_duration=self.record_seconds + 3,
                stage_title="MENDENGARKAN PERINTAH",
                context_hint="Katakan tujuan Anda, misal: 'Antar saya ke ruang ICU'",
            )

            if self.stop_event.is_set():
                return

            if not command_text:
                self.set_user_ui_state("(•_•)", "Tidak ada suara terdeteksi", "Silakan coba lagi atau pilih tujuan di bawah.", COLOR_RED)
                self.ui_call(lambda: self.admin_view.status_label.config(text="● Tidak ada suara terdeteksi", fg=COLOR_RED))
                self.ui_call(lambda: self.status_badge.config(
                    text="● TIDAK ADA SUARA", fg=COLOR_RED, bg=COLOR_RED_BG, highlightbackground=COLOR_RED_BORDER
                ))
                self.ui_call(lambda: self.admin_view.stt_label.config(text='(tidak ada suara terdeteksi)', fg=COLOR_RED))
                update_conv_log("🎤 Anda", "(tidak ada suara terdeteksi)", COLOR_RED)
                self.ui_call(lambda: self.admin_view.robot_label.config(text="Robot: Tidak ada instruksi yang terdengar", fg=COLOR_RED))

                no_sound_msg = "Maaf, saya tidak mendengar perintah. Silakan coba lagi."
                update_conv_log("🤖 Robot", no_sound_msg, COLOR_RED)
                if tts_ok:
                    speak_and_wait(no_sound_msg)
                return

            logger.info(f"[TASK] Command text: \"{command_text}\"")
            self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{command_text}”', fg=FG_WHITE))
            update_conv_log("🎤 Anda", f'"{command_text}"', COLOR_GREEN)

            # === STEP 3: NLU Processing ===
            self.set_user_ui_state("(・_・)", "Memproses...", f"Robot sedang memahami: \"{command_text}\"", COLOR_AMBER)
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Memproses pemahaman bahasa (NLU)...", fg=COLOR_AMBER))
            result = self.pipeline.process(command_text)

            if self.stop_event.is_set():
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
                    clarify_question = f"Mohon informasikan {translate_slot_names(missing)}."

                self._show_partial_result(result)

                self.set_user_ui_state("(・_・)", "Mohon informasikan tujuan", clarify_question, COLOR_AMBER)
                self.ui_call(lambda: self.admin_view.status_label.config(text="● Robot meminta klarifikasi...", fg=COLOR_AMBER))
                self.ui_call(lambda: self.status_badge.config(
                    text="● KLARIFIKASI", fg=COLOR_AMBER, bg=COLOR_AMBER_BG, highlightbackground=COLOR_AMBER_BORDER
                ))
                update_conv_log("🤖 Robot", clarify_question, COLOR_AMBER)
                self.ui_call(lambda: self.admin_view.robot_label.config(text=f"Robot: \"{clarify_question}\"", fg=COLOR_AMBER))

                if tts_ok:
                    speak_and_wait(clarify_question)
                else:
                    time.sleep(0.5)

                if self.stop_event.is_set():
                    return

                try:
                    from audio.tts import play_mic_cue
                    play_mic_cue()
                except Exception:
                    pass

                self.set_user_ui_state("(👂‿👂)", "Silakan Jawab Sekarang 🎙️", clarify_question, COLOR_SKY)
                self.ui_call(lambda: self.admin_view.status_label.config(text="● Mendengarkan jawaban klarifikasi...", fg=COLOR_SKY))
                self.ui_call(lambda: self.status_badge.config(
                    text="● MENDENGARKAN", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
                ))
                self.ui_call(lambda: self.admin_view.stt_label.config(text='(mendengarkan jawaban klarifikasi...)', fg=COLOR_SKY))
                update_conv_log("🎤 Anda", "(mendengarkan jawaban...)", COLOR_SKY)

                answer_text = self.stt_listen_once(
                    listen_duration=self.record_seconds + 3,
                    stage_title="KLARIFIKASI TUJUAN",
                    context_hint=clarify_question,
                )

                if self.stop_event.is_set():
                    return

                if answer_text:
                    logger.info(f"[TASK] Answer text: \"{answer_text}\"")
                    self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{answer_text}”', fg=FG_WHITE))
                    update_conv_log("🎤 Anda", f'"{answer_text}"', COLOR_GREEN)

                    combined_text = merge_dialog_context(command_text, answer_text, missing)
                    logger.info(f"[TASK] Combined text: \"{combined_text}\"")

                    self.ui_call(lambda: self.admin_view.status_label.config(text="● Memproses ulang NLU...", fg=COLOR_AMBER))
                    result = self.pipeline.process(combined_text)
                else:
                    logger.info("[TASK] Tidak ada jawaban — gunakan hasil parsial")
                    update_conv_log("🎤 Anda", "(tidak ada jawaban)", COLOR_RED)

            # === STEP 5: Display final result ===
            self._show_final_result(result, tts_ok)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[TASK ERROR] {err_msg}")
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Terjadi kesalahan pada sistem", fg=COLOR_RED))
            self.ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            self.is_task_running = False
            self.stop_event.clear()
            self._reset_task_button()

    def _show_partial_result(self, result: dict):
        """Tampilkan hasil NLU parsial (saat masih ada slot yang kurang)."""
        intent_text = f"INTENT: {result['intent']} ({result['confidence']:.2f})"
        slots_str = json.dumps(result["slots"], ensure_ascii=False) if result["slots"] else "(kosong)"
        missing_id = translate_slot_names(result.get("missing_slots", []))

        def apply_partial():
            self.admin_view.intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
            self.admin_view.slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
            self.admin_view.validation_label.config(text=f"⚠ KURANG: {missing_id}", fg=COLOR_AMBER)
        self.ui_call(apply_partial)

    def _show_final_result(self, result: dict, tts_ok: bool = False):
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
                self.set_user_ui_state("(★‿★)", f"Mengantar ke {loc or 'tujuan'}...", confirm_msg, COLOR_GREEN)
            elif intent == "MOVE":
                direction = slots.get("direction", "?")
                distance = slots.get("distance", "")
                unit = slots.get("unit", "")
                dist_str = f" {distance} {unit}" if distance else ""
                rtext = f"Robot: Bergerak {direction}{dist_str}"
                confirm_msg = f"Baik, saya bergerak ke {direction}{dist_str}."
                self.set_user_ui_state("(🚗)", f"Bergerak {direction}", confirm_msg, COLOR_GREEN)
            elif intent == "STOP":
                rtext = "Robot: Berhenti"
                confirm_msg = "Baik, saya berhenti."
                self.set_user_ui_state("(🛑)", "Robot Berhenti", confirm_msg, COLOR_RED)
            elif intent == "GO_BACK":
                rtext = "Robot: Kembali ke posisi awal"
                confirm_msg = "Baik, saya kembali."
                self.set_user_ui_state("(🏠)", "Kembali ke Pos", confirm_msg, COLOR_GREEN)
            elif intent == "WAIT":
                rtext = "Robot: Menunggu"
                confirm_msg = "Baik, saya akan menunggu."
                self.set_user_ui_state("(⏳)", "Menunggu", confirm_msg, COLOR_AMBER)
            elif intent == "FOLLOW":
                person = slots.get('person', 'user')
                rtext = f"Robot: Mengikuti {person}"
                confirm_msg = f"Baik, saya akan mengikuti {person}."
                self.set_user_ui_state("(🚶)", f"Mengikuti {person}", confirm_msg, COLOR_GREEN)
            elif intent == "EMERGENCY":
                loc = slots.get('location', '')
                if loc:
                    rtext = f"Robot: [DARURAT] Menuju {loc}"
                    confirm_msg = f"Peringatan kondisi darurat. Segera menuju {loc} sekarang juga."
                else:
                    rtext = "Robot: [DARURAT] Mode darurat diaktifkan"
                    confirm_msg = "Peringatan kondisi darurat. Mode prioritas darurat diaktifkan."
                self.set_user_ui_state("(🚨)", "MODE DARURAT", confirm_msg, COLOR_RED)
            elif intent == "STATUS":
                rtext = "Robot: Status normal (baterai 85%)"
                confirm_msg = "Status robot normal. Daya baterai delapan puluh lima persen dan siap beroperasi."
                self.set_user_ui_state("(🔋)", "Status Normal (85%)", confirm_msg, COLOR_GREEN)
            elif intent == "GREETING":
                rtext = "Robot: Menyapa pengguna"
                confirm_msg = "Halo, ada yang bisa saya bantu?"
                self.set_user_ui_state("(＾▽＾)", "Halo!", confirm_msg, COLOR_PURPLE)
            elif intent == "THANK":
                rtext = "Robot: Merespons terima kasih"
                confirm_msg = "Sama-sama, senang bisa membantu."
                self.set_user_ui_state("(◕‿◕)", "Sama-sama!", confirm_msg, COLOR_GREEN)
            else:
                rtext = "Robot: Siap siaga (stand by)"
                confirm_msg = "Perintah diterima."
                self.set_user_ui_state("(◕‿◕)", "Siap Melayani", confirm_msg, COLOR_GREEN)
        elif vstatus == "INCOMPLETE":
            rcolor = COLOR_AMBER
            missing_raw = ", ".join(result.get("missing_slots", []))
            missing_id = translate_slot_names(result.get("missing_slots", []))
            rtext = f"Robot: Perintah belum lengkap (kurang: {missing_raw})"
            confirm_msg = f"Maaf, perintah belum lengkap. Informasi {missing_id} belum ada."
            self.set_user_ui_state("(・_・)", "Perintah Belum Lengkap", confirm_msg, COLOR_AMBER)
        elif vstatus == "INVALID_LOCATION":
            rcolor = COLOR_RED
            loc = result['slots'].get('location', '?')
            rtext = f'Robot: Ruangan "{loc}" tidak ditemukan'
            confirm_msg = f"Maaf, lokasi {loc} tidak ditemukan."
            self.set_user_ui_state("(×_×)", "Lokasi Tidak Ditemukan", confirm_msg, COLOR_RED)
        elif vstatus == "UNKNOWN_INTENT":
            rcolor = COLOR_RED
            confirm_msg = responses.unknown_intent()
            rtext = f"Robot: {confirm_msg}"
            self.set_user_ui_state("(•_•)", "Perintah Tidak Dikenali", confirm_msg, COLOR_RED)
        else:
            rcolor = COLOR_RED
            rtext = "Robot: Perintah tidak dikenali"
            confirm_msg = "Maaf, saya tidak mengenali perintah tersebut."
            self.set_user_ui_state("(•_•)", "Perintah Tidak Dikenali", confirm_msg, COLOR_RED)

        def apply_final():
            self.admin_view.intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
            self.admin_view.slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
            self.admin_view.validation_label.config(text=f"{vicon} VALIDASI: {vstatus}", fg=vcolor)
            self.admin_view.robot_label.config(text=rtext, fg=rcolor)
            self.admin_view.status_label.config(text="● Status: Siap beroperasi", fg=COLOR_GREEN)
            self.status_badge.config(
                text="● SIAP", fg=COLOR_GREEN, bg=COLOR_GREEN_BG, highlightbackground=COLOR_GREEN_BORDER
            )
            self.admin_view.conv_log_label.config(text=f"🤖 Robot: {confirm_msg}", fg=rcolor)
        self.ui_call(apply_final)
        self.update_ros2_ui_status(result)

        # Dua arah konfirmasi jika NAVIGATE membawa objek penting
        if vstatus == "VALID" and result.get("intent") == "NAVIGATE" and result.get("slots", {}).get("object"):
            obj = result["slots"]["object"]
            loc = result["slots"].get("location", "")
            loc_str = f" ke {loc}" if loc else ""
            ask_confirm = f"Saya akan mengantarkan {obj}{loc_str}. Apakah sudah benar?"
            guide_confirm = "Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal."

            self.ui_call(lambda: self.admin_view.conv_log_label.config(text=f"🤖 Robot: {ask_confirm} ({guide_confirm})", fg=COLOR_AMBER))
            self.append_dialog_history("🤖 Robot", f"{ask_confirm}\n   💡 ({guide_confirm})", COLOR_AMBER)
            self.set_user_ui_state(
                "(・_・)",
                "Menunggu Konfirmasi Anda",
                f"{ask_confirm}\n({guide_confirm})",
                COLOR_AMBER
            )
            if self.user_view.user_action_pill:
                self.ui_call(lambda: self.user_view.user_action_pill.config(
                    text="● Menunggu Konfirmasi (Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal)",
                    fg=COLOR_AMBER
                ))
            if self.user_view.user_voice_btn:
                self.ui_call(lambda: self.user_view.user_voice_btn.config(
                    text="🎙️   MENUNGGU KONFIRMASI ANDA\nUcapkan \"Benar\" untuk lanjut, atau \"Tidak\" untuk batal",
                    fg=COLOR_AMBER,
                    bg=COLOR_AMBER_BG,
                    highlightbackground=COLOR_AMBER_BORDER,
                ))

            if tts_ok:
                try:
                    from audio.tts import speak_and_wait
                    speak_and_wait(ask_confirm)
                except Exception:
                    pass

            if self.stop_event.is_set():
                if self.user_view.user_voice_btn:
                    self.ui_call(lambda: self.user_view.user_voice_btn.config(
                        text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
                        fg=COLOR_PRIMARY,
                        bg=COLOR_PRIMARY_BG,
                        highlightbackground=COLOR_PRIMARY_BORDER,
                    ))
                return

            try:
                from audio.tts import play_mic_cue
                play_mic_cue()
            except Exception:
                pass

            self.ui_call(lambda: self.admin_view.status_label.config(
                text="● Menunggu konfirmasi: (Ucapkan 'benar' untuk lanjut, 'tidak' untuk batal)...",
                fg=COLOR_PURPLE
            ))
            self.ui_call(lambda: self.status_badge.config(
                text="● MENUNGGU KONFIRMASI",
                fg=COLOR_PURPLE,
                bg=COLOR_PURPLE_BG,
                highlightbackground=COLOR_PURPLE_BORDER
            ))
            ans_confirm = self.stt_listen_once(
                listen_duration=self.record_seconds + 2,
                stage_title="MENUNGGU KONFIRMASI",
                context_hint="Ucapkan 'Benar' untuk lanjut, atau 'Tidak' untuk batal",
            )
            ans_clean = (ans_confirm or "").lower().strip()
            user_spoken = ans_confirm.strip() if ans_confirm else "(tidak ada respon)"

            if self.user_view.user_voice_btn:
                self.ui_call(lambda: self.user_view.user_voice_btn.config(
                    text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
                    fg=COLOR_PRIMARY,
                    bg=COLOR_PRIMARY_BG,
                    highlightbackground=COLOR_PRIMARY_BORDER,
                ))

            if any(w in ans_clean for w in ["tidak", "bukan", "batal", "jangan", "salah", "no"]):
                cancel_msg = "Baik, pengantaran dibatalkan."
                self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{user_spoken}” [konfirmasi: batal]', fg=COLOR_RED))
                self.append_dialog_history("👤 Anda", f'"{user_spoken}" (Konfirmasi: Batal)', COLOR_RED)
                self.ui_call(lambda: self.admin_view.conv_log_label.config(text=f'👤 Anda: "{user_spoken}" → 🤖 Robot: {cancel_msg}', fg=COLOR_RED))
                self.ui_call(lambda: self.admin_view.robot_label.config(text=f'Robot: Tugas dibatalkan (kata: "{user_spoken}")', fg=COLOR_RED))
                self.ui_call(lambda: self.admin_view.status_label.config(text=f'● Status: Tugas dibatalkan (respon: "{user_spoken}")', fg=COLOR_RED))
                self.ui_call(lambda: self.status_badge.config(
                    text="● DIBATALKAN", fg=COLOR_RED, bg=COLOR_RED_BG, highlightbackground=COLOR_RED_BORDER
                ))
                self.append_dialog_history("🤖 Robot", cancel_msg, COLOR_RED)
                self.set_user_ui_state("(•_•)", "Tugas Dibatalkan", f'Anda mengatakan: "{user_spoken}". Robot kembali siaga.', COLOR_RED)
                if tts_ok:
                    try:
                        from audio.tts import speak_and_wait
                        speak_and_wait(cancel_msg)
                    except Exception:
                        pass
                return
            else:
                confirmed_word = user_spoken if ans_confirm else "benar"
                self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{confirmed_word}” [konfirmasi: lanjut]', fg=COLOR_GREEN))
                self.append_dialog_history("👤 Anda", f'"{confirmed_word}" (Konfirmasi: Lanjut)', COLOR_GREEN)
                self.ui_call(lambda: self.admin_view.conv_log_label.config(text=f'✓ Konfirmasi disetujui: "{confirmed_word}"', fg=COLOR_GREEN))

        # TTS konfirmasi akhir
        if tts_ok and confirm_msg:
            try:
                from audio.tts import speak_and_wait
                speak_and_wait(confirm_msg)
            except Exception as e:
                logger.error(f"[TTS] Gagal konfirmasi: {e}")
        self.append_dialog_history("🤖 Robot", confirm_msg, rcolor)

        # Follow-up kedatangan
        if vstatus == "VALID" and result.get("intent") == "NAVIGATE":
            target_loc = result.get("slots", {}).get("location", "tujuan")
            arrival_msg = f"Robot telah tiba di {target_loc}. Ada yang bisa saya bantu lagi?"
            time.sleep(1.0)
            self.ui_call(lambda: self.admin_view.conv_log_label.config(text=f"🤖 Robot: {arrival_msg}", fg=COLOR_GREEN))
            self.ui_call(lambda: self.admin_view.robot_label.config(text=f"Robot: Tiba di {target_loc}", fg=COLOR_GREEN))
            self.append_dialog_history("🤖 Robot", arrival_msg, COLOR_GREEN)
            self.set_user_ui_state("(◕‿◕)✓", f"Tiba di {target_loc}!", arrival_msg, COLOR_GREEN)
            if tts_ok:
                try:
                    from audio.tts import speak_and_wait
                    speak_and_wait(arrival_msg)
                except Exception:
                    pass

    def _reset_task_button(self):
        self.ui_call(lambda: self.admin_view.task_button.config(
            text="💬  DIALOG TUGAS\nMode interaktif dua arah (TTS ↔ STT)",
            bg=BG_CARD,
            fg=COLOR_PURPLE,
            highlightbackground=BORDER,
            state="normal",
        ))
        if self.user_view.user_voice_btn:
            self.ui_call(lambda: self.user_view.user_voice_btn.config(
                text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Poli Jantung\"",
                bg=COLOR_PRIMARY_BG,
                fg=COLOR_PRIMARY,
                highlightbackground=COLOR_PRIMARY_BORDER,
            ))
        if self.user_view.user_live_speech_label:
            self.ui_call(lambda: self.user_view.user_live_speech_label.config(
                text="🎙️ Audio: Standby (Sentuh tombol suara untuk berbicara)",
                fg=FG_MUTED,
            ))
        if self.user_view.user_chat_status_label:
            self.ui_call(lambda: self.user_view.user_chat_status_label.config(
                text="● Siaga",
                fg=COLOR_GREEN,
            ))
        self.ui_call(lambda: self.admin_view.talk_button.config(state="normal"))

    def on_task_button_clicked(self):
        if self.is_task_running:
            self.stop_event.set()
            return

        if self.is_recording:
            return

        self.is_task_running = True
        self.stop_event.clear()

        self.ui_call(lambda: self.admin_view.talk_button.config(state="disabled"))
        self.ui_call(lambda: self.admin_view.task_button.config(
            text="⏹  BATALKAN DIALOG\nKlik untuk menghentikan percakapan",
            bg=COLOR_RED_BG,
            fg=COLOR_RED,
            highlightbackground=COLOR_RED_BORDER,
        ))
        if self.user_view.user_voice_btn:
            self.ui_call(lambda: self.user_view.user_voice_btn.config(
                text="⏹️   SELESAI / BERHENTI BICARA\n● Sedang mendengarkan ucapan Anda... (Ketuk jika selesai)",
                bg=COLOR_SKY_BG,
                fg=COLOR_SKY,
                highlightbackground=COLOR_SKY_BORDER,
            ))

        self.task_thread = threading.Thread(
            target=self.interactive_task_flow,
            daemon=True,
        )
        self.task_thread.start()

    def record_and_process_stream(self):
        """Perekaman streaming audio kontinu dan NLU."""
        import sounddevice as sd
        self.stop_event.clear()
        if not self.stt_is_ready():
            self.ui_call(lambda: self.admin_view.status_label.config(text="> status: menyiapkan model STT...", fg=COLOR_AMBER))
            self.ui_call(lambda: self.admin_view.stt_label.config(text="$ stt: (menyiapkan model...)", fg=COLOR_AMBER))

        stream = self.stt_create_stream()
        audio_q = _queue.Queue()

        self.ui_call(lambda: self.admin_view.talk_button.config(
            text="⏹  SELESAI BICARA\nKlik untuk memproses perintah suara",
            bg=COLOR_SKY_BG,
            fg=COLOR_SKY,
            highlightbackground=COLOR_SKY_BORDER,
        ))
        self.ui_call(lambda: self.admin_view.status_label.config(text="● Mendengarkan secara langsung (streaming)...", fg=COLOR_SKY))
        self.ui_call(lambda: self.status_badge.config(
            text="● STREAMING", fg=COLOR_SKY, bg=COLOR_SKY_BG, highlightbackground=COLOR_SKY_BORDER
        ))
        self.ui_call(lambda: self.admin_view.stt_label.config(text='(silakan berbicara langsung ke mikrofon...)', fg=COLOR_SKY))
        self.ui_call(lambda: self.admin_view.intent_label.config(text="🎯 INTENT: -", fg=FG_MUTED))
        self.ui_call(lambda: self.admin_view.slots_label.config(text="📦 PARAMETER: -", fg=FG_MUTED))
        self.ui_call(lambda: self.admin_view.validation_label.config(text="🛡️ VALIDASI: -", fg=FG_MUTED))
        self.ui_call(lambda: self.admin_view.robot_label.config(text="Robot: Mendengarkan audio real-time...", fg=FG_MUTED))

        capture_rate = get_supported_sample_rate(self.device)
        optimal_channels = get_optimal_channels(self.device, capture_rate)

        def audio_callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                return
            audio_q.put(indata.copy())

        def gui_decode_worker():
            while not self.stop_event.is_set():
                try:
                    indata = audio_q.get(timeout=0.1)
                except _queue.Empty:
                    continue

                try:
                    if indata.shape[1] > 1:
                        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
                    else:
                        samples = np.ascontiguousarray(indata.flatten(), dtype=np.float32)

                    if capture_rate != self.sample_rate:
                        samples = resample_to_16k(samples, capture_rate)
                    stream.accept_waveform(self.sample_rate, samples)
                    while self.stt_recognizer.is_ready(stream):
                        self.stt_recognizer.decode_stream(stream)

                    is_endpoint = self.stt_recognizer.is_endpoint(stream)
                    partial = self.stt_recognizer.get_result(stream).strip()

                    if partial:
                        self.ui_call(lambda p=partial: self.admin_view.stt_label.config(text=f'“{p}...”', fg=COLOR_SKY))
                        sys.stdout.write(f"\r   [STREAM TALK] \"\033[36m{partial}\033[0m...\" ")
                        sys.stdout.flush()

                    if is_endpoint:
                        if partial:
                            self.stop_event.set()
                            return
                        else:
                            self.stt_recognizer.reset(stream)

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
                device=self.device,
                callback=audio_callback,
            ):
                start_time = time.time()
                while time.time() - start_time < self.record_seconds and not self.stop_event.is_set():
                    time.sleep(0.05)

            self.stop_event.set()
            decode_thread.join(timeout=2)

            t_decode_start = time.time()
            text = self.stt_recognizer.get_result(stream).strip()
            decode_latency_ms = (time.time() - t_decode_start) * 1000

            if text:
                sys.stdout.write(f"\r   [✓ STREAM TALK] \"\033[32m{text}\033[0m\"" + " " * 25 + "\n")
            else:
                sys.stdout.write("\r   [✕ STREAM TALK] (tidak ada suara terdeteksi)" + " " * 25 + "\n")
            sys.stdout.flush()

            logger.info(f"[STT] text = \"{text}\" (final latency: {decode_latency_ms:.2f}ms)")
            self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{text}”', fg=FG_WHITE))
            self.append_dialog_history("🎤 Anda", text or "(kosong)", COLOR_SKY)

            # NLU Pipeline
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Memproses NLU...", fg=COLOR_AMBER))
            result = self.pipeline.process(text)

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
                    else:
                        rtext = "Robot: Menuju lokasi"
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
                self.admin_view.intent_label.config(text=f"🎯 {intent_text}", fg=COLOR_PRIMARY)
                self.admin_view.slots_label.config(text=f"📦 PARAMETER: {slots_str}", fg=FG_SUB)
                self.admin_view.validation_label.config(text=f"{vicon} VALIDASI: {vstatus}", fg=vcolor)
                self.admin_view.robot_label.config(text=rtext, fg=rcolor)
                self.admin_view.status_label.config(text="● Status: Siap beroperasi", fg=COLOR_GREEN)
                self.status_badge.config(
                    text="● SIAP", fg=COLOR_GREEN, bg=COLOR_GREEN_BG, highlightbackground=COLOR_GREEN_BORDER
                )
                self.admin_view.conv_log_label.config(text=f"🤖 {rtext}", fg=rcolor)

            self.ui_call(apply_results)
            self.append_dialog_history("🤖 Robot", rtext, rcolor)
            self.update_ros2_ui_status(result)

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[GUI ERROR] {err_msg}")
            self.ui_call(lambda: self.admin_view.status_label.config(text="● Terjadi kesalahan perekaman audio", fg=COLOR_RED))
            self.ui_call(lambda e_msg=err_msg: messagebox.showerror("Error", e_msg))

        finally:
            self.is_recording = False
            self.stop_event.clear()
            self.ui_call(lambda: self.admin_view.talk_button.config(
                text="🎙️  STREAM TALK\nBicara langsung (real-time continuous STT)",
                bg=BG_CARD,
                fg=COLOR_SKY,
                highlightbackground=BORDER,
            ))
            self.ui_call(lambda: self.admin_view.task_button.config(state="normal"))

    def on_talk_button_clicked(self):
        if self.is_recording:
            self.stop_event.set()
        elif self.is_task_running:
            return
        else:
            self.is_recording = True
            self.stop_event.clear()
            self.ui_call(lambda: self.admin_view.task_button.config(state="disabled"))
            self.record_thread = threading.Thread(
                target=self.record_and_process_stream,
                daemon=True,
            )
            self.record_thread.start()

    def execute_quick_destination(self, dest_name: str, task_mode: str = "arah"):
        """Eksekusi tujuan cepat sekali sentuh oleh end-user."""
        if self.is_recording or self.is_task_running:
            return

        if task_mode == "obat":
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

        self.set_user_ui_state(face, title, sub, color)
        self.append_dialog_history("👤 Anda", cmd, COLOR_SKY)
        self.ui_call(lambda: self.admin_view.status_label.config(text=f"● MEMPROSES TUJUAN CEPAT: {cmd}...", fg=COLOR_AMBER))

        def _worker():
            res = self.pipeline.process(cmd)
            try:
                from audio.tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            self._show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    def execute_quick_action(self, action_cmd: str):
        """Eksekusi aksi cepat sekali sentuh (misal: Berhenti atau Kembali)."""
        if self.is_recording or self.is_task_running:
            self.stop_event.set()
        self.append_dialog_history("👤 Anda", action_cmd, COLOR_SKY)
        def _worker():
            res = self.pipeline.process(action_cmd)
            try:
                from audio.tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            self._show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    def execute_user_text(self, txt: str):
        """Kirim teks dari input pencarian Mode Pengguna."""
        self.set_user_ui_state("(・_・)", "Memproses...", f"Robot menerima: \"{txt}\"", COLOR_AMBER)
        self.append_dialog_history("👤 Anda", txt, COLOR_SKY)
        def _worker():
            res = self.pipeline.process(txt)
            try:
                from audio.tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            self._show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    def execute_manual_text(self, txt: str):
        """Kirim teks dari input toolbar Mode Teknisi."""
        self.ui_call(lambda: self.admin_view.status_label.config(text="● Memproses instruksi teks...", fg=COLOR_AMBER))
        self.ui_call(lambda: self.admin_view.stt_label.config(text=f'“{txt}” [manual]', fg=COLOR_PRIMARY))
        self.append_dialog_history("⌨️ Manual", txt, COLOR_PRIMARY)

        def _worker():
            res = self.pipeline.process(txt)
            try:
                from audio.tts import is_tts_available
                tts_ok = is_tts_available()
            except Exception:
                tts_ok = False
            self._show_final_result(res, tts_ok)
        threading.Thread(target=_worker, daemon=True).start()

    def run_mic_diagnostics(self):
        """Uji level mikrofon langsung dari GUI."""
        import sounddevice as sd
        self.admin_view.status_label.config(text="● Menguji level volume audio mikrofon...", fg=COLOR_AMBER)
        self.admin_view.robot_label.config(text="Robot: Merekam sampel audio 3 detik untuk pengujian hardware...", fg=COLOR_AMBER)
        self.admin_view.conv_log_label.config(text="[UJI MIKROFON] Silakan berbicara sekarang...", fg=COLOR_SKY)
        self.append_dialog_history("ℹ️ Diagnosa", "Merekam suara 3 detik untuk uji hardware mic...", COLOR_SKY)

        def test_worker():
            try:
                sr = get_supported_sample_rate(self.device)
                ch = get_optimal_channels(self.device, sr)
                recording = sd.rec(int(3 * sr), samplerate=sr, channels=ch, dtype="float32", device=self.device)
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
                self.ui_call(lambda: self.admin_view.status_label.config(text="● Pengujian mikrofon selesai", fg=COLOR_GREEN))
                self.ui_call(lambda: self.admin_view.robot_label.config(text=f"Hasil Tes Mic: {diag_res}", fg=col))
                self.ui_call(lambda: self.admin_view.conv_log_label.config(text=f"Hasil Tes: {diag_res}", fg=col))
                self.append_dialog_history("📊 Hardware Mic", res_detail, col)
            except Exception as e:
                self.ui_call(lambda: self.admin_view.status_label.config(text=f"● Gagal menguji mikrofon ({e})", fg=COLOR_RED))

        threading.Thread(target=test_worker, daemon=True).start()

    def clear_dialog_history(self):
        """Bersihkan riwayat percakapan."""
        self.admin_view.dialog_history_box.config(state="normal")
        self.admin_view.dialog_history_box.delete("1.0", tk.END)
        self.admin_view.dialog_history_box.config(state="disabled")
        self.admin_view.conv_log_label.config(text="Riwayat percakapan telah dibersihkan", fg=FG_MUTED)
        self.admin_view.ros2_status_badge.config(
            text="● MENUNGGU",
            fg=FG_MUTED,
            bg=BG_CARD,
            highlightbackground=BORDER,
        )
        self.admin_view.ros2_detail_label.config(
            text="Belum ada payload dikirim. Kirim instruksi untuk memicu transmisi TCP ke ROS 2.",
            fg=FG_SUB,
        )

    def check_ros2_ping(self):
        """Periksa ping koneksi socket TCP ROS 2."""
        def _bg_ping():
            target_str = getattr(self.pipeline.ros2_adapter, "host", "Node ROS 2")
            if hasattr(self.pipeline.ros2_adapter, "port"):
                target_str = f"{target_str}:{self.pipeline.ros2_adapter.port}"
            self.append_dialog_history("🌐 ROS 2", f"Memeriksa koneksi TCP ke {target_str}...", COLOR_SKY)
            ok, msg = self.pipeline.ros2_adapter.ping(timeout=1.5)
            def _ui():
                if ok:
                    self.admin_view.ros2_status_badge.config(
                        text="● TERHUBUNG ✓",
                        fg=COLOR_GREEN,
                        bg=COLOR_GREEN_BG,
                        highlightbackground=COLOR_GREEN_BORDER,
                    )
                    self.admin_view.ros2_detail_label.config(
                        text=f"✓ Node ROS 2 terjangkau ({msg})",
                        fg=COLOR_GREEN,
                    )
                    self.append_dialog_history("🌐 ROS 2", f"✓ {msg}", COLOR_GREEN)
                else:
                    self.admin_view.ros2_status_badge.config(
                        text="✕ TERPUTUS",
                        fg=COLOR_RED,
                        bg=COLOR_RED_BG,
                        highlightbackground=COLOR_RED_BORDER,
                    )
                    self.admin_view.ros2_detail_label.config(
                        text=f"✕ Node ROS 2 tidak terjangkau ({msg})",
                        fg=COLOR_RED,
                    )
                    self.append_dialog_history("🌐 ROS 2", f"✕ Gagal ping: {msg}", COLOR_RED)
            self.ui_call(_ui)
        threading.Thread(target=_bg_ping, daemon=True).start()

    def run(self):
        """Mulai main event loop Tkinter."""
        self.root.mainloop()


def run_gui_mode(duration: int = 5, device: Optional[int] = None, ros2_host: Optional[str] = None, ros2_port: Optional[int] = None):
    """Jalankan GUI Tkinter dengan microphone + real-time streaming STT + NLU pipeline."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:
        logger.warning(f"\n[WARN] Gagal membuka GUI Tkinter (tidak ada Display Server / $DISPLAY): {e}")
        logger.info("[INFO] Beralih otomatis ke Mode CLI Interaktif (Terminal)...\n")
        from cli import run_cli_interactive
        run_cli_interactive(duration=duration, device=device, ros2_host=ros2_host, ros2_port=ros2_port)
        return

    app = RobotGUIApp(root, duration=duration, device=device, ros2_host=ros2_host, ros2_port=ros2_port)
    app.run()
