"""
Admin View — Antarmuka Mode Teknisi (Dashboard Diagnostik, Telemetri NLU, dan Payload ROS 2).
"""

import tkinter as tk
from gui.theme import (
    BG_BASE,
    BG_CARD,
    BG_CARD_ALT,
    BG_CARD_GLOW,
    BORDER,
    BORDER_HOVER,
    FG_WHITE,
    FG_SUB,
    FG_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_BORDER,
    COLOR_GREEN,
    COLOR_GREEN_BG,
    COLOR_GREEN_BORDER,
    COLOR_AMBER,
    COLOR_AMBER_BG,
    COLOR_AMBER_BORDER,
    COLOR_SKY,
    COLOR_SKY_BG,
    COLOR_SKY_BORDER,
    COLOR_PURPLE,
    COLOR_PURPLE_BG,
    COLOR_PURPLE_BORDER,
    add_hover_effect,
)


class AdminView:
    """Komponen Tampilan Dashboard Teknisi."""

    def __init__(self, parent: tk.Widget, controller, fonts: dict):
        self.controller = controller
        self.fonts = fonts

        self.frame = tk.Frame(parent, bg=BG_BASE)

        # 1. Action Buttons (Talk & Task)
        hero_frame = tk.Frame(self.frame, bg=BG_BASE)
        hero_frame.pack(fill="x", pady=(0, 10))

        self.talk_button = tk.Button(
            hero_frame,
            text="🎙️  STREAM TALK\nBicara langsung (real-time continuous STT)",
            font=fonts["F_BTN_TITLE"],
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
            command=self.controller.on_talk_button_clicked,
        )
        self.talk_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        add_hover_effect(self.talk_button, BG_CARD, BG_CARD_ALT, BORDER, COLOR_SKY_BORDER)

        self.task_button = tk.Button(
            hero_frame,
            text="💬  DIALOG TUGAS\nMode interaktif dua arah (TTS ↔ STT)",
            font=fonts["F_BTN_TITLE"],
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
            command=self.controller.on_task_button_clicked,
        )
        self.task_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        add_hover_effect(self.task_button, BG_CARD, BG_CARD_ALT, BORDER, COLOR_PURPLE_BORDER)

        # 2. Command Input Bar & Diagnostics Toolbar
        toolbar_frame = tk.Frame(self.frame, bg=BG_BASE)
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
            font=(fonts["FONT_FAMILY_UI"], 11),
            bg=BG_CARD_ALT,
            fg=COLOR_PRIMARY,
        ).pack(side="left", padx=(2, 6))

        self.text_entry = tk.Entry(
            input_box,
            font=fonts["F_CODE"],
            bg=BG_CARD_ALT,
            fg=FG_WHITE,
            insertbackground=COLOR_PRIMARY,
            relief="flat",
            borderwidth=0,
        )
        self.text_entry.pack(side="left", fill="x", expand=True, ipady=4)

        def on_enter_pressed(event=None):
            txt = self.text_entry.get().strip()
            if txt:
                self.text_entry.delete(0, tk.END)
                self.controller.execute_manual_text(txt)

        self.text_entry.bind("<Return>", on_enter_pressed)

        send_btn = tk.Button(
            input_box,
            text="KIRIM ↵",
            font=(fonts["FONT_FAMILY_UI"], 9, "bold"),
            fg=FG_WHITE,
            bg=COLOR_PRIMARY,
            activeforeground=FG_WHITE,
            activebackground=COLOR_PRIMARY_BORDER,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=2,
            cursor="hand2",
            command=on_enter_pressed,
        )
        send_btn.pack(side="right")
        add_hover_effect(send_btn, COLOR_PRIMARY, COLOR_PRIMARY_BORDER)

        btn_mic = tk.Button(
            toolbar_frame,
            text="🎧 PILIH MIC",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=self.controller.open_devices_dialog,
        )
        btn_mic.pack(side="left", padx=(0, 6))
        add_hover_effect(btn_mic, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

        btn_diag = tk.Button(
            toolbar_frame,
            text="📊 TES MIC",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=self.controller.run_mic_diagnostics,
        )
        btn_diag.pack(side="left", padx=(0, 6))
        add_hover_effect(btn_diag, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

        btn_clear = tk.Button(
            toolbar_frame,
            text="🧹 BERSIHKAN",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=self.controller.clear_dialog_history,
        )
        btn_clear.pack(side="left")
        add_hover_effect(btn_clear, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

        btn_ros2 = tk.Button(
            toolbar_frame,
            text="🌐 CEK ROS 2",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=self.controller.check_ros2_ping,
        )
        btn_ros2.pack(side="left", padx=(6, 0))
        add_hover_effect(btn_ros2, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

        btn_ros2_cfg = tk.Button(
            toolbar_frame,
            text="⚙️ IP & PORT",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=self.controller.open_ros2_dialog,
        )
        btn_ros2_cfg.pack(side="left", padx=(6, 0))
        add_hover_effect(btn_ros2_cfg, BG_CARD, BG_CARD_ALT, BORDER, BORDER_HOVER)

        # 2.5 Hardware Acceleration & Compute Telemetry Card
        hw_card = tk.Frame(
            self.frame,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=8,
        )
        hw_card.pack(fill="x", pady=(0, 10))

        hw_header = tk.Frame(hw_card, bg=BG_CARD)
        hw_header.pack(fill="x", pady=(0, 6))

        tk.Label(
            hw_header,
            text="⚡ MONITOR AKSELERASI KOMPUTASI (GPU / CPU)",
            font=fonts["F_SECTION"],
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(side="left")

        self.hw_accel_badge = tk.Label(
            hw_header,
            text="● GPU (CUDA) AKTIF",
            font=fonts["F_CODE_BOLD"],
            fg=COLOR_GREEN,
            bg=BG_CARD,
        )
        self.hw_accel_badge.pack(side="right")

        hw_badges_bar = tk.Frame(hw_card, bg=BG_CARD)
        hw_badges_bar.pack(fill="x", pady=(0, 6))

        self.hw_chip_label = tk.Label(
            hw_badges_bar,
            text="📟 Jetson Orin",
            font=fonts["F_CODE"],
            fg=FG_WHITE,
            bg=BG_CARD_ALT,
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.hw_chip_label.pack(side="left", padx=(0, 6))

        self.hw_stt_label = tk.Label(
            hw_badges_bar,
            text="🎙️ STT: CUDA",
            font=fonts["F_CODE"],
            fg=COLOR_SKY,
            bg=BG_CARD_ALT,
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.hw_stt_label.pack(side="left", padx=(0, 6))

        self.hw_tts_label = tk.Label(
            hw_badges_bar,
            text="🔊 TTS: CUDA",
            font=fonts["F_CODE"],
            fg=COLOR_PURPLE,
            bg=BG_CARD_ALT,
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.hw_tts_label.pack(side="left", padx=(0, 6))

        self.hw_mem_label = tk.Label(
            hw_badges_bar,
            text="💾 RAM: -",
            font=fonts["F_CODE"],
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.hw_mem_label.pack(side="right")

        # Mode selector & Restart control bar
        hw_control_bar = tk.Frame(hw_card, bg=BG_CARD_ALT, padx=10, pady=5, highlightthickness=1, highlightbackground=BORDER)
        hw_control_bar.pack(fill="x", pady=(0, 6))

        ctrl_left = tk.Frame(hw_control_bar, bg=BG_CARD_ALT)
        ctrl_left.pack(side="left", fill="x", expand=True)

        tk.Label(
            ctrl_left,
            text="⚙️ PILIH AKSELERASI:",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_SUB,
            bg=BG_CARD_ALT,
        ).pack(side="left", padx=(0, 8))

        from audio.provider import is_cuda_available, load_compute_mode
        self.cuda_supported = is_cuda_available()
        current_mode = load_compute_mode()

        # Tombol Mode GPU (CUDA)
        gpu_btn_text = "⚡ GPU (CUDA)" if self.cuda_supported else "⚡ GPU (No CUDA)"
        self.btn_mode_gpu = tk.Button(
            ctrl_left,
            text=gpu_btn_text,
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            relief="flat",
            borderwidth=0,
            cursor="hand2" if self.cuda_supported else "arrow",
            padx=10,
            pady=3,
            state="normal" if self.cuda_supported else "disabled",
            command=lambda: self.controller.set_compute_mode("gpu"),
        )
        self.btn_mode_gpu.pack(side="left", padx=(0, 6))

        # Tombol Mode CPU
        self.btn_mode_cpu = tk.Button(
            ctrl_left,
            text="💻 CPU",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=10,
            pady=3,
            command=lambda: self.controller.set_compute_mode("cpu"),
        )
        self.btn_mode_cpu.pack(side="left", padx=(0, 10))

        # Status mode label
        self.hw_mode_status_label = tk.Label(
            ctrl_left,
            text="Default: GPU" if self.cuda_supported else "Mode: CPU (No GPU)",
            font=(fonts["FONT_FAMILY_UI"], 8),
            fg=FG_MUTED,
            bg=BG_CARD_ALT,
        )
        self.hw_mode_status_label.pack(side="left")

        # Tombol Restart Aplikasi
        ctrl_right = tk.Frame(hw_control_bar, bg=BG_CARD_ALT)
        ctrl_right.pack(side="right")

        self.btn_restart_app = tk.Button(
            ctrl_right,
            text="🔄 Restart Aplikasi",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=COLOR_AMBER,
            bg=BG_CARD,
            activeforeground=FG_WHITE,
            activebackground=COLOR_AMBER_BG,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLOR_AMBER_BORDER,
            cursor="hand2",
            padx=10,
            pady=3,
            command=self.controller.on_restart_app_clicked,
        )
        self.btn_restart_app.pack(side="right")
        add_hover_effect(self.btn_restart_app, BG_CARD, COLOR_AMBER_BG, COLOR_AMBER_BORDER, COLOR_AMBER)

        self.update_compute_mode_ui(current_mode)

        # Live meters row: GPU (GR3D) & CPU
        meters_row = tk.Frame(hw_card, bg=BG_CARD_ALT, padx=10, pady=6, highlightthickness=1, highlightbackground=BORDER)
        meters_row.pack(fill="x")

        # GPU Meter
        gpu_col = tk.Frame(meters_row, bg=BG_CARD_ALT)
        gpu_col.pack(side="left", fill="x", expand=True, padx=(0, 10))

        gpu_top = tk.Frame(gpu_col, bg=BG_CARD_ALT)
        gpu_top.pack(fill="x")
        tk.Label(gpu_top, text="GPU (GR3D CORE):", font=(fonts["FONT_FAMILY_UI"], 8, "bold"), fg=COLOR_GREEN, bg=BG_CARD_ALT).pack(side="left")
        self.hw_gpu_val = tk.Label(gpu_top, text="0.0% (Standby)", font=fonts["F_CODE_BOLD"], fg=COLOR_GREEN, bg=BG_CARD_ALT)
        self.hw_gpu_val.pack(side="right")

        self.gpu_bar_canvas = tk.Canvas(gpu_col, height=6, bg=BG_BASE, highlightthickness=0)
        self.gpu_bar_canvas.pack(fill="x", pady=(3, 0))

        # CPU Meter
        cpu_col = tk.Frame(meters_row, bg=BG_CARD_ALT)
        cpu_col.pack(side="left", fill="x", expand=True, padx=(10, 0))

        cpu_top = tk.Frame(cpu_col, bg=BG_CARD_ALT)
        cpu_top.pack(fill="x")
        tk.Label(cpu_top, text="CPU MULTI-CORE:", font=(fonts["FONT_FAMILY_UI"], 8, "bold"), fg=COLOR_SKY, bg=BG_CARD_ALT).pack(side="left")
        self.hw_cpu_val = tk.Label(cpu_top, text="0.0%", font=fonts["F_CODE_BOLD"], fg=COLOR_SKY, bg=BG_CARD_ALT)
        self.hw_cpu_val.pack(side="right")

        self.cpu_bar_canvas = tk.Canvas(cpu_col, height=6, bg=BG_BASE, highlightthickness=0)
        self.cpu_bar_canvas.pack(fill="x", pady=(3, 0))

        # 3. Telemetry & NLU Status Card
        telemetry_card = tk.Frame(
            self.frame,
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
            font=fonts["F_SECTION"],
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(side="left")

        self.status_label = tk.Label(
            card1_header,
            text="● Status: Siap",
            font=fonts["F_CODE_BOLD"],
            fg=COLOR_GREEN,
            bg=BG_CARD,
        )
        self.status_label.pack(side="right")

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
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD_ALT,
        ).pack(anchor="w")

        self.stt_label = tk.Label(
            speech_frame,
            text='“(menunggu perintah suara dari mikrofon atau ketikan manual...)”',
            font=(fonts["FONT_FAMILY_CODE"], 10, "italic"),
            fg=FG_MUTED,
            bg=BG_CARD_ALT,
            anchor="w",
            justify="left",
        )
        self.stt_label.pack(fill="x", pady=(2, 2))

        tags_bar = tk.Frame(telemetry_card, bg=BG_CARD)
        tags_bar.pack(fill="x", pady=(0, 6))

        self.intent_label = tk.Label(
            tags_bar,
            text="🎯 INTENT: -",
            font=fonts["F_CODE_BOLD"],
            fg=COLOR_PRIMARY,
            bg=BG_CARD_ALT,
            padx=10,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.intent_label.pack(side="left", padx=(0, 6))

        self.validation_label = tk.Label(
            tags_bar,
            text="🛡️ VALIDASI: -",
            font=fonts["F_CODE_BOLD"],
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            padx=10,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.validation_label.pack(side="left", padx=(0, 6))

        self.slots_label = tk.Label(
            tags_bar,
            text="📦 PARAMETER: -",
            font=fonts["F_CODE"],
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            padx=10,
            pady=3,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.slots_label.pack(side="left", fill="x", expand=True)

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
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_SUB,
            bg=BG_CARD_GLOW,
        ).pack(anchor="w")

        self.robot_label = tk.Label(
            robot_banner,
            text="Robot siap menerima instruksi.",
            font=fonts["F_ROBOT"],
            fg=COLOR_GREEN,
            bg=BG_CARD_GLOW,
            anchor="w",
        )
        self.robot_label.pack(fill="x", pady=(2, 2))

        # ROS 2 Node Payload Transmission Banner
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

        self.ros2_title_label = tk.Label(
            ros2_banner_top,
            text="🌐 TRANSMISI PAYLOAD NODE ROS 2:",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD_ALT,
        )
        self.ros2_title_label.pack(side="left")

        btn_cfg_inline = tk.Button(
            ros2_banner_top,
            text="⚙️ Ubah IP/Port",
            font=(fonts["FONT_FAMILY_UI"], 8),
            fg=COLOR_SKY,
            bg=BG_CARD_ALT,
            activeforeground=FG_WHITE,
            activebackground=BG_CARD,
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=6,
            command=self.controller.open_ros2_dialog,
        )
        btn_cfg_inline.pack(side="left", padx=(6, 0))

        self.ros2_status_badge = tk.Label(
            ros2_banner_top,
            text="● MENUNGGU",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD,
            padx=8,
            pady=2,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.ros2_status_badge.pack(side="right")

        self.ros2_detail_label = tk.Label(
            ros2_banner,
            text="Belum ada payload dikirim. Kirim instruksi untuk memicu transmisi TCP ke ROS 2.",
            font=fonts["F_CODE"],
            fg=FG_SUB,
            bg=BG_CARD_ALT,
            anchor="w",
            justify="left",
        )
        self.ros2_detail_label.pack(fill="x", pady=(2, 0))

        # 4. Detailed Conversation Debug Log
        dialog_card = tk.Frame(
            self.frame,
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
            font=fonts["F_SECTION"],
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(side="left")

        self.conv_log_label = tk.Label(
            card2_header,
            text="Tekan [STREAM TALK] atau [DIALOG TUGAS] untuk memulai",
            font=(fonts["FONT_FAMILY_UI"], 9),
            fg=FG_MUTED,
            bg=BG_CARD,
        )
        self.conv_log_label.pack(side="right")

        log_scroll_frame = tk.Frame(
            dialog_card,
            bg=BG_BASE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        log_scroll_frame.pack(fill="both", expand=True, pady=(4, 2))

        self.dialog_history_box = tk.Text(
            log_scroll_frame,
            font=fonts["F_CODE"],
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
            command=self.dialog_history_box.yview,
            bg=BG_CARD,
            troughcolor=BG_BASE,
            width=12,
        )
        self.dialog_history_box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.dialog_history_box.pack(side="left", fill="both", expand=True)

        self.dialog_history_box.tag_configure("tag_time", foreground=FG_MUTED)
        self.dialog_history_box.tag_configure("tag_robot", foreground=COLOR_PURPLE, font=fonts["F_CODE_BOLD"])
        self.dialog_history_box.tag_configure("tag_user", foreground=COLOR_SKY, font=fonts["F_CODE_BOLD"])
        self.dialog_history_box.tag_configure("tag_manual", foreground=COLOR_PRIMARY, font=fonts["F_CODE_BOLD"])
        self.dialog_history_box.tag_configure("tag_ros2", foreground=COLOR_SKY, font=fonts["F_CODE_BOLD"])
        self.dialog_history_box.tag_configure("tag_info", foreground=FG_SUB)

    def update_hw_telemetry(self, snap: dict):
        """Update visual telemetri hardware GPU vs CPU secara real-time."""
        chip = snap.get("device_model", "Tegra Device")
        if len(chip) > 30:
            chip = chip[:28] + ".."
        self.hw_chip_label.config(text=f"📟 {chip}")

        stt_p = snap.get("stt_provider", "cpu").upper()
        tts_p = snap.get("tts_provider", "cpu").upper()
        self.hw_stt_label.config(text=f"🎙️ STT: {stt_p}")
        self.hw_tts_label.config(text=f"🔊 TTS: {tts_p}")

        ram_used = snap.get("ram_used_mb", 0)
        ram_total = snap.get("ram_total_mb", 0)
        ram_pct = snap.get("ram_pct", 0.0)
        self.hw_mem_label.config(text=f"💾 RAM: {ram_used}/{ram_total} MB ({ram_pct:.0f}%)")

        gpu_pct = snap.get("gpu_load", 0.0)
        cpu_pct = snap.get("cpu_load", 0.0)
        gpu_temp = snap.get("gpu_temp")
        cpu_temp = snap.get("cpu_temp")

        g_temp_str = f" • {gpu_temp:.1f}°C" if gpu_temp else ""
        c_temp_str = f" • {cpu_temp:.1f}°C" if cpu_temp else ""

        if gpu_pct > 2.0:
            self.hw_gpu_val.config(text=f"{gpu_pct:.1f}% (INFERENCE ACTIVE{g_temp_str})", fg=COLOR_GREEN)
            self.hw_accel_badge.config(text="● GPU (CUDA) AKTIF", fg=COLOR_GREEN)
        else:
            self.hw_gpu_val.config(text=f"{gpu_pct:.1f}% (Standby{g_temp_str})", fg=FG_SUB)
            self.hw_accel_badge.config(text="● GPU (CUDA) STANDBY", fg=FG_SUB)

        self.hw_cpu_val.config(text=f"{cpu_pct:.1f}%{c_temp_str}")

        # Update visual progress bar canvas
        try:
            gw = self.gpu_bar_canvas.winfo_width()
            if gw > 1:
                self.gpu_bar_canvas.delete("all")
                g_fill = max(2, int(gw * (gpu_pct / 100.0)))
                bar_color = COLOR_GREEN if gpu_pct > 2.0 else COLOR_PRIMARY
                self.gpu_bar_canvas.create_rectangle(0, 0, g_fill, 6, fill=bar_color, width=0)

            cw = self.cpu_bar_canvas.winfo_width()
            if cw > 1:
                self.cpu_bar_canvas.delete("all")
                c_fill = max(2, int(cw * (cpu_pct / 100.0)))
                self.cpu_bar_canvas.create_rectangle(0, 0, c_fill, 6, fill=COLOR_SKY, width=0)
        except Exception:
            pass

    def update_compute_mode_ui(self, active_mode: str, status_msg: str = None):
        """Update styling tombol pemilih mode komputasi GPU vs CPU."""
        is_gpu = active_mode in ("gpu", "cuda")
        if is_gpu and self.cuda_supported:
            self.btn_mode_gpu.config(
                bg=COLOR_GREEN_BG,
                fg=COLOR_GREEN,
                highlightthickness=1,
                highlightbackground=COLOR_GREEN_BORDER,
            )
            self.btn_mode_cpu.config(
                bg=BG_CARD,
                fg=FG_MUTED,
                highlightthickness=1,
                highlightbackground=BORDER,
            )
            if status_msg:
                self.hw_mode_status_label.config(text=status_msg, fg=COLOR_GREEN)
            else:
                self.hw_mode_status_label.config(text="● Mode aktif: GPU (CUDA)", fg=COLOR_GREEN)
        else:
            self.btn_mode_gpu.config(
                bg=BG_CARD,
                fg=FG_MUTED,
                highlightthickness=1,
                highlightbackground=BORDER,
            )
            self.btn_mode_cpu.config(
                bg=COLOR_SKY_BG,
                fg=COLOR_SKY,
                highlightthickness=1,
                highlightbackground=COLOR_SKY_BORDER,
            )
            if status_msg:
                self.hw_mode_status_label.config(text=status_msg, fg=COLOR_SKY)
            else:
                self.hw_mode_status_label.config(text="● Mode aktif: CPU (Standard)", fg=COLOR_SKY)
