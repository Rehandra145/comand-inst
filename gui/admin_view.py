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
    COLOR_AMBER,
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
