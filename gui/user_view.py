"""
User View — Antarmuka Mode Pengguna (Kiosk Layanan Mandiri / Pasien & Pengunjung).
"""

import threading
import tkinter as tk
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
    add_hover_effect,
)


class UserView:
    """Komponen Tampilan Kiosk Pengguna."""

    def __init__(self, parent: tk.Widget, controller, fonts: dict):
        self.controller = controller
        self.fonts = fonts
        self.current_quick_mode = "arah"

        self.frame = tk.Frame(parent, bg=BG_BASE)

        # 1. Persona & Expression Hero Card
        user_hero_card = tk.Frame(
            self.frame,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=20,
            pady=14,
        )
        user_hero_card.pack(fill="x", pady=(0, 10))

        hero_card_left = tk.Frame(user_hero_card, bg=BG_CARD)
        hero_card_left.pack(side="left")

        self.user_face_label = tk.Label(
            hero_card_left,
            text="(◕‿◕)",
            font=(fonts["FONT_FAMILY_CODE"], 28, "bold"),
            fg=COLOR_GREEN,
            bg=BG_CARD_ALT,
            padx=16,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.user_face_label.pack(side="left", padx=(0, 16))

        hero_card_text = tk.Frame(user_hero_card, bg=BG_CARD)
        hero_card_text.pack(side="left", fill="both", expand=True)

        self.user_status_title = tk.Label(
            hero_card_text,
            text="Halo! Saya asisten robot pemandu rumah sakit.",
            font=(fonts["FONT_FAMILY_UI"], 13, "bold"),
            fg=FG_WHITE,
            bg=BG_CARD,
        )
        self.user_status_title.pack(anchor="w")

        self.user_status_sub = tk.Label(
            hero_card_text,
            text="Katakan tujuan Anda, sentuh tombol suara di bawah, atau pilih salah satu tujuan cepat.",
            font=(fonts["FONT_FAMILY_UI"], 9),
            fg=FG_SUB,
            bg=BG_CARD,
        )
        self.user_status_sub.pack(anchor="w", pady=(2, 4))

        pill_and_speech_row = tk.Frame(hero_card_text, bg=BG_CARD)
        pill_and_speech_row.pack(fill="x", pady=(2, 0))

        self.user_action_pill = tk.Label(
            pill_and_speech_row,
            text="● Siap Melayani",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=COLOR_GREEN,
            bg=COLOR_GREEN_BG,
            padx=8,
            pady=2,
            highlightthickness=1,
            highlightbackground=COLOR_GREEN_BORDER,
        )
        self.user_action_pill.pack(side="left")

        self.user_live_speech_label = tk.Label(
            pill_and_speech_row,
            text="🎙️ Audio: Standby (Sentuh tombol suara untuk berbicara)",
            font=(fonts["FONT_FAMILY_CODE"], 9),
            fg=FG_MUTED,
            bg=BG_CARD_ALT,
            padx=10,
            pady=2,
            highlightthickness=1,
            highlightbackground=BORDER,
            anchor="w",
        )
        self.user_live_speech_label.pack(side="left", padx=(8, 0), fill="x", expand=True)

        # 2. Giant Voice Interaction Touchpad (1-Tap to Speak)
        self.user_voice_btn = tk.Button(
            self.frame,
            text="🎙️   SENTUH UNTUK BERBICARA (DIALOG SUARA)\nKatakan: \"Antar saya ke ruang ICU\" atau \"Di mana Poli Jantung?\"",
            font=(fonts["FONT_FAMILY_UI"], 11, "bold"),
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
            command=self.controller.on_task_button_clicked,
        )
        self.user_voice_btn.pack(fill="x", pady=(0, 10))
        add_hover_effect(self.user_voice_btn, COLOR_PRIMARY_BG, COLOR_PRIMARY_BORDER, COLOR_PRIMARY_BORDER, COLOR_PRIMARY)

        # 3. Quick Destination Tiles (1-Touch Instant Kiosk)
        tiles_card = tk.Frame(
            self.frame,
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
            font=(fonts["FONT_FAMILY_UI"], 9, "bold"),
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(side="left")

        mode_switch_frame = tk.Frame(tiles_header, bg=BG_CARD)
        mode_switch_frame.pack(side="right")

        tk.Label(
            mode_switch_frame,
            text="Opsi Tugas:",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=FG_MUTED,
            bg=BG_CARD,
        ).pack(side="left", padx=(0, 6))

        self.btn_mode_arah = tk.Button(
            mode_switch_frame,
            text="🧭 Penunjuk Arah",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=lambda: self.select_quick_mode("arah"),
        )
        self.btn_mode_arah.pack(side="left", padx=(0, 4))

        self.btn_mode_obat = tk.Button(
            mode_switch_frame,
            text="💊 Antar Obat",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
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
            command=lambda: self.select_quick_mode("obat"),
        )
        self.btn_mode_obat.pack(side="left")

        self.quick_mode_hint = tk.Label(
            tiles_card,
            text="● Mode Aktif: 🧭 Penunjuk Arah (Robot memandu dan mengantar Anda langsung ke lokasi)",
            font=(fonts["FONT_FAMILY_UI"], 8),
            fg=COLOR_SKY,
            bg=BG_CARD,
            anchor="w",
        )
        self.quick_mode_hint.pack(fill="x", pady=(0, 6))

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
                    return lambda: self.controller.execute_quick_action("berhenti")
                elif target == "__HOME__":
                    return lambda: self.controller.execute_quick_action("kembali")
                else:
                    return lambda: self.controller.execute_quick_destination(target, self.current_quick_mode)

            qbtn = tk.Button(
                tiles_grid,
                text=label_txt,
                font=(fonts["FONT_FAMILY_UI"], 9, "bold"),
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
            self.frame,
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
            font=(fonts["FONT_FAMILY_UI"], 9, "bold"),
            fg=FG_SUB,
            bg=BG_CARD,
        ).pack(side="left")

        self.user_chat_status_label = tk.Label(
            user_chat_header,
            text="● Siaga",
            font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
            fg=COLOR_GREEN,
            bg=BG_CARD,
        )
        self.user_chat_status_label.pack(side="right")

        user_chat_scroll = tk.Frame(user_chat_card, bg=BG_BASE, highlightthickness=1, highlightbackground=BORDER)
        user_chat_scroll.pack(fill="both", expand=True, pady=(2, 8))

        self.user_chat_box = tk.Text(
            user_chat_scroll,
            font=(fonts["FONT_FAMILY_UI"], 10),
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
        user_scrollbar = tk.Scrollbar(user_chat_scroll, command=self.user_chat_box.yview, bg=BG_CARD, troughcolor=BG_BASE, width=12)
        self.user_chat_box.configure(yscrollcommand=user_scrollbar.set)
        user_scrollbar.pack(side="right", fill="y")
        self.user_chat_box.pack(side="left", fill="both", expand=True)

        self.user_chat_box.tag_configure("user_tag_time", foreground=FG_MUTED, font=(fonts["FONT_FAMILY_CODE"], 8))
        self.user_chat_box.tag_configure("user_tag_robot", foreground=COLOR_PURPLE, font=(fonts["FONT_FAMILY_UI"], 10, "bold"))
        self.user_chat_box.tag_configure("user_tag_user", foreground=COLOR_SKY, font=(fonts["FONT_FAMILY_UI"], 10, "bold"))
        self.user_chat_box.tag_configure("user_tag_info", foreground=FG_SUB, font=(fonts["FONT_FAMILY_UI"], 9, "italic"))
        self.user_chat_box.tag_configure("user_tag_body", foreground=FG_WHITE)

        # Touch Search / Type Bar
        user_input_box = tk.Frame(user_chat_card, bg=BG_CARD_ALT, highlightthickness=1, highlightbackground=BORDER, padx=8, pady=4)
        user_input_box.pack(fill="x")

        tk.Label(user_input_box, text="⌨️", font=(fonts["FONT_FAMILY_UI"], 11), bg=BG_CARD_ALT, fg=COLOR_PRIMARY).pack(side="left", padx=(2, 6))

        self.user_text_entry = tk.Entry(
            user_input_box,
            font=(fonts["FONT_FAMILY_UI"], 10),
            bg=BG_CARD_ALT,
            fg=FG_WHITE,
            insertbackground=COLOR_PRIMARY,
            relief="flat",
            borderwidth=0,
        )
        self.user_text_entry.pack(side="left", fill="x", expand=True, ipady=4)

        def on_enter_pressed(event=None):
            txt = self.user_text_entry.get().strip()
            if txt:
                self.user_text_entry.delete(0, tk.END)
                self.controller.execute_user_text(txt)

        self.user_text_entry.bind("<Return>", on_enter_pressed)

        user_send_btn = tk.Button(
            user_input_box,
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
        user_send_btn.pack(side="right")
        add_hover_effect(user_send_btn, COLOR_PRIMARY, COLOR_PRIMARY_BORDER)

    def select_quick_mode(self, mode: str):
        """Pilih opsi tugas tujuan cepat: 'arah' (Penunjuk Arah) atau 'obat' (Antar Obat)."""
        self.current_quick_mode = mode
        if mode == "arah":
            self.btn_mode_arah.config(
                bg=COLOR_SKY_BG,
                fg=COLOR_SKY,
                highlightbackground=COLOR_SKY_BORDER,
            )
            self.btn_mode_obat.config(
                bg=BG_CARD_ALT,
                fg=FG_MUTED,
                highlightbackground=BORDER,
            )
            if self.quick_mode_hint:
                self.quick_mode_hint.config(
                    text="● Mode Aktif: 🧭 Penunjuk Arah (Robot memandu dan mengantar Anda langsung ke lokasi)",
                    fg=COLOR_SKY,
                )
        elif mode == "obat":
            self.btn_mode_obat.config(
                bg=COLOR_GREEN_BG,
                fg=COLOR_GREEN,
                highlightbackground=COLOR_GREEN_BORDER,
            )
            self.btn_mode_arah.config(
                bg=BG_CARD_ALT,
                fg=FG_MUTED,
                highlightbackground=BORDER,
            )
            if self.quick_mode_hint:
                self.quick_mode_hint.config(
                    text="● Mode Aktif: 💊 Antar Obat (Robot membawa dan mengantarkan obat medis ke lokasi)",
                    fg=COLOR_GREEN,
                )
