"""
GUI Modals — Dialog pemilihan perangkat mikrofon dan konfigurasi endpoint ROS 2.
"""

import threading
import time
import socket
import tkinter as tk
from typing import Callable, Optional

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
    COLOR_AMBER,
    COLOR_RED,
    COLOR_SKY,
)
from nlu.ros2_adapter import save_ros2_config, SocketROS2Adapter


def open_devices_modal(
    root: tk.Tk,
    current_device: Optional[int],
    fonts: dict,
    on_device_selected: Callable[[int, str], None],
):
    """Buka modal modern untuk memilih mikrofon input."""
    modal = tk.Toplevel(root)
    modal.title("Pilih Perangkat Input Audio")
    modal.geometry("560x380")
    modal.configure(bg=BG_BASE)
    modal.transient(root)
    modal.grab_set()

    tk.Label(
        modal,
        text="DAFTAR PERANGKAT INPUT AUDIO (MIKROFON)",
        font=(fonts["FONT_FAMILY_UI"], 11, "bold"),
        fg=FG_WHITE,
        bg=BG_BASE,
    ).pack(pady=(16, 8))

    dev_listbox = tk.Listbox(
        modal,
        font=fonts["F_CODE"],
        bg=BG_CARD_ALT,
        fg=FG_WHITE,
        selectbackground=COLOR_PRIMARY_BG,
        selectforeground=COLOR_PRIMARY,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    dev_listbox.pack(fill="both", expand=True, padx=20, pady=(0, 14))

    in_devs = []
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                in_devs.append((i, d))
                marker = " [*] " if i == current_device else "     "
                dev_listbox.insert(tk.END, f"{marker}[{i}] {d['name']} ({d['max_input_channels']} ch)")
    except Exception as e:
        dev_listbox.insert(tk.END, f"Gagal membaca perangkat: {e}")

    def select_device():
        sel = dev_listbox.curselection()
        if sel and in_devs:
            idx = sel[0]
            dev_id = in_devs[idx][0]
            dev_name = in_devs[idx][1]['name']
            on_device_selected(dev_id, dev_name)
            modal.destroy()

    btn_select = tk.Button(
        modal,
        text="Gunakan Perangkat Terpilih",
        font=fonts["F_BTN_TITLE"],
        fg=FG_WHITE,
        bg=COLOR_PRIMARY,
        activeforeground=FG_WHITE,
        activebackground=COLOR_PRIMARY_BORDER,
        relief="flat",
        borderwidth=0,
        padx=16,
        pady=6,
        cursor="hand2",
        command=select_device,
    )
    btn_select.pack(pady=(0, 16))


def open_ros2_config_modal(
    root: tk.Tk,
    pipeline,
    fonts: dict,
    ui_call: Callable,
    on_endpoint_updated: Callable[[str, int], None],
):
    """Buka modal modern untuk mengatur Custom IP & Port Node ROS 2."""
    modal = tk.Toplevel(root)
    modal.title("⚙️ Pengaturan Endpoint Node ROS 2")
    modal.configure(bg=BG_CARD)
    modal.resizable(False, False)
    modal.transient(root)
    modal.grab_set()

    mw, mh = 480, 440
    rx = root.winfo_x() + (root.winfo_width() // 2) - (mw // 2)
    ry = root.winfo_y() + (root.winfo_height() // 2) - (mh // 2)
    modal.geometry(f"{mw}x{mh}+{max(0, rx)}+{max(0, ry)}")

    container = tk.Frame(modal, bg=BG_CARD, padx=24, pady=20)
    container.pack(fill="both", expand=True)

    # Header
    tk.Label(
        container,
        text="⚙️ KONFIGURASI ENDPOINT ROS 2",
        font=(fonts["FONT_FAMILY_UI"], 12, "bold"),
        fg=FG_WHITE,
        bg=BG_CARD,
    ).pack(anchor="w")

    tk.Label(
        container,
        text="Atur IP address & port TCP node ROS 2 di robot tujuan.",
        font=(fonts["FONT_FAMILY_UI"], 9),
        fg=FG_SUB,
        bg=BG_CARD,
    ).pack(anchor="w", pady=(2, 14))

    # Current Host & Port values
    cur_host = getattr(pipeline.ros2_adapter, "host", "10.107.245.221")
    cur_port = getattr(pipeline.ros2_adapter, "port", 5000)

    # Field IP Host
    tk.Label(
        container,
        text="ALAMAT IP / HOSTNAME:",
        font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD,
    ).pack(anchor="w")

    host_entry = tk.Entry(
        container,
        font=(fonts["FONT_FAMILY_CODE"], 10),
        bg=BG_CARD_ALT,
        fg=FG_WHITE,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    host_entry.pack(fill="x", ipady=4, pady=(3, 10))
    host_entry.insert(0, str(cur_host))

    # Field Port
    tk.Label(
        container,
        text="PORT TCP:",
        font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
        fg=FG_MUTED,
        bg=BG_CARD,
    ).pack(anchor="w")

    port_entry = tk.Entry(
        container,
        font=(fonts["FONT_FAMILY_CODE"], 10),
        bg=BG_CARD_ALT,
        fg=FG_WHITE,
        insertbackground=COLOR_PRIMARY,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    port_entry.pack(fill="x", ipady=4, pady=(3, 10))
    port_entry.insert(0, str(cur_port))

    # Preset Buttons
    preset_frame = tk.Frame(container, bg=BG_CARD)
    preset_frame.pack(fill="x", pady=(0, 12))

    tk.Label(
        preset_frame,
        text="Preset Cepat:",
        font=(fonts["FONT_FAMILY_UI"], 8),
        fg=FG_MUTED,
        bg=BG_CARD,
    ).pack(side="left", padx=(0, 6))

    def set_preset(h, p):
        host_entry.delete(0, tk.END)
        host_entry.insert(0, h)
        port_entry.delete(0, tk.END)
        port_entry.insert(0, str(p))

    btn_pre_robot = tk.Button(
        preset_frame,
        text="🤖 Robot (10.107.245.221)",
        font=(fonts["FONT_FAMILY_UI"], 8),
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=BORDER,
        relief="flat",
        borderwidth=0,
        padx=8,
        pady=2,
        cursor="hand2",
        command=lambda: set_preset("10.107.245.221", 5000),
    )
    btn_pre_robot.pack(side="left", padx=(0, 4))

    btn_pre_local = tk.Button(
        preset_frame,
        text="💻 Localhost (127.0.0.1)",
        font=(fonts["FONT_FAMILY_UI"], 8),
        fg=FG_SUB,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=BORDER,
        relief="flat",
        borderwidth=0,
        padx=8,
        pady=2,
        cursor="hand2",
        command=lambda: set_preset("127.0.0.1", 5000),
    )
    btn_pre_local.pack(side="left")

    # Feedback test label
    test_feedback = tk.Label(
        container,
        text="💡 Tekan 'Uji Ping' untuk menguji port terbuka sebelum menyimpan.",
        font=(fonts["FONT_FAMILY_UI"], 8),
        fg=FG_MUTED,
        bg=BG_CARD,
        anchor="w",
    )
    test_feedback.pack(fill="x", pady=(0, 14))

    def run_test_ping():
        h = host_entry.get().strip()
        p_str = port_entry.get().strip()
        if not h or not p_str:
            test_feedback.config(text="⚠ Alamat IP dan Port tidak boleh kosong", fg=COLOR_AMBER)
            return
        try:
            p = int(p_str)
        except ValueError:
            test_feedback.config(text="⚠ Port harus berupa angka bulat (1-65535)", fg=COLOR_RED)
            return

        test_feedback.config(text=f"Menghubungi {h}:{p}...", fg=COLOR_SKY)

        def _worker():
            t0 = time.time()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.5)
                    s.connect((h, p))
                latency_ms = int((time.time() - t0) * 1000)
                msg = f"✓ Berhasil terhubung ke {h}:{p} ({latency_ms} ms)"
                col = COLOR_GREEN
            except Exception as e:
                msg = f"✕ Gagal terhubung ke {h}:{p} ({e})"
                col = COLOR_RED

            def _apply():
                if modal.winfo_exists():
                    test_feedback.config(text=msg, fg=col)
            ui_call(_apply)

        threading.Thread(target=_worker, daemon=True).start()

    def save_and_apply():
        h = host_entry.get().strip()
        p_str = port_entry.get().strip()
        if not h or not p_str:
            test_feedback.config(text="⚠ Alamat IP dan Port tidak boleh kosong", fg=COLOR_AMBER)
            return
        try:
            p = int(p_str)
        except ValueError:
            test_feedback.config(text="⚠ Port harus berupa angka bulat", fg=COLOR_RED)
            return

        save_ros2_config(h, p)

        if hasattr(pipeline.ros2_adapter, "set_target"):
            pipeline.ros2_adapter.set_target(h, p)
        else:
            pipeline.ros2_adapter = SocketROS2Adapter(host=h, port=p)

        on_endpoint_updated(h, p)
        modal.destroy()

    # Action Buttons
    btn_box = tk.Frame(container, bg=BG_CARD)
    btn_box.pack(fill="x")

    btn_test = tk.Button(
        btn_box,
        text="🔍 UJI PING",
        font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
        fg=COLOR_SKY,
        bg=BG_CARD_ALT,
        activeforeground=FG_WHITE,
        activebackground=BORDER,
        relief="flat",
        borderwidth=0,
        padx=12,
        pady=6,
        cursor="hand2",
        command=run_test_ping,
    )
    btn_test.pack(side="left", padx=(0, 8))

    btn_save = tk.Button(
        btn_box,
        text="💾 SIMPAN & TERAPKAN",
        font=(fonts["FONT_FAMILY_UI"], 8, "bold"),
        fg=FG_WHITE,
        bg=COLOR_PRIMARY,
        activeforeground=FG_WHITE,
        activebackground=COLOR_PRIMARY_BORDER,
        relief="flat",
        borderwidth=0,
        padx=14,
        pady=6,
        cursor="hand2",
        command=save_and_apply,
    )
    btn_save.pack(side="left")

    btn_cancel = tk.Button(
        btn_box,
        text="BATAL",
        font=(fonts["FONT_FAMILY_UI"], 8),
        fg=FG_SUB,
        bg=BG_CARD,
        activeforeground=FG_WHITE,
        activebackground=BG_CARD_ALT,
        relief="flat",
        borderwidth=0,
        padx=12,
        pady=6,
        cursor="hand2",
        command=modal.destroy,
    )
    btn_cancel.pack(side="right")
