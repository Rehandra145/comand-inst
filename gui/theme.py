"""
GUI Theme — Color tokens, typography, and styling helpers for Tkinter.
"""

# =========================
# COLORS (Modern Clean Studio / Calm Slate Palette)
# =========================
BG_BASE = "#111317"          # Deep warm charcoal slate
BG_CARD = "#191c23"          # Elevated matte card surface
BG_CARD_ALT = "#212530"      # Inner container & inputs
BG_CARD_GLOW = "#1d222d"     # Robot execution banner container
BORDER = "#2b3140"           # Subtle hairline separator
BORDER_HOVER = "#40495d"     # Subtle border hover
FG_WHITE = "#f1f5f9"         # Crisp readable soft white
FG_SUB = "#94a3b8"           # Slate secondary text
FG_MUTED = "#64748b"         # Subtle hint / muted text

# Status & Accent Colors (Calm, desaturated, gentle on eyes)
COLOR_PRIMARY = "#6366f1"    # Modern Indigo
COLOR_PRIMARY_BG = "#1e2238"
COLOR_PRIMARY_BORDER = "#3730a3"

COLOR_SKY = "#38bdf8"        # Soft Sky Blue
COLOR_SKY_BG = "#142533"
COLOR_SKY_BORDER = "#0369a1"

COLOR_GREEN = "#10b981"      # Calm Emerald Green
COLOR_GREEN_BG = "#132d24"
COLOR_GREEN_BORDER = "#047857"

COLOR_AMBER = "#f59e0b"      # Soft Honey Amber
COLOR_AMBER_BG = "#2e2415"
COLOR_AMBER_BORDER = "#b45309"

COLOR_RED = "#f43f5e"        # Soft Rose Red
COLOR_RED_BG = "#311820"
COLOR_RED_BORDER = "#be123c"

COLOR_PURPLE = "#a78bfa"     # Soft Lilac / Violet
COLOR_PURPLE_BG = "#262138"
COLOR_PURPLE_BORDER = "#6d28d9"

# ============================================================================
# LINUX X11 EMOJI / NON-BMP SANITIZER (RenderAddGlyphs Crash Fix)
# ============================================================================
# Pada platform Linux X11, libtk8.6 (Xft) mengalami 'BadLength' protocol error
# dan segmentation fault / abort ketika merender karakter di luar BMP (> U+FFFF),
# seperti color emoji. Patch transparan ini memetakan emoji umum ke label teks aman
# dan menyaring karakter astral sebelum dikirimkan ke X server.
import sys
import tkinter as tk

if sys.platform.startswith("linux"):
    _EMOJI_REPLACEMENTS = {
        "🤖": "[Robot]",
        "🎙️": "[Mic]",
        "🎙": "[Mic]",
        "🛠️": "[Teknisi]",
        "🛠": "[Teknisi]",
        "📍": "[Lokasi]",
        "🧭": "[Arah]",
        "💊": "[Obat]",
        "🏥": "[ICU]",
        "🚨": "[IGD]",
        "🩺": "[Poli]",
        "👶": "[Anak]",
        "🚪": "[Lobi]",
        "🛑": "[Stop]",
        "🏠": "[Pos]",
        "🌐": "[ROS 2]",
        "🔊": "[TTS]",
        "⚙️": "[Config]",
        "⚙": "[Config]",
        "📋": "[Log]",
        "💡": "[Tip]",
        "✨": "[*]",
        "⏳": "[...]",
        "🔍": "[Cari]",
        "💬": "[Chat]",
        "🗑️": "[Hapus]",
        "🗑": "[Hapus]",
        "🔄": "[Reload]",
    }

    def _sanitize_tk_str(s):
        if not isinstance(s, str):
            return s
        for k, v in _EMOJI_REPLACEMENTS.items():
            if k in s:
                s = s.replace(k, v)
        s = s.replace("\ufe0f", "").replace("\ufe0e", "")
        # Filter karakter non-BMP (> U+FFFF)
        return "".join(ch for ch in s if ord(ch) <= 0xFFFF)

    _orig_options = tk.Misc._options

    def _safe_options(self, cnf, kw=None):
        if cnf:
            cnf = {k: _sanitize_tk_str(v) if isinstance(v, str) else v for k, v in cnf.items()}
        if kw:
            kw = {k: _sanitize_tk_str(v) if isinstance(v, str) else v for k, v in kw.items()}
        return _orig_options(self, cnf, kw)

    tk.Misc._options = _safe_options

    _orig_text_insert = tk.Text.insert

    def _safe_text_insert(self, index, chars, *args):
        return _orig_text_insert(self, index, _sanitize_tk_str(chars), *args)

    tk.Text.insert = _safe_text_insert

    _orig_entry_insert = tk.Entry.insert

    def _safe_entry_insert(self, index, string):
        return _orig_entry_insert(self, index, _sanitize_tk_str(string))

    tk.Entry.insert = _safe_entry_insert

    _orig_wm_title = tk.Wm.wm_title

    def _safe_wm_title(self, string=None):
        if string is not None:
            string = _sanitize_tk_str(string)
        return _orig_wm_title(self, string)

    tk.Wm.wm_title = _safe_wm_title


def get_theme_fonts(tkfont):
    """Deteksi font modern sistem dengan fallback yang aman di Linux, Windows, dan macOS."""
    installed_fonts = set(tkfont.families())

    # Prioritas font UI
    for f in ["Segoe UI", "Ubuntu", "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]:
        if f in installed_fonts:
            font_ui = f
            break
    else:
        font_ui = "Helvetica"

    # Prioritas font Monospace / Code
    for f in ["Cascadia Code", "JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono", "Consolas"]:
        if f in installed_fonts:
            font_code = f
            break
    else:
        font_code = "Courier"

    return {
        "FONT_FAMILY_UI": font_ui,
        "FONT_FAMILY_CODE": font_code,
        "F_TITLE": (font_ui, 13, "bold"),
        "F_SUBTITLE": (font_ui, 9),
        "F_SECTION": (font_ui, 9, "bold"),
        "F_BTN_TITLE": (font_ui, 10, "bold"),
        "F_CODE": (font_code, 9),
        "F_CODE_BOLD": (font_code, 9, "bold"),
        "F_ROBOT": (font_ui, 10, "bold"),
    }


def add_hover_effect(widget, normal_bg, hover_bg, normal_border=None, hover_border=None):
    """Menambahkan efek hover halus pada widget Button Tkinter."""
    def on_enter(e):
        if str(widget.cget("state")) != "disabled":
            widget.config(bg=hover_bg)
            if hover_border:
                widget.config(highlightbackground=hover_border, highlightcolor=hover_border)

    def on_leave(e):
        if str(widget.cget("state")) != "disabled":
            widget.config(bg=normal_bg)
            if normal_border:
                widget.config(highlightbackground=normal_border, highlightcolor=normal_border)

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)
