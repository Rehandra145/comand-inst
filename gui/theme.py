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


def get_theme_fonts(tkfont):
    """Deteksi font modern sistem (Segoe UI / Cascadia / Arial / Consolas)."""
    installed_fonts = set(tkfont.families())
    font_ui = "Segoe UI" if "Segoe UI" in installed_fonts else ("Arial" if "Arial" in installed_fonts else "Helvetica")
    font_code = "Cascadia Code" if "Cascadia Code" in installed_fonts else (
        "JetBrains Mono" if "JetBrains Mono" in installed_fonts else "Consolas"
    )

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
