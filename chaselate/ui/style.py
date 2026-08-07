"""Qt style sheets for the overlay.

The look is "glass": a translucent dark plate, hairline border, generous corner radius.
Real blur behind the window is not available through Qt on Windows without per-platform
DWM calls, so depth is faked with layered translucent fills instead.

Every colour lives in one of the two palettes below and every font size comes from
:class:`~chaselate.config.UiConfig`, so a theme or font change is one stylesheet rebuild
with no widget touched individually.
"""

from __future__ import annotations

from typing import Dict

from ..config import UiConfig

# Palettes are plain dicts of QSS-ready colour literals. rgba() everywhere because the
# window itself is translucent: an opaque fill anywhere would punch a hard rectangle
# through the glass.
DARK: Dict[str, str] = {
    "plate": "rgba(16, 18, 24, 216)",
    "plate_soft": "rgba(28, 32, 42, 150)",
    "bar": "rgba(24, 27, 36, 190)",
    "border": "rgba(255, 255, 255, 38)",
    "border_strong": "rgba(255, 255, 255, 66)",
    "text": "#f2f4f8",
    "text_dim": "rgba(233, 238, 247, 150)",
    "text_faint": "rgba(233, 238, 247, 105)",
    "accent": "#6fd0ff",
    "accent_dim": "rgba(111, 208, 255, 55)",
    "good": "#7ee2a8",
    "warn": "#ffc46b",
    "bad": "#ff8b8b",
    "bad_soft": "rgba(255, 90, 90, 46)",
    "hover": "rgba(255, 255, 255, 26)",
    "pressed": "rgba(255, 255, 255, 46)",
    "field": "rgba(255, 255, 255, 20)",
    "scroll": "rgba(255, 255, 255, 46)",
}

LIGHT: Dict[str, str] = {
    "plate": "rgba(248, 249, 252, 226)",
    "plate_soft": "rgba(255, 255, 255, 170)",
    "bar": "rgba(236, 239, 245, 214)",
    "border": "rgba(20, 24, 34, 40)",
    "border_strong": "rgba(20, 24, 34, 74)",
    "text": "#151922",
    "text_dim": "rgba(21, 25, 34, 165)",
    "text_faint": "rgba(21, 25, 34, 115)",
    "accent": "#0d6fa8",
    "accent_dim": "rgba(13, 111, 168, 46)",
    "good": "#1d8a52",
    "warn": "#9a6212",
    "bad": "#b32626",
    "bad_soft": "rgba(179, 38, 38, 34)",
    "hover": "rgba(20, 24, 34, 22)",
    "pressed": "rgba(20, 24, 34, 40)",
    "field": "rgba(255, 255, 255, 190)",
    "scroll": "rgba(20, 24, 34, 60)",
}


def palette(theme: str) -> Dict[str, str]:
    return LIGHT if str(theme).strip().lower() == "light" else DARK


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clamp_font_size(size: int) -> int:
    """Overlay caption size. The floor keeps captions legible on a 4K panel."""
    try:
        return _clamp(int(size), 10, 72)
    except (TypeError, ValueError):
        return UiConfig.font_size


def build_stylesheet(cfg: UiConfig) -> str:
    """Full stylesheet for the overlay window, sized and coloured from ``cfg``."""
    c = palette(getattr(cfg, "theme", "dark"))
    translation_pt = clamp_font_size(getattr(cfg, "font_size", 20))
    original_pt = clamp_font_size(getattr(cfg, "original_font_size", 15))
    # The chrome must not scale with the caption size or a large font would swallow the
    # window; it tracks the smaller original size, loosely bounded.
    ui_pt = _clamp(int(round(original_pt * 0.8)), 9, 18)
    small_pt = _clamp(ui_pt - 1, 8, 16)

    return f"""
/* --- window shell ------------------------------------------------------ */
#OverlayRoot {{
    background: {c["plate"]};
    border: 1px solid {c["border"]};
    border-radius: 14px;
}}
QWidget {{
    color: {c["text"]};
    font-family: "Segoe UI", "Microsoft JhengHei UI", sans-serif;
    font-size: {ui_pt}pt;
}}

/* --- top bar ----------------------------------------------------------- */
#TopBar {{
    background: {c["bar"]};
    border: none;
    border-bottom: 1px solid {c["border"]};
    border-top-left-radius: 13px;
    border-top-right-radius: 13px;
}}
#TopBar QToolButton, #TopBar QPushButton {{
    background: {c["field"]};
    border: 1px solid {c["border"]};
    border-radius: 7px;
    padding: 3px 10px;
    color: {c["text"]};
}}
#TopBar QToolButton:hover, #TopBar QPushButton:hover {{
    background: {c["hover"]};
    border-color: {c["border_strong"]};
}}
#TopBar QToolButton:pressed, #TopBar QPushButton:pressed {{
    background: {c["pressed"]};
}}
#PrimaryButton {{
    background: {c["accent_dim"]};
    border: 1px solid {c["accent"]};
    color: {c["text"]};
    font-weight: 600;
    min-width: 62px;
}}
#IconButton {{
    padding: 3px 6px;
    min-width: 22px;
}}
#StatusText {{
    color: {c["text_dim"]};
    font-size: {small_pt}pt;
}}
#StatusText[severity="error"] {{ color: {c["bad"]}; }}
#StatusText[severity="busy"] {{ color: {c["warn"]}; }}
#StatusText[severity="live"] {{ color: {c["good"]}; }}

#TopBar QComboBox {{
    background: {c["field"]};
    border: 1px solid {c["border"]};
    border-radius: 7px;
    padding: 2px 8px;
    min-width: 118px;
}}
#TopBar QComboBox:hover {{ border-color: {c["border_strong"]}; }}
#TopBar QComboBox::drop-down {{ width: 14px; border: none; }}
QComboBox QAbstractItemView {{
    background: {c["plate"]};
    border: 1px solid {c["border_strong"]};
    selection-background-color: {c["accent_dim"]};
    color: {c["text"]};
}}

/* --- error banner ------------------------------------------------------ */
#Banner {{
    background: {c["bad_soft"]};
    border: 1px solid {c["bad"]};
    border-radius: 8px;
}}
#BannerText {{
    color: {c["bad"]};
    font-size: {small_pt}pt;
}}
#Banner QToolButton {{
    background: transparent;
    border: none;
    color: {c["bad"]};
    padding: 0px 4px;
}}

/* --- caption area ------------------------------------------------------ */
#CaptionScroll {{
    background: transparent;
    border: none;
}}
#CaptionCanvas {{ background: transparent; }}

#CaptionBlock {{
    background: {c["plate_soft"]};
    border: 1px solid {c["border"]};
    border-radius: 10px;
}}
#CaptionBlock[failed="true"] {{ border-color: {c["bad"]}; }}

#OriginalText {{
    color: {c["text_dim"]};
    font-size: {original_pt}pt;
    background: transparent;
}}
#TranslationText {{
    color: {c["text"]};
    font-size: {translation_pt}pt;
    background: transparent;
}}
#TranslationText[pending="true"] {{ color: {c["text_faint"]}; }}
#BlockError {{
    color: {c["bad"]};
    font-size: {small_pt}pt;
    background: transparent;
}}
#Placeholder {{
    color: {c["text_faint"]};
    font-size: {original_pt}pt;
}}

QSplitter::handle {{ background: {c["border"]}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {c["scroll"]};
    border-radius: 4px;
    min-height: 26px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {c["scroll"]};
    border-radius: 4px;
    min-width: 26px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}

/* --- bottom status ----------------------------------------------------- */
#StatusBar {{
    background: {c["bar"]};
    border: none;
    border-top: 1px solid {c["border"]};
    border-bottom-left-radius: 13px;
    border-bottom-right-radius: 13px;
}}
#MetricsText {{
    color: {c["text_faint"]};
    font-size: {small_pt}pt;
}}
#MetricsWarning {{
    color: {c["warn"]};
    font-size: {small_pt}pt;
}}
"""


def build_dialog_stylesheet(cfg: UiConfig) -> str:
    """Stylesheet for the settings dialog.

    Kept separate from the overlay's: the dialog is an opaque, ordinary window and
    inheriting the overlay's translucent plate colours makes its fields unreadable.
    """
    c = palette(getattr(cfg, "theme", "dark"))
    dark = palette(getattr(cfg, "theme", "dark")) is DARK
    base = "#1b1f28" if dark else "#f6f7fa"
    field = "#252a35" if dark else "#ffffff"
    return f"""
QDialog {{ background: {base}; color: {c["text"]}; }}
QDialog, QDialog QWidget {{
    font-family: "Segoe UI", "Microsoft JhengHei UI", sans-serif;
    font-size: 10pt;
    color: {c["text"]};
}}
QTabWidget::pane {{
    border: 1px solid {c["border"]};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    border: 1px solid transparent;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    padding: 6px 14px;
    color: {c["text_dim"]};
}}
QTabBar::tab:selected {{
    background: {field};
    border-color: {c["border"]};
    border-bottom-color: {field};
    color: {c["text"]};
}}
QTabBar::tab:hover {{ color: {c["text"]}; }}
QGroupBox {{
    border: 1px solid {c["border"]};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0px 4px;
    color: {c["text_dim"]};
}}
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {field};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 3px 6px;
    selection-background-color: {c["accent_dim"]};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c["accent"]}; }}
QComboBox QAbstractItemView {{
    background: {field};
    border: 1px solid {c["border_strong"]};
    selection-background-color: {c["accent_dim"]};
    color: {c["text"]};
}}
QPushButton {{
    background: {field};
    border: 1px solid {c["border_strong"]};
    border-radius: 6px;
    padding: 5px 14px;
}}
QPushButton:hover {{ background: {c["hover"]}; }}
QPushButton:pressed {{ background: {c["pressed"]}; }}
QPushButton:disabled {{ color: {c["text_faint"]}; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {c["border_strong"]};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {c["accent"]};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {c["accent"]}; border-radius: 2px; }}
#HintLabel {{ color: {c["text_faint"]}; font-size: 9pt; }}
#ErrorLabel {{ color: {c["bad"]}; font-size: 9pt; }}
#OkLabel {{ color: {c["good"]}; font-size: 9pt; }}
"""
