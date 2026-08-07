"""PyQt5 presentation layer: the caption overlay and its settings dialog."""

from .overlay import OverlayWindow
from .settings import SettingsDialog
from .style import build_stylesheet

__all__ = ["OverlayWindow", "SettingsDialog", "build_stylesheet"]
