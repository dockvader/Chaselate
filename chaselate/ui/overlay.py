"""The floating caption window.

A frameless, translucent, always-on-top strip that sits over whatever the user is
watching. It owns no threads and does no work of its own: everything arrives as a Qt
signal from :class:`~chaselate.pipeline.Pipeline`, already marshalled onto the GUI thread.

Two things drive most of the structure here:

* Captions arrive incrementally. A translation is streamed token by token, so each
  utterance gets a widget that is looked up by id and mutated in place, never re-created.
* The window has no title bar, so moving, resizing, closing and every command have to be
  provided by hand -- hence the drag handling, the explicit size grip and the tray menu.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QShortcut,
    QSizeGrip,
    QSizePolicy,
    QSystemTrayIcon,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..config import AppConfig, UiConfig
from ..languages import AUTO, LANGUAGES, display_name
from ..pipeline import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPING,
    Pipeline,
    Utterance,
)
from .settings import SettingsDialog, wait_for_background_jobs
from .style import build_stylesheet, palette

log = logging.getLogger(__name__)

#: Default strip size when there is no saved geometry, in logical pixels.
DEFAULT_WIDTH_RATIO = 0.62
DEFAULT_HEIGHT = 300
#: Gap between the strip and the bottom of the screen, so it clears a taskbar.
BOTTOM_MARGIN = 96
#: Smallest the user may drag the window to.
MIN_SIZE = QSize(420, 150)
#: Smallest size the app will *choose* on its own. MIN_SIZE fits barely two words per column,
#: which is fine if the user deliberately shrank it but is not a sane default.
USABLE_MIN_SIZE = QSize(900, 260)
#: An area smaller than this in either dimension is not a real monitor, so a screen reporting
#: one is assumed to be mid-reconfiguration and skipped.
PLAUSIBLE_SCREEN = QSize(800, 480)
#: Minimum clearance left either side when the strip cannot have its preferred width.
EDGE_MARGIN = 16
#: Slack in pixels within which the caption view counts as "scrolled to the bottom".
AUTOSCROLL_SLACK = 6


class LevelMeter(QWidget):
    """Input level bar. Turns green while the VAD reports speech."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._level = 0.0
        self._speech = False
        self._colors = palette("dark")
        self.setFixedSize(64, 8)
        self.setToolTip("Input level. Green while speech is detected.")

    def set_palette_colors(self, colors: Dict[str, str]) -> None:
        self._colors = colors
        self.update()

    def set_level(self, rms: float) -> None:
        # RMS of speech sits around 0.02-0.2; scaling by 6 puts normal talking near full
        # scale without a log curve that would make silence look busy.
        value = max(0.0, min(1.0, float(rms) * 6.0))
        # Decay slowly so the bar reads as a meter rather than flickering at 30 Hz.
        self._level = value if value > self._level else self._level * 0.82 + value * 0.18
        self.update()

    def set_speech(self, speaking: bool) -> None:
        self._speech = bool(speaking)
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        radius = rect.height() / 2.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 34))
        painter.drawRoundedRect(rect, radius, radius)
        if self._level > 0.01:
            filled = QRect(rect)
            filled.setWidth(max(2, int(rect.width() * self._level)))
            key = "good" if self._speech else "accent"
            painter.setBrush(QColor(self._colors.get(key, "#6fd0ff")))
            painter.drawRoundedRect(filled, radius, radius)
        painter.end()


class CaptionBlock(QFrame):
    """One utterance: its original text, its translation, and any failure note."""

    def __init__(self, utterance: Utterance, ui: UiConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("CaptionBlock")
        self.utterance = utterance
        self._failed = False

        self.original = QLabel(utterance.original)
        self.original.setObjectName("OriginalText")
        self.original.setWordWrap(True)
        self.original.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.original.setVisible(bool(ui.show_original))

        self.translation = QLabel(utterance.translation or "...")
        self.translation.setObjectName("TranslationText")
        self.translation.setWordWrap(True)
        self.translation.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.translation.setProperty("pending", "true" if not utterance.translation else "false")

        self.note = QLabel("")
        self.note.setObjectName("BlockError")
        self.note.setWordWrap(True)
        self.note.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 7, 10, 7)
        outer.setSpacing(4)

        if ui.layout == "side" and ui.show_original:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(14)
            row.addWidget(self.original, 1)
            row.addWidget(self.translation, 1)
            outer.addLayout(row)
        else:
            if ui.show_original:
                outer.addWidget(self.original)
            outer.addWidget(self.translation)
        outer.addWidget(self.note)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def set_translation(self, text: str, pending: bool) -> None:
        self.utterance.translation = text
        self.translation.setText(text or "...")
        self._set_prop(self.translation, "pending", "true" if pending or not text else "false")

    def set_failed(self, message: str) -> None:
        self._failed = True
        self.utterance.state = "failed"
        self.utterance.error = message
        # The original stays: a failed translation still leaves a useful transcript.
        self.note.setText(message)
        self.note.setVisible(True)
        if not self.utterance.translation:
            self.translation.setText("—")
            self._set_prop(self.translation, "pending", "true")
        self._set_prop(self, "failed", "true")

    def plain_text(self) -> str:
        parts = [self.utterance.original]
        if self.utterance.translation:
            parts.append(self.utterance.translation)
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _set_prop(widget: QWidget, name: str, value: str) -> None:
        # Qt only re-evaluates property selectors after an explicit unpolish/polish.
        if widget.property(name) == value:
            return
        widget.setProperty(name, value)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)


class OverlayWindow(QWidget):
    """The caption strip. Create once, on the GUI thread, and ``show()`` it."""

    #: Emitted just before the application really exits, after config has been saved.
    about_to_quit = pyqtSignal()

    def __init__(self, config: AppConfig, pipeline: Pipeline):
        super().__init__(None)
        self.config = config
        self.pipeline = pipeline

        self._blocks: Dict[int, CaptionBlock] = {}
        self._order: List[int] = []
        self._settings_dialog: Optional[SettingsDialog] = None
        self._quitting = False
        self._drag_origin: Optional[QPoint] = None
        self._suppress_lang_signal = False

        self.setWindowTitle(f"{APP_NAME} - Live Captions")
        self.setObjectName("OverlayShell")
        self.setMinimumSize(MIN_SIZE)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._build_ui()
        self._build_tray()
        self._build_shortcuts()
        self._apply_window_flags()
        self._apply_ui_config(config.ui, rebuild_blocks=False)
        self._restore_geometry()
        self._connect_pipeline()
        self._update_start_button(pipeline.state)

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)

        self.root = QFrame(self)
        self.root.setObjectName("OverlayRoot")
        shell.addWidget(self.root)

        column = QVBoxLayout(self.root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        column.addWidget(self._build_top_bar())
        column.addWidget(self._build_banner())
        column.addWidget(self._build_captions(), 1)
        column.addWidget(self._build_status_bar())

        # Parented to the window rather than the status bar so it survives the status bar
        # being hidden; positioned by hand in resizeEvent.
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setToolTip("Drag to resize")

    def _build_top_bar(self) -> QWidget:
        bar = QFrame(self.root)
        bar.setObjectName("TopBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 8, 6)
        row.setSpacing(8)

        self.start_button = QToolButton(bar)
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setText("Start")
        self.start_button.setToolTip("Start or stop capturing (Ctrl+Space)")
        self.start_button.clicked.connect(self.toggle_capture)
        row.addWidget(self.start_button)

        self.status_label = QLabel("Idle", bar)
        self.status_label.setObjectName("StatusText")
        # Ignored width: a long device name must shrink the status text rather than push
        # the language picker and buttons off the bar.
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_label.setMinimumWidth(0)
        row.addWidget(self.status_label, 1)

        self.meter = LevelMeter(bar)
        row.addWidget(self.meter, 0, Qt.AlignVCenter)

        self.lang_combo = self._build_lang_combo(bar)
        row.addWidget(self.lang_combo)

        self.settings_button = self._icon_button(bar, "⚙", "Settings (Ctrl+,)")
        self.settings_button.clicked.connect(self.open_settings)
        row.addWidget(self.settings_button)

        self.hide_button = self._icon_button(bar, "✕", "Hide to tray (Esc)")
        self.hide_button.clicked.connect(self.hide_to_tray)
        row.addWidget(self.hide_button)
        return bar

    def _build_lang_combo(self, parent: QWidget):
        from PyQt5.QtWidgets import QComboBox  # local: only needed here

        combo = QComboBox(parent)
        combo.setToolTip("Translate into this language")
        # Without a cap the combo eats every spare pixel in the bar.
        combo.setMaximumWidth(240)
        combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # No auto-detect entry: a target language must be concrete. Endonyms only here --
        # the full "native (English)" label does not fit the bar; Settings shows both.
        for lang in LANGUAGES:
            combo.addItem(lang.native, lang.code)
            combo.setItemData(combo.count() - 1, display_name(lang.code), Qt.ToolTipRole)
        idx = combo.findData(self.config.translate.target_lang)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._on_target_lang_changed)
        return combo

    def _icon_button(self, parent: QWidget, glyph: str, tip: str) -> QToolButton:
        button = QToolButton(parent)
        button.setObjectName("IconButton")
        button.setText(glyph)
        button.setToolTip(tip)
        button.setCursor(Qt.ArrowCursor)
        return button

    def _build_banner(self) -> QWidget:
        self.banner = QFrame(self.root)
        self.banner.setObjectName("Banner")
        self.banner.setVisible(False)
        row = QHBoxLayout(self.banner)
        row.setContentsMargins(10, 5, 6, 5)
        row.setSpacing(6)
        self.banner_label = QLabel("", self.banner)
        self.banner_label.setObjectName("BannerText")
        self.banner_label.setWordWrap(True)
        self.banner_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.banner_label, 1)
        close = QToolButton(self.banner)
        close.setText("✕")
        close.setToolTip("Dismiss")
        close.clicked.connect(self.banner.hide)
        row.addWidget(close)
        return self.banner

    def _build_captions(self) -> QWidget:
        self.scroll = QScrollArea(self.root)
        self.scroll.setObjectName("CaptionScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canvas = QWidget()
        self.canvas.setObjectName("CaptionCanvas")
        self.canvas_layout = QVBoxLayout(self.canvas)
        self.canvas_layout.setContentsMargins(10, 8, 10, 8)
        self.canvas_layout.setSpacing(6)
        self.canvas_layout.setAlignment(Qt.AlignTop)

        self.placeholder = QLabel(
            "Press Start (Ctrl+Space) to caption whatever is playing.", self.canvas
        )
        self.placeholder.setObjectName("Placeholder")
        self.placeholder.setWordWrap(True)
        self.canvas_layout.addWidget(self.placeholder)

        self.scroll.setWidget(self.canvas)
        return self.scroll

    def _build_status_bar(self) -> QWidget:
        self.status_bar = QFrame(self.root)
        self.status_bar.setObjectName("StatusBar")
        row = QHBoxLayout(self.status_bar)
        row.setContentsMargins(10, 3, 26, 3)
        row.setSpacing(10)
        self.metrics_label = QLabel("", self.status_bar)
        self.metrics_label.setObjectName("MetricsText")
        self.metrics_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.metrics_label, 1)
        self.drops_label = QLabel("", self.status_bar)
        self.drops_label.setObjectName("MetricsWarning")
        self.drops_label.setVisible(False)
        row.addWidget(self.drops_label)
        return self.status_bar

    def _build_shortcuts(self) -> None:
        bindings = (
            ("Ctrl+Space", self.toggle_capture),
            ("Ctrl+,", self.open_settings),
            ("Ctrl+H", self.toggle_original),
            ("Esc", self.hide_to_tray),
            ("Ctrl+Q", self.quit),
            ("Ctrl+C", self.copy_captions),
            ("Ctrl++", lambda: self.adjust_font(1)),
            ("Ctrl+=", lambda: self.adjust_font(1)),
            ("Ctrl+Shift+=", lambda: self.adjust_font(1)),
            ("Ctrl+-", lambda: self.adjust_font(-1)),
        )
        self._shortcuts = []
        for keys, slot in bindings:
            shortcut = QShortcut(QKeySequence(keys), self)
            # The overlay can be click-through; keep shortcuts working while it has focus
            # only, so we do not steal keys from the app the user is actually watching.
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self._make_icon(), self)
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        self._act_show = QAction("Show", menu)
        self._act_show.triggered.connect(self.show_overlay)
        self._act_hide = QAction("Hide", menu)
        self._act_hide.triggered.connect(self.hide_to_tray)
        self._act_toggle = QAction("Start", menu)
        self._act_toggle.triggered.connect(self.toggle_capture)
        self._act_clickthrough = QAction("Click-through", menu)
        self._act_clickthrough.setCheckable(True)
        self._act_clickthrough.setChecked(bool(self.config.ui.mouse_transparent))
        self._act_clickthrough.toggled.connect(self.set_click_through)
        act_settings = QAction("Settings...", menu)
        act_settings.triggered.connect(self.open_settings)
        act_quit = QAction("Quit", menu)
        act_quit.triggered.connect(self.quit)

        menu.addAction(self._act_show)
        menu.addAction(self._act_hide)
        menu.addSeparator()
        menu.addAction(self._act_toggle)
        # Click-through disables the toolbar, so the only way back is this menu entry.
        menu.addAction(self._act_clickthrough)
        menu.addAction(act_settings)
        menu.addSeparator()
        menu.addAction(act_quit)
        # Keep a reference: QMenu is not owned by the tray icon.
        self._tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        try:
            self.tray.show()
        except Exception:  # noqa: BLE001 - no tray on this session; menu just goes unused
            log.debug("system tray unavailable", exc_info=True)

    def _make_icon(self) -> QIcon:
        """Draw the tray icon so the app needs no image files on disk."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colors = palette(self.config.ui.theme)
        painter.setBrush(QColor(colors.get("accent", "#6fd0ff")))
        painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
        painter.drawEllipse(3, 3, 58, 58)
        font = QFont("Segoe UI", 30, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(12, 16, 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "C")
        painter.end()
        return QIcon(pixmap)

    # -- signal wiring -----------------------------------------------------

    def _connect_pipeline(self) -> None:
        p = self.pipeline
        p.status_changed.connect(self._on_status)
        p.error.connect(self.show_banner)
        p.level.connect(self.meter.set_level)
        p.speech_state.connect(self.meter.set_speech)
        p.utterance_added.connect(self._on_utterance)
        p.translation_delta.connect(self._on_delta)
        p.translation_done.connect(self._on_done)
        p.translation_failed.connect(self._on_failed)
        p.metrics.connect(self._on_metrics)

    # -- appearance --------------------------------------------------------

    def _apply_window_flags(self) -> None:
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.config.ui.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        visible = self.isVisible()
        self.setWindowFlags(flags)
        self.setAttribute(
            Qt.WA_TransparentForMouseEvents, bool(self.config.ui.mouse_transparent)
        )
        if visible:
            # setWindowFlags hides a visible window on Windows.
            self.show()

    def _apply_ui_config(self, ui: UiConfig, rebuild_blocks: bool = True) -> None:
        self.setStyleSheet(build_stylesheet(ui))
        self.setWindowOpacity(max(0.2, min(1.0, float(ui.opacity))))
        self.meter.set_palette_colors(palette(ui.theme))
        self.status_bar.setVisible(bool(ui.show_status_bar))
        if hasattr(self, "_act_clickthrough"):
            self._act_clickthrough.setChecked(bool(ui.mouse_transparent))
        if rebuild_blocks:
            self._rebuild_blocks()
        self._position_grip()

    def apply_config(self, config: AppConfig) -> None:
        """Adopt an edited config: restyle, relayout, and hand it to the pipeline."""
        layout_changed = (
            config.ui.layout != self.config.ui.layout
            or config.ui.show_original != self.config.ui.show_original
        )
        flags_changed = (
            config.ui.always_on_top != self.config.ui.always_on_top
            or config.ui.mouse_transparent != self.config.ui.mouse_transparent
        )
        target_changed = config.translate.target_lang != self.config.translate.target_lang

        self.config = config
        if flags_changed:
            self._apply_window_flags()
        self._apply_ui_config(config.ui, rebuild_blocks=layout_changed)
        if target_changed:
            self._sync_lang_combo()
        self.pipeline.apply_config(config)

    def apply_preview(self, config: AppConfig) -> None:
        """Live preview from the settings dialog. Appearance only, never the pipeline."""
        layout_changed = (
            config.ui.layout != self.config.ui.layout
            or config.ui.show_original != self.config.ui.show_original
        )
        flags_changed = (
            config.ui.always_on_top != self.config.ui.always_on_top
            or config.ui.mouse_transparent != self.config.ui.mouse_transparent
        )
        # Preview must not mutate the dialog's working copy, so take our own snapshot of
        # the UI section only and leave the rest of self.config alone.
        self.config.ui = UiConfig(**vars(config.ui))
        if flags_changed:
            self._apply_window_flags()
        self._apply_ui_config(self.config.ui, rebuild_blocks=layout_changed)

    def _rebuild_blocks(self) -> None:
        """Re-create every caption block, e.g. after the layout mode changed."""
        utterances = [self._blocks[uid].utterance for uid in self._order if uid in self._blocks]
        for block in self._blocks.values():
            block.setParent(None)
            block.deleteLater()
        self._blocks.clear()
        kept = list(self._order)
        self._order.clear()
        for utterance in utterances:
            self._add_block(utterance, autoscroll=False)
        if kept:
            self._scroll_to_bottom()

    # -- caption plumbing --------------------------------------------------

    def _at_bottom(self) -> bool:
        bar = self.scroll.verticalScrollBar()
        return bar.value() >= bar.maximum() - AUTOSCROLL_SLACK

    def _scroll_to_bottom(self) -> None:
        # Deferred: the new block has no height until the layout has run.
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def _add_block(self, utterance: Utterance, autoscroll: bool = True) -> CaptionBlock:
        stick = autoscroll and self._at_bottom()
        self.placeholder.setVisible(False)
        block = CaptionBlock(utterance, self.config.ui, self.canvas)
        self.canvas_layout.addWidget(block)
        self._blocks[utterance.id] = block
        self._order.append(utterance.id)
        self._trim_history()
        if stick:
            self._scroll_to_bottom()
        return block

    def _trim_history(self) -> None:
        limit = max(1, int(self.config.ui.max_history))
        while len(self._order) > limit:
            uid = self._order.pop(0)
            block = self._blocks.pop(uid, None)
            if block is not None:
                block.setParent(None)
                block.deleteLater()

    def _on_utterance(self, utterance: object) -> None:
        if not isinstance(utterance, Utterance):
            log.debug("ignoring unexpected utterance payload: %r", type(utterance))
            return
        self._add_block(utterance)

    def _on_delta(self, uid: int, text: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            stick = self._at_bottom()
            block.set_translation(text, pending=True)
            if stick:
                self._scroll_to_bottom()

    def _on_done(self, uid: int, text: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            stick = self._at_bottom()
            block.utterance.state = "done"
            block.set_translation(text, pending=False)
            if stick:
                self._scroll_to_bottom()

    def _on_failed(self, uid: int, message: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            block.set_failed(message)

    # -- status ------------------------------------------------------------

    def _on_status(self, state: str, detail: str) -> None:
        self.status_label.setText(detail or state.capitalize())
        self.status_label.setToolTip(detail or state)
        severity = {
            STATE_ERROR: "error",
            STATE_STARTING: "busy",
            STATE_STOPPING: "busy",
            STATE_RUNNING: "live",
        }.get(state, "idle")
        CaptionBlock._set_prop(self.status_label, "severity", severity)
        self._update_start_button(state)
        if state == STATE_IDLE:
            self.meter.set_level(0.0)
            self.meter.set_speech(False)

    def _update_start_button(self, state: str) -> None:
        # STATE_ERROR must fall through to "Start": the pipeline is no longer running and a
        # button still reading "Stop" would leave the user with no way to retry.
        running = state in (STATE_STARTING, STATE_RUNNING, STATE_STOPPING)
        self.start_button.setText("Stop" if running else "Start")
        self.start_button.setEnabled(state != STATE_STOPPING)
        if hasattr(self, "_act_toggle"):
            self._act_toggle.setText("Stop" if running else "Start")

    def _on_metrics(self, data: dict) -> None:
        if not self.config.ui.show_status_bar:
            return
        bits = []
        device = data.get("device")
        if device:
            bits.append(str(device))
        asr = data.get("asr")
        if asr:
            bits.append(str(asr))
        rtf = data.get("asr_rtf") or 0.0
        if rtf:
            bits.append(f"RTF {rtf:.2f}")
        bits.append(
            "queue {}/{}/{}".format(
                data.get("audio_queue", 0),
                data.get("asr_queue", 0),
                data.get("translate_queue", 0),
            )
        )
        count = data.get("asr_count")
        if count:
            bits.append(f"{count} utt")
        self.metrics_label.setText("   ".join(bits))
        self.metrics_label.setToolTip("device / model / realtime factor / audio-ASR-translate backlog")

        dropped = [
            ("audio", int(data.get("dropped_audio") or 0)),
            ("segments", int(data.get("dropped_segments") or 0)),
            ("sentences", int(data.get("dropped_sentences") or 0)),
        ]
        losses = [f"{n} {name} dropped" for name, n in dropped if n > 0]
        self.drops_label.setText("  ".join(losses))
        self.drops_label.setVisible(bool(losses))
        self.drops_label.setToolTip(
            "The pipeline is behind and discarding the oldest work. Try a smaller "
            "Whisper model or a faster translation model."
        )

    def show_banner(self, message: str) -> None:
        """Non-modal error strip. Deliberately not a dialog: a modal would cover captions."""
        if not message:
            return
        self.banner_label.setText(message)
        self.banner_label.setToolTip(message)
        self.banner.setVisible(True)

    # -- commands ----------------------------------------------------------

    def toggle_capture(self) -> None:
        if self.pipeline.running:
            self.pipeline.stop()
        else:
            self.banner.setVisible(False)
            self.pipeline.start()

    def open_settings(self) -> None:
        if self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        dialog = SettingsDialog(self.config, self.pipeline, self)
        self._settings_dialog = dialog
        dialog.preview_requested.connect(self.apply_preview)
        dialog.finished.connect(lambda _r: self._on_settings_finished(dialog))
        dialog.accepted_config.connect(self._on_settings_applied)
        dialog.show()

    def _on_settings_applied(self, config: object) -> None:
        if isinstance(config, AppConfig):
            self.apply_config(config)
            config.save()

    def _on_settings_finished(self, dialog: "SettingsDialog") -> None:
        # Cancel leaves nothing behind, but a live preview may already have been applied,
        # so restore the appearance the config actually holds.
        self._apply_ui_config(self.config.ui)
        self._apply_window_flags()
        if self._settings_dialog is dialog:
            self._settings_dialog = None
        dialog.deleteLater()

    def toggle_original(self) -> None:
        self.config.ui.show_original = not self.config.ui.show_original
        self._apply_ui_config(self.config.ui)

    def adjust_font(self, step: int) -> None:
        ui = self.config.ui
        ui.font_size = max(10, min(72, int(ui.font_size) + step))
        ui.original_font_size = max(10, min(72, int(ui.original_font_size) + step))
        self._apply_ui_config(ui, rebuild_blocks=False)

    def copy_captions(self) -> None:
        text = "\n\n".join(
            self._blocks[uid].plain_text() for uid in self._order if uid in self._blocks
        )
        QGuiApplication.clipboard().setText(text)
        if text:
            self.tray.setToolTip(f"{APP_NAME} - captions copied")

    def set_click_through(self, enabled: bool) -> None:
        self.config.ui.mouse_transparent = bool(enabled)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, bool(enabled))

    def show_overlay(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_to_tray(self) -> None:
        self._save_geometry()
        self.hide()

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._save_geometry()
        self.config.save()
        # A settings probe may still be inside WASAPI or an HTTP call; tearing the process
        # down under it crashes rather than exits.
        wait_for_background_jobs()
        try:
            self.pipeline.shutdown()
        except Exception:  # noqa: BLE001 - never block exit on a stuck worker
            log.exception("pipeline shutdown failed")
        self.tray.hide()
        self.about_to_quit.emit()
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            if self.isVisible():
                self.hide_to_tray()
            else:
                self.show_overlay()

    def _on_target_lang_changed(self, index: int) -> None:
        if self._suppress_lang_signal or index < 0:
            return
        code = self.lang_combo.itemData(index)
        if not code or code == AUTO or code == self.config.translate.target_lang:
            return
        self.config.translate.target_lang = code
        self.pipeline.apply_config(self.config)
        self.config.save()

    def _sync_lang_combo(self) -> None:
        idx = self.lang_combo.findData(self.config.translate.target_lang)
        if idx < 0:
            return
        self._suppress_lang_signal = True
        try:
            self.lang_combo.setCurrentIndex(idx)
        finally:
            self._suppress_lang_signal = False

    # -- geometry ----------------------------------------------------------

    def _restore_geometry(self) -> None:
        saved = list(self.config.ui.geometry or [])
        if len(saved) == 4:
            try:
                x, y, w, h = (int(v) for v in saved)
            except (TypeError, ValueError):
                saved = []
            else:
                rect = QRect(x, y, max(MIN_SIZE.width(), w), max(MIN_SIZE.height(), h))
                if self._on_some_screen(rect):
                    self.setGeometry(rect)
                    return
        self._center_bottom()

    def _on_some_screen(self, rect: QRect) -> bool:
        """Reject geometry for a monitor that is no longer attached."""
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().intersects(rect):
                return True
        return False

    def _center_bottom(self) -> None:
        """Place a caption strip along the bottom of the most likely screen.

        Deliberately paranoid about what Qt reports. On a multi-monitor setup that is being
        reconfigured (a display waking up, a resolution change), ``availableGeometry`` can
        briefly return a rectangle matching no real monitor; trusting it once produced a
        425x150 window positioned above the top of every screen, which is unusable and, since
        geometry is persisted on exit, sticky. So: sanity-check the area, insist on a size
        that can actually hold two columns of captions, and clamp the result back inside the
        screen.
        """
        area = self._best_screen_area()
        if area is None:
            self.resize(1000, DEFAULT_HEIGHT)
            return

        width = max(USABLE_MIN_SIZE.width(), int(area.width() * DEFAULT_WIDTH_RATIO))
        width = min(width, max(MIN_SIZE.width(), area.width() - 2 * EDGE_MARGIN))
        height = max(USABLE_MIN_SIZE.height(), DEFAULT_HEIGHT)
        height = min(height, max(MIN_SIZE.height(), int(area.height() * 0.5)))

        x = area.x() + (area.width() - width) // 2
        y = area.y() + area.height() - height - BOTTOM_MARGIN
        # Keep the whole strip on the chosen screen even if the margins do not fit.
        x = max(area.x(), min(x, area.right() - width + 1))
        y = max(area.y(), min(y, area.bottom() - height + 1))
        rect = QRect(x, y, width, height)
        log.debug("placing overlay at %s within screen area %s", rect.getRect(), area.getRect())
        self.setGeometry(rect)

    def _best_screen_area(self) -> Optional[QRect]:
        """Available area of the screen the user is most likely looking at.

        Prefers the screen under the pointer over the "primary" one, which on a multi-monitor
        desk is often not the one being used. Falls back through primary, then the largest
        screen, and rejects any area too small to be a real monitor.
        """
        candidates = []
        try:
            under_cursor = QGuiApplication.screenAt(QCursor.pos())
        except Exception:  # noqa: BLE001 - screenAt can fail while displays are changing
            under_cursor = None
        for screen in (under_cursor, QGuiApplication.primaryScreen()):
            if screen is not None:
                candidates.append(screen)
        candidates.extend(
            sorted(
                QGuiApplication.screens(),
                key=lambda s: s.geometry().width() * s.geometry().height(),
                reverse=True,
            )
        )

        for screen in candidates:
            for rect in (screen.availableGeometry(), screen.geometry()):
                if rect.width() >= PLAUSIBLE_SCREEN.width() and rect.height() >= PLAUSIBLE_SCREEN.height():
                    return rect

        # Nothing plausible: use whatever primary claims rather than refusing to place.
        primary = QGuiApplication.primaryScreen()
        if primary is not None:
            log.warning(
                "no plausible screen geometry reported; falling back to %s",
                primary.availableGeometry().getRect(),
            )
            return primary.availableGeometry()
        return None

    def _save_geometry(self) -> None:
        rect = self.geometry()
        self.config.ui.geometry = [rect.x(), rect.y(), rect.width(), rect.height()]

    def _position_grip(self) -> None:
        if not hasattr(self, "_grip"):
            return
        size = self._grip.size()
        self._grip.move(self.width() - size.width() - 3, self.height() - size.height() - 3)
        self._grip.raise_()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._position_grip()

    # -- window behaviour --------------------------------------------------

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            # globalPos - frameGeometry origin, so the window keeps its offset under the
            # cursor for the whole drag.
            self._drag_origin = event.globalPos() - self.frameGeometry().topLeft()
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_origin)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self.unsetCursor()
            self._save_geometry()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):  # noqa: N802
        if self._quitting:
            event.accept()
            return
        # The X button and the window manager mean "get out of the way", not "exit"; only
        # Quit really exits.
        event.ignore()
        self.hide_to_tray()

    # -- introspection (used by tests) -------------------------------------

    def caption_block_count(self) -> int:
        return len(self._blocks)

    def caption_block(self, uid: int) -> Optional[CaptionBlock]:
        return self._blocks.get(uid)
