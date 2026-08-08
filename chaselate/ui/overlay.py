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
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
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
from .style import build_stylesheet, palette, qcolor

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
#: Slack in pixels within which a scrollbar value counts as "where we last put it ourselves"
#: rather than a manual scroll away from it. See _on_scroll_value_changed.
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
        #: id of the block currently marked "active" (still awaiting translation), or None.
        self._active_uid: Optional[int] = None
        #: Whether the view should keep repositioning itself as new content arrives. A manual
        #: scroll turns this off; the jump pill turns it back on. See _on_scroll_value_changed.
        self._auto_follow = True
        #: True only for the instant _scroll_to's own bar.setValue() call is on the stack, so
        #: _on_scroll_value_changed can tell our own moves apart from a genuine user scroll.
        self._applying_scroll = False
        #: Bumped to invalidate any in-flight _scroll_to "catch up" still connected; see
        #: _scroll_to for why a plain disconnect is not enough on its own.
        self._scroll_chase_gen = 0
        self._scroll_chase_conn: Optional[Callable] = None
        #: Bumped on every _defer_canvas_height_sync call so a later call's re-check ticks can
        #: tell an earlier, superseded call's ticks apart from their own and skip.
        self._canvas_height_sync_gen = 0
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

        self._build_jump_pill()

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

        # Trailing room so the most recent (likely still-active) sentence can actually be
        # scrolled to the middle of the view. Centring the *last* item in an AlignTop stack has
        # nowhere to go without this: there is nothing below it to make room by scrolling
        # further. Kept as a real widget rather than a QSpacerItem so _sync_canvas_height's
        # heightForWidth() aggregation picks it up like any other item, no special-casing
        # needed. Height is kept at roughly half the viewport by _sync_canvas_height; the
        # bottom-follow target (see _content_bottom / _follow_target) deliberately does not
        # scroll into it, so it never reintroduces the blank-space-at-the-bottom bug this
        # exists alongside a fix for -- it is scroll range that only centring ever uses.
        self.canvas_tail_spacer = QWidget(self.canvas)
        self.canvas_tail_spacer.setObjectName("CaptionTailSpacer")
        self.canvas_layout.addWidget(self.canvas_tail_spacer)

        self.scroll.setWidget(self.canvas)
        bar = self.scroll.verticalScrollBar()
        # Deliberately two different slots, not one shared by both signals: a rangeChanged
        # event fires *before* a chase (see _scroll_to_bottom) has had a chance to catch the
        # scrollbar up to a still-growing range, so checking "are we at the bottom" from a
        # rangeChanged handler would see a false negative and cancel the very chase meant to
        # fix it. Only a genuine value change -- the scrollbar actually moving, which is what a
        # manual scroll looks like -- is trustworthy evidence that the user moved away.
        bar.valueChanged.connect(self._on_scroll_value_changed)
        # New content changes what "at the bottom" means even if the user has not touched the
        # scrollbar, so a caption arriving while they are reading up must be able to reveal
        # the pill immediately rather than waiting for the next manual scroll. Visibility only,
        # no cancellation here -- see above.
        bar.rangeChanged.connect(self._update_jump_pill)
        return self.scroll

    def _build_jump_pill(self) -> None:
        """The "you have scrolled away" affordance.

        Parented to ``self.root`` rather than the scroll area: QScrollArea manages its own
        children (viewport, scrollbars), and an extra widget dropped into that hierarchy is
        liable to be fought over by its layout. A sibling positioned by hand -- the same
        approach already used for the resize grip -- has no such conflict.
        """
        self._jump_pill = QToolButton(self.root)
        self._jump_pill.setObjectName("JumpPill")
        self._jump_pill.setCursor(Qt.PointingHandCursor)
        self._jump_pill.clicked.connect(self._jump_to_latest)
        self._jump_pill.hide()

    def _pill_text(self) -> str:
        return "▾ Translating — click to follow" if self._active_uid is not None else "▾ New captions"

    def _jump_to_latest(self) -> None:
        self._auto_follow = True
        self._scroll_to(self._follow_target())
        # Optimistic: the real scroll is deferred a frame (see _scroll_to), but hiding
        # immediately avoids a click feeling like it did nothing for that frame.
        self._jump_pill.hide()

    def _on_scroll_value_changed(self, _value: int) -> None:
        # self._applying_scroll brackets our own bar.setValue() calls (see _scroll_to), so
        # anything landing here while it is False is unambiguously not us -- a manual scroll, a
        # wheel event, a drag, anything. Comparing the new value against where we last put it
        # was tried first and is not reliable enough: a chase from a still-loading block can be
        # armed and waiting when the user scrolls, and if their landing spot happens to be
        # within a few pixels of that stale target, a value-only check would wrongly treat the
        # manual scroll as "still following". A flag set only for the instant of our own call
        # has no such coincidence to worry about.
        if not self._applying_scroll:
            self._auto_follow = False
            self._cancel_scroll_chase()
        self._update_jump_pill()

    def _update_jump_pill(self, *_args) -> None:
        if not hasattr(self, "_jump_pill"):
            return
        show = bool(self._order) and not self._auto_follow
        if show:
            self._jump_pill.setText(self._pill_text())
            self._jump_pill.adjustSize()
            self._position_jump_pill()
        self._jump_pill.setVisible(show)

    def _position_jump_pill(self) -> None:
        if not hasattr(self, "_jump_pill") or not hasattr(self, "scroll"):
            return
        size = self._jump_pill.sizeHint()
        # scroll's geometry is in self.root's coordinate space, which is exactly the parent
        # the pill lives in, so no mapping is needed -- just anchor to its bottom edge.
        area = self.scroll.geometry()
        x = area.x() + (area.width() - size.width()) // 2
        y = area.y() + area.height() - size.height() - 10
        self._jump_pill.move(max(area.x(), x), max(area.y(), y))
        self._jump_pill.resize(size)
        self._jump_pill.raise_()

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
        """Draw the tray icon so the app needs no image files on disk.

        Same three-chevron mark as the packaged exe's icon (see
        packaging/scripts/make_icon.py for the full account of what it is meant to evoke) --
        drawn here at runtime instead of loaded from a file so the accent colour can follow
        the user's light/dark theme choice the same way the rest of the chrome does.
        """
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # qcolor(), not QColor(...) directly: several palette entries (including "plate",
        # used right below) are written in CSS's rgba(r, g, b, a) form for the stylesheet's
        # sake, which QColor's own string constructor does not parse -- it fails silently to
        # an invalid, opaque-black colour rather than raising. See style.qcolor's docstring.
        colors = palette(self.config.ui.theme)
        plate = qcolor(colors.get("plate", "#0e1016"))
        plate.setAlpha(255)
        accent = qcolor(colors.get("accent", "#6fd0ff"))
        text = qcolor(colors.get("text", "#f2f4f8"))

        radius = size * 0.225
        painter.setPen(Qt.NoPen)
        painter.setBrush(plate)
        painter.drawRoundedRect(0, 0, size, size, radius, radius)

        glow_cx, glow_cy, glow_r = size * 0.665, size * 0.5, size * 0.34
        glow = QRadialGradient(glow_cx, glow_cy, glow_r)
        glow_color = QColor(accent)
        glow_color.setAlpha(90)
        glow.setColorAt(0.0, glow_color)
        glow_edge = QColor(accent)
        glow_edge.setAlpha(0)
        glow.setColorAt(1.0, glow_edge)
        painter.setBrush(glow)
        painter.drawEllipse(
            int(glow_cx - glow_r), int(glow_cy - glow_r), int(glow_r * 2), int(glow_r * 2)
        )

        half_w, half_h = size * 0.115, size * 0.195
        stroke_w = size * 0.075
        spacing = size * 0.185
        start_x = size * 0.5 - spacing
        cy = size * 0.5
        for i in range(3):
            cx = start_x + spacing * i
            t = i / 2.0
            color = QColor(
                round(text.red() + (accent.red() - text.red()) * t),
                round(text.green() + (accent.green() - text.green()) * t),
                round(text.blue() + (accent.blue() - text.blue()) * t),
            )
            path = QPainterPath()
            path.moveTo(cx - half_w, cy - half_h)
            path.lineTo(cx + half_w, cy)
            path.lineTo(cx - half_w, cy + half_h)
            pen = QPen(color, stroke_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.strokePath(path, pen)

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
        else:
            # A font-size-only change (no layout/rebuild) still changes each block's needed
            # height at the same width; the style needs a layout pass to actually apply to the
            # existing labels first, hence the deferred sync rather than an immediate one.
            self._defer_canvas_height_sync()
        self._position_grip()
        self._position_jump_pill()

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
            self._add_block(utterance)
        self._defer_canvas_height_sync()  # each _add_block already schedules this; once more for good measure
        active_survived = self._active_uid in self._blocks
        if not active_survived:
            self._active_uid = None
        if kept:
            # Block heights can change completely under a rebuild (side <-> stacked layout,
            # original text shown or hidden), so the old scroll position does not mean
            # anything in the new one -- always reposition rather than respecting a pause
            # that applied to a layout that no longer exists.
            self._auto_follow = True
            if active_survived:
                # Re-applies the "active" highlight to the freshly-built block (a rebuild
                # replaces every CaptionBlock instance) and, since _auto_follow is now True,
                # centres on it via _set_active's own call to _follow().
                self._set_active(self._active_uid)
            else:
                self._follow()
        self._update_jump_pill()

    # -- caption plumbing --------------------------------------------------

    def _cancel_scroll_chase(self) -> None:
        """Invalidate any in-flight "catch up" from a previous _scroll_to call.

        Bumping the generation makes the chase closure below a no-op even if its disconnect
        has not run yet, which matters because the two can otherwise race: a chase armed by an
        earlier reposition must not override a scroll the user makes a moment later, just
        because a range-changed event from unrelated new content happens to land inside its
        window.
        """
        self._scroll_chase_gen += 1
        if self._scroll_chase_conn is not None:
            try:
                self.scroll.verticalScrollBar().rangeChanged.disconnect(self._scroll_chase_conn)
            except TypeError:
                pass  # already disconnected
            self._scroll_chase_conn = None

    def _scroll_to(self, compute_target: Callable[[], int]) -> None:
        """Move to ``compute_target()``, and stay there through any layout passes still in
        flight, marking this as the new auto-follow position.

        A single deferred tick is not enough: a brand-new block (or one that just grew from a
        streamed translation) has no correct height/position until Qt has actually laid it out,
        which for a multi-line label can take more than one event-loop turn (word wrap needs a
        real width first). Setting the scrollbar once, immediately, can therefore land on the
        wrong value with no second chance to correct it -- confirmed by a test that emits a
        block and then polls for up to 400ms without the scrollbar ever reaching the right
        value. So: apply the target immediately for the common case (content already laid out),
        and re-evaluate and reapply it on every range change for a short window afterwards, in
        case this one has not settled yet. ``compute_target`` is called fresh each time rather
        than evaluated once up front for exactly this reason.

        The chase is tagged with a generation so a later call -- including one made because the
        user scrolled away in the meantime -- can invalidate it instead of two chases fighting
        over the scrollbar.
        """
        bar = self.scroll.verticalScrollBar()

        def _apply() -> None:
            # Safety net: whichever content-changing path led here, make sure the canvas's
            # height is honest before computing a target against it (see _sync_canvas_height).
            # Idempotent and cheap when it is already correct, so unconditional is fine.
            self._sync_canvas_height()
            value = compute_target()
            self._applying_scroll = True
            try:
                bar.setValue(value)
            finally:
                self._applying_scroll = False

        self._auto_follow = True
        _apply()

        self._cancel_scroll_chase()
        gen = self._scroll_chase_gen

        def _chase(_min=None, _max=None, gen=gen) -> None:
            if gen == self._scroll_chase_gen:
                _apply()

        self._scroll_chase_conn = _chase
        bar.rangeChanged.connect(_chase)
        QTimer.singleShot(150, self._cancel_scroll_chase_if_current(gen))

    def _cancel_scroll_chase_if_current(self, gen: int) -> Callable[[], None]:
        def _end() -> None:
            if gen == self._scroll_chase_gen:
                self._cancel_scroll_chase()
        return _end

    def _scroll_to_bottom(self) -> None:
        self._scroll_to(lambda: self.scroll.verticalScrollBar().maximum())

    def _centered_scroll_value(self, uid: int) -> int:
        """Scrollbar value that puts block ``uid`` in the middle of the visible area."""
        bar = self.scroll.verticalScrollBar()
        block = self._blocks.get(uid)
        if block is None:
            return bar.maximum()
        # block.y() is already in self.canvas's coordinate space, which is exactly what the
        # scrollbar's value represents (QScrollArea offsets the widget it owns by that many
        # pixels), so no coordinate mapping is needed here.
        viewport_height = self.scroll.viewport().height()
        center_y = block.y() + block.height() / 2.0
        value = int(round(center_y - viewport_height / 2.0))
        return max(0, min(value, bar.maximum()))

    def _follow_target(self) -> Callable[[], int]:
        """What "the current position" means right now.

        Centred on the actively-translating sentence when there is one -- so the line the user
        cares about most sits in a comfortable reading position rather than jammed against the
        bottom edge next to the status bar and the jump pill -- otherwise the bottom, so a
        transcribe-only session (nothing ever becomes "active") still tracks new lines.

        "The bottom" here means the last real caption's bottom edge lined up with the viewport,
        via _content_bottom() -- not bar.maximum(), which also spans the trailing spacer
        reserved for centring (see _build_captions). Landing there would put the view exactly
        in the blank space this whole mechanism exists to stop happening.
        """
        uid = self._active_uid
        if uid is not None:
            return lambda: self._centered_scroll_value(uid)
        return self._bottom_of_content_target

    def _bottom_of_content_target(self) -> int:
        bar = self.scroll.verticalScrollBar()
        value = self._content_bottom() - self.scroll.viewport().height()
        return max(0, min(value, bar.maximum()))

    def _follow(self) -> None:
        """Reposition to the current focus (see _follow_target), but only while following.

        Call this whenever what "the current focus" is could have changed -- a new sentence
        becoming active, a translation streaming in and changing that block's height. A no-op
        while the user has scrolled away, which is the entire point of auto-follow being
        pausable: this is safe to call unconditionally from those events without re-litigating
        whether the user wanted to be moved.
        """
        if self._auto_follow:
            self._scroll_to(self._follow_target())

    def _add_block(self, utterance: Utterance) -> CaptionBlock:
        """Add the widget for one utterance. Purely mechanical: callers decide whether and
        where to scroll afterwards (see _follow), since that depends on things -- is
        translation enabled, is this utterance becoming the active one -- that this method
        has no business knowing about."""
        self.placeholder.setVisible(False)
        block = CaptionBlock(utterance, self.config.ui, self.canvas)
        # Insert before the trailing spacer (always the last item), not append -- appending
        # would push the spacer into the middle of the stack instead of leaving it trailing.
        self.canvas_layout.insertWidget(self.canvas_layout.count() - 1, block)
        self._blocks[utterance.id] = block
        self._order.append(utterance.id)
        self._trim_history()  # always resyncs the canvas height, even when nothing was trimmed
        # The scroll range has not been recalculated for this block yet (it has no height
        # until the next layout pass), but if the user is already away from wherever we'd
        # follow to, there is no need to wait for that to show the pill.
        self._update_jump_pill()
        return block

    def _sync_canvas_height(self) -> None:
        """Keep the caption canvas's actual height matched to what it needs at its actual width.

        QScrollArea's ``setWidgetResizable(True)`` sizes the content widget from its
        ``sizeHint()``, but ``sizeHint()`` is computed without any width in mind -- it cannot
        be, since the layout does not yet know how wide it will end up. For content with
        wrapped, multi-line labels this can measurably overshoot the height the widget actually
        needs at its real (viewport-constrained) width, since heightForWidth-aware negotiation
        only happens once a width is actually assigned. The gap does not correct itself and
        grows with every block added, eventually leaving a stretch of true blank canvas below
        the last real caption that scrolling to "the bottom" lands in.

        ``heightForWidth()``, given the widget's actual current width, computes the correct
        answer (confirmed empirically: it matches the measured position of the last block's
        bottom edge, while ``sizeHint()`` at the same moment does not). Forcing the canvas to
        that height keeps it honest. This does not fight ``setWidgetResizable``'s WIDTH
        tracking -- only height is set here, and QScrollArea keeps sizing the width to the
        viewport as normal.
        """
        self._sync_tail_spacer()
        canvas = self.canvas
        width = canvas.width()
        if width <= 0:
            return
        correct_height = canvas.heightForWidth(width)
        if correct_height > 0 and correct_height != canvas.height():
            canvas.setFixedHeight(correct_height)

    def _sync_tail_spacer(self) -> None:
        """Keep the trailing spacer at roughly half the viewport, so the most recent sentence
        always has room to be scrolled up to the middle. See its creation in _build_captions."""
        viewport_height = self.scroll.viewport().height()
        target = max(0, viewport_height // 2)
        if self.canvas_tail_spacer.minimumHeight() != target:
            self.canvas_tail_spacer.setFixedHeight(target)

    def _content_bottom(self) -> int:
        """Bottom edge of the last real caption block, in canvas coordinates -- i.e. excluding
        the trailing spacer. This is "the bottom" for jump-to-latest / transcribe-only-mode
        follow purposes; bar.maximum() is not, since it also covers the spacer's reserved
        centring room and landing there is exactly the blank-space bug this method exists to
        avoid reintroducing.
        """
        if not self._order:
            return 0
        block = self._blocks.get(self._order[-1])
        if block is None:
            return 0
        return block.y() + block.height()

    #: Delays (ms) at which a deferred canvas-height sync re-checks itself. A single deferred
    #: tick is enough for a plain text/content change, but a structural change -- a layout mode
    #: switch restructuring every block's internal QHBoxLayout/QVBoxLayout, in particular --
    #: can take Qt more than one event-loop turn to fully settle; confirmed empirically, where
    #: one deferred tick landed on an intermediate height thirty-odd pixels short of the real
    #: one and nothing ever corrected it afterwards. Chasing across a few frame-paced ticks
    #: catches that without guessing a single "surely long enough" delay.
    _CANVAS_HEIGHT_CHASE_DELAYS_MS = (0, 16, 50, 120)

    def _defer_canvas_height_sync(self) -> None:
        """Schedule _sync_canvas_height across the next few event-loop turns, not run it now.

        Calling it synchronously in the same call stack as canvas_layout.addWidget() (or a
        label's setText()) reads a *stale* heightForWidth(): Qt's layout invalidation from that
        change has not actually been processed yet, so the query answers "what did you need
        before this change", not after. A *single* deferred tick fixes the common case (one new
        block, one text update) but is not always enough for a structural change like a layout
        mode switch, which can take Qt more than one turn to fully re-settle -- so this re-syncs
        at each delay in _CANVAS_HEIGHT_CHASE_DELAYS_MS rather than trusting the first one.

        Coalesced via a generation counter: a new call supersedes any still-pending re-checks
        from an earlier one instead of running two overlapping chases, the same pattern
        _scroll_to uses for its own catch-up and for the same reason -- a later call reflects
        more current intent than an earlier one still in flight.
        """
        self._canvas_height_sync_gen += 1
        gen = self._canvas_height_sync_gen

        def _run(gen=gen) -> None:
            if gen != self._canvas_height_sync_gen:
                return  # superseded by a later call; that one's ticks will finish the job
            self._sync_canvas_height()

        for delay in self._CANVAS_HEIGHT_CHASE_DELAYS_MS:
            QTimer.singleShot(delay, _run)

        QTimer.singleShot(0, _run)

    def _trim_history(self) -> None:
        limit = max(1, int(self.config.ui.max_history))
        while len(self._order) > limit:
            uid = self._order.pop(0)
            block = self._blocks.pop(uid, None)
            if block is not None:
                block.setParent(None)
                block.deleteLater()
            if self._active_uid == uid:
                self._active_uid = None
        self._defer_canvas_height_sync()

    def _set_active(self, uid: Optional[int]) -> None:
        """Move the "currently translating" highlight to ``uid`` (or clear it for None)."""
        if self._active_uid is not None and self._active_uid != uid:
            prev = self._blocks.get(self._active_uid)
            if prev is not None:
                CaptionBlock._set_prop(prev, "active", "false")
        self._active_uid = uid
        if uid is not None:
            block = self._blocks.get(uid)
            if block is not None:
                CaptionBlock._set_prop(block, "active", "true")
        # The pill's wording depends on whether something is actively translating; refresh it
        # in place so it does not show stale text while it happens to already be visible.
        self._update_jump_pill()
        if uid is not None:
            # A newly active sentence is a new "current focus" to centre on. Deliberately not
            # called when clearing to None (a translation finishing): the view should hold
            # still until something new actually happens, not jump to the bottom the instant
            # a block resolves with nothing else having arrived yet.
            self._follow()

    def _mark_resolved(self, uid: int) -> None:
        """Drop the highlight once a block's translation has settled (done/failed/skipped).

        Only acts if ``uid`` is still the active one -- a newer sentence may already have
        taken the highlight over, in which case this arrival is old news and should not
        touch anything.
        """
        if self._active_uid == uid:
            self._set_active(None)

    def _on_utterance(self, utterance: object) -> None:
        if not isinstance(utterance, Utterance):
            log.debug("ignoring unexpected utterance payload: %r", type(utterance))
            return
        self._add_block(utterance)
        if self.config.translate.enabled:
            # _set_active follows to centre on it. Only when translation is actually running:
            # with it disabled, no translation_done/_failed signal will ever arrive for this
            # utterance to clear the highlight, so it would stay lit forever.
            self._set_active(utterance.id)
        else:
            # Nothing will ever become "active" in this mode, so _follow's fallback target
            # (the bottom) is what keeps a transcribe-only session tracking new lines.
            self._follow()

    def _on_delta(self, uid: int, text: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            block.set_translation(text, pending=True)
            self._defer_canvas_height_sync()
            # A streamed translation can grow the block by several lines, shifting where its
            # centre is; re-centre on it as it grows rather than only once when it arrives.
            # _scroll_to's own chase picks up the deferred sync above once it lands.
            self._follow()

    def _on_done(self, uid: int, text: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            block.utterance.state = "done"
            block.set_translation(text, pending=False)
            self._defer_canvas_height_sync()
            # The final text can differ slightly from the last streamed delta (cleanup applied
            # to the completed translation), so absorb one last possible height change before
            # the highlight -- and with it, further re-centring -- goes away.
            self._follow()
        self._mark_resolved(uid)

    def _on_failed(self, uid: int, message: str) -> None:
        block = self._blocks.get(uid)
        if block is not None:
            block.set_failed(message)
        self._mark_resolved(uid)

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
        self._position_jump_pill()
        # heightForWidth is, as the name says, a function of width: a window resize changes
        # the canvas's viewport width, which changes the correct height too. Deferred because
        # the scroll area's own width-to-viewport sync may not have propagated down to the
        # canvas yet at the moment this outer window's resizeEvent fires.
        self._defer_canvas_height_sync()

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
