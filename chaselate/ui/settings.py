"""The settings dialog.

Edits a *copy* of the config, so Cancel is genuinely free and the running pipeline never
sees a half-edited state; the caller gets the edited copy through
:attr:`SettingsDialog.accepted_config` or :meth:`SettingsDialog.result_config`.

Two operations here are slow and must never touch the GUI thread: enumerating audio
devices (WASAPI/COM enumeration, hundreds of milliseconds) and talking to Ollama (seconds,
or a full TCP timeout when it is not running). Both go through :class:`_Job` onto the
global :class:`QThreadPool`, and results come back as signals, which Qt delivers on the
GUI thread.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PyQt5.QtCore import (
    QCoreApplication,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..audio.devices import KIND_LOOPBACK, KIND_MIC, list_devices
from ..config import (
    ASR_DEVICES,
    AUDIO_BACKENDS,
    COMPUTE_TYPES,
    LAYOUTS,
    WHISPER_MODELS,
    AppConfig,
)
from ..languages import AUTO, LANGUAGES, display_name
from ..pipeline import Pipeline
from .style import build_dialog_stylesheet

log = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

JOB_DEVICES = "devices"
JOB_MODELS = "models"
JOB_HEALTH = "health"


def _flat_row() -> QHBoxLayout:
    """A row layout for composite form fields.

    QFormLayout already provides the margins; a nested layout keeping its own default
    11 px inset makes those rows sit visibly lower than their labels.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    return row


class _JobSignals(QObject):
    done = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


class _Job(QRunnable):
    """Run one blocking call on the thread pool and report back through signals.

    The signals object is parented to the application, not to the dialog and not to the
    job: the job is destroyed by Qt as soon as ``run`` returns, and a dialog can be closed
    mid-flight, either of which would leave the worker thread emitting from freed memory.
    Qt drops the connection when the dialog dies, and ``deleteLater`` after the emit
    reclaims the emitter on the GUI thread once the queued signal has been delivered.
    """

    def __init__(self, kind: str, work: Callable[[], object]):
        super().__init__()
        self.kind = kind
        self._work = work
        self.signals = _JobSignals(QCoreApplication.instance())
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - exercised through the UI
        signals = self.signals
        try:
            result = self._work()
        except Exception as exc:  # noqa: BLE001 - any failure is a message for the user
            log.debug("%s job failed", self.kind, exc_info=True)
            signals.failed.emit(self.kind, str(exc) or exc.__class__.__name__)
        else:
            signals.done.emit(self.kind, result)
        signals.deleteLater()


def wait_for_background_jobs(timeout_ms: int = 5000) -> bool:
    """Let in-flight device/Ollama probes finish. Call before the process exits.

    Killing the interpreter while a worker is inside the WASAPI COM enumeration crashes
    the process, so exit waits for the pool instead.
    """
    return QThreadPool.globalInstance().waitForDone(timeout_ms)


class SettingsDialog(QDialog):
    """Tabbed editor for every :class:`~chaselate.config.AppConfig` field bar geometry."""

    #: Appearance changed while the dialog is open; payload is the working AppConfig copy.
    preview_requested = pyqtSignal(object)
    #: OK or Apply pressed; payload is a fresh AppConfig for the caller to adopt and save.
    accepted_config = pyqtSignal(object)

    def __init__(self, config: AppConfig, pipeline: Pipeline, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._working = config.copy()
        self.pipeline = pipeline
        # Suppresses preview signals while widgets are being populated from the config.
        self._loading = True

        self.setWindowTitle("Chaselate Settings")
        self.setModal(False)
        # Wide enough that the five tab labels fit without scroll arrows.
        self.setMinimumWidth(720)
        self.setStyleSheet(build_dialog_stylesheet(self._working.ui))

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_audio_tab(), "Audio")
        self.tabs.addTab(self._build_recognition_tab(), "Recognition")
        self.tabs.addTab(self._build_translation_tab(), "Translation")
        self.tabs.addTab(self._build_appearance_tab(), "Appearance")
        self.tabs.addTab(self._build_advanced_tab(), "Advanced")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply,
            Qt.Horizontal,
            self,
        )
        self.buttons.accepted.connect(self._on_ok)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)

        column = QVBoxLayout(self)
        column.addWidget(self.tabs)
        column.addWidget(self.buttons)

        self._load()
        self._loading = False
        self._connect_preview()
        # Enumeration is slow, so the dialog paints first and fills the device list after.
        QTimer.singleShot(0, self.refresh_devices)

    # -- tab construction --------------------------------------------------

    @staticmethod
    def _form(parent: QWidget) -> QFormLayout:
        form = QFormLayout(parent)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setSpacing(8)
        return form

    @staticmethod
    def _spin(low: float, high: float, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _int_spin(low: int, high: int, step: int = 1, suffix: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _build_audio_tab(self) -> QWidget:
        page = QWidget()
        form = self._form(page)

        self.audio_backend = QComboBox()
        self.audio_backend.addItems(AUDIO_BACKENDS)
        self.audio_backend.setToolTip(
            "Capture library. 'auto' prefers soundcard and falls back to pyaudiowpatch, "
            "which sometimes copes with drivers soundcard cannot enumerate."
        )
        form.addRow("Backend", self.audio_backend)

        self.source_loopback = QRadioButton("System audio (loopback)")
        self.source_loopback.setToolTip(
            "Capture what the speakers are playing: videos, calls, streams."
        )
        self.source_mic = QRadioButton("Microphone")
        self.source_mic.setToolTip("Capture an input device instead of the speaker mix.")
        source_row = _flat_row()
        source_row.addWidget(self.source_loopback)
        source_row.addWidget(self.source_mic)
        source_row.addStretch(1)
        source_box = QWidget()
        source_box.setLayout(source_row)
        form.addRow("Source", source_box)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "Capture endpoint. Leave on the system default unless you need a specific "
            "device; a saved name that has disappeared falls back to the default."
        )
        self.device_refresh = QPushButton("Refresh")
        self.device_refresh.clicked.connect(self.refresh_devices)
        device_row = _flat_row()
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.device_refresh)
        device_box = QWidget()
        device_box.setLayout(device_row)
        form.addRow("Device", device_box)

        self.device_status = QLabel("")
        self.device_status.setObjectName("HintLabel")
        self.device_status.setWordWrap(True)
        form.addRow("", self.device_status)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(10, 400)  # 0.10x .. 4.00x, stored as a float
        self.gain_slider.setToolTip(
            "Multiplier applied to the captured signal before voice detection. Raise it "
            "for quiet sources; too high clips and hurts recognition."
        )
        self.gain_value = QLabel("1.00x")
        self.gain_slider.valueChanged.connect(
            lambda v: self.gain_value.setText(f"{v / 100.0:.2f}x")
        )
        gain_row = _flat_row()
        gain_row.addWidget(self.gain_slider, 1)
        gain_row.addWidget(self.gain_value)
        gain_box = QWidget()
        gain_box.setLayout(gain_row)
        form.addRow("Gain", gain_box)

        self.queue_seconds = self._spin(2.0, 300.0, 1.0, 1, " s")
        self.queue_seconds.setToolTip(
            "How much unprocessed audio may pile up before the oldest is discarded. "
            "Larger means fewer drops but staler captions when recognition is behind."
        )
        form.addRow("Audio buffer", self.queue_seconds)

        self.block_frames = self._int_spin(128, 4096, 128, " frames")
        self.block_frames.setToolTip(
            "Frames of 16 kHz mono audio per block (512 = 32 ms). Silero VAD expects 512; "
            "change only if capture is glitching."
        )
        form.addRow("Block size", self.block_frames)

        self.capture_rate = self._int_spin(0, 192000, 1000, " Hz")
        self.capture_rate.setToolTip(
            "Force a capture sample rate. 0 asks the device for its native rate, which is "
            "almost always what you want."
        )
        form.addRow("Capture rate", self.capture_rate)
        return page

    def _build_recognition_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        model_box = QGroupBox("Whisper")
        form = self._form(model_box)

        self.asr_model = QComboBox()
        self.asr_model.setEditable(True)
        self.asr_model.addItems(WHISPER_MODELS)
        self.asr_model.setToolTip(
            "Bigger is more accurate and slower. 'small' is the usual sweet spot on GPU; "
            "use 'tiny' or 'base' on CPU. A CTranslate2 model path also works."
        )
        form.addRow("Model", self.asr_model)

        self.asr_device = QComboBox()
        self.asr_device.addItems(ASR_DEVICES)
        self.asr_device.setToolTip(
            "Where to run recognition. 'auto' uses CUDA when it is usable and falls back "
            "to CPU. Changing this restarts the pipeline."
        )
        form.addRow("Device", self.asr_device)

        self.compute_type = QComboBox()
        self.compute_type.addItems(COMPUTE_TYPES)
        self.compute_type.setToolTip(
            "Numeric precision. float16 on GPU, int8 on CPU; 'auto' picks per device. "
            "Lower precision is faster and slightly less accurate."
        )
        form.addRow("Precision", self.compute_type)

        self.source_lang = QComboBox()
        self.source_lang.addItem(display_name(AUTO), AUTO)
        for lang in LANGUAGES:
            self.source_lang.addItem(display_name(lang.code), lang.code)
        self.source_lang.setToolTip(
            "Spoken language. Pinning it is noticeably more stable than auto-detect, "
            "which can switch language mid-conversation."
        )
        form.addRow("Language", self.source_lang)

        self.beam_size = self._int_spin(1, 10)
        self.beam_size.setToolTip(
            "Beam search width. 1 (greedy) is fastest and usually enough for live use; "
            "higher costs proportionally more time per utterance."
        )
        form.addRow("Beam size", self.beam_size)

        self.vad_filter = QCheckBox("Run Whisper's own VAD filter")
        self.vad_filter.setToolTip(
            "A second silence-trimming pass inside Whisper. Cheap, and removes some "
            "invented text at segment edges."
        )
        form.addRow("", self.vad_filter)
        outer.addWidget(model_box)

        vad_box = QGroupBox("Voice activity detection (Silero)")
        vform = self._form(vad_box)

        self.vad_threshold = self._spin(0.05, 0.95, 0.05, 2)
        self.vad_threshold.setToolTip(
            "Speech probability above which a frame counts as speech. Raise it in a noisy "
            "room; lower it if quiet speech is being missed."
        )
        vform.addRow("Threshold", self.vad_threshold)

        self.min_speech_ms = self._int_spin(0, 3000, 50, " ms")
        self.min_speech_ms.setToolTip("Ignore speech bursts shorter than this, e.g. clicks and coughs.")
        vform.addRow("Min speech", self.min_speech_ms)

        self.min_silence_ms = self._int_spin(100, 5000, 50, " ms")
        self.min_silence_ms.setToolTip(
            "Trailing silence that ends an utterance. Lower reacts faster but chops "
            "sentences mid-thought; higher gives better sentences with more delay."
        )
        vform.addRow("Min silence", self.min_silence_ms)

        self.speech_pad_ms = self._int_spin(0, 1000, 25, " ms")
        self.speech_pad_ms.setToolTip(
            "Audio kept either side of detected speech so leading and trailing consonants "
            "are not clipped."
        )
        vform.addRow("Speech padding", self.speech_pad_ms)

        self.max_segment_s = self._spin(3.0, 60.0, 1.0, 1, " s")
        self.max_segment_s.setToolTip(
            "Hard cut for a speaker who never pauses, so captions keep appearing. The next "
            "segment overlaps slightly and the repeat is removed automatically."
        )
        vform.addRow("Max segment", self.max_segment_s)

        self.silence_rms = self._spin(0.0, 0.05, 0.0005, 4)
        self.silence_rms.setToolTip(
            "Below this RMS a block is treated as silence without invoking Silero at all. "
            "A cheap gate; raise it if a noisy line keeps waking the detector."
        )
        vform.addRow("Silence RMS gate", self.silence_rms)
        outer.addWidget(vad_box)
        outer.addStretch(1)
        return page

    def _build_translation_tab(self) -> QWidget:
        page = QWidget()
        form = self._form(page)

        self.translate_enabled = QCheckBox("Translate captions")
        self.translate_enabled.setToolTip(
            "Off leaves recognition running and shows the transcript only."
        )
        form.addRow("", self.translate_enabled)

        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("http://127.0.0.1:11434")
        self.base_url.setToolTip("Address of the Ollama server serving the translation model.")
        form.addRow("Ollama URL", self.base_url)

        self.model_combo = QComboBox()
        # Editable so a model that is not pulled yet, or a server that cannot be reached
        # right now, does not stop the user configuring it.
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setToolTip(
            "Ollama model used for translation. Translation-specialist models "
            "(translategemma, madlad...) are listed first."
        )
        self.model_refresh = QPushButton("Refresh")
        self.model_refresh.clicked.connect(self.refresh_models)
        model_row = _flat_row()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.model_refresh)
        model_box = QWidget()
        model_box.setLayout(model_row)
        form.addRow("Model", model_box)

        self.test_button = QPushButton("Test connection")
        self.test_button.clicked.connect(self.test_connection)
        self.ollama_status = QLabel("")
        self.ollama_status.setObjectName("HintLabel")
        self.ollama_status.setWordWrap(True)
        test_row = _flat_row()
        test_row.addWidget(self.test_button)
        test_row.addWidget(self.ollama_status, 1)
        test_box = QWidget()
        test_box.setLayout(test_row)
        form.addRow("", test_box)

        self.target_lang = QComboBox()
        for lang in LANGUAGES:
            self.target_lang.addItem(display_name(lang.code), lang.code)
        self.target_lang.setToolTip("Language the captions are translated into.")
        form.addRow("Target language", self.target_lang)

        self.context_sentences = self._int_spin(0, 10)
        self.context_sentences.setToolTip(
            "Previously translated sentences replayed as context so pronouns and "
            "terminology stay consistent. 0 disables it and is fastest."
        )
        form.addRow("Context sentences", self.context_sentences)

        self.temperature = self._spin(0.0, 2.0, 0.05, 2)
        self.temperature.setToolTip(
            "Sampling randomness. Keep it low for translation: high values invent wording."
        )
        form.addRow("Temperature", self.temperature)

        self.stream_check = QCheckBox("Stream translations as they are generated")
        self.stream_check.setToolTip(
            "Show tokens as they arrive instead of waiting for the whole sentence. Feels "
            "much faster; the final text is identical."
        )
        form.addRow("", self.stream_check)

        self.keep_alive = QLineEdit()
        self.keep_alive.setPlaceholderText("10m")
        self.keep_alive.setToolTip(
            "How long Ollama keeps the model in VRAM between utterances ('10m', '1h', "
            "'0' to unload immediately). Too short and every caption pays reload time."
        )
        form.addRow("Keep alive", self.keep_alive)

        self.extra_instructions = QPlainTextEdit()
        self.extra_instructions.setFixedHeight(72)
        self.extra_instructions.setToolTip(
            "Appended to the system prompt. Use it for tone or a glossary, e.g. "
            "'keep product names in English'. Long text slows every translation."
        )
        form.addRow("Extra instructions", self.extra_instructions)
        return page

    def _build_appearance_tab(self) -> QWidget:
        page = QWidget()
        form = self._form(page)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setToolTip("Window opacity. Lower lets more of the video through.")
        self.opacity_value = QLabel("92%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_value.setText(f"{v}%")
        )
        op_row = _flat_row()
        op_row.addWidget(self.opacity_slider, 1)
        op_row.addWidget(self.opacity_value)
        op_box = QWidget()
        op_box.setLayout(op_row)
        form.addRow("Opacity", op_box)

        self.font_size = self._int_spin(10, 72, 1, " pt")
        self.font_size.setToolTip("Translation text size.")
        form.addRow("Translation size", self.font_size)

        self.original_font_size = self._int_spin(10, 72, 1, " pt")
        self.original_font_size.setToolTip(
            "Original transcript size. The window chrome scales with this value."
        )
        form.addRow("Original size", self.original_font_size)

        self.layout_combo = QComboBox()
        for name in LAYOUTS:
            self.layout_combo.addItem(
                "Side by side" if name == "side" else "Stacked", name
            )
        self.layout_combo.setToolTip(
            "Side by side puts the original left and the translation right; stacked puts "
            "the original above. Side needs a wider window."
        )
        form.addRow("Layout", self.layout_combo)

        self.show_original = QCheckBox("Show the original transcript (Ctrl+H)")
        self.show_original.setToolTip("Off shows translations only, for a smaller strip.")
        form.addRow("", self.show_original)

        self.always_on_top = QCheckBox("Keep the window above other windows")
        self.always_on_top.setToolTip(
            "Off lets a full-screen player cover the captions; some full-screen games "
            "misbehave with it on."
        )
        form.addRow("", self.always_on_top)

        self.mouse_transparent = QCheckBox("Click-through overlay")
        self.mouse_transparent.setToolTip(
            "The overlay stops receiving clicks entirely, including its own toolbar. Use "
            "the tray icon menu to switch it back off."
        )
        form.addRow("", self.mouse_transparent)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.setToolTip("Caption plate colours.")
        form.addRow("Theme", self.theme_combo)

        self.max_history = self._int_spin(10, 5000, 10, " lines")
        self.max_history.setToolTip(
            "Captions kept before the oldest are removed. Large values grow memory use."
        )
        form.addRow("History", self.max_history)

        self.show_status_bar = QCheckBox("Show the metrics bar at the bottom")
        self.show_status_bar.setToolTip(
            "Device, model, realtime factor and backlog. Useful while tuning, noise "
            "afterwards."
        )
        form.addRow("", self.show_status_bar)
        return page

    def _build_advanced_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        app_box = QGroupBox("Application")
        form = self._form(app_box)
        self.autostart = QCheckBox("Start capturing as soon as the window opens")
        self.autostart.setToolTip("Skips pressing Start; the models still need to load first.")
        form.addRow("", self.autostart)

        self.log_level = QComboBox()
        self.log_level.addItems(LOG_LEVELS)
        self.log_level.setToolTip(
            "Verbosity of the log file. DEBUG is for diagnosing capture or model problems "
            "and is noisy."
        )
        form.addRow("Log level", self.log_level)
        outer.addWidget(app_box)

        quality_box = QGroupBox("Recognition filtering")
        qform = self._form(quality_box)

        self.min_logprob = self._spin(-5.0, 0.0, 0.1, 2)
        self.min_logprob.setToolTip(
            "Discard segments whose average token log-probability is below this. Closer to "
            "0 is stricter and drops more real speech; -1.0 is a good default."
        )
        qform.addRow("Min log-probability", self.min_logprob)

        self.max_no_speech_prob = self._spin(0.0, 1.0, 0.05, 2)
        self.max_no_speech_prob.setToolTip(
            "Discard segments Whisper itself rates as more likely than this to be silence."
        )
        qform.addRow("Max no-speech probability", self.max_no_speech_prob)

        self.filter_hallucinations = QCheckBox("Drop known hallucinated phrases")
        self.filter_hallucinations.setToolTip(
            "Rejects the boilerplate Whisper invents over music or silence "
            "('Thanks for watching', subtitle credits, and so on)."
        )
        qform.addRow("", self.filter_hallucinations)

        self.condition_on_previous = QCheckBox("Condition on previous text")
        self.condition_on_previous.setToolTip(
            "Feeds the previous transcript back into Whisper. Better continuity, but one "
            "bad segment can send it into a repeating loop, so it is off by default."
        )
        qform.addRow("", self.condition_on_previous)

        self.initial_prompt = QLineEdit()
        self.initial_prompt.setToolTip(
            "Vocabulary hint for Whisper: names, jargon, acronyms it keeps mishearing."
        )
        qform.addRow("Initial prompt", self.initial_prompt)

        self.cpu_threads = self._int_spin(0, 64)
        self.cpu_threads.setToolTip(
            "CPU threads for recognition. 0 lets CTranslate2 decide, which is right unless "
            "you are sharing the machine with something else."
        )
        qform.addRow("CPU threads", self.cpu_threads)
        outer.addWidget(quality_box)

        net_box = QGroupBox("Translation limits")
        nform = self._form(net_box)
        self.num_predict = self._int_spin(16, 8192, 16, " tokens")
        self.num_predict.setToolTip(
            "Cap on tokens per translation. A sentence needs far fewer; the cap only stops "
            "a runaway model from stalling the queue."
        )
        nform.addRow("Max new tokens", self.num_predict)

        self.request_timeout = self._spin(5.0, 600.0, 5.0, 0, " s")
        self.request_timeout.setToolTip(
            "Give up on an Ollama request after this long. Must exceed the first-call model "
            "load time, which can be a minute or more for a large model."
        )
        nform.addRow("Request timeout", self.request_timeout)
        outer.addWidget(net_box)
        outer.addStretch(1)
        return page

    # -- config <-> widgets ------------------------------------------------

    def _load(self) -> None:
        cfg = self._working
        a = cfg.audio
        self._set_combo_text(self.audio_backend, a.backend)
        self.source_mic.setChecked(a.source == KIND_MIC)
        self.source_loopback.setChecked(a.source != KIND_MIC)
        self.gain_slider.setValue(int(round(max(0.1, min(4.0, a.gain)) * 100)))
        self.gain_value.setText(f"{self.gain_slider.value() / 100.0:.2f}x")
        self.queue_seconds.setValue(a.queue_seconds)
        self.block_frames.setValue(a.block_frames)
        self.capture_rate.setValue(a.capture_rate)
        # Placeholder until the async enumeration lands, so the current choice is visible.
        self.device_combo.clear()
        self.device_combo.addItem("System default", "")
        if a.device_name:
            self.device_combo.addItem(a.device_name, a.device_name)
            self.device_combo.setCurrentIndex(1)

        s = cfg.asr
        self._set_combo_text(self.asr_model, s.model)
        self._set_combo_text(self.asr_device, s.device)
        self._set_combo_text(self.compute_type, s.compute_type)
        self._set_combo_data(self.source_lang, s.source_lang, fallback=AUTO)
        self.beam_size.setValue(s.beam_size)
        self.vad_filter.setChecked(s.vad_filter)
        self.min_logprob.setValue(s.min_logprob)
        self.max_no_speech_prob.setValue(s.max_no_speech_prob)
        self.filter_hallucinations.setChecked(s.filter_hallucinations)
        self.condition_on_previous.setChecked(s.condition_on_previous_text)
        self.initial_prompt.setText(s.initial_prompt)
        self.cpu_threads.setValue(s.cpu_threads)

        v = cfg.vad
        self.vad_threshold.setValue(v.threshold)
        self.min_speech_ms.setValue(v.min_speech_ms)
        self.min_silence_ms.setValue(v.min_silence_ms)
        self.speech_pad_ms.setValue(v.speech_pad_ms)
        self.max_segment_s.setValue(v.max_segment_s)
        self.silence_rms.setValue(v.silence_rms)

        t = cfg.translate
        self.translate_enabled.setChecked(t.enabled)
        self.base_url.setText(t.base_url)
        self.model_combo.clear()
        if t.model:
            self.model_combo.addItem(t.model)
        self.model_combo.setEditText(t.model)
        self._set_combo_data(self.target_lang, t.target_lang, fallback="zh-TW")
        self.context_sentences.setValue(t.context_sentences)
        self.temperature.setValue(t.temperature)
        self.stream_check.setChecked(t.stream)
        self.keep_alive.setText(t.keep_alive)
        self.extra_instructions.setPlainText(t.extra_instructions)
        self.num_predict.setValue(t.num_predict)
        self.request_timeout.setValue(t.request_timeout)

        u = cfg.ui
        self.opacity_slider.setValue(int(round(max(0.2, min(1.0, u.opacity)) * 100)))
        self.opacity_value.setText(f"{self.opacity_slider.value()}%")
        self.font_size.setValue(u.font_size)
        self.original_font_size.setValue(u.original_font_size)
        self._set_combo_data(self.layout_combo, u.layout, fallback="side")
        self.show_original.setChecked(u.show_original)
        self.always_on_top.setChecked(u.always_on_top)
        self.mouse_transparent.setChecked(u.mouse_transparent)
        self._set_combo_data(self.theme_combo, u.theme, fallback="dark")
        self.max_history.setValue(u.max_history)
        self.show_status_bar.setChecked(u.show_status_bar)

        self.autostart.setChecked(cfg.autostart)
        self._set_combo_text(self.log_level, cfg.log_level.upper())

    def _collect(self) -> AppConfig:
        cfg = self._working
        a = cfg.audio
        a.backend = self.audio_backend.currentText()
        a.source = KIND_MIC if self.source_mic.isChecked() else KIND_LOOPBACK
        a.device_name = self.device_combo.currentData() or ""
        a.gain = self.gain_slider.value() / 100.0
        a.queue_seconds = float(self.queue_seconds.value())
        a.block_frames = int(self.block_frames.value())
        a.capture_rate = int(self.capture_rate.value())

        s = cfg.asr
        s.model = self.asr_model.currentText().strip() or "small"
        s.device = self.asr_device.currentText()
        s.compute_type = self.compute_type.currentText()
        s.source_lang = self.source_lang.currentData() or AUTO
        s.beam_size = int(self.beam_size.value())
        s.vad_filter = self.vad_filter.isChecked()
        s.min_logprob = float(self.min_logprob.value())
        s.max_no_speech_prob = float(self.max_no_speech_prob.value())
        s.filter_hallucinations = self.filter_hallucinations.isChecked()
        s.condition_on_previous_text = self.condition_on_previous.isChecked()
        s.initial_prompt = self.initial_prompt.text()
        s.cpu_threads = int(self.cpu_threads.value())

        v = cfg.vad
        v.threshold = float(self.vad_threshold.value())
        v.min_speech_ms = int(self.min_speech_ms.value())
        v.min_silence_ms = int(self.min_silence_ms.value())
        v.speech_pad_ms = int(self.speech_pad_ms.value())
        v.max_segment_s = float(self.max_segment_s.value())
        v.silence_rms = float(self.silence_rms.value())

        t = cfg.translate
        t.enabled = self.translate_enabled.isChecked()
        t.base_url = self.base_url.text().strip() or "http://127.0.0.1:11434"
        t.model = self.model_combo.currentText().strip()
        t.target_lang = self.target_lang.currentData() or "zh-TW"
        t.context_sentences = int(self.context_sentences.value())
        t.temperature = float(self.temperature.value())
        t.stream = self.stream_check.isChecked()
        t.keep_alive = self.keep_alive.text().strip() or "10m"
        t.extra_instructions = self.extra_instructions.toPlainText().strip()
        t.num_predict = int(self.num_predict.value())
        t.request_timeout = float(self.request_timeout.value())

        u = cfg.ui
        u.opacity = self.opacity_slider.value() / 100.0
        u.font_size = int(self.font_size.value())
        u.original_font_size = int(self.original_font_size.value())
        u.layout = self.layout_combo.currentData() or "side"
        u.show_original = self.show_original.isChecked()
        u.always_on_top = self.always_on_top.isChecked()
        u.mouse_transparent = self.mouse_transparent.isChecked()
        u.theme = self.theme_combo.currentData() or "dark"
        u.max_history = int(self.max_history.value())
        u.show_status_bar = self.show_status_bar.isChecked()

        cfg.autostart = self.autostart.isChecked()
        cfg.log_level = self.log_level.currentText()
        return cfg

    def result_config(self) -> AppConfig:
        """The edited config. Safe to call at any time; reflects the widgets right now."""
        return self._collect().copy()

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value, Qt.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable():
            combo.setEditText(value)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str, fallback: str = "") -> None:
        idx = combo.findData(value)
        if idx < 0 and fallback:
            idx = combo.findData(fallback)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # -- live preview ------------------------------------------------------

    def _connect_preview(self) -> None:
        for widget in (
            self.opacity_slider,
            self.font_size,
            self.original_font_size,
            self.max_history,
        ):
            widget.valueChanged.connect(self._emit_preview)
        for combo in (self.layout_combo, self.theme_combo):
            combo.currentIndexChanged.connect(self._emit_preview)
        for check in (
            self.show_original,
            self.always_on_top,
            self.mouse_transparent,
            self.show_status_bar,
        ):
            check.toggled.connect(self._emit_preview)

    def _emit_preview(self, *_args) -> None:
        if self._loading:
            return
        self.preview_requested.emit(self._collect().copy())

    # -- background jobs ---------------------------------------------------

    def _submit(self, job: _Job) -> None:
        job.signals.done.connect(self._on_job_done)
        job.signals.failed.connect(self._on_job_failed)
        QThreadPool.globalInstance().start(job)

    def refresh_devices(self) -> None:
        backend = self.audio_backend.currentText()
        kind = KIND_MIC if self.source_mic.isChecked() else KIND_LOOPBACK
        self.device_refresh.setEnabled(False)
        self.device_refresh.setText("...")
        self.device_status.setObjectName("HintLabel")
        self.device_status.setText("Enumerating devices...")
        self._submit(_Job(JOB_DEVICES, lambda: list_devices(backend, kind=kind)))

    def refresh_models(self) -> None:
        self.model_refresh.setEnabled(False)
        self.model_refresh.setText("...")
        self.ollama_status.setObjectName("HintLabel")
        self.ollama_status.setText("Checking...")
        # Snapshot on this (GUI) thread; the job must not read widgets itself.
        translate_cfg = self._collect().translate
        self._submit(_Job(JOB_MODELS, lambda: self._fetch_models(translate_cfg)))

    def test_connection(self) -> None:
        self.test_button.setEnabled(False)
        self.test_button.setText("Checking...")
        self.ollama_status.setObjectName("HintLabel")
        self.ollama_status.setText("Contacting Ollama...")
        translate_cfg = self._collect().translate
        self._submit(_Job(JOB_HEALTH, lambda: self._probe_health(translate_cfg)))

    def _client(self, translate_cfg):
        """A throwaway Ollama client for the *edited* URL.

        The pipeline's own client is bound to the config that is currently applied, so it
        would test the old address after the user edits the URL; it also owns a requests
        session that the GUI must not share.

        Takes the config as an argument rather than reading the widgets, because this runs on a
        thread-pool thread and touching a QWidget off the GUI thread is undefined behaviour.
        Callers snapshot the settings on the GUI thread before submitting the job.
        """
        from ..translate import OllamaClient

        return OllamaClient(translate_cfg)

    def _fetch_models(self, translate_cfg) -> List:
        client = self._client(translate_cfg)
        try:
            return client.list_models(timeout=8.0)
        finally:
            client.close()

    def _probe_health(self, translate_cfg):
        client = self._client(translate_cfg)
        try:
            ok, message = client.health(timeout=4.0)
            models: List = []
            if ok:
                try:
                    models = client.list_models(timeout=8.0)
                except Exception:  # noqa: BLE001 - version is still worth reporting
                    log.debug("model list failed during health check", exc_info=True)
            return ok, message, models
        finally:
            client.close()

    def _on_job_done(self, kind: str, result: object) -> None:
        if kind == JOB_DEVICES:
            self._fill_devices(result or [])
        elif kind == JOB_MODELS:
            self._fill_models(result or [])
            self._set_status(self.ollama_status, f"{len(result or [])} models available", ok=True)
        elif kind == JOB_HEALTH:
            ok, message, models = result
            if ok:
                self._fill_models(models)
                self._set_status(
                    self.ollama_status, f"{message} - {len(models)} models", ok=True
                )
            else:
                self._set_status(self.ollama_status, message, ok=False)
        self._reset_job_buttons()

    def _on_job_failed(self, kind: str, message: str) -> None:
        if kind == JOB_DEVICES:
            self._set_status(self.device_status, f"Device enumeration failed: {message}", ok=False)
        else:
            self._set_status(self.ollama_status, message, ok=False)
        self._reset_job_buttons()

    def _reset_job_buttons(self) -> None:
        self.device_refresh.setEnabled(True)
        self.device_refresh.setText("Refresh")
        self.model_refresh.setEnabled(True)
        self.model_refresh.setText("Refresh")
        self.test_button.setEnabled(True)
        self.test_button.setText("Test connection")

    def _fill_devices(self, devices) -> None:
        wanted = self._working.audio.device_name
        self.device_combo.clear()
        self.device_combo.addItem("System default", "")
        for dev in devices:
            self.device_combo.addItem(dev.label, dev.name)
        if wanted:
            idx = self.device_combo.findData(wanted)
            if idx < 0:
                # Keep an unavailable saved device selectable so opening Settings while a
                # headset is unplugged does not silently reset it.
                self.device_combo.addItem(f"{wanted} [not connected]", wanted)
                idx = self.device_combo.count() - 1
            self.device_combo.setCurrentIndex(idx)
        if devices:
            self._set_status(self.device_status, f"{len(devices)} devices found", ok=True)
        else:
            self._set_status(
                self.device_status,
                "No capture devices found. Check that soundcard or PyAudioWPatch is "
                "installed and that an output device is enabled.",
                ok=False,
            )

    def _fill_models(self, models) -> None:
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        for index, model in enumerate(models):
            # The visible text must be exactly the model name: the combo is editable, so
            # its text is what gets saved. Size and quantisation go in the tooltip.
            self.model_combo.addItem(model.name, model.name)
            self.model_combo.setItemData(index, model.label, Qt.ToolTipRole)
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setEditText(current)
        elif models:
            self.model_combo.setCurrentIndex(0)

    def _set_status(self, label: QLabel, text: str, ok: bool) -> None:
        label.setObjectName("OkLabel" if ok else "ErrorLabel")
        label.setText(text)
        label.setToolTip(text)
        # Object-name selectors are resolved at polish time, so force a re-polish.
        style = label.style()
        style.unpolish(label)
        style.polish(label)

    # -- accept / apply ----------------------------------------------------

    def _on_apply(self) -> None:
        self.accepted_config.emit(self._collect().copy())

    def _on_ok(self) -> None:
        self.accepted_config.emit(self._collect().copy())
        self.accept()
