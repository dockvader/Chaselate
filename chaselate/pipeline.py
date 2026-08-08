"""The pipeline: audio in, captions out.

Four stages, each on its own thread, joined by bounded queues::

    capture thread   WASAPI -> 16 kHz mono blocks        (audio_queue)
    segment thread   Silero VAD -> padded utterances     (asr_queue)
    asr thread       faster-whisper -> sentences         (translate_queue)
    translate thread Ollama -> translated text           (Qt signals)

The stages are split rather than chained inline because they run at wildly different
speeds. Audio arrives every 32 ms and WASAPI drops samples if we are late collecting them;
Whisper takes a second or two per utterance; Ollama takes several. Anything slow behind the
audio callback shows up as clicks in the capture, so the callback does nothing but enqueue.

Queues are bounded and drop *oldest* under pressure. That is deliberate: for live captions,
audio from thirty seconds ago has no value, and an unbounded queue would grow until the
process died. Every drop is counted and surfaced through :attr:`Pipeline.metrics` so the UI
can tell the user it is falling behind instead of silently lying.

All cross-thread communication out of here is Qt signals, which Qt marshals onto the GUI
thread automatically.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Import order below is load-bearing and must not be sorted.
#
# PyQt5 puts its own older Visual C++ runtime on the DLL search path when imported, and
# CTranslate2's bundled Intel OpenMP then binds to it and takes the process down with an
# access violation. Loading the native ML stack first makes the system runtime win.
# asr.preload_native_libraries has the details; it is idempotent, and it lives here as well
# as in __main__ so that importing this module from a script or a test is also safe.
from .asr import preload_native_libraries as _preload_native_libraries

_preload_native_libraries()

from PyQt5.QtCore import QObject, QTimer, pyqtSignal  # noqa: E402

from .asr import AsrError, WhisperEngine  # noqa: E402
from .audio import AudioCapture
from .audio.devices import DeviceInfo
from .config import AppConfig
from .languages import AUTO
from .textutils import dedupe_overlap, normalize_ws, split_sentences
from .translate import (
    ContextPair,
    OllamaClient,
    OllamaError,
    TranslationCancelled,
)
from .vad import SAMPLE_RATE, Segment, Segmenter

log = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_ERROR = "error"

#: Utterances waiting for Whisper. Beyond this we are hopelessly behind.
ASR_QUEUE_MAX = 12
#: Sentences waiting for Ollama.
TRANSLATE_QUEUE_MAX = 16
#: How often the metrics signal fires, in milliseconds.
METRICS_INTERVAL_MS = 1000

_SENTINEL = object()
#: Queued like a normal item so it lands in order relative to already-queued utterances,
#: rather than mutating _context directly from the GUI thread -- see clear_context().
_CLEAR_CONTEXT = object()


@dataclass
class Utterance:
    """One caption line: the recognised original and its translation."""

    id: int
    original: str
    language: str = ""
    start: float = 0.0
    end: float = 0.0
    translation: str = ""
    #: ``pending`` -> ``translating`` -> ``done`` | ``failed`` | ``skipped``
    state: str = "pending"
    error: str = ""
    asr_rtf: float = 0.0
    created_at: float = field(default_factory=time.time)


class Pipeline(QObject):
    """Owns the worker threads and reports everything through signals.

    Create on the GUI thread. :meth:`start` and :meth:`stop` are safe to call repeatedly and
    from the GUI thread only.
    """

    #: ``(state, human readable detail)`` -- state is one of the ``STATE_*`` constants.
    status_changed = pyqtSignal(str, str)
    #: A message worth putting in front of the user.
    error = pyqtSignal(str)
    #: Input RMS, roughly 30 times a second, for the level meter.
    level = pyqtSignal(float)
    #: True while the VAD believes someone is speaking.
    speech_state = pyqtSignal(bool)
    #: A finalised original is ready; payload is an :class:`Utterance`.
    utterance_added = pyqtSignal(object)
    #: ``(utterance id, cumulative translation so far)`` while streaming.
    translation_delta = pyqtSignal(int, str)
    #: ``(utterance id, final translation)``
    translation_done = pyqtSignal(int, str)
    #: ``(utterance id, error message)``
    translation_failed = pyqtSignal(int, str)
    #: Throughput and backlog numbers; see :meth:`_emit_metrics` for the keys.
    metrics = pyqtSignal(dict)

    def __init__(self, config: AppConfig, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config = config

        self._state = STATE_IDLE
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._capture: Optional[AudioCapture] = None
        #: Set once every model is loaded and capture has been handed the go-ahead.
        self._warm = False

        self._audio_queue: "queue.Queue" = queue.Queue()
        self._asr_queue: "queue.Queue" = queue.Queue()
        self._translate_queue: "queue.Queue" = queue.Queue()

        self._engine = WhisperEngine(config.asr)
        self._ollama = OllamaClient(config.translate)
        self._segmenter = Segmenter(config.vad)

        # Assembly state, touched only by the ASR thread.
        self._pending_text = ""
        self._previous_text = ""
        self._next_id = 1

        # Context for the translator, touched only by the translate thread.
        self._context: List[ContextPair] = []
        self._cancel = threading.Event()

        # Counters read by the metrics timer; ints are atomic enough under the GIL for this.
        self._dropped_audio = 0
        self._dropped_segments = 0
        self._dropped_sentences = 0
        self._asr_count = 0
        self._asr_time = 0.0
        self._asr_audio = 0.0
        self._last_level = 0.0
        self._device_label = ""

        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(METRICS_INTERVAL_MS)
        self._metrics_timer.timeout.connect(self._emit_metrics)

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def running(self) -> bool:
        return self._state in (STATE_STARTING, STATE_RUNNING)

    def _set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self.status_changed.emit(state, detail)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        # A brand-new Event, not clear() on the old one: a worker that outlived the previous
        # stop() still holds the old event, and clearing it would wake that thread back up.
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._warm = False
        self._set_state(STATE_STARTING, "Starting...")

        cfg = self.config
        self._engine.config = cfg.asr
        self._ollama.config = cfg.translate
        self._segmenter = Segmenter(cfg.vad, self._segmenter.vad)
        self._pending_text = ""
        self._previous_text = ""
        self._context.clear()
        self._dropped_audio = self._dropped_segments = self._dropped_sentences = 0
        self._asr_count = 0
        self._asr_time = self._asr_audio = 0.0

        audio_max = max(16, int(cfg.audio.queue_seconds * SAMPLE_RATE / max(1, cfg.audio.block_frames)))
        self._audio_queue = queue.Queue(maxsize=audio_max)
        self._asr_queue = queue.Queue(maxsize=ASR_QUEUE_MAX)
        self._translate_queue = queue.Queue(maxsize=TRANSLATE_QUEUE_MAX)

        # Every worker gets its run's stop event and queues as arguments rather than reading
        # them off self. A thread that outlives the join in stop() therefore keeps a reference
        # to the *old*, already-set event and the old queues: it exits on its next iteration
        # and can never attach itself to the next run as a second consumer. Reading them from
        # self would let such a thread pick work off the new queue and translate the same
        # sentence twice.
        stop = self._stop
        audio_q, asr_q, translate_q = self._audio_queue, self._asr_queue, self._translate_queue

        self._capture = AudioCapture(
            cfg.audio,
            on_audio=lambda block: self._on_audio(block, audio_q),
            on_error=self._on_capture_error,
            on_level=self._on_level,
            on_started=self._on_capture_started,
        )
        capture = self._capture

        self._threads = [
            threading.Thread(
                target=self._segment_loop, args=(stop, audio_q, asr_q),
                name="chaselate-segment", daemon=True,
            ),
            threading.Thread(
                target=self._asr_loop, args=(stop, asr_q, translate_q),
                name="chaselate-asr", daemon=True,
            ),
            threading.Thread(
                target=self._translate_loop, args=(stop, translate_q),
                name="chaselate-translate", daemon=True,
            ),
            # Starts the capture itself once the models are ready -- see _warmup_loop.
            threading.Thread(
                target=self._warmup_loop, args=(stop, capture),
                name="chaselate-warmup", daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        self._metrics_timer.start()

    def stop(self) -> None:
        if self._state == STATE_IDLE:
            return
        self._set_state(STATE_STOPPING, "Stopping...")
        self._metrics_timer.stop()
        self._stop.set()
        self._cancel.set()

        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        # Unblock any thread parked on a get().
        for q in (self._audio_queue, self._asr_queue, self._translate_queue):
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=4.0)
        self._threads = []
        self._set_state(STATE_IDLE, "Stopped")

    def shutdown(self) -> None:
        """Stop and release the model and HTTP session. Call once, at exit."""
        self.stop()
        self._engine.unload()
        self._ollama.close()

    def clear_context(self) -> None:
        """Drop the translator's rolling context of recent sentence pairs.

        Call this whenever the visible caption history is wiped (see
        OverlayWindow.clear_captions), so a translation made right afterwards does not pull
        pronouns/continuity from utterances the user can no longer see. _context is otherwise
        touched only by the translate thread, so this enqueues a marker for that thread to act
        on rather than mutating it directly from the GUI thread -- it also keeps ordering
        correct relative to any utterance already queued ahead of it.
        """
        try:
            self._translate_queue.put_nowait(_CLEAR_CONTEXT)
        except queue.Full:
            log.debug("translate queue full, dropping clear-context request")

    def apply_config(self, config: AppConfig) -> bool:
        """Adopt new settings. Returns True if a restart was needed to apply them.

        Translation and VAD settings take effect immediately; changing the audio device or
        the Whisper model means rebuilding the stage that owns it, so the pipeline is
        cycled.
        """
        old = self.config
        needs_restart = self.running and (
            old.audio != config.audio
            or old.asr.model != config.asr.model
            or old.asr.device != config.asr.device
            or old.asr.compute_type != config.asr.compute_type
            or old.asr.cpu_threads != config.asr.cpu_threads
        )
        self.config = config
        self._engine.config = config.asr
        self._ollama.config = config.translate
        self._segmenter.config = config.vad

        if needs_restart:
            self.stop()
            self.start()
        return needs_restart

    # -- capture callbacks (capture thread) --------------------------------

    def _on_capture_started(self, device: DeviceInfo) -> None:
        self._device_label = device.name
        self._set_state(STATE_RUNNING, f"Listening to {device.name}")

    def _on_capture_error(self, message: str) -> None:
        self._set_state(STATE_ERROR, message)
        self.error.emit(message)

    def _on_level(self, rms: float) -> None:
        self._last_level = rms
        self.level.emit(rms)

    def _on_audio(self, block: np.ndarray, audio_q: "queue.Queue") -> None:
        try:
            audio_q.put_nowait(block)
        except queue.Full:
            # Discard the oldest block to make room; recency beats completeness here.
            try:
                audio_q.get_nowait()
                self._dropped_audio += 1
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(block)
            except queue.Full:
                self._dropped_audio += 1

    # -- stage 2: segmentation ---------------------------------------------

    def _segment_loop(
        self, stop: threading.Event, audio_q: "queue.Queue", asr_q: "queue.Queue"
    ) -> None:
        while not stop.is_set():
            try:
                item = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            was_speaking = self._segmenter.speech_active
            try:
                segments = self._segmenter.feed(item)
            except Exception as exc:  # noqa: BLE001
                log.exception("segmentation failed")
                self.error.emit(f"Voice detection failed: {exc}")
                continue
            if self._segmenter.speech_active != was_speaking:
                self.speech_state.emit(self._segmenter.speech_active)
            for segment in segments:
                self._enqueue_segment(segment, asr_q)

        # Flush whatever was mid-utterance so the last words are not lost.
        try:
            for segment in self._segmenter.flush():
                self._enqueue_segment(segment, asr_q)
        except Exception:  # noqa: BLE001
            log.debug("flush failed during shutdown", exc_info=True)

    def _enqueue_segment(self, segment: Segment, asr_q: "queue.Queue") -> None:
        try:
            asr_q.put_nowait(segment)
        except queue.Full:
            try:
                asr_q.get_nowait()
                self._dropped_segments += 1
                log.warning("ASR backlog full; dropped an utterance")
            except queue.Empty:
                pass
            try:
                asr_q.put_nowait(segment)
            except queue.Full:
                self._dropped_segments += 1

    # -- stage 3: recognition ----------------------------------------------

    def _warmup_loop(self, stop: threading.Event, capture: AudioCapture) -> None:
        """Load every model, then start the capture.

        Capture deliberately does not begin until this finishes, for two reasons.

        The first is correctness. Importing onnxruntime (for Silero) while CTranslate2 is
        pulling the CUDA libraries into the process can fail on the Windows loader lock. With
        lazy loading the two happened concurrently -- audio arrived while Whisper was still
        loading -- and the VAD silently fell back to a crude energy gate for the whole
        session. Loading both here, sequentially, on one thread removes the race.

        The second is that all of this is slow exactly once: Whisper may download weights and
        JIT GPU kernels, and Ollama reads gigabytes into VRAM. Better to spend it during
        startup, where the status line explains the wait, than to lose the first sentence.
        """
        try:
            self._set_state(STATE_STARTING, "Loading voice detector...")
            if not self._segmenter.vad.load():
                # Not fatal: the energy gate still segments, just less precisely.
                self.error.emit(
                    "Voice detector unavailable; falling back to a simpler energy-based "
                    "detector. Captions may break at odd places."
                )

            self._set_state(STATE_STARTING, "Loading speech model...")
            elapsed = self._engine.warmup()
            log.info("Whisper warm in %.1fs (%s)", elapsed, self._engine.description)
        except AsrError as exc:
            self._set_state(STATE_ERROR, str(exc))
            self.error.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("warmup failed")
            self._set_state(STATE_ERROR, str(exc))
            self.error.emit(f"Speech model warmup failed: {exc}")
            return

        if stop.is_set():
            return

        if self.config.translate.enabled:
            try:
                self._set_state(STATE_STARTING, "Loading translation model...")
                self._ollama.translate("Hello.", source_lang="en", cancel=stop)
            except TranslationCancelled:
                return
            except OllamaError as exc:
                # Recognition still works; captions just show no translation.
                log.warning("translation preload failed: %s", exc)
                self.error.emit(str(exc))
            except Exception:  # noqa: BLE001
                log.exception("translation preload failed")

        if stop.is_set():
            return
        self._warm = True
        capture.start()

    def _asr_loop(
        self, stop: threading.Event, asr_q: "queue.Queue", translate_q: "queue.Queue"
    ) -> None:
        while not stop.is_set():
            try:
                item = asr_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            try:
                self._transcribe(item, translate_q)
            except AsrError as exc:
                log.error("%s", exc)
                self.error.emit(str(exc))
            except Exception as exc:  # noqa: BLE001
                log.exception("unexpected ASR failure")
                self.error.emit(f"Recognition failed: {exc}")

    def _transcribe(self, segment: Segment, translate_q: "queue.Queue") -> None:
        result = self._engine.transcribe(segment.audio, self.config.asr.source_lang)
        self._asr_count += 1
        self._asr_time += result.elapsed
        self._asr_audio += result.duration
        if result.empty:
            return

        text = result.text
        # Forced mid-speech cuts overlap the previous segment on purpose; strip the repeat.
        if segment.continues_previous and self._previous_text:
            text = dedupe_overlap(self._previous_text, text, result.language)
            if not text:
                return
        self._previous_text = result.text

        combined = normalize_ws(" ".join(p for p in (self._pending_text, text) if p))
        sentences, remainder = split_sentences(combined, result.language)

        # A real pause is a sentence boundary even without punctuation, so flush the tail.
        if segment.reason in ("silence", "flush") and remainder:
            sentences.append(remainder)
            remainder = ""
        self._pending_text = remainder

        for sentence in sentences:
            self._emit_utterance(sentence, segment, result, translate_q)

    def _emit_utterance(
        self, sentence: str, segment: Segment, result, translate_q: "queue.Queue"
    ) -> None:
        utterance = Utterance(
            id=self._next_id,
            original=sentence,
            language=result.language,
            start=segment.start,
            end=segment.end,
            asr_rtf=result.rtf,
        )
        self._next_id += 1
        self.utterance_added.emit(utterance)

        if not self.config.translate.enabled:
            utterance.state = "skipped"
            return
        try:
            translate_q.put_nowait(utterance)
        except queue.Full:
            try:
                stale = translate_q.get_nowait()
                stale.state = "skipped"
                self._dropped_sentences += 1
                self.translation_failed.emit(stale.id, "skipped - translator behind")
            except queue.Empty:
                pass
            try:
                translate_q.put_nowait(utterance)
            except queue.Full:
                self._dropped_sentences += 1
                self.translation_failed.emit(utterance.id, "skipped - translator behind")

    # -- stage 4: translation ----------------------------------------------

    def _translate_loop(self, stop: threading.Event, translate_q: "queue.Queue") -> None:
        while not stop.is_set():
            try:
                item = translate_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            if item is _CLEAR_CONTEXT:
                self._context.clear()
                continue
            self._translate_one(item, stop)

    def _translate_one(self, utterance: Utterance, stop: threading.Event) -> None:
        cfg = self.config.translate
        source = utterance.language or self.config.asr.source_lang
        if source == AUTO:
            source = utterance.language or None

        utterance.state = "translating"
        try:
            text = self._ollama.translate(
                utterance.original,
                source_lang=source,
                target_lang=cfg.target_lang,
                context=self._context,
                on_delta=lambda partial, uid=utterance.id: self.translation_delta.emit(
                    uid, partial
                ),
                cancel=stop,
            )
        except TranslationCancelled:
            utterance.state = "skipped"
            return
        except OllamaError as exc:
            utterance.state = "failed"
            utterance.error = str(exc)
            log.warning("translation failed: %s", exc)
            self.translation_failed.emit(utterance.id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            utterance.state = "failed"
            utterance.error = str(exc)
            log.exception("unexpected translation failure")
            self.translation_failed.emit(utterance.id, str(exc))
            return

        utterance.translation = text
        utterance.state = "done"
        self.translation_done.emit(utterance.id, text)

        if text:
            self._context.append(ContextPair(utterance.original, text))
            keep = max(0, int(cfg.context_sentences))
            if keep and len(self._context) > keep:
                del self._context[:-keep]
            elif not keep:
                self._context.clear()

    # -- metrics -----------------------------------------------------------

    def _emit_metrics(self) -> None:
        rtf = self._asr_time / self._asr_audio if self._asr_audio > 0 else 0.0
        capture = self._capture
        self.metrics.emit(
            {
                "state": self._state,
                "warm": self._warm,
                "device": self._device_label,
                "asr": self._engine.description,
                "asr_rtf": rtf,
                "asr_count": self._asr_count,
                "level": self._last_level,
                "audio_queue": self._audio_queue.qsize(),
                "asr_queue": self._asr_queue.qsize(),
                "translate_queue": self._translate_queue.qsize(),
                "dropped_audio": self._dropped_audio,
                "dropped_segments": self._dropped_segments,
                "dropped_sentences": self._dropped_sentences,
                "discontinuities": capture.discontinuities if capture is not None else 0,
                "speech": self._segmenter.speech_active,
            }
        )

    # -- helpers for the UI ------------------------------------------------

    def list_models(self) -> List:
        """Ollama models available locally. Blocking; call from a worker."""
        return self._ollama.list_models()

    def ollama_health(self):
        return self._ollama.health()
