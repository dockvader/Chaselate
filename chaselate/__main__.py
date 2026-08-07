"""Entry point: ``python -m chaselate``.

Command-line flags mirror the macOS original's where they still make sense, so existing
notes and habits carry over. They act as one-shot overrides on top of the saved config and
are not persisted unless ``--save`` is given -- handy for trying a bigger model once without
committing to it.

``--list-devices`` and ``--list-models`` run headless and are the first thing to reach for
when nothing is being captured or translated.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from typing import List, Optional

from . import APP_NAME, __version__
from .config import (
    ASR_DEVICES,
    AUDIO_BACKENDS,
    COMPUTE_TYPES,
    LAYOUTS,
    WHISPER_MODELS,
    AppConfig,
    config_dir,
    config_path,
)
from .languages import AUTO, LANGUAGES, display_name

log = logging.getLogger("chaselate")

LOG_FILENAME = "chaselate.log"


def _language_codes() -> List[str]:
    return [lang.code for lang in LANGUAGES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chaselate",
        description=(
            "Live speech translation overlay for Windows. Captures system audio via WASAPI "
            "loopback, transcribes it with faster-whisper, and translates it with a local "
            "Ollama model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")

    lang = parser.add_argument_group("languages")
    lang.add_argument(
        "--target", metavar="CODE",
        help="Translate into this language, e.g. zh-TW, en, ja. zh-TW and zh-CN are "
             "distinct: zh-TW yields Traditional Chinese.",
    )
    lang.add_argument(
        "--source", metavar="CODE",
        help=f"Spoken language, or '{AUTO}' to detect. Pinning it is noticeably more stable "
             f"than auto-detection on multilingual audio.",
    )

    audio = parser.add_argument_group("audio")
    audio.add_argument("--device", metavar="NAME", help="Capture device name (substring match).")
    audio.add_argument(
        "--source-type", choices=["loopback", "mic"],
        help="loopback captures what the speakers play; mic captures an input device.",
    )
    audio.add_argument("--audio-backend", choices=AUDIO_BACKENDS)
    audio.add_argument("--gain", type=float, metavar="X", help="Input gain multiplier.")

    asr = parser.add_argument_group("recognition")
    asr.add_argument("--whisper", metavar="MODEL", help=f"One of: {', '.join(WHISPER_MODELS)}")
    asr.add_argument("--asr-device", choices=ASR_DEVICES)
    asr.add_argument("--compute-type", choices=COMPUTE_TYPES)
    asr.add_argument("--beam-size", type=int)
    asr.add_argument(
        "--silence-rms", type=float, metavar="X",
        help="Blocks quieter than this RMS are treated as silence without consulting the VAD.",
    )
    asr.add_argument("--vad-threshold", type=float, metavar="X")
    asr.add_argument("--vad-min-speech-ms", type=int, metavar="MS")
    asr.add_argument(
        "--min-silence-ms", type=int, metavar="MS",
        help="Trailing silence that closes an utterance. Lower is snappier but choppier.",
    )
    asr.add_argument("--speech-pad-ms", type=int, metavar="MS")
    asr.add_argument("--max-segment", type=float, metavar="SECONDS")

    tr = parser.add_argument_group("translation")
    tr.add_argument("--ollama-model", metavar="NAME", help="e.g. translategemma:latest")
    tr.add_argument("--ollama-url", metavar="URL")
    tr.add_argument(
        "--context-sentences", type=int, metavar="N",
        help="Previous sentence pairs given to the model for continuity. 0 = strict "
             "single-sentence translation.",
    )
    tr.add_argument("--temperature", type=float)
    tr.add_argument("--no-translate", action="store_true", help="Transcribe only.")
    tr.add_argument("--no-stream", action="store_true", help="Wait for the whole translation.")

    ui = parser.add_argument_group("appearance")
    ui.add_argument("--layout", choices=LAYOUTS)
    ui.add_argument("--opacity", type=float, metavar="0-1")
    ui.add_argument("--font-size", type=int)
    ui.add_argument("--no-original", action="store_true", help="Show only the translation.")
    ui.add_argument("--click-through", action="store_true", help="Let clicks pass through.")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--autostart", action="store_true", help="Begin capturing immediately.")
    misc.add_argument("--save", action="store_true", help="Persist these overrides.")
    misc.add_argument("--reset-config", action="store_true", help="Restore defaults and exit.")
    misc.add_argument("--config", metavar="PATH", help="Use an alternate config file.")
    misc.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    misc.add_argument("--list-devices", action="store_true", help="Print devices and exit.")
    misc.add_argument("--list-models", action="store_true", help="Print Ollama models and exit.")
    misc.add_argument("--list-languages", action="store_true", help="Print languages and exit.")
    return parser


def setup_logging(level: str) -> None:
    numeric = getattr(logging, (level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        directory = config_dir()
        directory.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            directory / LOG_FILENAME, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        rotating.setFormatter(fmt)
        root.addHandler(rotating)
    except OSError as exc:
        log.warning("file logging disabled: %s", exc)

    # These two are chatty at DEBUG and drown out everything we care about.
    logging.getLogger("urllib3").setLevel(max(numeric, logging.INFO))
    logging.getLogger("faster_whisper").setLevel(max(numeric, logging.INFO))


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Fold command-line flags into the loaded config. Only set flags take effect."""
    if args.target:
        cfg.translate.target_lang = args.target
    if args.source:
        cfg.asr.source_lang = args.source

    if args.device is not None:
        cfg.audio.device_name = args.device
    if args.source_type:
        cfg.audio.source = args.source_type
    if args.audio_backend:
        cfg.audio.backend = args.audio_backend
    if args.gain is not None:
        cfg.audio.gain = args.gain

    if args.whisper:
        cfg.asr.model = args.whisper
    if args.asr_device:
        cfg.asr.device = args.asr_device
    if args.compute_type:
        cfg.asr.compute_type = args.compute_type
    if args.beam_size is not None:
        cfg.asr.beam_size = args.beam_size

    if args.silence_rms is not None:
        cfg.vad.silence_rms = args.silence_rms
    if args.vad_threshold is not None:
        cfg.vad.threshold = args.vad_threshold
    if args.vad_min_speech_ms is not None:
        cfg.vad.min_speech_ms = args.vad_min_speech_ms
    if args.min_silence_ms is not None:
        cfg.vad.min_silence_ms = args.min_silence_ms
    if args.speech_pad_ms is not None:
        cfg.vad.speech_pad_ms = args.speech_pad_ms
    if args.max_segment is not None:
        cfg.vad.max_segment_s = args.max_segment

    if args.ollama_model:
        cfg.translate.model = args.ollama_model
    if args.ollama_url:
        cfg.translate.base_url = args.ollama_url
    if args.context_sentences is not None:
        cfg.translate.context_sentences = args.context_sentences
    if args.temperature is not None:
        cfg.translate.temperature = args.temperature
    if args.no_translate:
        cfg.translate.enabled = False
    if args.no_stream:
        cfg.translate.stream = False

    if args.layout:
        cfg.ui.layout = args.layout
    if args.opacity is not None:
        cfg.ui.opacity = max(0.2, min(1.0, args.opacity))
    if args.font_size is not None:
        cfg.ui.font_size = args.font_size
    if args.no_original:
        cfg.ui.show_original = False
    if args.click_through:
        cfg.ui.mouse_transparent = True

    if args.autostart:
        cfg.autostart = True
    if args.log_level:
        cfg.log_level = args.log_level
    return cfg


# -- headless helpers ------------------------------------------------------------


def cmd_list_devices() -> int:
    from .audio.devices import available_backends, list_devices

    backends = available_backends()
    print(f"Audio backends available: {', '.join(backends) or 'NONE'}")
    if not backends:
        print("\nInstall one:  pip install soundcard    (or)    pip install PyAudioWPatch")
        return 1

    devices = list_devices()
    if not devices:
        print("\nNo capture devices found. Check Windows sound settings.")
        return 1
    print(f"\n{len(devices)} device(s):\n")
    for dev in devices:
        mark = "*" if dev.is_default else " "
        print(f" {mark} [{dev.kind:8s}] {dev.name}")
        print(f"     backend={dev.backend} channels={dev.channels} rate={dev.rate}")
    print("\n  * = system default. Pass a name substring to --device.")
    return 0


def cmd_list_models(cfg: AppConfig) -> int:
    from .translate import OllamaClient, OllamaError

    client = OllamaClient(cfg.translate)
    try:
        ok, message = client.health()
        print(message)
        if not ok:
            return 1
        models = client.list_models()
    except OllamaError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        client.close()

    if not models:
        print("\nNo models installed. Get the default translation model with:")
        print("    ollama pull translategemma")
        return 1
    print(f"\n{len(models)} model(s):\n")
    for model in models:
        tag = " [translation specialist]" if model.is_translation_specialist else ""
        print(f"  {model.label}{tag}")
    return 0


def cmd_list_languages() -> int:
    print(f"{len(LANGUAGES)} languages (use the code with --target / --source):\n")
    for lang in LANGUAGES:
        print(f"  {lang.code:6s}  {lang.english:34s} {lang.native}")
    print(f"\n  {AUTO:6s}  detect the spoken language automatically (--source only)")
    return 0


# -- main ------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    path = config_path()
    if args.config:
        from pathlib import Path

        path = Path(args.config).expanduser()

    if args.reset_config:
        setup_logging("INFO")
        AppConfig().save(path)
        print(f"Configuration reset: {path}")
        return 0

    cfg = AppConfig.load(path)
    cfg = apply_overrides(cfg, args)
    setup_logging(cfg.log_level)

    if args.list_devices:
        return cmd_list_devices()
    if args.list_models:
        return cmd_list_models(cfg)
    if args.list_languages:
        return cmd_list_languages()

    if args.save:
        cfg.save(path)
        log.info("saved configuration to %s", path)

    log.info(
        "%s %s starting: %s -> %s via %s",
        APP_NAME, __version__,
        display_name(cfg.asr.source_lang), display_name(cfg.translate.target_lang),
        cfg.translate.model,
    )

    # Import the native ML stack now, while PyQt5 is still absent from sys.modules.
    #
    # Order matters and the failure is a hard crash, not an exception: PyQt5 ships an older
    # Visual C++ runtime and puts it on the DLL search path, which CTranslate2's bundled Intel
    # OpenMP then binds to, killing the process with an access violation. See
    # asr.preload_native_libraries for the full account. Nothing above this line may import
    # PyQt5, and run_gui is where PyQt5 first appears.
    from .asr import preload_native_libraries

    if not preload_native_libraries():
        print(
            "error: the speech recognition backend could not be loaded.\n"
            "Reinstall dependencies with:  .venv\\Scripts\\python.exe -m pip install -r "
            "requirements.txt",
            file=sys.stderr,
        )
        return 1

    return run_gui(cfg, path)


def run_gui(cfg: AppConfig, config_file) -> int:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # Must be set before the QApplication exists, or they are ignored.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_NAME)
    # The overlay hides to the tray instead of exiting, so the last visible window closing
    # must not end the process.
    app.setQuitOnLastWindowClosed(False)

    from .pipeline import Pipeline
    from .ui import OverlayWindow
    from .ui.settings import wait_for_background_jobs

    pipeline = Pipeline(cfg)
    window = OverlayWindow(cfg, pipeline)
    window.show()

    if cfg.autostart:
        pipeline.start()

    def teardown() -> None:
        # Settings probes run on the global thread pool and may be inside a WASAPI
        # enumeration or an HTTP call. Tearing the interpreter down under one of those
        # crashes instead of exiting, so drain them first. OverlayWindow.quit does this too;
        # hooking aboutToQuit covers the exit paths that do not go through it.
        wait_for_background_jobs()
        pipeline.shutdown()
        if config_file != config_path():
            # quit() saves window geometry to the default location; mirror it to the
            # explicitly requested file.
            cfg.save(config_file)

    app.aboutToQuit.connect(teardown)
    try:
        return app.exec_()
    finally:
        teardown()


if __name__ == "__main__":
    sys.exit(main())
