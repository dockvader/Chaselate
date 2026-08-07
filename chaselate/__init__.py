"""Chaselate - live speech translation overlay for Windows.

System audio (WASAPI loopback) -> Silero VAD -> faster-whisper -> Ollama -> PyQt5 overlay.
"""

# This runs on import, before PyQt5 can be imported, and it must stay at the top of the file.
# PyQt5 bundles an older Visual C++ runtime that CTranslate2's OpenMP will bind to if it loads
# first, which crashes the process with an access violation. See chaselate._runtime.
from ._runtime import pin_system_msvc_runtime

pin_system_msvc_runtime()

__version__ = "0.1.0"
APP_NAME = "Chaselate"

__all__ = ["APP_NAME", "__version__", "pin_system_msvc_runtime"]
