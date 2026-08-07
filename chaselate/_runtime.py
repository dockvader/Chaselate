"""Pin the system Visual C++ runtime before anything else can load a different copy.

This exists to stop a hard crash, and it has to run before PyQt5 is imported.

**The problem.** PyQt5 ships its own copies of the Visual C++ redistributable in
``PyQt5/Qt5/bin`` -- ``msvcp140.dll``, ``msvcp140_1.dll``, ``msvcp140_2.dll``,
``vcruntime140.dll``, ``vcruntime140_1.dll``, ``concrt140.dll`` -- and puts that directory on
the DLL search path when it is imported. They are older than the copies in ``System32``
(590,112 vs 643,512 bytes for ``msvcp140.dll`` on the machine this was diagnosed on).
CTranslate2, the backend inside faster-whisper, bundles Intel OpenMP (``libiomp5md.dll``),
which links against that runtime. If PyQt5 is imported first, CTranslate2's OpenMP binds to
PyQt5's older runtime and the process dies with an access violation the first time a model is
constructed.

The failure is unusually nasty to diagnose:

* it is a Windows access violation, so it looks like a GPU or driver fault, not an import
  problem, and Python raises nothing catchable;
* it happens on CPU as well as CUDA, and on the main thread as well as worker threads, so
  every plausible-looking hypothesis about threads or CUDA is a dead end;
* merely ``import PyQt5`` is enough -- no ``QApplication`` required;
* ``KMP_DUPLICATE_LIB_OK=TRUE``, the usual advice for duplicate OpenMP runtimes, does not
  help.

**The fix.** Load the ``System32`` copies by absolute path, first. The Windows loader matches
already-loaded modules by name, so every later request for ``msvcp140.dll`` -- including
PyQt5's -- resolves to the system copy that is already in the process, and nothing gets two
incompatible runtimes.

Called from :mod:`chaselate.__init__`, so importing anything from this package is enough to be
safe. Importing PyQt5 *before* any part of ``chaselate`` defeats it; that case is detected and
logged rather than left to crash mysteriously.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import List

log = logging.getLogger(__name__)

#: Order matters within this list: vcruntime is the dependency of the msvcp libraries.
_RUNTIME_DLLS = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "concrt140.dll",
)

_pinned: List[str] = []
_done = False


def was_pinned() -> bool:
    """Whether the System32 runtime was successfully pinned before PyQt5 could load its own.

    Lets callers distinguish "PyQt5 was imported first and that is fine because we already
    won the race" from "PyQt5 was imported first and nothing protected us".
    """
    return bool(_pinned)


def pin_system_msvc_runtime() -> List[str]:
    """Load the System32 C++ runtime DLLs so no bundled copy can win. Idempotent.

    Returns the names successfully pinned. Silent no-op off Windows, and non-fatal if the
    system copies are missing -- a machine without the redistributable has other problems, and
    failing here would turn a crash risk into a certain startup failure.
    """
    global _done
    if _done:
        return list(_pinned)
    _done = True

    if not sys.platform.startswith("win"):
        return []

    if "PyQt5.QtCore" in sys.modules:
        log.warning(
            "PyQt5 was imported before chaselate, so its bundled Visual C++ runtime is "
            "already loaded. CTranslate2 may crash with an access violation. Import chaselate "
            "(or call chaselate.pin_system_msvc_runtime()) before importing PyQt5."
        )

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in _RUNTIME_DLLS:
        path = os.path.join(system32, name)
        if not os.path.exists(path):
            continue
        try:
            # Absolute path, so the search order cannot substitute a different copy. The
            # handle is intentionally dropped: the module stays loaded for the process
            # lifetime, which is the entire point.
            ctypes.WinDLL(path)
            _pinned.append(name)
        except OSError as exc:
            log.debug("could not pin %s: %s", name, exc)

    if _pinned:
        log.debug("pinned system C++ runtime: %s", ", ".join(_pinned))
    else:
        log.warning(
            "no system Visual C++ runtime found in %s; if the app crashes when starting "
            "recognition, install the Microsoft Visual C++ 2015-2022 Redistributable",
            system32,
        )
    return list(_pinned)
