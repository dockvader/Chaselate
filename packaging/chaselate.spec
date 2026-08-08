# PyInstaller spec for Chaselate. Build with:
#   .venv\Scripts\python.exe -m PyInstaller packaging\chaselate.spec --noconfirm
#
# onedir, not onefile: onefile re-extracts the whole bundle to a temp dir on every launch,
# which is a real cost given how large ctranslate2/onnxruntime/PyQt5 are together, and it
# makes the CUDA-DLL-drop-in trick the installer does afterward (see packaging/installer.nsi)
# awkward -- onedir gives a stable directory the installer can add files to post-install.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# SPECPATH is injected by PyInstaller as this file's own directory; the project root (where
# the chaselate package lives) is one level up, regardless of the caller's cwd.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

block_cipher = None

datas = []
binaries = []

# faster_whisper ships its Silero VAD model as package data (assets/silero_vad_v6.onnx), not
# code, so PyInstaller's import-graph analysis never sees it on its own.
datas += collect_data_files("faster_whisper")

# certifi's CA bundle (cacert.pem) is read from disk by requests at runtime, not imported.
datas += collect_data_files("certifi")

# Native DLLs whose loading isn't visible to PyInstaller's static import analysis: ctranslate2
# lazy-loads cudnn64_9.dll/libiomp5md.dll rather than link-time-linking them, and onnxruntime's
# capi DLLs are loaded by its own C extension at runtime, not via a Python import PyInstaller's
# analyzer can trace.
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")

a = Analysis(
    [os.path.join(SPECPATH, "run_chaselate.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "chaselate",
        "chaselate.ui",
        "chaselate.ui.overlay",
        "chaselate.ui.settings",
        "chaselate.ui.style",
        # cffi's C extension backend: soundcard uses cffi at runtime, and PyInstaller's
        # analyzer does not always follow cffi's dynamic backend loading.
        "_cffi_backend",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Pulled in transitively by some dependency's optional path but never used here;
        # excluding keeps the bundle smaller and avoids torch's own heavyweight native DLLs
        # being dragged in if anything ever probes for it.
        "torch",
        "torchaudio",
        "matplotlib",
        "tkinter",
        "test",
        "tests",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Chaselate",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed: no console flash, matches run.bat's pythonw.exe behaviour
    icon=os.path.join(SPECPATH, "assets", "chaselate.ico"),
)

# PyQt5 bundles its own copy of the Visual C++ runtime (msvcp140.dll, vcruntime140.dll,
# vcruntime140_1.dll, concrt140.dll) and PyInstaller collects it like any other dependency.
# In the dev environment chaselate/_runtime.py works around exactly this DLL by pinning the
# System32 copy before PyQt5 can be imported -- but in a frozen build, PyInstaller's own
# bootstrap touches PyQt5 (and loads its bundled runtime) before any of that Python-level
# code gets a chance to run, so the workaround can't win the race here. Stripping these files
# from the collected output instead of trying to out-race the bootloader means there is
# nothing bundled for anything to load *except* the System32 copy, which Windows resolves
# automatically once no local override shadows it. See chaselate/_runtime.py's docstring for
# the full account of why the mismatch causes ctranslate2/onnxruntime to fail with a DLL
# initialization error rather than a normal "file not found".
_MSVC_RUNTIME_DLLS = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"}
a.binaries = [
    entry for entry in a.binaries if os.path.basename(entry[0]).lower() not in _MSVC_RUNTIME_DLLS
]

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Chaselate",
)
