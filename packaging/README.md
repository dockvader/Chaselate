# Building the Windows installer

Produces `ChaselateSetup-<version>.exe`: a self-contained NSIS installer that shows up in
Windows' "Add or Remove Programs", with an optional GPU-acceleration component and a
best-effort Ollama presence/update check. **Not code-signed** -- Windows SmartScreen will
show an "unknown publisher" warning on first run of the installer. Signing needs a paid
Authenticode certificate; this was a deliberate scope decision, not an oversight.

**Per-user install, no admin/UAC.** Installs to `%LOCALAPPDATA%\Programs\Chaselate` and
registers under `HKCU`, not `HKLM`. Nothing here needs machine-wide installation (no
service, no driver), and skipping the UAC prompt is a better experience for what is a
single-user desktop tool. It still shows up in the unified "Add or Remove Programs" /
Settings > Apps list exactly the same as a machine-wide install would.

## Prerequisites

- The project's own `.venv` (see the main README's install instructions), plus:
  ```bat
  .venv\Scripts\python.exe -m pip install pyinstaller
  ```
- [NSIS](https://nsis.sourceforge.io/) 3.x. Installed here via `winget install --id NSIS.NSIS`;
  `makensis.exe` ends up at `C:\Program Files (x86)\NSIS\makensis.exe`.
- `pip install pillow`, only if you need to regenerate `assets/chaselate.ico` (see below) --
  not needed for a normal build, since the generated files are checked into `packaging/assets/`.

## Build

From the project root:

```bat
:: 1. Freeze the app (onedir, not onefile -- see chaselate.spec for why)
.venv\Scripts\python.exe -m PyInstaller packaging\chaselate.spec --noconfirm ^
    --distpath packaging\dist --workpath packaging\build

:: 2. Wrap it into an installer
"C:\Program Files (x86)\NSIS\makensis.exe" packaging\installer.nsi
```

Output: `packaging\ChaselateSetup-<version>.exe`, roughly 89 MB (compressed from a ~320 MB
CPU-only frozen build -- the base install never bundles CUDA; that is what the optional
component in the installer downloads separately, on the target machine).

## What's in each piece

| File | Purpose |
|---|---|
| `chaselate.spec` | PyInstaller build spec. onedir mode, explicitly collects faster-whisper's bundled Silero VAD model and certifi's CA bundle (neither is visible to static import analysis), and **strips PyQt5's bundled MSVC runtime DLLs from the output** -- see the spec's own comment for why that one matters: without it, ctranslate2/onnxruntime fail with a DLL initialization error the first time either is used, not a normal crash. |
| `run_chaselate.py` | Thin entry point PyInstaller actually analyzes (`python -m chaselate` isn't something PyInstaller can target directly). |
| `installer.nsi` | The NSIS script: base install (required), optional CUDA component, post-install Ollama check, uninstaller. |
| `scripts/download_cuda.ps1` | Run by the CUDA component. Resolves whichever version PyPI currently reports as *latest* for `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` (no version pin to maintain), extracts just the DLLs into `<install dir>\nvidia\...`. Compares against a small manifest (`nvidia\.chaselate_cuda_versions.json`) left by the last run and skips re-downloading (~1.9 GB) when already current -- matters because installer.nsi re-runs this on every reinstall/repair/upgrade, not just the first install. Failure here is non-fatal -- the app still runs on CPU. |
| `scripts/check_ollama.ps1` | Run after install regardless of components chosen. Detects Ollama, compares its version against GitHub's latest release tag, and emits one line (`MISSING` / `OK:<v>` / `UPDATE:<old>:<new>` / `UNKNOWN:<reason>`) that `installer.nsi` turns into a native Yes/No prompt to open the download page. Never fails the install -- a network hiccup here just skips the prompt. |
| `scripts/make_icon.py` | Generates `assets/chaselate.ico` / `assets/chaselate.png`. Drawn programmatically (no external art), matching `chaselate/ui/overlay.py`'s own runtime-drawn tray icon -- same three-chevron mark, same palette as `chaselate/ui/style.py`'s dark theme. Re-run only if you want to change the design; the output is checked in, not regenerated on every build. |
| `assets/chaselate.ico` | The exe/installer icon `chaselate.spec` and `installer.nsi` both point at. |

## Known landmines (already worked around, but worth knowing about if this breaks again)

- **PyQt5 vs. ctranslate2/onnxruntime DLL conflict.** Covered above; if a frozen build starts
  logging "voice activity detector unavailable: DLL load failed..." or crashes outright the
  first time a model is touched, check that `_MSVC_RUNTIME_DLLS` in `chaselate.spec` is still
  stripping `msvcp140.dll` / `vcruntime140.dll` / `vcruntime140_1.dll` / `concrt140.dll` from
  `a.binaries` before `COLLECT`. This is the same root cause `chaselate/_runtime.py` works
  around for `python -m chaselate` runs; a frozen build's own bootstrap can import PyQt5
  before that Python-level fix gets a chance to run, so removing the bundled DLL entirely
  (forcing resolution to the System32 copy) is what actually fixes it here.
- **CUDA DLL discovery in a frozen build.** `chaselate.asr.ensure_cuda_libraries()` normally
  finds the `nvidia-*-cu12` wheels via `importlib.util.find_spec("nvidia")`, which only works
  for a real pip-installed package. `download_cuda.ps1` drops the DLLs at
  `<install dir>\nvidia\<lib>\bin\`, and `ensure_cuda_libraries()` has an explicit
  `sys.frozen` fallback that checks that exact path next to the running executable. If you
  change where the installer puts these DLLs, that fallback needs to change to match.
- **`SF_SELECTED` / component defaults.** The CUDA component is unchecked by default
  (`Section /o "..."`) given the ~1.9 GB download; don't flip that without a good reason, a
  lot of users don't have an NVIDIA GPU.
- **`Invoke-WebRequest`'s default progress bar is not just cosmetic overhead -- it can make a
  download nearly hang.** Confirmed the hard way: with stdout redirected (exactly how
  `nsExec::ExecToLog` runs `download_cuda.ps1`), a 528 MB download sat at 0 MB for 10+ minutes
  while the request thread visibly burned CPU on per-chunk progress updates, not network I/O.
  `download_cuda.ps1` sets `$ProgressPreference = "SilentlyContinue"` at the top for exactly
  this reason -- don't remove it, and set the same thing in any new script here that downloads
  anything non-trivial.
- **No version pin on the CUDA wheels, deliberately.** `download_cuda.ps1` always resolves
  PyPI's current *latest* `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` rather than a hand-maintained
  pinned version -- a deliberate choice to trade "guaranteed-tested combination" for "always
  current, no manual bump commits". If a future PyPI release turns out to be incompatible with
  the CTranslate2 build Chaselate ships, that surfaces at install time on a user's machine, not
  in testing here -- if that ever happens, the fix is back at the wheel-selection logic in this
  script, not somewhere in `chaselate/asr.py`.
- **`QColor(str)` cannot parse `rgba(r, g, b, a)`.** `chaselate/ui/style.py`'s palettes write
  translucent colours in that CSS function form because QSS (Qt's stylesheet language)
  understands it fine -- but PyQt5's plain `QColor(some_string)` constructor does not, and
  fails *silently* to an invalid, opaque-black colour instead of raising. Found the hard way:
  the light-theme tray icon rendered with a black background because `"plate"` is written
  that way for both themes. Anything drawing with `QPainter` that needs a real `QColor` from
  a palette entry should go through `chaselate.ui.style.qcolor()`, not `QColor()` directly.

## Testing a build

Don't skip this -- `packaging/dist/Chaselate/Chaselate.exe` can be run directly (no installer
needed) to sanity-check the freeze before wrapping it:

```bat
packaging\dist\Chaselate\Chaselate.exe --log-level DEBUG
```

Check `%APPDATA%\Chaselate\chaselate.log` for `Silero VAD loaded` (confirms the onnxruntime
fix took) and `Whisper ready: ...` (confirms faster-whisper/ctranslate2 loaded).

For the installer itself:

```bat
packaging\ChaselateSetup-0.1.0.exe /S    :: silent install, for scripting
packaging\ChaselateSetup-0.1.0.exe       :: normal interactive install
```

After installing, confirm it shows up under Settings > Apps (or the classic Control Panel
"Programs and Features") as "Chaselate", that the Start Menu shortcut launches it, and that
uninstalling removes the install directory, the Start Menu folder, and the registry entry.

**Verified**, end to end, via silent install/launch/uninstall on the machine this was built
on: 341 files land at `%LOCALAPPDATA%\Programs\Chaselate`, the `HKCU` uninstall entry carries
the right name/version/publisher/size, the launched exe stays running and logs real
recognition activity, and a silent uninstall removes the install directory, the registry key,
and the Start Menu shortcut completely. `scripts/download_cuda.ps1` was also run standalone
against a scratch directory twice in a row: the first run resolved and fetched both
latest-version wheels, extracted them, and landed `cublas64_*.dll`/`cudnn64_*.dll` at the
expected paths (1.9 GB, seconds once `$ProgressPreference` was fixed rather than 10+ minutes
stalled); the second run against the same directory correctly detected both packages already
matched latest and finished in well under a second without touching the network. The icon was
checked by extracting it back out of
both the built `Chaselate.exe` and `ChaselateSetup-<version>.exe` with
`[System.Drawing.Icon]::ExtractAssociatedIcon(...)` -- the one thing *not* separately verified
is clicking through the installer's own Components page to select CUDA interactively (the
silent-mode tests exercise the script directly, not NSIS's checkbox wiring around it).
