# Chaselate

Live speech translation overlay for **Windows**. Captures whatever your speakers are
playing, transcribes it, translates it with a local model, and shows both texts in a
floating always-on-top caption bar.

Everything runs on your machine. No API keys, no accounts, nothing leaves the computer.

```
System audio (WASAPI loopback) -> Silero VAD -> faster-whisper -> Ollama -> PyQt5 overlay
```

This is a Windows port of [KazKozDev/live-translation](https://github.com/KazKozDev/live-translation),
which is macOS/Apple-silicon only. The pipeline design is the same; three layers had to be
replaced outright.

| | Original (macOS) | Chaselate (Windows) |
|---|---|---|
| Audio capture | BlackHole virtual device, manual Multi-Output setup | **WASAPI loopback** — no virtual cable, nothing to configure |
| Speech recognition | MLX Whisper (Apple silicon only) | **faster-whisper** / CTranslate2, CUDA or CPU |
| Interface | PyObjC + Cocoa | **PyQt5** |
| Translation | Ollama | Ollama, **with model selection at runtime** |

The audio change is the one you feel: on macOS you have to install BlackHole and build a
Multi-Output Device before anything works. WASAPI exposes a loopback endpoint for every
output device directly, so Chaselate just records the speakers.

## Requirements

- Windows 10 or 11
- Python 3.10–3.12
- [Ollama](https://ollama.com/download) for translation
- Optional: an NVIDIA GPU (about 3× faster recognition; CPU is fine for live speech)

## Install

```bat
setup.bat
```

That creates `.venv`, installs dependencies, offers the GPU extras, and offers to pull the
default translation model. Then:

```bat
run.bat
```

Doing it by hand instead:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt   :: optional, NVIDIA only
ollama pull translategemma
.venv\Scripts\python.exe -m chaselate
```

## Usage

Press **Start** (or `Ctrl+Space`), then play something with speech in it.

```bat
run.bat --target ja --autostart              :: translate into Japanese, start immediately
run.bat --target en --source zh --whisper medium
run.bat --no-translate                       :: transcribe only, no LLM
run.bat --source-type mic                    :: caption your own microphone instead
```

### Keyboard

| | |
|---|---|
| `Ctrl+Space` | start / stop |
| `Ctrl+,` | settings |
| `Ctrl+H` | show/hide the original text |
| `Ctrl+C` | copy all captions |
| `Ctrl++` / `Ctrl+-` | font size |
| `Esc` | hide to the system tray |
| `Ctrl+Q` | quit |

Closing the window hides it to the tray; quit from the tray menu or with `Ctrl+Q`.

### Diagnostics

```bat
.venv\Scripts\python.exe -m chaselate --list-devices     :: what can be captured
.venv\Scripts\python.exe -m chaselate --list-models      :: what Ollama has pulled
.venv\Scripts\python.exe -m chaselate --list-languages   :: language codes
```

`run.bat` uses `pythonw.exe`, which has no console — so startup errors are invisible. When
something is wrong, run it with the console attached and watch the output:

```bat
.venv\Scripts\python.exe -m chaselate --log-level DEBUG
```

A rotating log is also written to `%APPDATA%\Chaselate\chaselate.log`.

## Models

### Translation

The default is [translategemma](https://ollama.com/library/translategemma) (Gemma 3 tuned
for translation across 55 languages). Any model Ollama has pulled can be chosen from the
Translation tab, or with `--ollama-model`:

```bat
ollama pull translategemma:12b     :: 8.1 GB, better quality
ollama pull qwen3:8b               :: a general model works too
run.bat --ollama-model qwen3:8b
```

Prompts follow TranslateGemma's published format exactly, including naming both the language
and its code, because that is what the model was tuned on.

**Context.** By default the previous 2 sentence pairs are replayed as chat turns, which is
what lets pronouns and dropped subjects resolve correctly — the difference between
translating a sentence and translating a conversation. Set `--context-sentences 0` for
strict one-sentence-at-a-time behaviour.

### Recognition

| Model | Size | Notes |
|---|---|---|
| `tiny`, `base` | 75 MB / 145 MB | fast, noticeably worse |
| `small` | 460 MB | **default**, good balance |
| `medium` | 1.5 GB | better on accents and noise |
| `large-v3` | 3 GB | best, GPU strongly recommended |
| `large-v3-turbo` | 1.6 GB | close to large-v3, much faster |

Measured here on an RTX 5080 with `small`, transcribing a 13-second clip:

| | realtime factor |
|---|---|
| CPU, `int8` | 0.25 |
| CUDA, `float16` | 0.076 |

Both keep up with live speech (anything below 1.0 does). The first GPU call is slow — see
"first run is slow" below.

## Language codes

`zh-TW`, `zh-CN` and `zh-HK` are **separate languages** here, not aliases of `zh`. Asking for
`zh` gets you whatever the model prefers, which in practice is Simplified; asking for `zh-TW`
puts "Traditional Chinese" in the prompt and gets you Traditional. Use the specific code.

Whisper has no token for these variants (or for Cantonese), so the audio side falls back to
`zh` automatically while the translation side keeps the specific code.

## Settings

Everything is adjustable in the settings dialog (`Ctrl+,`) and persisted to
`%APPDATA%\Chaselate\config.json`. The ones worth knowing about:

| Setting | Effect |
|---|---|
| `min_silence_ms` (700) | Trailing silence that ends an utterance. Lower = captions appear sooner but break mid-sentence more. |
| `speech_pad_ms` (200) | Audio kept either side of detected speech. Too low clips leading consonants and Whisper starts guessing. |
| `max_segment_s` (14) | Hard cut for a speaker who never pauses. The next segment overlaps the cut so no audio is lost. |
| `silence_rms` (0.0009) | Below this, a block is silence and the VAD is not consulted. Raise it if a noisy room produces phantom captions. |
| `context_sentences` (2) | Previous pairs given to the translator for continuity. |
| `mouse_transparent` | Clicks pass through the overlay. Re-enable from the tray menu, since the toolbar stops responding. |

Command-line flags override the saved config for one run; add `--save` to persist them.

## Troubleshooting

**Nothing is captured / captions never appear.** Check `--list-devices` shows a `loopback`
device with `*` next to it. WASAPI loopback records a *specific* output device, so if Windows
is playing through headphones and Chaselate is recording the speakers you get silence. Pick
the right one in Settings → Audio, or change the Windows default output.

**"Cannot reach Ollama".** Start it: `ollama serve`. Confirm with `--list-models`.

**"Model 'translategemma:latest' is not installed".** `ollama pull translategemma`

**First run is slow.** Three separate one-time costs: Whisper downloads its weights
(460 MB for `small`), Ollama loads 3.3 GB into VRAM (this alone took 94 s here on a cold
start), and on a GPU newer than the shipped CUDA kernels the driver JIT-compiles them (~15 s
on an RTX 5080). All three are cached. Chaselate does the model loading during startup rather
than on the first utterance, so the status line tells you what it is waiting on. Steady-state
latency after that is about 2–3 s per sentence.

**Captions lag further and further behind.** The status bar shows queue depths; when
recognition cannot keep up the pipeline drops the *oldest* audio and reports it, rather than
growing a backlog forever. Use a smaller Whisper model or enable the GPU.

**`ImportError: DLL load failed while importing onnxruntime_pybind11_state`.** onnxruntime
1.28.0 does not load on this platform, and it takes the voice activity detector with it.
`requirements.txt` pins `<1.27` for exactly this reason. Fix with:

```bat
.venv\Scripts\python.exe -m pip install "onnxruntime<1.27"
```

**`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`.** The GPU
libraries are missing: `pip install -r requirements-cuda.txt`. If they *are* installed and
you still see this, something is stripping `PATH` — Chaselate prepends the wheel directories
at startup because CTranslate2 loads cuBLAS with a plain `LoadLibrary`, which searches `PATH`
but ignores `os.add_dll_directory`. Setting `--asr-device cpu` sidesteps it entirely.

**The process dies instantly with an access violation when recognition starts.** Something
imported PyQt5 before Chaselate. PyQt5 bundles an older Visual C++ runtime and puts it on the
DLL search path; CTranslate2's bundled Intel OpenMP then binds to it and the process is killed
outright — no Python traceback, and it happens on CPU as well as GPU. `chaselate/__init__.py`
prevents this by loading the `System32` copies first, which only works if `chaselate` is
imported first. If you are writing your own script, make `import chaselate` the first import;
the log will warn you (`PyQt5 was imported before chaselate`) when it is too late. Running via
`run.bat` or `python -m chaselate` always gets the order right.

**Whisper invents "Thanks for watching" over music.** Known Whisper behaviour on non-speech
audio. Chaselate filters the common stock phrases and collapses repetition loops; raise
`silence_rms` or `vad.threshold` if music still gets through.

## Layout

```
chaselate/
  __main__.py      CLI, logging, application startup
  _runtime.py      pins the system C++ runtime before PyQt5 loads its own (read this one)
  config.py        settings tree, JSON persistence
  languages.py     language table (codes, names, script properties)
  textutils.py     overlap dedup, sentence splitting, hallucination filter, output cleanup
  vad.py           Silero VAD wrapper and the utterance segmenter
  asr.py           faster-whisper wrapper, device selection, CUDA library discovery
  translate.py     Ollama client, prompt construction, streaming
  pipeline.py      the four worker threads and their queues
  audio/
    devices.py     device enumeration across both backends
    capture.py     the capture thread
    resample.py    anti-aliased streaming resampler to 16 kHz
  ui/
    overlay.py     the caption window
    settings.py    settings dialog
    style.py       stylesheet and palette
tests/             unit tests (no audio hardware or network needed)
```

## Licence

MIT, matching the upstream project.
