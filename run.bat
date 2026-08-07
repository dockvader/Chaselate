@echo off
REM Launch Chaselate. Any arguments are passed through, e.g.:  run.bat --target ja --autostart
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

REM Start Ollama if it is installed but not yet listening. Without this the first
REM translation fails with "Cannot reach Ollama" on a fresh boot.
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul
if errorlevel 1 (
    where ollama >nul 2>&1
    if not errorlevel 1 (
        echo Starting Ollama in the background ...
        start "" /b ollama serve
    )
)

REM pythonw.exe keeps the console window from appearing behind the overlay.
start "Chaselate" ".venv\Scripts\pythonw.exe" -m chaselate %*
