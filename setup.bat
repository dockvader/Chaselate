@echo off
REM One-time setup: creates .venv, installs dependencies, offers the GPU extras.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Chaselate setup
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python is not on PATH.
    echo Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Using Python !PYVER!
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

set PY=.venv\Scripts\python.exe

echo.
echo Upgrading pip ...
"%PY%" -m pip install --upgrade pip --quiet

echo.
echo Installing dependencies ^(this downloads a few hundred MB^) ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency installation failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Optional: NVIDIA GPU acceleration
echo ============================================================
echo Speech recognition runs about three times faster on an NVIDIA GPU.
echo This downloads roughly 1.9 GB of CUDA libraries. Skip it to use the CPU
echo (still fast enough for live speech), you can install it later with:
echo     .venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
echo.
set /p CUDA="Install GPU support now? [y/N] "
if /i "!CUDA!"=="y" (
    "%PY%" -m pip install -r requirements-cuda.txt
    if errorlevel 1 echo WARNING: GPU libraries failed to install; CPU will be used.
)

echo.
echo ============================================================
echo  Checking for Ollama
echo ============================================================
where ollama >nul 2>&1
if errorlevel 1 (
    echo Ollama is NOT installed. Translation needs it.
    echo Download it from https://ollama.com/download  then run:
    echo     ollama pull translategemma
) else (
    echo Ollama found. Checking for the default translation model ...
    ollama list 2>nul | findstr /i "translategemma" >nul
    if errorlevel 1 (
        echo.
        echo The default model 'translategemma' is not installed ^(3.3 GB^).
        set /p PULL="Pull it now? [y/N] "
        if /i "!PULL!"=="y" ollama pull translategemma
    ) else (
        echo translategemma is already installed.
    )
)

echo.
echo Verifying the installation ...
"%PY%" -m chaselate --list-devices
echo.
echo ============================================================
echo  Setup complete. Start Chaselate with:  run.bat
echo ============================================================
endlocal
