# Downloads the two nvidia-*-cu12 wheels from PyPI and drops their DLLs next to the
# installed Chaselate.exe, matching the nvidia/<lib>/bin layout chaselate.asr.
# ensure_cuda_libraries() looks for (both in a normal venv and in a frozen build -- see that
# function's frozen-build fallback).
#
# Run by installer.nsi's optional "GPU Acceleration" component via nsExec::ExecToLog, so
# everything here is Write-Host progress output + a numeric exit code; there is no interactive
# UI. Never throws past its own boundary: CUDA is optional, so any failure here should leave
# the base (CPU) install intact and working, not abort the installer.
#
# Usage: download_cuda.ps1 -InstallDir "C:\Program Files\Chaselate"

param(
    [Parameter(Mandatory = $true)]
    [string]$InstallDir
)

$ErrorActionPreference = "Stop"

# Pinned rather than "latest": these exact versions were verified against faster-whisper /
# CTranslate2 on an RTX 5080 (see chaselate/CLAUDE.md). Bumping them is a deliberate,
# re-tested decision, not something that should silently drift because PyPI published a
# newer release between one user's install and another's.
$Packages = @(
    @{ Name = "nvidia-cublas-cu12"; Version = "12.9.2.10" },
    @{ Name = "nvidia-cudnn-cu12";  Version = "9.24.0.43" }
)

function Write-Step($Message) {
    Write-Host "[CUDA] $Message"
}

$TempRoot = Join-Path $env:TEMP ("chaselate-cuda-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

try {
    $NvidiaDest = Join-Path $InstallDir "nvidia"

    foreach ($pkg in $Packages) {
        $name = $pkg.Name
        $version = $pkg.Version
        Write-Step "Looking up $name $version on PyPI..."

        $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$name/$version/json" -TimeoutSec 30
        $file = $meta.urls | Where-Object { $_.filename -match "win_amd64" } | Select-Object -First 1
        if (-not $file) {
            throw "no win_amd64 wheel found for $name $version"
        }

        $wheelPath = Join-Path $TempRoot $file.filename
        Write-Step "Downloading $($file.filename) ($([math]::Round($file.size / 1MB)) MB)..."
        Invoke-WebRequest -Uri $file.url -OutFile $wheelPath -TimeoutSec 600

        # A .whl is a zip file; Expand-Archive needs a .zip extension to cooperate.
        $zipPath = [System.IO.Path]::ChangeExtension($wheelPath, ".zip")
        Rename-Item -Path $wheelPath -NewName (Split-Path $zipPath -Leaf)
        $extractDir = Join-Path $TempRoot ($name -replace "[^\w]", "_")
        Write-Step "Extracting..."
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        # Each wheel unpacks a top-level nvidia\<lib>\bin\*.dll tree; copy just that, merged
        # into one shared nvidia\ folder at the install root so both packages' DLLs sit
        # together the same way they do in a normal pip-installed venv.
        $srcNvidia = Join-Path $extractDir "nvidia"
        if (-not (Test-Path $srcNvidia)) {
            throw "$name $version did not unpack the expected nvidia/ folder"
        }
        Write-Step "Installing DLLs to $NvidiaDest ..."
        New-Item -ItemType Directory -Path $NvidiaDest -Force | Out-Null
        Copy-Item -Path (Join-Path $srcNvidia "*") -Destination $NvidiaDest -Recurse -Force
    }

    $cublasDll = Get-ChildItem -Path $NvidiaDest -Filter "cublas64_*.dll" -Recurse -ErrorAction SilentlyContinue
    if (-not $cublasDll) {
        throw "cublas64_*.dll not found after extraction -- something about the wheel layout changed"
    }

    Write-Step "CUDA support installed successfully."
    exit 0
} catch {
    Write-Host "[CUDA] Setup failed: $($_.Exception.Message)"
    Write-Host "[CUDA] Chaselate will still work on CPU. Retry later from Settings, or see README.md."
    exit 1
} finally {
    Remove-Item -Path $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
