# Downloads the two nvidia-*-cu12 wheels from PyPI and drops their DLLs next to the
# installed Chaselate.exe, matching the nvidia/<lib>/bin layout chaselate.asr.
# ensure_cuda_libraries() looks for (both in a normal venv and in a frozen build -- see that
# function's frozen-build fallback).
#
# Always resolves whatever PyPI currently reports as each package's *latest* version -- no
# pinned version list to maintain here. Before downloading anything, compares that latest
# version against a small manifest file left by the last successful run
# (nvidia\.chaselate_cuda_versions.json) and the DLLs it actually installed; if both already
# match latest, this is a fast no-op instead of a ~1.9 GB re-download, which matters because
# the installer re-runs this on every reinstall/repair/upgrade, not just the first install.
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
# Invoke-WebRequest's default progress bar renders per-chunk, which is fine in an interactive
# console but degrades to a near-crawl when stdout is redirected (exactly how installer.nsi's
# nsExec::ExecToLog runs this) -- confirmed the hard way: a 528 MB download sat at 0 MB for
# 10+ minutes with the request thread visibly burning CPU on progress updates, not the network.
$ProgressPreference = "SilentlyContinue"

$PackageNames = @("nvidia-cublas-cu12", "nvidia-cudnn-cu12")

function Write-Step($Message) {
    Write-Host "[CUDA] $Message"
}

$NvidiaDest = Join-Path $InstallDir "nvidia"
$ManifestPath = Join-Path $NvidiaDest ".chaselate_cuda_versions.json"

function Get-InstalledManifest {
    if (-not (Test-Path $ManifestPath)) { return @{} }
    try {
        $raw = Get-Content $ManifestPath -Raw | ConvertFrom-Json
        $result = @{}
        foreach ($prop in $raw.PSObject.Properties) { $result[$prop.Name] = $prop.Value }
        return $result
    } catch {
        # A corrupt/hand-edited manifest must not block re-downloading; treat as "nothing
        # recorded" rather than failing the whole component.
        return @{}
    }
}

function Save-Manifest($Versions) {
    $Versions | ConvertTo-Json | Set-Content -Path $ManifestPath -Encoding utf8
}

$TempRoot = Join-Path $env:TEMP ("chaselate-cuda-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

try {
    $installed = Get-InstalledManifest
    $newVersions = @{}
    $anyDownloaded = $false

    foreach ($name in $PackageNames) {
        Write-Step "Checking latest version of $name on PyPI..."
        try {
            $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$name/json" -TimeoutSec 30
        } catch {
            # PyPI unreachable: if we already have a working copy, keep it and move on rather
            # than failing the whole component over a transient network hiccup. If we have
            # nothing installed yet, there is nothing useful to fall back to.
            if ($installed[$name] -and (Get-ChildItem -Path $NvidiaDest -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1)) {
                Write-Step "Could not reach PyPI to check for updates; keeping the installed $name $($installed[$name])."
                $newVersions[$name] = $installed[$name]
                continue
            }
            throw "could not reach PyPI for $name`: $($_.Exception.Message)"
        }
        $latest = $meta.info.version

        $dllMarker = if ($name -eq "nvidia-cublas-cu12") { "cublas64_*.dll" } else { "cudnn64_*.dll" }
        $dllPresent = Get-ChildItem -Path $NvidiaDest -Filter $dllMarker -Recurse -ErrorAction SilentlyContinue

        if ($installed[$name] -eq $latest -and $dllPresent) {
            Write-Step "$name $latest already installed, up to date -- skipping."
            $newVersions[$name] = $latest
            continue
        }

        Write-Step "Installing $name $latest (installed: $(if ($installed[$name]) { $installed[$name] } else { 'none' }))..."
        $file = $meta.urls | Where-Object { $_.filename -match "win_amd64" } | Select-Object -First 1
        if (-not $file) {
            # Fall back to this version's own release metadata in case "latest" (project-wide)
            # and this version's own asset list differ in shape.
            $verMeta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$name/$latest/json" -TimeoutSec 30
            $file = $verMeta.urls | Where-Object { $_.filename -match "win_amd64" } | Select-Object -First 1
        }
        if (-not $file) {
            throw "no win_amd64 wheel found for $name $latest"
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
            throw "$name $latest did not unpack the expected nvidia/ folder"
        }
        Write-Step "Installing DLLs to $NvidiaDest ..."
        New-Item -ItemType Directory -Path $NvidiaDest -Force | Out-Null
        Copy-Item -Path (Join-Path $srcNvidia "*") -Destination $NvidiaDest -Recurse -Force

        $newVersions[$name] = $latest
        $anyDownloaded = $true
    }

    $cublasDll = Get-ChildItem -Path $NvidiaDest -Filter "cublas64_*.dll" -Recurse -ErrorAction SilentlyContinue
    if (-not $cublasDll) {
        throw "cublas64_*.dll not found after setup -- something about the wheel layout changed"
    }

    Save-Manifest $newVersions
    if ($anyDownloaded) {
        Write-Step "CUDA support installed successfully."
    } else {
        Write-Step "CUDA support already up to date, nothing to download."
    }
    exit 0
} catch {
    Write-Host "[CUDA] Setup failed: $($_.Exception.Message)"
    Write-Host "[CUDA] Chaselate will still work on CPU. Retry later from Settings, or see README.md."
    exit 1
} finally {
    Remove-Item -Path $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
