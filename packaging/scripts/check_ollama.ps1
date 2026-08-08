# Checks whether Ollama is installed and, if so, whether it is up to date. Pure detection --
# no installing, no popups. Emits exactly one line to stdout for installer.nsi to parse (via
# nsExec::ExecToStack) and act on with native NSIS message boxes, since NSIS controls the
# actual user-facing prompts and the "open the download page" follow-up.
#
# Output contract (always exactly one line, always exit 0 -- this must never fail the install):
#   MISSING                     Ollama not found on PATH or in its default install location.
#   OK:<version>                Installed and (as far as this check could tell) current.
#   UPDATE:<installed>:<latest> An update is available.
#   UNKNOWN:<reason>            Installed, but the check could not be completed (offline,
#                                GitHub API rate-limited, unparseable output, etc.) -- treated
#                                as "don't bother the user", not as "missing".

$ErrorActionPreference = "Stop"

function Find-OllamaExe {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $default) { return $default }
    return $null
}

try {
    $exe = Find-OllamaExe
    if (-not $exe) {
        Write-Output "MISSING"
        exit 0
    }

    $raw = & $exe --version 2>&1 | Out-String
    # Typical output: "ollama version is 0.31.2" (client) sometimes followed by a server-version
    # line; the client line is what matters here since that's what an update would replace.
    $match = [regex]::Match($raw, "(\d+\.\d+\.\d+)")
    if (-not $match.Success) {
        Write-Output "UNKNOWN:could not parse 'ollama --version' output"
        exit 0
    }
    $installed = $match.Groups[1].Value

    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ollama/ollama/releases/latest" `
            -TimeoutSec 10 -Headers @{ "User-Agent" = "Chaselate-Installer" }
        $latest = $release.tag_name -replace "^v", ""
    } catch {
        # Offline, rate-limited, GitHub down -- an installed Ollama is not a problem worth
        # surfacing over a check that itself failed.
        Write-Output "UNKNOWN:update check failed: $($_.Exception.Message)"
        exit 0
    }

    if ([version]$latest -gt [version]$installed) {
        Write-Output "UPDATE:${installed}:${latest}"
    } else {
        Write-Output "OK:$installed"
    }
    exit 0
} catch {
    Write-Output "UNKNOWN:$($_.Exception.Message)"
    exit 0
}
