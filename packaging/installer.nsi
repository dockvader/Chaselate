; Chaselate installer.
;
; Wraps the PyInstaller onedir build (packaging\dist\Chaselate\) into a self-contained .exe
; installer: Start Menu shortcut, registers in "Add or Remove Programs", clean uninstall.
; No code signing -- Windows SmartScreen will show an "unknown publisher" prompt on first run;
; that is expected and was a deliberate scope decision (see packaging/README.md).
;
; Build (from the project root):
;   .venv\Scripts\python.exe -m PyInstaller packaging\chaselate.spec --noconfirm ^
;       --distpath packaging\dist --workpath packaging\build
;   "C:\Program Files (x86)\NSIS\makensis.exe" packaging\installer.nsi
;
; Produces packaging\ChaselateSetup-<version>.exe.

!define PRODUCT_NAME "Chaselate"
!define PRODUCT_VERSION "0.1.0"
!define PRODUCT_PUBLISHER "Chaselate"
!define PRODUCT_WEB_SITE "https://github.com/dockvader/Chaselate"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

Name "${PRODUCT_NAME}"
OutFile "ChaselateSetup-${PRODUCT_VERSION}.exe"
; Per-user install (no admin/UAC): nothing here needs machine-wide installation -- no service,
; no driver, just files plus a Start Menu shortcut. HKCU still shows up in Windows' unified
; "Add or Remove Programs" / Settings > Apps list exactly the same as an HKLM entry does; this
; is the same pattern VS Code's default per-user installer uses.
InstallDir "$LOCALAPPDATA\Programs\${PRODUCT_NAME}"
InstallDirRegKey HKCU "${UNINST_KEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma

; ---------------------------------------------------------------------------------------------
; Utility includes (must precede any use of the macros they define, below)
; ---------------------------------------------------------------------------------------------

!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "WordFunc.nsh"
!insertmacro GetSize

; ---------------------------------------------------------------------------------------------
; UI
; ---------------------------------------------------------------------------------------------

!include "MUI2.nsh"

!define MUI_ICON "assets\chaselate.ico"
!define MUI_UNICON "assets\chaselate.ico"
!define MUI_ABORTWARNING
!define MUI_UNABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\Chaselate.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch Chaselate now"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "TradChinese"

; ---------------------------------------------------------------------------------------------
; Base install (always runs)
; ---------------------------------------------------------------------------------------------

Section "Chaselate (required)" SEC_BASE
    SectionIn RO  ; not deselectable in the components list

    SetOutPath "$INSTDIR"
    File /r "dist\Chaselate\*.*"

    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\Chaselate.exe"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

    WriteUninstaller "$INSTDIR\Uninstall.exe"

    WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
    WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\Chaselate.exe"
    WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
    WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" "$\"$INSTDIR\Uninstall.exe$\" /S"
    WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
    WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1

    ; EstimatedSize is in KB and is what Windows shows in Settings > Apps; Add/Remove Programs
    ; otherwise has no idea how big an NSIS install is (unlike a real MSI, which tracks this
    ; itself from its file table).
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

; ---------------------------------------------------------------------------------------------
; Optional: GPU acceleration
; ---------------------------------------------------------------------------------------------

Section /o "GPU acceleration (CUDA, downloads ~1.9 GB)" SEC_CUDA
    SetOutPath "$INSTDIR"
    File "scripts\download_cuda.ps1"

    DetailPrint "Downloading CUDA libraries from PyPI -- this can take a while..."
    nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\download_cuda.ps1" -InstallDir "$INSTDIR"'
    Pop $0
    Delete "$INSTDIR\download_cuda.ps1"

    ${If} $0 != 0
        ; /SD IDOK: in a silent (/S) install there is no one to click this, so answer it
        ; automatically rather than hanging the installer waiting for input that never comes.
        MessageBox MB_OK|MB_ICONEXCLAMATION|MB_DEFBUTTON1 \
            "GPU acceleration setup did not complete (see the install log above for details).$\r$\n$\r$\n\
            Chaselate is still fully usable on CPU -- Settings > Recognition lets you pick the \
            processing device any time, and you can retry this from Settings if you get CUDA \
            working manually later." \
            /SD IDOK
    ${EndIf}
SectionEnd

; ---------------------------------------------------------------------------------------------
; Post-install: best-effort Ollama presence/update check. Runs regardless of which optional
; components were selected -- translation needs Ollama either way.
; ---------------------------------------------------------------------------------------------

Section "-Post"
    SetOutPath "$INSTDIR"
    File "scripts\check_ollama.ps1"

    DetailPrint "Checking for Ollama..."
    nsExec::ExecToStack '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\check_ollama.ps1"'
    Pop $0  ; exit code (always 0 by the script's own contract; not acted on)
    Pop $1  ; captured stdout
    Delete "$INSTDIR\check_ollama.ps1"

    ; $1 is one line: MISSING | OK:<v> | UPDATE:<old>:<new> | UNKNOWN:<reason>. OK and UNKNOWN
    ; are deliberately silent -- see check_ollama.ps1's contract comment for why.
    ${WordFind} "$1" ":" "+1" $2  ; the part before the first ":" (or the whole string if none)

    ; /SD IDNO on both: a silent/unattended install has no one to answer these, so default to
    ; not popping a browser open on someone's machine mid-automation.
    ${If} $2 == "MISSING"
        MessageBox MB_YESNO|MB_ICONQUESTION \
            "Chaselate translates using a local Ollama server, which is not installed.$\r$\n$\r$\n\
            Open the Ollama download page now?" \
            /SD IDNO IDYES open_ollama_missing IDNO ollama_done
        open_ollama_missing:
            ExecShell "open" "https://ollama.com/download/windows"
        ollama_done:
    ${ElseIf} $2 == "UPDATE"
        MessageBox MB_YESNO|MB_ICONINFORMATION \
            "An Ollama update is available ($1).$\r$\n$\r$\nOpen the download page now?" \
            /SD IDNO IDYES open_ollama_update IDNO update_done
        open_ollama_update:
            ExecShell "open" "https://ollama.com/download/windows"
        update_done:
    ${EndIf}
SectionEnd

; ---------------------------------------------------------------------------------------------
; Component descriptions (shown on the components page as each item is highlighted)
; ---------------------------------------------------------------------------------------------

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SEC_BASE} "The application itself. Runs on CPU out of the box; keeps up with live speech (faster-whisper 'small', ~0.25x realtime)."
    !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CUDA} "NVIDIA GPU support for faster recognition (roughly 3x on a recent card). Downloads nvidia-cublas-cu12 and nvidia-cudnn-cu12 directly from PyPI during install; skip this if you don't have an NVIDIA GPU, or add it later by re-running this installer."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ---------------------------------------------------------------------------------------------
; Uninstaller
; ---------------------------------------------------------------------------------------------

Section "Uninstall"
    RMDir /r "$INSTDIR"
    RMDir /r "$SMPROGRAMS\${PRODUCT_NAME}"
    DeleteRegKey HKCU "${UNINST_KEY}"

    ; /SD IDNO: silent uninstall keeps user data by default rather than hanging on a prompt
    ; nobody is there to answer -- the safer default when deletion can't be confirmed.
    MessageBox MB_YESNO|MB_ICONQUESTION \
        "Also delete settings and logs ($APPDATA\Chaselate)?$\r$\n\
        (Whisper model weights Chaselate downloaded on first run live in your Hugging Face \
        cache and are not affected either way.)" \
        /SD IDNO IDNO skip_appdata
        RMDir /r "$APPDATA\Chaselate"
    skip_appdata:
SectionEnd
