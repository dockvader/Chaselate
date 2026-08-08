# Chaselate — 專案守則

Windows 版即時語音翻譯字幕條。移植自 macOS 的 KazKozDev/live-translation。
管線：WASAPI loopback → Silero VAD → faster-whisper → Ollama → PyQt5 overlay。

## 怎麼跑

```powershell
# 永遠用專案 venv，不要用系統 python
.venv\Scripts\python.exe -m chaselate                      # GUI
.venv\Scripts\python.exe -m chaselate --log-level DEBUG    # 除錯（run.bat 用 pythonw，看不到錯誤）
.venv\Scripts\python.exe -m chaselate --list-devices       # 有哪些可擷取的裝置
.venv\Scripts\python.exe -m chaselate --list-models        # Ollama 有哪些模型
.venv\Scripts\python.exe -m pytest tests -q                # 測試
```

翻譯需要 Ollama 在跑（`ollama serve`）並已 `ollama pull translategemma`。

打包成安裝檔（PyInstaller 凍結＋NSIS）在 `packaging/`，細節見 `packaging/README.md`。
凍結後的 exe 有一個跟原始碼不一樣的坑：PyQt5 自帶的 msvcp140.dll 會被 PyInstaller 自動收集
進去，而且它的開機流程會搶在 `chaselate/_runtime.py` 的修正碼之前載入 PyQt5，所以
`chaselate.spec` 改用「打包時直接排除那幾個 DLL」而非「搶時間先載入系統版」。

## 怎麼驗證（這個專案沒有 CI，驗證靠手動實跑）

改動後**至少**跑對應那一層：

| 改了什麼 | 怎麼驗 |
|---|---|
| `textutils.py` / `languages.py` / `config.py` / `resample.py` | `pytest tests -q`，這些有單元測試覆蓋 |
| `audio/` | 播一段有語音的音訊，跑 `--list-devices` 確認裝置在，再實際 Start 看字幕出不出來 |
| `vad.py` | 看 log 的 segment `reason` 與長度；靜音時不該產生 segment |
| `asr.py` | 實跑一次並看 metrics 的 `asr_rtf`（<1 才跟得上實時） |
| `translate.py` | 用真 Ollama 跑一句，**特別確認 `zh-TW` 輸出是繁體**（見下方陷阱） |
| `ui/` | `QT_QPA_PLATFORM=offscreen` 手動 emit signal 跑 smoke，並實際開一次 GUI 目視 |

「程式碼寫好了」不算完成。要有實跑輸出或測試結果。

## 這個環境的既知陷阱（踩過，別再踩）

0. **最重要：`import chaselate` 必須早於 `import PyQt5`。**
   PyQt5 夾帶自己那份較舊的 `msvcp140.dll` / `vcruntime140.dll`（`PyQt5/Qt5/bin/`，
   590,112 bytes vs System32 的 643,512），import 時會把該目錄加進 DLL 搜尋路徑。
   ctranslate2 夾帶的 Intel OpenMP (`libiomp5md.dll`) 一旦連到那份舊 runtime，
   **建構模型時整個行程 access violation**（0xC0000005，Python 抓不到）。
   `chaselate/__init__.py` 會呼叫 `_runtime.pin_system_msvc_runtime()` 先把 System32 那份
   釘進行程，之後同名請求都解析到它。所以：
   - 寫任何腳本/測試，第一個 import 就是 `import chaselate`（或它的子模組）。
   - 順序錯了會印 warning（`PyQt5 was imported before chaselate`），看到就是要修順序。
   - 這個崩潰跟執行緒、CUDA、CPU/GPU **都無關**（全部組合都會崩），別往那些方向查。
   - `KMP_DUPLICATE_LIB_OK=TRUE` 沒有用，試過了。
   診斷紀錄與完整證據見 `chaselate/_runtime.py` 的 docstring。

1. **`onnxruntime` 必須 `<1.27`**。1.28.0 在本機 import 就炸（DLL 初始化失敗），
   而 Silero VAD 靠它，一炸 VAD 就整個失效。`requirements.txt` 已釘住上界。
2. **CUDA DLL 要靠 PATH，不是 `os.add_dll_directory`**。CTranslate2 用原生
   `LoadLibrary("cublas64_12.dll")`，只搜 PATH。`asr.py:ensure_cuda_libraries` 負責前置
   nvidia wheel 的目錄。改動 asr.py 的載入順序時不要把它移到 GPU 操作之後。
3. **RTX 50 系首次 GPU 推論要 ~15 秒**（driver JIT sm_120 kernel），之後才快。
   看到「第一次很慢」不要當成 bug 去修。
4. **`zh-TW` / `zh-CN` 在 `languages.py` 是獨立語言，不可併回 `zh`**。
   併回去 prompt 會寫 "Chinese"，translategemma 就輸出簡體——對使用者是錯答案。
   `whisper_code()` 才做 `zh-TW → zh` 的降級（Whisper 沒有這個 token）。
5. **`sounddevice` 不支援 WASAPI loopback**，別想改用它。可用的是 `soundcard`（主）
   與 `pyaudiowpatch`（備），兩者都實測過。
6. WASAPI loopback 錄的是**特定輸出裝置**。Windows 在放耳機、程式錄喇叭 → 全靜音。
   查「沒聲音」先確認裝置對不對。
7. **不要在擷取執行緒自己呼叫 `CoInitializeEx`**。試過，結果 soundcard 自己的
   `CoInitializeEx` 回 `S_FALSE`，而它只容忍 `RPC_E_CHANGED_MODE`，於是列舉整個失敗、
   `_COMLibrary.__del__` 一路噴 AttributeError，並靜默降級到 pyaudiowpatch。
   兩個後端都會自己管 COM，別插手。見 `audio/capture.py` 該處註解。
8. **結束前要排空 QThreadPool**。設定對話框的探測（裝置列舉、Ollama 查詢）跑在
   global thread pool；解譯器在 worker 還在 WASAPI/COM 裡面時開始拆解會崩（0xC0000409）。
   `OverlayWindow.quit()` 與 `__main__.teardown()` 都會呼叫 `wait_for_background_jobs()`，
   新增退出路徑時別漏掉。

## 目錄裡什麼別亂碰

- **`audio/resample.py` 的 `StreamResampler.process`**：區塊邊界的 phase/history 進位很細，
  改壞了症狀是音訊有 click 或 ASR 準確度莫名下降。有區塊不變性測試（chunked 必須逐樣本
  等於 one-shot），改完一定跑。`i1 = minimum(i0+1, size-1)` 那行是修過的 off-by-one，
  整數比例（48k→16k）每個區塊都會踩到，不要「簡化」掉。
- **`pipeline.py` 的四執行緒/佇列結構**：音訊 callback 裡只能 `put_nowait`。
  把 VAD 或 ASR 搬進 callback 會讓 WASAPI 掉樣本（爆音）。
- **佇列滿時丟最舊**是刻意的（即時字幕要新不要全），不是 bug。丟棄有計數並回報到 metrics。
- `.venv/`、`%APPDATA%\Chaselate\`（config 與 log）不進 git。

## 風格

- 註解與 docstring 用英文；只寫「為什麼」和不明顯的約束，不要逐行解說。
- 對外訊息（錯誤、狀態）要可行動：說出下一步該做什麼（例如 "Run: ollama pull X"）。
- 例外處理不要吞掉錯誤又不回報；callback 內的例外要 log 但不能弄死執行緒。
