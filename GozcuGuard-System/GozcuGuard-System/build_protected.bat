@echo off
REM OPTIONAL and UNTESTED: compiles the app to native code with Nuitka (harder to decompile than PyInstaller).
REM Needs a C compiler (Nuitka offers to download MinGW on first run). The build can take 10+ minutes.
python -m venv .venv || exit /b 1
call .venv\Scripts\activate
pip install -r requirements.txt nuitka || exit /b 1
python download_models.py
set PYTHONPATH=src
python -m nuitka --onefile --windows-console-mode=disable --enable-plugin=tk-inter ^
  --include-package=guvenli_kamera --include-package-data=nudenet --include-package-data=customtkinter ^
  --include-package=mediapipe --include-data-dir=assets=assets ^
  --output-filename=SafeCamera.exe run.py
