@echo off
REM WARNING: the .exe bundles GPL-2.0 / AGPL-3.0 libraries. Do not redistribute it before reading THIRD_PARTY_LICENSES.md
REM SafeCamera - builds SafeCamera.exe (PyInstaller) and copies it to the Desktop
python -m venv .venv || exit /b 1
call .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt || exit /b 1
python download_models.py
set PYTHONPATH=src
python -m pytest tests -q || exit /b 1
pyinstaller --noconfirm --onefile --windowed --name SafeCamera --paths src ^
  --collect-data nudenet --collect-all mediapipe --collect-data customtkinter ^
  --add-data "assets;assets" run.py || exit /b 1
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set DESK=%%D
copy /Y dist\SafeCamera.exe "%DESK%\SafeCamera.exe"
echo Done: %DESK%\SafeCamera.exe
