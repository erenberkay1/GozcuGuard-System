# SafeCamera

**Created by Eren Berkay Gün**

SafeCamera sits between your webcam and your video-call app (Zoom, Meet, Teams). It can

- **blur nudity** automatically (optional),
- **blur your background** (optional),
- **put a virtual shirt or blazer on you** (optional, lightweight 2D overlay).

Everything runs locally on your computer. Nothing is recorded, uploaded or sent anywhere.

## You are in control

On first start SafeCamera asks which protections you want. You can change them any time in the main window:

| Protection | What it does | Default |
|---|---|---|
| **Nudity blur** | Detects nudity with NudeNet and blurs it (region, whole frame, cooldown or protection screen — see *Safety Policy*). | Recommended, you are asked |
| **Background blur** | Keeps you sharp and blurs everything behind you. | Off |

While *Nudity blur* is on, a failure of the detection model never lets the raw camera image through: the camera shows a protection screen instead.

## Install and run (Windows 10/11)

1. Install **Python 3.12** (python.org, tick "Add Python to PATH") and **OBS Studio**. Open OBS once, then close it (this installs the *OBS Virtual Camera* driver).
2. Open **Command Prompt** (cmd) and run:

```
git clone https://github.com/YOUR-NAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python run.py
```

3. In Zoom/Meet/Teams choose **OBS Virtual Camera** as the camera.

Next time: `cd YOUR-REPO`, `.venv\Scripts\activate`, `python run.py`.

Shortcuts: **F11** full screen, **Esc** leave full screen. The red **Panic Key** button blacks out the camera instantly.

`python download_models.py` downloads the MediaPipe models (background blur and outfit placement). The nudity model is installed with `pip`.

## Build an .exe (read the legal note first)

`build.bat` builds `SafeCamera.exe` with PyInstaller for **your own use**. Do **not** upload that `.exe` to GitHub
Releases or anywhere else without reading `THIRD_PARTY_LICENSES.md`: the bundle contains GPL-2.0 and AGPL-3.0
libraries. The safe way to share SafeCamera is the source code plus the install steps above.

## Performance tips

- The status bar shows `FPS (work X ms, camera wait Y ms, WxH)`. If **camera wait** is large (e.g. 60+ ms), your
  webcam itself is the limit (dim room = webcams lower their frame rate, or the USB port is slow): add light and use
  a USB 3 port. If **work** is large, lower **Quality (speed)** to *Balanced* or *Fast*.
- Turn off features you do not need. Background blur and outfit overlay each cost CPU time.
- Close other programs that use the camera or the CPU (browser tabs with video calls, games).
- SafeCamera requests MJPG from the webcam, which fixes the common "10 FPS at 720p" problem of uncompressed USB video.

## Honest limitations

- Nudity detection is **not 100% reliable**. Treat it as an extra safety net, not a guarantee.
- The virtual outfit is a 2D overlay. It can look pasted on during fast movement or when your arms cross it.
- Without MediaPipe (e.g. very new Python versions) background blur and outfit placement use a rougher fallback.
- If you use the outfit feature in an interview, consider telling the other side.
- Antivirus/SmartScreen may warn about the unsigned .exe.

## License

MIT License. Copyright (c) 2026 Eren Berkay Gün. The copyright and attribution notice must be kept in copies of the software.
See `THIRD_PARTY_LICENSES.md` for the libraries used (please verify model licenses before redistributing).


## Author credit and protection
SafeCamera is **Created by Eren Berkay Gün**. The credit is shown in the window title, the footer and the
welcome dialog, and it is verified with a SHA-256 check; a copy with a removed or edited credit shows a warning
and will not start the camera.

Honest note: no Python program can be made impossible to copy or decompile. The checks above only make casual
removal harder. The real protection is the MIT license (the copyright and attribution notice must stay in all
copies) plus compiling with Nuitka instead of PyInstaller (`build_protected.bat`, slower to build, much harder
to decompile).

## Legal and copyright

- SafeCamera's own code is released under the MIT license (`LICENSE`). **Created by Eren Berkay Gün.**
- Third-party libraries keep their own licenses, see `THIRD_PARTY_LICENSES.md`.
- This repository intentionally contains **no third-party images, logos or model files**. The built-in outfit
  templates are drawn by the program itself.
- Garment images you load are your responsibility: only use images you made yourself or have the right to use,
  and avoid brand logos. AI-generated images may have unclear ownership.
- SafeCamera is an independent project and is not affiliated with or endorsed by OBS Project, Zoom, Google,
  Microsoft or any other company. Product names are used only to describe compatibility.
- The software is provided "as is", without warranty (see `LICENSE`). Nudity detection is not guaranteed.
