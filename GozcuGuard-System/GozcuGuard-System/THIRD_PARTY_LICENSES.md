# Third-party software

SafeCamera's own source code is MIT licensed (see `LICENSE`). It depends on the libraries below, which keep
their **own** licenses. The list was read from the installed packages' metadata on 2026-11; please re-check
the upstream projects before you publish anything, because licenses can change.

| Library | Version checked | License (as found) |
|---|---|---|
| NudeNet | 3.4.2 | Metadata says "MIT", but the `LICENSE` file shipped inside the package is the **GNU AGPL-3.0**. Treat it as AGPL-3.0 until the authors clarify. |
| pyvirtualcam | 0.15.0 | **GPL-2.0** |
| MediaPipe | 0.10.x | Apache-2.0 (model files are provided by Google and downloaded by `download_models.py`) |
| OpenCV (opencv-python) | 5.x | Apache-2.0 |
| ONNX Runtime | 1.x | MIT |
| customtkinter | 6.x | MIT |
| keyboard | 0.13 | MIT |
| psutil | 7.x | BSD-3-Clause |
| Pillow | 13.x | MIT-CMU (HPND) |
| NumPy | 2.x | BSD-3-Clause (plus bundled permissive licenses) |
| requests | 2.x | Apache-2.0 |

## What this means (not legal advice)

- This repository contains **only SafeCamera's own source code**. It does not contain the libraries or model files
  above; users install them with `pip` and `python download_models.py`, and each stays under its own license.
- **Be careful with binary releases.** A PyInstaller/Nuitka `.exe` *bundles* those libraries. Redistributing such
  a bundle triggers the GPL-2.0 / AGPL-3.0 conditions of pyvirtualcam and NudeNet (license texts, source offer,
  license compatibility of the combined work). GPL-2.0-only and AGPL-3.0 are generally considered incompatible
  with each other, so a single bundled `.exe` may not be distributable at all. If you want to publish an `.exe`,
  get proper legal advice first, or only share the source code and the install instructions in `README.md`.
- Running the app on your own computer is not a distribution and is not affected.
- "OBS", "Zoom", "Google Meet", "Microsoft Teams", "NudeNet" and "MediaPipe" are names of their respective
  owners. SafeCamera is an independent project and is not affiliated with or endorsed by them.
