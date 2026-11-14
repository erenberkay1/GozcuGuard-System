## 1.0.0
- Two independent, opt-in blur systems: nudity blur and background blur (first-run welcome dialog + switches)
- Nudity detection only loads/runs while its switch is on; fail-closed (protection screen) while it is on
- Whole interface in English; "Created by Eren Berkay Gün" credit in the window title, footer and welcome dialog
- Fixed: undefined toggle variables (app did not start), broken `blur.py` import, template name mismatch
- Background blur falls back to an approximate mask when MediaPipe is unavailable
- Scrollable, resizable window, F11 full screen, live outfit size/position sliders

- Credit integrity check (SHA-256) in title, footer, welcome dialog and pipeline start
- Loaded garments apply immediately while the camera runs; removed dead code (blur.py, gui_fix.py) and empty placeholder assets
- Console log messages translated to English
- Background blur: soft, smoothed person mask plus a protected head area (ears and hair stay sharp)
- Outfit fit: the rough fallback mask no longer clips the garment; the face fallback now finds its OpenCV cascade reliably (OpenCV pinned to 4.x, which ships it)
- First-run and on-demand "virtual camera needed" dialog with OBS instructions and download link
- Copyright hygiene: removed third-party sample images, real license audit in THIRD_PARTY_LICENSES.md, legal notes in README
- Performance: webcam opened in MJPG mode, selectable capture quality (Fast / Balanced / HD), nudity detection in its own thread (never shows the raw camera before the first detection result), pose/mask analysis on a small frame every 2nd frame, 10x faster background blur, FPS/timing diagnostics in the status bar

