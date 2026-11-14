import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from guvenli_kamera.segmentation import refine_person_mask, blur_background, head_ellipse

POSE = {"left_shoulder": (300, 400), "right_shoulder": (700, 400), "nose": (500, 250),
        "left_ear": (430, 245), "right_ear": (570, 245)}


def test_head_protection_covers_ears_but_not_far_background():
    mask = np.zeros((720, 1280), np.uint8)
    out = refine_person_mask(mask, POSE)
    assert out[245, 430] > 200 and out[245, 570] > 200   # kulaklar
    assert out[245, 50] == 0 and out[700, 1200] == 0       # uzak arka plan


def test_background_blur_keeps_head_sharp():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    out = blur_background(frame, refine_person_mask(np.zeros((720, 1280), np.uint8), POSE))
    assert np.array_equal(out[240:250, 495:505], frame[240:250, 495:505])  # kafa merkezi keskin
    assert not np.array_equal(out[600:700, 50:150], frame[600:700, 50:150])  # arka plan bulanık


def test_head_ellipse_uses_face_box_when_available():
    he = head_ellipse((720, 1280), {"face_box": (400, 100, 200, 220)})
    assert he is not None and abs(he[0] - 500) < 5


def test_no_third_party_images_are_shipped():
    d = os.path.join(os.path.dirname(__file__), "..", "assets", "outfit")
    files = [f for f in os.listdir(d)] if os.path.isdir(d) else []
    assert all(f.lower() == "readme.txt" for f in files)


@pytest.fixture
def gui_module(monkeypatch):
    for m in ["tkinter", "tkinter.filedialog", "tkinter.colorchooser", "customtkinter"]:
        monkeypatch.setitem(sys.modules, m, MagicMock())
    sys.modules.pop("guvenli_kamera.gui", None)
    import guvenli_kamera.gui as g
    yield g
    sys.modules.pop("guvenli_kamera.gui", None)


def test_obs_help_dialog_opens_once_and_mentions_obs(gui_module, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    app = gui_module.SafeCameraGUI()
    app._obs_help_open = False
    app._show_obs_help("Virtual camera could not be opened.")
    assert app._obs_help_open is True
    before = gui_module.ctk.CTkToplevel.call_count
    app._show_obs_help()          # ikinci çağrı yeni pencere açmaz
    assert gui_module.ctk.CTkToplevel.call_count == before
    assert "obsproject.com" in app.OBS_URL
    app.cleanup()


def test_pose_fallback_loader_never_crashes():
    from guvenli_kamera.pose import PoseEstimator
    assert PoseEstimator()._ensure_fallback_loaded() in (True, False)


def test_requirements_pin_opencv4_for_face_cascade():
    path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    assert "opencv-python>=4.8,<5" in open(path, encoding="utf-8").read()
