import time

import cv2
import numpy as np
import pytest

from guvenli_kamera.pipeline import Pipeline
from guvenli_kamera.prefs import load_prefs, save_prefs, DEFAULTS
from guvenli_kamera.segmentation import approx_person_mask, blur_background
from guvenli_kamera.safety import SafetyEngine
from guvenli_kamera.detector import Detection


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    rng = np.random.default_rng(1)
    for _ in range(40):
        vw.write(rng.integers(100, 255, (240, 320, 3), dtype=np.uint8))
    vw.release()
    return path


def run_pipeline(video, **kw):
    p = Pipeline(use_fake_camera=True, fake_source=video, enable_panik=False, use_virtual_camera=False, **kw)
    return p


def test_nudity_blur_off_loads_no_model_and_passes_frames_untouched(video):
    p = run_pipeline(video, enable_nudity_blur=False)
    loaded = []
    p.detector.load = lambda: loaded.append(1) or True
    assert p.start()
    time.sleep(0.8)
    frame = p.get_last_frame()
    st = p.status()
    p.stop()
    assert not loaded, "detector must not be loaded while the nudity blur is OFF"
    assert frame is not None and frame.mean() > 60  # real picture, not a protection screen
    assert st["nudity_blur"] is False and st["failed"] is False


def test_detector_failure_only_blocks_when_nudity_blur_is_on(video):
    p = run_pipeline(video, enable_nudity_blur=True)
    p.detector.load = lambda: False
    assert p.start()
    time.sleep(0.8)
    frame = p.get_last_frame()
    p.stop()
    assert frame is not None and frame.mean() < 20  # protection screen


def test_background_blur_blurs_outside_the_person_only():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    mask[60:200, 100:220] = 255
    out = blur_background(frame, mask)
    assert np.array_equal(out[100:160, 130:190], frame[100:160, 130:190])      # person untouched
    assert not np.array_equal(out[:40, :40], frame[:40, :40])                  # background blurred


def test_approx_person_mask_covers_head_and_torso_but_not_corners():
    pose = {"left_shoulder": (200, 300), "right_shoulder": (440, 300)}
    m = approx_person_mask((480, 640, 3), pose)
    assert m.shape == (480, 640) and m[470, 320] > 200 and m[190, 320] > 200 and m[10, 10] == 0
    assert approx_person_mask((480, 640, 3), None) is None


def test_prefs_roundtrip_and_defaults(tmp_path):
    path = str(tmp_path / "p.json")
    assert load_prefs(path) == DEFAULTS
    assert save_prefs({"nudity_blur": False, "background_blur": True, "onboarding_done": True}, path)
    got = load_prefs(path)
    assert got["nudity_blur"] is False and got["background_blur"] is True and got["onboarding_done"] is True


def test_safety_reset_clears_held_boxes_and_failure():
    eng = SafetyEngine({"policy": "regional"})
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    eng.evaluate(frame, [Detection("X", 0.9, 10, 10, 60, 60)])
    eng.report_failure()
    eng.reset()
    assert np.array_equal(eng.process(frame), frame)
