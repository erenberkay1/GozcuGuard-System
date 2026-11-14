import os
import time

import cv2
import numpy as np

from guvenli_kamera.pipeline import Pipeline
from guvenli_kamera.segmentation import blur_background


def _video(tmp_path, size=(320, 240), n=90):
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    rng = np.random.default_rng(0)
    for _ in range(n):
        vw.write(rng.integers(100, 255, (size[1], size[0], 3), dtype=np.uint8))
    vw.release()
    return path


def test_fast_background_blur_keeps_person_and_blurs_rest():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    full = np.full((540, 960), 255, np.uint8)
    assert np.array_equal(blur_background(frame, full), frame)          # person everywhere -> unchanged
    out = blur_background(frame, np.zeros((540, 960), np.uint8))
    assert out.shape == frame.shape and not np.array_equal(out, frame)  # background blurred
    small_mask = np.full((135, 240), 255, np.uint8)                      # small masks are scaled up
    assert blur_background(frame, small_mask).shape == frame.shape


def test_pose_scaling_helper():
    pose = {"left_shoulder": (100, 200), "face_box": (10, 20, 30, 40), "nose": None}
    out = Pipeline._scale_pose(pose, 2.0)
    assert out["left_shoulder"] == (200, 400) and out["face_box"] == (20, 40, 60, 80) and out["nose"] is None


def test_no_raw_frame_before_first_detection_result(tmp_path):
    p = Pipeline(use_fake_camera=True, fake_source=_video(tmp_path), enable_panik=False,
                 use_virtual_camera=False, enable_nudity_blur=True)

    def slow_load():
        time.sleep(1.2)
        return True
    p.detector.load = slow_load
    p.detector.detect = lambda frame, threshold=0.35: []
    assert p.start()
    time.sleep(0.5)
    early = p.get_last_frame()
    assert early is not None and early.mean() < 20      # still protected while the model loads
    time.sleep(2.0)
    late = p.get_last_frame()
    p.stop()
    assert late is not None and late.mean() > 60        # normal video once detection is running


def test_status_reports_timing_and_size(tmp_path):
    p = Pipeline(use_fake_camera=True, fake_source=_video(tmp_path), enable_panik=False,
                 use_virtual_camera=False, enable_nudity_blur=False)
    assert p.start()
    time.sleep(1.0)
    st = p.status()
    p.stop()
    assert st["proc_ms"] >= 0 and st["wait_ms"] >= 0 and st["size"][0] > 0
