"""Güvenlik düzeltmelerini doğrulayan testler."""
import time

import cv2
import numpy as np
import pytest

from guvenli_kamera.detector import Detector, Detection, DetectorError
from guvenli_kamera.safety import SafetyEngine
from guvenli_kamera.outfit.overlay import OverlayProvider
from guvenli_kamera.pipeline import Pipeline


class FakeModel:
    def detect(self, f):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [400, 300, 200, 150]},
            {"class": "FACE_FEMALE", "score": 0.99, "box": [10, 10, 50, 50]},   # blurlanmamalı
            {"class": "BUTTOCKS_EXPOSED", "score": 0.1, "box": [0, 0, 20, 20]},  # eşik altı
        ]


def loaded_detector():
    d = Detector()
    d._model, d._loaded = FakeModel(), True
    return d


def test_detector_parses_nudenet_box_format():
    dets = loaded_detector().detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (400, 300, 600, 450)


def test_detector_raises_when_model_cannot_load():
    d = Detector()
    d._ensure_loaded = lambda: False
    with pytest.raises(DetectorError):
        d.detect(np.zeros((10, 10, 3), np.uint8))


def test_blur_actually_changes_pixels():
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    det = loaded_detector().detect(frame)
    out = SafetyEngine({"policy": "regional"}).evaluate(frame, det)
    assert not np.array_equal(out, frame)
    assert np.array_equal(out[:100], frame[:100])  # kutu dışı değişmez


def test_temporal_hold_keeps_blur_without_new_detections():
    eng = SafetyEngine({"policy": "regional", "temporal_hold_frames": 5})
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    eng.evaluate(frame, [Detection("X", 0.9, 50, 50, 120, 120)])
    for _ in range(3):  # algılama yapılmayan kareler
        out = eng.evaluate(frame, [])
        assert not np.array_equal(out, frame)
    for _ in range(5):
        out = eng.evaluate(frame, [])
    assert np.array_equal(out, frame)  # süre bitince temiz


def test_failure_and_panic_never_return_raw_frame():
    frame = np.random.randint(100, 255, (200, 200, 3), dtype=np.uint8)
    eng = SafetyEngine()
    out = eng.evaluate(frame, None)  # algılama başarısız
    assert out.mean() < 20
    eng.report([])  # kurtarma
    assert np.array_equal(eng.process(frame), frame)
    eng.set_panic(True)
    assert eng.process(frame).mean() < 20


@pytest.mark.parametrize("policy", ["solid", "cooldown", "overlay"])
def test_full_frame_policies_cover_everything(policy):
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    eng = SafetyEngine({"policy": policy})
    out = eng.evaluate(frame, [Detection("X", 0.9, 5, 5, 20, 20)])
    assert not np.array_equal(out[150:, 150:], frame[150:, 150:])


def test_overlay_without_pose_returns_original_and_with_pose_changes_below_shoulders():
    g = np.zeros((200, 300, 4), np.uint8)
    g[..., 2] = 255
    g[..., 3] = 255
    prov = OverlayProvider({"mode": "overlay"})
    prov.garment, prov._ready = g, True
    frame = np.full((480, 640, 3), 90, np.uint8)
    assert np.array_equal(prov.process(frame, None), frame)
    pose = {"left_shoulder": (220, 200), "right_shoulder": (420, 200), "nose": (320, 120)}
    out = prov.process(frame, pose)
    assert np.array_equal(out[:150], frame[:150])      # baş/boyun bölgesi dokunulmaz
    assert not np.array_equal(out[260:400, 250:400], frame[260:400, 250:400])


def test_pipeline_never_outputs_raw_when_detector_fails(tmp_path):
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(30):
        vw.write(np.random.randint(100, 255, (240, 320, 3), dtype=np.uint8))
    vw.release()
    p = Pipeline(use_fake_camera=True, fake_source=path, enable_panik=False, use_virtual_camera=False)
    p.detector.load = lambda: False
    assert p.start()
    time.sleep(1.0)
    frame = p.get_last_frame()
    p.stop()
    assert frame is not None and frame.mean() < 20
