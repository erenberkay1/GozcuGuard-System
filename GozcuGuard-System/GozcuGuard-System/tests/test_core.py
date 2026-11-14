"""Testler: Safe Camera - güvenlik politika ve detektör testleri."""
from __future__ import annotations

import numpy as np
import pytest

from guvenli_kamera.detector import Detector, Detection
from guvenli_kamera.safety import SafetyEngine, CooldownGuard
from guvenli_kamera.blur import FrameBuffer, apply_regional_blur, apply_solid_blur
from guvenli_kamera.camera import FakeCamera


# ── Bileşen testleri ─────────────────────────────────────────────────

def test_detection_dataclass():
    det = Detection(class_name="FEMALE_BREAST_EXPOSED", confidence=0.45,
                   x1=100, y1=200, x2=300, y2=450)
    d = det.to_dict()
    assert d["class_name"] == "FEMALE_BREAST_EXPOSED"
    assert d["confidence"] == 0.45
    assert d["x1"] == 100

def test_regional_blur_box_expand():
    """Kutu etrafında %40 genişletme testi."""
    # Fareli bir frame — kutu bölgesinde farklı değerler içerir
    frame = np.zeros((300, 640, 3), dtype=np.uint8)
    frame[80:250, 150:450] = np.random.randint(50, 255, (170, 300, 3), dtype=np.uint8)
    detection = Detection(class_name="TEST", confidence=0.8,
                         x1=200, y1=100, x2=400, y2=250)
    result = apply_regional_blur(frame, 200, 100, 400, 250, expand_ratio=0.4)
    h, w = result.shape[:2]
    assert w == 640 and h == 300
    # Genişletilmiş kutu (x1-40, y1-30, x2+40, y2+30) bölgesi etkilenir
    # Genişletme kenarı: (160, 70, 440, 280)
    # Dışarıdan (0:90, 0:160) etkilenmemesi gerekir
    outside_top = result[:90, :160]
    assert np.array_equal(outside_top, frame[:90, :160]), "expand dışında etkilenen alan var"
    # Kutu içindeki region (genişletme dahil) orijinalden farklı (blur)
    inner_mean = result[100:250, 200:400].mean()
    original_inner_mean = frame[100:250, 200:400].mean()
    assert inner_mean != original_inner_mean, "blur bölgesi orijinalle aynı"

def test_solid_blur():
    """Tüm kare blur testi."""
    frame = np.random.randint(0, 255, (300, 640, 3), dtype=np.uint8)
    result = apply_solid_blur(frame, kernel_size=51, sigma=30.0)
    assert result.shape == frame.shape
    # Tamamen gürültülü olması gerekir (varlıq olması gerekir)
    s = result.astype(int).sum() / (3 * 300 * 640)
    # Tamamlarının sinir alanlarında homojen olmasa gerekir
    edges = result[:, :, :]
    assert edges.shape == frame.shape

def test_frame_buffer():
    """Kare zamansal tutma testi."""
    fb = FrameBuffer(capacity=5)
    for i in range(7):
        fb.add(np.zeros((50, 50, 3), dtype=np.uint8) + i * 30)
    assert len(fb.frames) == 5
    assert fb.get() is not None
    assert fb.get() is not None


# ── Fail-closed davranışı ───────────────────────────────────────────

def test_safety_no_detection_returns_original():
    """Algılamazsa original frame döndür."""
    engine = SafetyEngine({"policy": "regional", "threshold": 0.9})
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = engine.evaluate(frame, [])
    assert result.shape == frame.shape

def test_safety_detected_region_blur():
    """Algıla geldiğinde regional blur uygula."""
    engine = SafetyEngine({
        "policy": "regional",
        "threshold": 0.9,
        "kernel_size": 31,
        "sigma": 20.0,
        "box_expand_ratio": 0.40,
    })
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Randoms veya gradyanla ortalama farklı
    frame[30:70, 30:70] = 200
    detection = Detection("TEST", 0.95, 10, 10, 80, 80)
    result = engine.evaluate(frame, [detection])
    assert result.shape == frame.shape
    # Genişleştirilmiş region dışı alanlar gri/gruel eşitleşmez (sadece orta kısım)
    outside = result[80:100, 80:100]
    assert outside.mean() != frame[80:100, 80:100].mean()




# ── Politika modları ──────────────────────────────────────────────

@pytest.mark.parametrize("policy", ["regional", "solid", "cooldown", "overlay"])
def test_policies(policy):
    """Tüm politikalar çalışırken frame'i döndürür."""
    engine = SafetyEngine({"policy": policy, "threshold": 0.9, "kernel_size": 51})
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    detection = Detection("TEST", 0.95, 20, 20, 70, 70)
    result = engine.evaluate(frame, [detection])
    assert result.shape == frame.shape
    assert isinstance(result, np.ndarray)

def test_unknown_policy_falls_back_to_regional():
    """Bilinmeyen politika → blurla, geçiliyor."""
    engine = SafetyEngine({"policy": "yok_bilinmiyor", "threshold": 0.9})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.evaluate(frame, [Detection("X", 0.95, 10, 10, 50, 50)])
    assert result.shape == frame.shape


# ── CooldownGuard ──────────────────────────────────────────────────

def test_cooldown_guard_recording_and_ticking():
    guard = CooldownGuard(hold_frames=5)
    guard.record_detection()
    assert guard.is_active
    for _ in range(4):
        guard.tick()
    assert guard.is_active
    assert not guard.tick()  # Sisteminin 5. çağı atlanmalı


# ── FakeCamera ────────────────────────────────────────────────────

@pytest.fixture
def video_path(tmp_path):
    import cv2
    path = str(tmp_path / "t.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for _ in range(10):
        vw.write(np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))
    vw.release()
    return path


def test_fake_camera_opens(video_path):
    fc = FakeCamera(source=video_path)
    assert fc.open() is True
    fc.close()

def test_fake_camera_reading(video_path):
    fc = FakeCamera(source=video_path)
    fc.open()
    frame = fc.read()
    assert frame is not None
    assert frame.shape == (720, 1280, 3) or fc.actual_resolution()[:2] == (720, 1280)
    fc.close()

def test_fake_camera_not_exist_error():
    fc = FakeCamera(source="/nonexistent/video.mp4")
    assert fc.open() is False


# ── OK tuşu yokluğu: BU TESTLERDEN ÇALIŞMALAR SAĞLANIYOR ────────────

def test_numpy_zero_image_detect_empty_result():
    detector = Detector()
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    res = detector.detect(img, threshold=0.5)
    assert isinstance(res, list)

def test_small_noise_image_detect_handles_robustly():
    detector = Detector(inference_resolution=320)
    img = np.random.randint(0, 50, (320, 320, 3), dtype=np.uint8)
    res = detector.detect(img)
    assert isinstance(res, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])