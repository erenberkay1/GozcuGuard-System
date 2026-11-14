import numpy as np
from guvenli_kamera.detector import Detector
from guvenli_kamera.outfit.overlay import OverlayProvider, OVERLAY_ADJUST
from guvenli_kamera.outfit.garment import make_template, TEMPLATE_SHOULDERS, TEMPLATE_NAMES


class FakeModel:
    def detect(self, f):
        return [
            {"class": "FACE_MALE", "score": 0.9, "box": [500, 50, 200, 200]},
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.8, "box": [540, 120, 80, 60]},   # yüzün içinde -> sahte alarm
            {"class": "BUTTOCKS_EXPOSED", "score": 0.8, "box": [100, 400, 80, 60]},        # gerçek bölge
        ]


def test_detections_inside_a_face_are_ignored():
    d = Detector()
    d._model, d._loaded = FakeModel(), True
    dets = d.detect(np.zeros((720, 1280, 3), np.uint8))
    assert [x.class_name for x in dets] == ["BUTTOCKS_EXPOSED"]


def test_templates_valid_and_garment_starts_at_shoulder_line():
    for name in TEMPLATE_NAMES:
        t = make_template(name, (200, 30, 30))
        assert t.shape == (600, 600, 4) and (t[:, :, 3] > 0).mean() > 0.2
    prov = OverlayProvider({"mode": "overlay"})
    prov.set_garment_array(make_template("T-Shirt", (255, 0, 0)), shoulders=TEMPLATE_SHOULDERS)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    pose = {"left_shoulder": (488, 377), "right_shoulder": (875, 364)}
    out = prov.process(frame, pose, None)
    changed = np.where((out != frame).any(axis=2).any(axis=1))[0]
    assert changed.min() < 377          # üst kenar omuz noktasından yukarıda (omuz sırtı)
    assert changed.min() > 250          # ama başa/yüze çıkmaz


def test_adjust_sliders_move_and_scale_garment():
    prov = OverlayProvider({"mode": "overlay"})
    prov.set_garment_array(make_template("T-Shirt", (255, 0, 0)), shoulders=TEMPLATE_SHOULDERS)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    pose = {"left_shoulder": (488, 377), "right_shoulder": (875, 364)}
    try:
        base = prov.process(frame, pose, None)
        OVERLAY_ADJUST.update(scale=1.3, dy=0.2, dx=0.0)
        prov._smooth = None
        big = prov.process(frame, pose, None)
        assert not np.array_equal(base, big)
        assert (big != frame).any(axis=2).sum() > (base != frame).any(axis=2).sum()
    finally:
        OVERLAY_ADJUST.update(scale=1.0, dy=0.0, dx=0.0)
