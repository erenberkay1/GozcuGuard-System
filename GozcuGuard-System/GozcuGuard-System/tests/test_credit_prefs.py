import numpy as np
from guvenli_kamera import _credit, credit_ok, CREDIT
from guvenli_kamera.pipeline import Pipeline
from guvenli_kamera.prefs import load_prefs, save_prefs, DEFAULTS
from guvenli_kamera.safety import SafetyEngine


def test_credit_is_intact():
    assert credit_ok()
    assert CREDIT == "Created by Eren Berkay Gün"


def test_tampered_credit_is_detected_and_blocks_pipeline(monkeypatch):
    monkeypatch.setattr(_credit, "_PARTS", ("Created by ", "Someone Else"))
    assert not _credit.credit_ok()
    assert "MODIFIED" in _credit.verified_credit()
    import guvenli_kamera.pipeline as pl
    monkeypatch.setattr(pl, "credit_ok", _credit.credit_ok)
    p = Pipeline(use_fake_camera=True, fake_source="nope.mp4", enable_panik=False, use_virtual_camera=False)
    assert p.start() is False and "modified" in p.error


def test_prefs_roundtrip_and_defaults(tmp_path):
    path = str(tmp_path / "prefs.json")
    assert load_prefs(path) == DEFAULTS
    prefs = dict(DEFAULTS, nudity_blur=False, background_blur=True, onboarding_done=True)
    assert save_prefs(prefs, path)
    assert load_prefs(path) == prefs


def test_safety_reset_clears_state():
    eng = SafetyEngine()
    eng.report_failure()
    assert eng.failed
    eng.reset()
    frame = np.full((50, 50, 3), 120, np.uint8)
    assert np.array_equal(eng.process(frame), frame)
