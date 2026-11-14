"""Politika motoru — kaçılmalı politika ve fail-closed davranış."""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from .detector import Detection

logger = logging.getLogger(__name__)


# ── RegionBlur (Bölgesel blur) ──────────────────────────────────────────────

def apply_regional_blur(
    frame: np.ndarray,
    detection: Detection,
    expand_ratio: float = 0.40,
    kernel_size: int = 51,
    sigma: float = 30.0,
) -> np.ndarray:
    """Algılanan kutu etrafında %40 genişletme ile güçlü blurlama uygula."""
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2

    # Kutu genişlet
    dx = int((x2 - x1) * expand_ratio / 2)
    dy = int((y2 - y1) * expand_ratio / 2)
    x1n = max(0, x1 - dx)
    yn1 = max(0, y1 - dy)
    x2n = min(w, x2 + dx)
    y2n = min(h, y2 + dy)

    # Kopya ve sadece kutuyu blurla
    out = frame.copy()
    roi = out[yn1:y2n, x1n:x2n]
    if roi.size == 0:
        return out
    # (x,y,w,h) formatında GaussianBlur
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=sigma, sigmaY=sigma)
    out[yn1:y2n, x1n:x2n] = blurred
    return out


# ── SolidBlur (Tüm kare blurlanır) ───────────────────────────────────────────

def apply_solid_blur(frame: np.ndarray, kernel_size: int = 51, sigma: float = 30.0) -> np.ndarray:
    """Tüm kareyi Gaussian blurlama."""
    import cv2
    h, w = frame.shape[:2]
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    out = frame.copy()
    blurred = cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)
    out[:] = blurred[:]
    return out


# ── CooldownGuard ───────────────────────────────────────────────────────────

@dataclass
class CooldownGuard:
    """Zamansal tutma ve cooldown yönetimi.

    - Son algılamadan sonra X saniye (t_frame) blur tut.
    - Yeni bir algılamaya önceki silinir, sayaç sıfırlanır.
    """

    hold_frames: int
    _frame_counter: int = 0
    _last_detection_time: float = field(default_factory=lambda: time.monotonic())
    _active: bool = False
    _hold_counter: int = 0

    def record_detection(self) -> None:
        """Yeni bir koruma algılaması kaydet: sayacı resetle ve tutmaya başla."""
        self._last_detection_time = time.monotonic()
        self._hold_counter = self.hold_frames
        self._active = True

    def tick(self) -> bool:
        """Bir frame'i ilerle. Blur aktif ise sayaç azalır; bittiğinde False döndür."""
        if not self._active:
            return False
        if self._hold_counter > 0:
            self._hold_counter -= 1
            if self._hold_counter == 0:
                # Tutma süresi bitti; aktif kalmasına izin vermeyiz, dışarıya False ver (tamamen aç)
                self._active = False
                return False
        return True

    @property
    def is_active(self) -> bool:
        return self._active


# ── Koruma ekranı ───────────────────────────────────────────────────────────

def protection_screen(shape, text: str = "Camera temporarily disabled") -> np.ndarray:
    """Siyah ekran + yazı (Hershey fontu Türkçe karakter desteklemez, bu yüzden ASCII)."""
    h, w = shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    scale = max(0.6, w / 1280.0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(out, text, (max(0, (w - tw) // 2), (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── SafetyEngine ────────────────────────────────────────────────────────────

def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return wrapper


class SafetyEngine:
    """Koruma kararları. Her karede: tick() -> (algılama karesinde) report()/report_failure() -> process().

    - Algılanan kutular `temporal_hold_frames` kare boyunca tutulur (algılama yapılmayan karelerde de blur sürer).
    - Algılama başarısızsa veya panik açıksa ham görüntü ASLA çıkmaz (koruma ekranı).
    """

    POLICY_REGIONAL = "regional"
    POLICY_SOLID = "solid"
    POLICY_COOLDOWN = "cooldown"
    POLICY_OVERLAY = "overlay"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.policy: str = cfg.get("policy", self.POLICY_REGIONAL)
        self.threshold: float = float(cfg.get("threshold", 0.35))
        self.box_expand_ratio: float = float(cfg.get("box_expand_ratio", 0.40))
        self.kernel_size: int = int(cfg.get("kernel_size", 51))
        self.sigma: float = float(cfg.get("sigma", 30.0))
        self.cooldown_seconds: float = float(cfg.get("cooldown_seconds", 3.0))
        self.temporal_hold_frames: int = int(cfg.get("temporal_hold_frames", 15))
        self.overlay_text: str = cfg.get("overlay_text", "Camera temporarily disabled")

        self._lock = threading.RLock()  # the detection thread reports while the main loop processes
        self.cooldown_guard = CooldownGuard(self.temporal_hold_frames)
        self._held: list = []          # [(Detection, kalan_kare)]
        self._cooldown_until: float = 0.0
        self._failed: bool = False
        self._panic: bool = False
        self.event_count: int = 0

    # -- durum ---------------------------------------------------------
    def set_panic(self, on: bool) -> None:
        self._panic = bool(on)

    @property
    def panic(self) -> bool:
        return self._panic

    @property
    def failed(self) -> bool:
        return self._failed

    @_locked
    def reset(self) -> None:
        """Clear held boxes, cooldown and failure state (used when the nudity blur is switched off)."""
        self._held = []
        self._cooldown_until = 0.0
        self._failed = False

    @_locked
    def tick(self) -> None:
        self._held = [(d, n - 1) for d, n in self._held if n - 1 > 0]

    @_locked
    def report(self, detections: list) -> None:
        self._failed = False
        if detections:
            self.event_count += 1
            self.cooldown_guard.record_detection()
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            for d in detections:
                self._held.append((d, self.temporal_hold_frames))

    @_locked
    def report_failure(self) -> None:
        self._failed = True

    # -- çıktı ---------------------------------------------------------
    @_locked
    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._panic:
            return protection_screen(frame.shape, "Camera off (panic)")
        if self._failed:
            return protection_screen(frame.shape, "Protection unavailable - camera disabled")

        active = bool(self._held)
        cooling = time.monotonic() < self._cooldown_until
        p = self.policy

        if p == self.POLICY_SOLID:
            return apply_solid_blur(frame, self.kernel_size, self.sigma) if active else frame
        if p == self.POLICY_COOLDOWN:
            return apply_solid_blur(frame, self.kernel_size, self.sigma) if (active or cooling) else frame
        if p == self.POLICY_OVERLAY:
            return protection_screen(frame.shape, self.overlay_text) if (active or cooling) else frame

        if p != self.POLICY_REGIONAL:
            if not getattr(self, "_warned_policy", False):
                self._warned_policy = True
                logger.warning("Bilinmeyen politika '%s', bölgesel blur kullanılıyor", p)
        out = frame
        for det, _ in self._held:
            out = apply_regional_blur(out, det, expand_ratio=self.box_expand_ratio,
                                      kernel_size=self.kernel_size, sigma=self.sigma)
        return out

    def evaluate(self, frame: np.ndarray, detections) -> np.ndarray:
        """Tek çağrılık kolaylık: detections None ise algılama başarısız sayılır."""
        self.tick()
        if detections is None:
            self.report_failure()
        else:
            self.report(list(detections))
        return self.process(frame)
