"""Gerçekçi (bulut) mod — Decart Virtual Try-On API entegrasyonu.

Uyarı: Bu bir iskelet (skeleton) implementasyondur.
Gerçek API dokümantasyonu (docs.platform.decart.ai, model Lucy-2 `lucy_2_rt`)
doğrulanana kadar NOT_IMPLEMENTEDError fırlatır.
API anahtarı asla loglanmaz; ortam değişkeni DECART_API_KEY veya
config üzerinden gizli alan ile alınır.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .base import OutfitProvider

logger = logging.getLogger(__name__)


class DecartTryOnError(RuntimeError):
    """Decart API hataları."""
    pass


class DecartTryOnProvider(OutfitProvider):
    """
    Decart Lucy-2 real-time virtual try-on sağlayıcısı (iskelet).

    Gerçek API kullanımı:
    - Endpoint: POST /v1/virtual-tryon (veya benzer)
    - Headers: Authorization: Bearer <API_KEY>
    - Body: { "model": "lucy_2_rt", "image": base64_jpeg, "prompt": "...", "garment_image": base64_jpeg }
    - Response: { "image": base64_jpeg, "mask": base64_png }

    NOT_IMPLEMENTEDError fırlatır — dokümantasyon doğrulandıktan sonra doldurulacak.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        super().__init__(cfg)
        self.mode = "realistic"
        self.api_key: str = cfg.get("realistic_api_key", "") or os.getenv("DECART_API_KEY", "")
        self.prompt: str = cfg.get("realistic_prompt",
                                   "Substitute the current top with a navy blazer and white shirt")
        self.garment_path: str = cfg.get("garment_path", "")
        self._session = None  # requests.Session placeholder
        self._endpoint = "https://api.decart.ai/v1/virtual-tryon"  # DOĞRULANMAMIŞ placeholder
        logger.warning("DecartTryOnProvider: iskelet mod — gerçek API entegrasyonu henüz yok")

    def _ensure_api_key(self) -> str:
        if not self.api_key:
            raise DecartTryOnError("API anahtarı eksik: DECART_API_KEY ortam değişkeni veya config.realistic_api_key ayarlayın")
        return self.api_key

    def _prepare_roi(self, frame: np.ndarray, pose: Optional[dict] = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Üst gövde ROI'sünü (omuz çizgisi ve altı) kes, küçült, base64'e çevir."""
        # Not: bu yöntem gerçek API için optimize edilecek
        h, w = frame.shape[:2]
        # Basit ROI: kare merkezi, %50 genişlik, %60 yükseklik
        roi_w = int(w * 0.5)
        roi_h = int(h * 0.6)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        roi = frame[y1:y1 + roi_h, x1:x1 + roi_w]
        # Küçült: API limitlerine göre (örn. 512x512 max)
        max_dim = 512
        rh, rw = roi.shape[:2]
        if max(rh, rw) > max_dim:
            scale = max_dim / max(rh, rw)
            roi = cv2.resize(roi, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_AREA)
        return roi, (x1, y1, x1 + roi_w, y1 + roi_h)

    @abstractmethod
    def _call_api(self, roi_bgr: np.ndarray, prompt: str, garment_bgr: Optional[np.ndarray] = None) -> np.ndarray:
        """Decart API'sini çağır, sonucu BGR numpy array olarak döndür."""
        raise NotImplementedError("Decart API henüz entegre edilmedi — docs.platform.decart.ai doğrulanmalı")

    def process(self, frame: np.ndarray, pose: Optional[dict] = None,
                person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Gerçekçi mod: ROI -> API -> maske ile birleştir -> frame döndür."""
        # API anahtarı kontrol
        try:
            self._ensure_api_key()
        except DecartTryOnError as exc:
            logger.error("Decart API anahtarı yok: %s", exc)
            # Fail-closed: gerçek görüntüyü döndür ama blur yine uygulanır (pipeline safety sonrası)
            return frame

        # Henüz implementasyon yok
        raise NotImplementedError(
            "Decart Virtual Try-On API (Lucy-2) henüz entegre edilmedi. "
            "docs.platform.decart.ai dokümantasyonu doğrulanmalı ve _call_api() doldurulmalı. "
            "Şimdilik 'Gerçekçi' modu arayüzde devre dışı (disabled)."
        )

    def get_status(self) -> str:
        return "Realistic (cloud) mode - not available yet (API integration pending)"


# Geçici import — OpenCV sadece process içinde lazım olacak
try:
    import cv2
except ImportError:
    cv2 = None