"""Kıyafet sistemi arayüzü."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class OutfitProvider(ABC):
    """Kıyafet modu arayüzü. Sağlayıcılar: Hafif (yerel) veya Gerçekçi (bulut).

    process() üyesi kareyi işleyip güvenli çıktı döndürür.
    Ham görüntünün çıkışa gitmediğinden emin olunur.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.mode: str = cfg.get("mode", "off")
        self.garment_path: str = cfg.get("garment_path", "")
        self.anchor_points: dict[str, list[float]] = cfg.get("anchor_points", {})
        self.profile_name: str = cfg.get("profile_name", "varsayılan")

    @abstractmethod
    def process(self, frame: np.ndarray, pose: Optional[dict] = None) -> np.ndarray:
        """Güvenli çıktı döndür. Ham frame'i döndürmez."""
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> str:
        """Durum mesajı döndürür (mod, hata olursa)."""
        raise NotImplementedError