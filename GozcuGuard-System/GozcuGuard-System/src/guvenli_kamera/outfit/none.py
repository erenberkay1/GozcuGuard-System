"""Üzgün mod — kıyafet değiştirme yapmaz."""

from __future__ import annotations

from .base import OutfitProvider


class NoOpOutfitProvider(OutfitProvider):
    """Kıyafet işlemi yapmaz; frame'i olduğu gibi döndürür (koruma blur'u ile)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)

    def process(self, frame, pose=None):
        # Kıyafet devre dışı; sadece bulanıkla (hem koruma hem poz)
        import cv2
        k = 51
        return cv2.GaussianBlur(frame, (k, k), sigmaX=30, sigmaY=30)

    def get_status(self):
        return "Outfit mode: off"