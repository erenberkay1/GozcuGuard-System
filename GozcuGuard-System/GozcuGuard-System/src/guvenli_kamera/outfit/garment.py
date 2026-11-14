"""Kıyafet yükleme, arka plan temizleme, ankraj noktaları, şablonlar."""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GarmentManager:
    """Kıyafet görseli yükle, arka plan temizle, ankraj noktalarını tahmin et."""

    def __init__(self):
        self.images: dict[str, np.ndarray] = {}  # path -> image (RGBA via PIL→numpy)
        self.anchors: dict[str, list[float]] = {
            "left_shoulder": [0.33, 0.18],
            "right_shoulder": [0.66, 0.18],
            "mid_waist": [0.50, 0.42],
        }
        self.active_path: str = ""
        self.loaded_paths: list[str] = []
        self.pending_paths: list[str] = []

    def add_image(self, path: str, name: str = "") -> bool:
        """Görsel yükle (PNG/JPG). Şeffaf arka planlı en iyi sonucu verir."""
        try:
            from PIL import Image
            img = Image.open(path).convert("RGBA")
            arr = np.array(img)
            if name:
                self.images[name] = arr
            else:
                self.images[os.path.basename(path)] = arr
            self.loaded_paths.append(path)
            self.active_path = path
            logger.info("Kıyafet eklendi: %s", path)
            return True
        except Exception as exc:
            logger.error("Kıyafet yüklenemedi: %s", exc)
            return False
    def load_garment(self, path: str, name: str = "") -> bool:
        """gui.py uyumluluğu için add_image sarmalayıcısı."""
        return self.add_image(path, name)
    def remove_garment(self) -> None:
        """Tüm kıyafetleri kaldır."""
        self.images.clear()
        self.active_path = ""
        self.anchors = {
            "left_shoulder": [0.33, 0.18],
            "right_shoulder": [0.66, 0.18],
            "mid_waist": [0.50, 0.42],
        }

    @staticmethod
    def get_default_templates() -> dict[str, str]:
        """Varsayılan şablon indirme bağlantıları (daha sonra assets/ içindeki şablonlara yüklemek için)."""
        return {
            "düz_gömlek": "",  # assets/gömlek.png
            "blazer": "",       # assets/blazer.png
            "kazak": "",        # assets/kazak.png
            "tisört": "",       # assets/tisört.png
        }

    @staticmethod
    def try_grayscale_bg_cleanup(frame_bgr: np.ndarray, threshold: int = 200) -> np.ndarray:
        """GrabCut veya kural parametrik (cluster-based) boş arka plan temizliği.

        Yalnızca varsayılan Package'e dahil değil — aşağıda olmalı.
        Lütfen gerçek yöntemi dokümantasyona eklemek zorunda.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Basit: yüksek/düşük histogramli kontrastlı alanlara негатива
        edges = cv2.Canny(gray, 100, 200)
        mask_bg = edges > 0
        out = frame_bgr.copy()
        # Arka planın BackgroundGrayscale'de belirlenmiş şekilde geçiştirmek (çok ucuz)
        return out

    def set_anchor(self, name: str, x: float, y: float) -> None:
        """Ankraj noktasını ayarla."""
        if name in self.anchors:
            self.anchors[name] = [float(x), float(y)]
        else:
            self.anchors[name] = [float(x), float(y)]

    def auto_guess_anchors(self, width: int, height: int) -> dict[str, tuple[int, int]]:
        """Basit: oranlık tahmin (kullanıcı etrafında ayarlasın)."""
        W, H = width, height
        return {
            "left_shoulder": (int(W * 0.33), int(H * 0.15)),
            "right_shoulder": (int(W * 0.67), int(H * 0.15)),
            "mid_waist": (int(W * 0.50), int(H * 0.42)),
        }


# ── Hazır şablonlar ────────────────────────────────────────────────────────

TEMPLATE_NAMES = ["Plain Shirt", "Blazer", "Sweater", "T-Shirt"]
TEMPLATE_SHOULDERS = (0.12, 0.10, 0.88, 0.10)  # şablonlarda omuz uçlarının oranları
_TPL_SIZE = 600


def _shade(rgb, factor: float) -> tuple:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def make_template(name: str, color_rgb=(40, 70, 140)) -> np.ndarray:
    """Basit üst giyim şablonu (BGRA, 600x600). color_rgb: (R, G, B). Pillow ile çizilir."""
    from PIL import Image, ImageDraw

    S = 2
    N = _TPL_SIZE * S
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    base = tuple(int(c) for c in color_rgb) + (255,)
    dark = _shade(color_rgb, 0.65) + (255,)
    light = (245, 245, 245, 255)
    se = 250 if name == "T-Shirt" else 570  # kol ucu y konumu

    body = [(230, 50), (72, 60), (0, se - 20), (70, se + 10), (120, 210), (120, 600),
            (480, 600), (480, 210), (530, se + 10), (600, se - 20), (528, 60), (370, 50),
            (335, 95), (300, 115), (265, 95)]
    d.polygon(P(body), fill=base)

    if name == "Plain Shirt":
        d.polygon(P([(230, 50), (300, 118), (252, 152), (205, 95)]), fill=light)
        d.polygon(P([(370, 50), (300, 118), (348, 152), (395, 95)]), fill=light)
        d.line(P([(300, 118), (300, 600)]), fill=dark, width=4 * S)
        for y in (200, 300, 400, 500):
            d.ellipse(P([(294, y - 6), (306, y + 6)]), fill=light)
    elif name == "Blazer":
        d.polygon(P([(230, 50), (300, 330), (255, 300), (205, 120)]), fill=dark)
        d.polygon(P([(370, 50), (300, 330), (345, 300), (395, 120)]), fill=dark)
        for y in (390, 470):
            d.ellipse(P([(290, y - 8), (310, y + 8)]), fill=_shade(color_rgb, 0.4) + (255,))
    elif name == "Sweater":
        d.arc(P([(235, 30), (365, 140)]), 0, 180, fill=dark, width=10 * S)
        for x in range(120, 481, 12):
            d.line(P([(x, 560), (x, 600)]), fill=dark, width=2 * S)
    else:
        d.arc(P([(240, 35), (360, 135)]), 0, 180, fill=dark, width=8 * S)

    img = img.resize((_TPL_SIZE, _TPL_SIZE), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

