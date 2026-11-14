"""Hafif kıyafet modu: kıyafet görselini omuz çizgisine oturtan 2D overlay (CPU).

Dürüst not: bu bir "yapıştırma"dır; yapay zeka ile yeniden çizim değildir.
Sakin, oturan mülakat senaryosunda makul görünür; hızlı harekette fark edilebilir.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .base import OutfitProvider

logger = logging.getLogger(__name__)

# Arayüz kaydırıcıları buraya yazar (tüm sağlayıcılar okur): boyut çarpanı, yatay/dikey kaydırma (omuz mesafesi oranı)
OVERLAY_ADJUST = {"scale": 1.0, "dx": 0.0, "dy": 0.0}


def remove_flat_background(bgr: np.ndarray) -> np.ndarray:
    """Düz (beyaz/tek renk) arka planlı görselden alfa kanalı üret. BGRA döndürür."""
    h, w = bgr.shape[:2]
    corners = np.array([bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]], np.float32)
    if corners.std(axis=0).max() > 30:  # karmaşık arka plan: dokunma
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    alpha = np.clip((dist - 18) / 30.0, 0, 1) * 255
    alpha = cv2.morphologyEx(alpha.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    return out


class OverlayProvider(OutfitProvider):
    """Kıyafet görselini omuz çizgisine oturtan 2D overlay (CPU, hafif)."""

    # Kıyafet görselinde omuz uçlarının oranları: (sol_x, sol_y, sağ_x, sağ_y)
    DEFAULT_SHOULDERS = (0.15, 0.06, 0.85, 0.06)
    PAD_RATIO = 0.22   # kıyafet omuz noktalarının dışına taşma payı (omuz mesafesine oranla)
    LIFT_RATIO = 0.15  # pose noktası omuz EKLEMİ, kıyafet üst kenarı omuz SIRTI: bu kadar yukarı al
    

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.garment: Optional[np.ndarray] = None  # BGRA
        self._ready = False
        self._smooth: Optional[np.ndarray] = None
        self._fade = 0.0
        self.shoulders = tuple(self.anchor_points.get("shoulders", self.DEFAULT_SHOULDERS))
        self._last_garment_w: int = 0
        self._last_garment_h: int = 0

        # BURASI EKLENECEK: Config içinden gelen dosya yolunu kontrol et ve otomatik yükle
        if config:
            path = config.get("garment_path") or config.get("path") or config.get("image_path")
            if path:
                self.load_garment(path)

    def load_garment(self, path: str) -> bool:
        """Kıyafet görselini yükle. Büyük dosyaları (genişlik > 1000px) indirgen.

        Bağlantı koparsa veya boyut çok büyükse anlık kare işlemine engel olur.
        """
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("görsel okunamadı")

            # Boyut sınırı: genişlik > 1000px ise orantılı küçült (her karede ağır işlem yapmamak için)
            h_orig, w_orig = img.shape[:2] if img.ndim >= 3 else (0, 0)
            max_w = 1000
            if w_orig > max_w:
                scale = max_w / float(w_orig)
                new_h = max(1, int(h_orig * scale))
                img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info("Kıyafet görseli küçültüldü: %d×%d -> %d×%d", w_orig, h_orig, max_w, new_h)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                img = remove_flat_background(img)
            # img.shape[2] == 4 zaten BGRA
            img = self._crop_to_content(img)  # ürün fotoğraflarındaki boş kenarları at
            if "shoulders" not in self.anchor_points:  # elle verilmediyse görselden bul
                self.shoulders = self._auto_shoulders(img)
                logger.info("Kıyafet omuz oranları otomatik bulundu: %s", self.shoulders)

            self.garment = img
            self._ready = True
            self._smooth = None
            self._last_garment_w = self.garment.shape[1]
            self._last_garment_h = self.garment.shape[0]
            return True
        except Exception as exc:
            logger.error("Kıyafet görseli yüklenemedi: %s (%s)", path, exc)
            self._ready = False
            return False

    def _grid(self, h: int, w: int):
        g = getattr(self, "_grid_cache", None)
        if g is None or g[0] != (h, w):
            yy, xx = np.mgrid[0:h, 0:w]
            self._grid_cache = ((h, w), xx.astype(np.float32), yy.astype(np.float32))
            g = self._grid_cache
        return g[1], g[2]

    @staticmethod
    def _auto_shoulders(bgra: np.ndarray) -> tuple:
        """Kıyafet görselinden omuz noktalarını otomatik bul (kollar yana açık düz fotoğraflar dahil).

        Mantık: merkez sütunu içeren sürekli dolu şeridin genişliğine bakılır. Etek hizasındaki şerit
        gövde genişliğidir; omuz çizgisi, şeridin gövde genişliğine ilk ulaştığı satırdır.
        """
        a = bgra[:, :, 3] > 40
        H, W = a.shape
        ys, xs = np.where(a)
        if len(xs) == 0:
            return OverlayProvider.DEFAULT_SHOULDERS
        cx = int(xs.mean())
        widths = np.zeros(H, np.float32)
        for y in range(H):
            row = a[y]
            if not row.any():
                continue
            idx = np.where(row)[0]
            c = int(idx[np.argmin(np.abs(idx - cx))])  # merkeze en yakın dolu piksel
            l = c
            while l > 0 and row[l - 1]:
                l -= 1
            r = c
            while r < W - 1 and row[r + 1]:
                r += 1
            widths[y] = r - l + 1
        lo, hi = int(0.80 * H), max(int(0.80 * H) + 1, int(0.95 * H))
        hem = float(np.median(widths[lo:hi]))
        if hem < 0.2 * W:
            hem = float(W)
        y_idx = np.where(widths >= 0.9 * hem)[0]
        y_sh = int(y_idx[0]) if len(y_idx) else int(0.1 * H)
        y_sh = min(y_sh, int(0.25 * H))
        x1 = max(0.0, (cx - 0.5 * hem) / W)
        x2 = min(1.0, (cx + 0.5 * hem) / W)
        return (x1, y_sh / H, x2, y_sh / H)

    @staticmethod
    def _crop_to_content(bgra: np.ndarray) -> np.ndarray:
        """Şeffaf/boş kenarları kırp: omuz oranları artık gerçek kıyafete göre hesaplanır."""
        a = bgra[:, :, 3]
        ys, xs = np.where(a > 20)
        if len(ys) == 0:
            return bgra
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) * (x1 - x0) < 0.02 * a.size:
            return bgra
        return bgra[y0:y1, x0:x1].copy()

    def set_garment_array(self, bgra: np.ndarray, shoulders=None) -> bool:
        """Hazır (örn. şablon) BGRA kıyafet dizisini ayarla."""
        try:
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                raise ValueError("BGRA (4 kanallı) dizi bekleniyor")
            self.garment = bgra.copy()
            if shoulders is not None:
                self.shoulders = tuple(shoulders)
            self._ready = True
            self._smooth = None
            self._last_garment_w = self.garment.shape[1]
            self._last_garment_h = self.garment.shape[0]
            return True
        except Exception as exc:
            logger.error("Kıyafet dizisi ayarlanamadı: %s", exc)
            self._ready = False
            return False

    def process(self, frame: np.ndarray, pose: Optional[dict] = None,
                person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._ready or self.garment is None:
            return frame

        ls = pose.get("left_shoulder") if pose else None
        rs = pose.get("right_shoulder") if pose else None
        if not (ls and rs):
            self._fade = max(0.0, self._fade - 0.15)  # poz kayıp: yavaşça gerçek görüntüye dön
            if self._fade <= 0 or self._smooth is None:
                return frame
        else:
            pts = sorted([ls, rs], key=lambda p: p[0])
            cur = np.array([*pts[0], *pts[1]], np.float32)
            self._smooth = cur if self._smooth is None else 0.6 * self._smooth + 0.4 * cur
            self._fade = min(1.0, self._fade + 0.2)

        lx, ly, rx, ry = [float(v) for v in self._smooth]
        L, R = np.array([lx, ly], np.float32), np.array([rx, ry], np.float32)
        d = R - L
        dist = float(np.linalg.norm(d))
        if dist < 20:
            return frame
        u = d / dist
        n = np.array([u[1], -u[0]], np.float32)  # omuz çizgisine dik, YUKARI yönlü

        adj = OVERLAY_ADJUST
        mid = (L + R) / 2 + n * (self.LIFT_RATIO * dist) + np.array([adj["dx"] * dist, adj["dy"] * dist], np.float32)
        half = (dist / 2 + self.PAD_RATIO * dist) * float(adj["scale"])
        dst_l, dst_r = mid - u * half, mid + u * half

        gh, gw = self.garment.shape[:2]
        sl = np.array([self.shoulders[0] * gw, self.shoulders[1] * gh], np.float32)
        sr = np.array([self.shoulders[2] * gw, self.shoulders[3] * gh], np.float32)
        sv, dv = sr - sl, dst_r - dst_l
        s = float(np.linalg.norm(dv) / max(np.linalg.norm(sv), 1e-3))
        th = math.atan2(dv[1], dv[0]) - math.atan2(sv[1], sv[0])
        c, sn = s * math.cos(th), s * math.sin(th)
        R_m = np.array([[c, -sn], [sn, c]], np.float32)
        t = dst_l - R_m @ sl
        M = np.hstack([R_m, t.reshape(2, 1)])

        h, w = frame.shape[:2]
        warped = cv2.warpAffine(self.garment, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        nz = cv2.findNonZero(np.ascontiguousarray(warped[:, :, 3]))
        if nz is None:
            return frame
        bx, by, bw, bh = cv2.boundingRect(nz)  # sadece kıyafetin kapladığı bölgede çalış (hız)
        x0, y0, x1, y1 = bx, by, bx + bw, by + bh
        roi = warped[y0:y1, x0:x1]
        alpha = roi[:, :, 3].astype(np.float32) / 255.0

        # 1) Yaka boşluğu: boyun/çene gerçek görüntü kalsın (omuz ortasına göre konumlanır)
        neck = np.zeros_like(alpha)
        angle = math.degrees(math.atan2(float(u[1]), float(u[0])))
        cv2.ellipse(neck, (int(mid[0]) - x0, int(mid[1]) - y0),
                    (max(1, int(0.20 * dist)), max(1, int(0.24 * dist))), angle, 0, 360, 1.0, -1)
        alpha *= (1.0 - neck)

        # 1b) Omuz çizgisinin belirgin üstü (yaka ucu, boyun) hiç çizilmesin: çene/ağız kapanmasın
        gx, gy = self._grid(h, w)
        up = (gx[y0:y1, x0:x1] - mid[0]) * n[0] + (gy[y0:y1, x0:x1] - mid[1]) * n[1]  # >0: omuz çizgisinin üstü
        alpha *= (up < 0.08 * dist).astype(np.float32)

        # 2) Kişi maskesi varsa kıyafet sadece kişinin siluetinde kalsın (arka plana taşmasın)
        if person_mask is not None:
            pm = cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
            pm = cv2.dilate(pm.astype(np.float32) / 255.0, np.ones((15, 15), np.uint8))
            alpha *= pm

        # 3) Kenar yumuşatma, ışığa kaba uyum, solma
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0) * self._fade
        gain = float(np.clip(frame[::4, ::4].mean() / 110.0, 0.8, 1.2))
        gar = np.clip(roi[:, :, :3].astype(np.float32) * gain, 0, 255)
        base = frame[y0:y1, x0:x1].astype(np.float32)
        out = frame.copy()
        out[y0:y1, x0:x1] = (base * (1 - alpha[:, :, None]) + gar * alpha[:, :, None]).astype(np.uint8)
        return out

    def get_status(self) -> str:
        return "No garment loaded" if not self._ready else "Light (overlay) mode active"