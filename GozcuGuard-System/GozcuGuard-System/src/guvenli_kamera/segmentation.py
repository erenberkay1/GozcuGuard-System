"""Kişi maskesi — MediaPipe ImageSegmenter (arkaplan blur/değiştir)."""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .paths import model_path
MODEL_PATH = model_path("selfie_segmenter.tflite")


class Segmentation:
    """Kişi maskesi üretir (0=arka plan, 1=kişi)."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._segmenter = None
        self._loaded = False
        self._error: Optional[str] = None
        self._prev: Optional[np.ndarray] = None  # zamansal yumuşatma

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = ImageSegmenterOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
            self._segmenter = ImageSegmenter.create_from_options(options)
            self._loaded = True
            logger.info("ImageSegmenter modeli yüklendi")
            return True
        except Exception as exc:
            self._error = str(exc)
            logger.warning("Segmenter modeli yüklenemedi: %s", exc)
            return False

    def get_mask(self, frame_bgr: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        """İkili maske (h, w) döndürür, uint8, 0-255."""
        if not self._ensure_loaded():
            return None
        try:
            import mediapipe as mp
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._segmenter.segment(mp_image)
            category_mask = result.confidence_masks[0].numpy_view() if hasattr(
                result.confidence_masks[0], 'numpy_view'
            ) else np.array(result.confidence_masks[0])
            # Sert eşik yerine YUMUŞAK maske: kulak/saç kenarları kesilmesin
            conf = np.asarray(category_mask, dtype=np.float32)
            lo = max(0.05, threshold - 0.25)
            soft = np.clip((conf - lo) / 0.5, 0.0, 1.0)
            fw = soft.shape[1]
            k = max(5, fw // 90) | 1
            soft = cv2.dilate(soft, np.ones((k, k), np.uint8))  # kişiyi hafif genişlet
            soft = cv2.GaussianBlur(soft, (0, 0), max(2.0, fw / 250.0))
            mask = (soft * 255).astype(np.uint8)
            if self._prev is not None and self._prev.shape == mask.shape:
                mask = cv2.addWeighted(mask, 0.6, self._prev, 0.4, 0)  # titremeyi azalt
            self._prev = mask
            return mask
        except Exception as exc:
            logger.error("Segmentasyon hatası: %s", exc)
            return None


def blur_background(frame: np.ndarray, mask: np.ndarray, kernel_size: int = 51,
                    sigma: float = 30.0) -> np.ndarray:
    """Blur the background and keep the person sharp.

    The blur is computed on a down-scaled copy and scaled back (looks the same, ~10x faster).
    """
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    f = 4 if w >= 960 else 2
    small = cv2.resize(frame, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = max(3, (kernel_size // f) | 1)
    small = cv2.GaussianBlur(small, (k, k), sigmaX=max(1.0, sigma / f), sigmaY=max(1.0, sigma / f))
    blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[:, :, None]
    out = frame.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)
    return out.astype(np.uint8)


def replace_background(frame: np.ndarray, mask: np.ndarray,
                       bg_image: Optional[np.ndarray] = None,
                       bg_color: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """Arka planı değiştir (görsel veya renk)."""
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_f = mask.astype(np.float32) / 255.0
    mask_3c = cv2.merge([mask_f, mask_f, mask_f])

    if bg_image is not None:
        bg = cv2.resize(bg_image, (w, h), interpolation=cv2.INTER_LINEAR)
    elif bg_color is not None:
        bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    else:
        return frame

    out = frame * mask_3c + bg * (1 - mask_3c)
    return out.astype(np.uint8)


def approx_person_mask(shape, pose) -> Optional[np.ndarray]:
    """Rough "portrait" mask (head + shoulders + torso) from shoulder points.

    Used for background blur when MediaPipe is unavailable. Returns uint8 (h, w) 0-255 or None.
    """
    if not pose:
        return None
    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    if not (ls and rs):
        return None
    h, w = shape[:2]
    mx, my = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    dist = max(1.0, abs(rs[0] - ls[0]))
    m = np.zeros((h, w), np.uint8)
    he = head_ellipse(shape, pose)
    if he:
        cv2.ellipse(m, (he[0], he[1]), (he[2], he[3]), 0, 0, 360, 255, -1)
    else:
        cv2.ellipse(m, (int(mx), int(my - 0.70 * dist)), (int(0.50 * dist), int(0.70 * dist)), 0, 0, 360, 255, -1)
    body = np.array([[mx - 0.75 * dist, my - 0.30 * dist], [mx + 0.75 * dist, my - 0.30 * dist],
                     [mx + 1.3 * dist, h], [mx - 1.3 * dist, h]], np.int32)
    cv2.fillConvexPoly(m, body, 255)
    return cv2.GaussianBlur(m, (0, 0), max(3.0, 0.05 * dist))



def head_ellipse(shape, pose):
    """Kafa (kulaklar ve saç dahil) için cömert bir elips: (cx, cy, yarı_genişlik, yarı_yükseklik) ya da None."""
    if not pose:
        return None
    fb = pose.get("face_box")
    if fb:
        x, y, w, h = fb
        return int(x + w / 2), int(y + h * 0.42), int(0.68 * w), int(0.85 * h)
    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    dist = abs(rs[0] - ls[0]) if (ls and rs) else 0
    le, re_, nose = pose.get("left_ear"), pose.get("right_ear"), pose.get("nose")
    if le and re_:
        ax = max(abs(re_[0] - le[0]) * 0.62, 0.22 * dist)
        cy = (le[1] + re_[1]) / 2.0
        return int((le[0] + re_[0]) / 2), int(cy), int(ax), int(1.25 * ax)
    if nose and dist:
        ax = 0.34 * dist
        return int(nose[0]), int(nose[1] - 0.1 * ax), int(ax), int(1.3 * ax)
    return None


def refine_person_mask(mask, pose):
    """Maskeye kafa bölgesini ekle: arka plan blur'u kulakları/saçı asla bulanıklaştırmasın."""
    if mask is None:
        return None
    he = head_ellipse(mask.shape, pose)
    if not he:
        return mask
    head = np.zeros(mask.shape[:2], np.uint8)
    cv2.ellipse(head, (he[0], he[1]), (he[2], he[3]), 0, 0, 360, 255, -1)
    head = cv2.GaussianBlur(head, (0, 0), max(2.0, 0.12 * he[2]))
    return np.maximum(mask, head)
