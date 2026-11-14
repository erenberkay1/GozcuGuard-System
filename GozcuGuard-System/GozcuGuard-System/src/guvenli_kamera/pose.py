"""Üst gövde/omuz noktaları — MediaPipe PoseLandmarker + OpenCV yüz tabanlı yedek."""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Landmark indeksleri: MediaPipe PoseModel'de omuz ve boyun
# 0 = nose, 11 = sol omuz, 12 = sağ omuz, 23 = sol kalça, 24 = sağ kalça
OMUZ_SOL = 11
OMUZ_SAG = 12
BAS = 0

from .paths import model_path
MODEL_PATH = model_path("pose_landmarker_lite.task")


class PoseEstimator:
    """Üst gövde noktalarını (omuz, boyun, kalça) tahmin eder.

    Öncelik: MediaPipe (model dosyası varsa) -> OpenCV yüz tabanlı yedek.
    Yedek: cv2.data.haarcascades içindeki Haar Cascade ile yüz tespiti, omuz oranlarına göre tahmini.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._landmarker = None
        self._loaded = False
        self._error: Optional[str] = None
        self._fallback_cascade = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._loaded = True
            logger.info("PoseLandmarker model loaded (%s)", self.model_path)
            return True
        except Exception as exc:
            self._error = str(exc)
            logger.warning("MediaPipe PoseLandmarker unavailable (%s), using the OpenCV fallback: %s",
                          self.model_path, exc)
            return False

    def _ensure_fallback_loaded(self) -> bool:
        if self._fallback_cascade is not None:
            return True
        name = "haarcascade_frontalface_default.xml"
        candidates = []
        try:
            candidates.append(os.path.join(cv2.data.haarcascades, name))  # pip wheel'lerde standart yer
        except Exception:
            pass
        try:
            candidates.append(cv2.samples.findFile(name))
        except Exception:
            pass
        for path in candidates:
            try:
                if path and os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self._fallback_cascade = cascade
                        logger.info("OpenCV face cascade loaded (fallback shoulder estimate)")
                        return True
            except Exception as exc:
                logger.warning("Could not prepare the OpenCV fallback: %s", exc)
        logger.warning("OpenCV face cascade file not found")
        return False

    def _fallback_estimate(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Yüz tabanlı omuz tahmini (CPU, her karede değil 2-3 karede bir).

        Kural: omuz genişliği ≈ yüz genişliğinin 2.4-3 katı, omuz çizgisi yüzün altından ≈
        0.7-0.9 yüz yüksekliği aşağıda, yüzün yatay merkezine simetrik.
        Oturan, kameraya bakan kişi senaryosu için yeterli.
        """
        if not self._ensure_fallback_loaded():
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._fallback_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                       minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # en büyük yüz

        # Omuz EKLEMİ tahmini (MediaPipe landmark'larıyla aynı anlam): yüz merkezine simetrik,
        # yüz genişliğinin ~1.84 katı mesafede, yüz üstünden ~1.7 yüz yüksekliği aşağıda.
        img_h, img_w = frame_bgr.shape[:2]
        cx = x + w * 0.5
        sy = int(y + h * 1.7)
        left_shoulder_x = int(cx - w * 0.92)
        right_shoulder_x = int(cx + w * 0.92)
        left_shoulder_x = max(0, min(left_shoulder_x, img_w - 1))
        right_shoulder_x = max(0, min(right_shoulder_x, img_w - 1))
        left_shoulder_y = right_shoulder_y = max(0, min(sy, img_h - 1))

        nose_x = int(x + w * 0.5)
        nose_y = int(y + h * 0.3)  # göz hizasında baş

        # Güvenlik: çok uzak noktalar olmasın, shoulders arası mesafe minimum
        shoulder_dist = ((right_shoulder_x - left_shoulder_x) ** 2 +
                        (right_shoulder_y - left_shoulder_y) ** 2) ** 0.5
        if shoulder_dist < 20:
            return None

        return {
            "left_shoulder": (left_shoulder_x, left_shoulder_y),
            "right_shoulder": (right_shoulder_x, right_shoulder_y),
            "nose": (nose_x, nose_y),
            "face_box": (int(x), int(y), int(w), int(h)),
        }

    def _mediapipe_estimate(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """MediaPipe ile tahmin (gerçek model kullanımı)."""
        try:
            import mediapipe as mp
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(mp_image)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            if not landmarks:
                return None
            h, w = frame_bgr.shape[:2]

            def pt(idx: int) -> Optional[tuple[int, int]]:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    return (int(lm.x * w), int(lm.y * h))
                return None

            left_shoulder = pt(OMUZ_SOL)
            right_shoulder = pt(OMUZ_SAG)
            nose = pt(BAS)

            return {
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder,
                "nose": nose,
                "left_ear": pt(7),
                "right_ear": pt(8),
            }
        except Exception as exc:
            logger.error("MediaPipe pose tahmini hatası: %s", exc)
            return None

    def estimate(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Frame'de omuz/kalça/boyun koordinatlarını döndürür.

        Dönüş: {"left_shoulder": (x, y), "right_shoulder": (x, y), ...} veya None
        Öncelik sırası: MediaPipe (model varsa) -> OpenCV yüz tabanlı yedek.
        """
        # MediaPipe dener
        if self._ensure_loaded():
            result = self._mediapipe_estimate(frame_bgr)
            if result:
                return result

        # Yedek yol: 3 karede bir yüz ara, aradaki karelerde son sonucu kullan (kıyafet titremesin/solmasın)
        # Basit state: self._fallback_counter her 3. called'ta sıfırlanır
        self._fallback_counter = getattr(self, "_fallback_counter", 0) + 1
        if self._fallback_counter >= 1 or getattr(self, "_fallback_last", None) is None:
            self._fallback_counter = 0
            result = self._fallback_estimate(frame_bgr)
            if result:
                self._fallback_last, self._fallback_age = result, 0
                return result
            self._fallback_age = getattr(self, "_fallback_age", 0) + 3
        else:
            self._fallback_age = getattr(self, "_fallback_age", 0) + 1
        last = getattr(self, "_fallback_last", None)
        if last is not None and getattr(self, "_fallback_age", 99) <= 15:
            return last
        self._fallback_last = None
        return None

    def get_status(self) -> str:
        """Modu ve hata durumunu bildir."""
        if self._loaded:
            return "MediaPipe shoulder detection"
        elif self._fallback_cascade is not None:
            return "OpenCV face-based shoulder estimate (fallback)"
        else:
            return "Shoulder detection unavailable"