"""NudeNet sarmalayıcı — çıplaklık algılama.

NudeNet v3 sonucu: {"class": str, "score": float, "box": [x, y, w, h]}.
Hata durumunda DetectorError fırlatılır; çağıran taraf fail-closed davranmalıdır.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Varsayılan olarak SADECE bunlar blurlanır (yüz, ayak, karın vb. blurlanmaz).
BLUR_CLASSES = [
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]


class DetectorError(RuntimeError):
    """Model yüklenemedi veya çıkarım başarısız oldu."""


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": self.confidence,
                "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


class Detector:
    def __init__(self, inference_resolution: int = 320, providers: Optional[list] = None,
                 include_male_breast: bool = False):
        self.inference_resolution = inference_resolution
        self.providers = providers
        self._model = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self.include_male_breast = include_male_breast

    def load(self) -> bool:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from nudenet import NudeDetector
            self._model = NudeDetector(inference_resolution=self.inference_resolution,
                                       providers=self.providers)
            self._loaded = True
            self._load_error = None
            logger.info("NudeNet model loaded (resolution=%d)", self.inference_resolution)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Could not load NudeNet: %s", exc)
            return False

    def set_include_male_breast(self, include: bool) -> None:
        """Üstsüz erkek sınıfını dahil et / hariç tut (runtime değişiklik)."""
        self.include_male_breast = bool(include)

    def _allowed_classes(self) -> set:
        """Küresel izin listesi: MALE_BREAST_EXPOSED opsiyonel olarak hariç tutulur."""
        base = list(BLUR_CLASSES)
        if not self.include_male_breast:
            base = [c for c in base if c != "MALE_BREAST_EXPOSED"]
        return set(base)

    def detect(self, frame: np.ndarray, threshold: float = 0.35,
               allowed_classes: Optional[list] = None) -> list:
        """Kareyi tara. Başarısızlıkta DetectorError fırlatır (asla sessizce [] dönmez)."""
        if not self._ensure_loaded():
            raise DetectorError(self._load_error or "model could not be loaded")
        try:
            results = self._model.detect(frame)
        except Exception as exc:
            raise DetectorError(str(exc)) from exc

        allowed = set(allowed_classes) if allowed_classes else self._allowed_classes()
        h, w = frame.shape[:2]
        out = []
        # NudeNet yüzleri de bulur (FACE_*). Hassas bir sınıf kutusu büyük oranda bir YÜZÜN içindeyse
        # bu sahte alarmdır (yüzü blurlamayalım).
        faces = []
        for r in results:
            try:
                if str(r["class"]).startswith("FACE_") and float(r["score"]) >= 0.3:
                    fx, fy, fw, fh = [int(v) for v in r["box"]]
                    faces.append((fx, fy, fx + fw, fy + fh))
            except Exception:
                continue
        for r in results:
            try:
                cls = r["class"]
                conf = float(r["score"])
                if "box" in r:
                    bx, by, bw, bh = [int(v) for v in r["box"]]
                    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                else:  # eski/alternatif biçim
                    x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            except Exception:
                continue
            if conf < threshold or cls not in allowed:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if self._mostly_inside_face((x1, y1, x2, y2), faces):
                logger.debug("Ignored false positive inside a face: %s %.2f", cls, conf)
                continue
            out.append(Detection(cls, conf, x1, y1, x2, y2))
        return out

    @staticmethod
    def _mostly_inside_face(box, faces, ratio: float = 0.5) -> bool:
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        for fx1, fy1, fx2, fy2 in faces:
            iw = min(x2, fx2) - max(x1, fx1)
            ih = min(y2, fy2) - max(y1, fy1)
            if iw > 0 and ih > 0 and (iw * ih) / area > ratio:
                return True
        return False

    def is_loaded(self) -> bool:
        return self._loaded

    def load_error(self) -> Optional[str]:
        return self._load_error
