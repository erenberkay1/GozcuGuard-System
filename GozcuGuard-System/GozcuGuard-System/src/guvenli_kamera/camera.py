"""Kamera listesi/yakalama, FakeCamera (test videosu/görselinden kare üretilir)."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Gerçek kamera (OpenCV VideoCapture)."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._opened = False

    def _open_once(self, use_mjpg: bool) -> bool:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.device_index, backend)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue
            if use_mjpg:
                # MJPG: sıkıştırılmamış (YUY2) modda birçok USB kamera 720p'de sadece ~10 FPS verir
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # düşük gecikme
            ok = False
            for _ in range(3):  # ilk kareler bazen boş gelir
                ok, _frame = cap.read()
                if ok:
                    break
            if ok:
                self.cap = cap
                return True
            cap.release()
        return False

    def open(self) -> bool:
        """Open the camera (tries MJPG first, then the default format)."""
        try:
            self.cap = None
            if not (self._open_once(True) or self._open_once(False)):
                logger.error("Could not open camera (device=%s)", self.device_index)
                self._opened = False
                return False
            self._opened = True
            logger.info("Camera opened (device=%s, %dx%d, %.0f fps reported)", self.device_index,
                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        self.cap.get(cv2.CAP_PROP_FPS))
            return True
        except Exception as exc:
            logger.error("Error while opening camera: %s", exc)
            self._opened = False
            return False

    def read(self) -> Optional[np.ndarray]:
        """Kare oku. Başarılı olursa BGR frame döndürür."""
        if not self._opened or self.cap is None:
            return None
        try:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
        except Exception as exc:
            logger.error("Camera read error: %s", exc)
        return None

    def close(self) -> None:
        """Kamerayı kapat."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened

    def actual_resolution(self) -> tuple[int, int]:
        """Kameranın gerçek çözünürlüğünü döndür."""
        if not self._opened or self.cap is None:
            return (self.width, self.height)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            return (w, h)
        return (self.width, self.height)


class FakeCamera:
    """Test amaçlı kamera: video/görsel dosyasından kare üretilir."""

    def __init__(self, source: str = "", width: int = 1280, height: int = 720, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._opened = False
        self._loop: bool = True
        self._frame_idx: int = 0
        self._start_time: float = 0.0

    def open(self) -> bool:
        if not self.source or not os.path.exists(self.source):
            logger.error("FakeCamera source yok: %s", self.source)
            self._opened = False
            return False
        ext = os.path.splitext(self.source)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self._opened = False
                return False
            self._opened = True
            self._loop = True
        else:
            # Görsel dosyası
            self._frame = cv2.imread(self.source)
            if self._frame is None:
                logger.error("Could not read image: %s", self.source)
                self._opened = False
                return False
            self._frame = cv2.resize(self._frame, (self.width, self.height))
            self._opened = True
            self._loop = True
        self._start_time = time.monotonic()
        return True

    def read(self) -> Optional[np.ndarray]:
        if not self._opened:
            return None
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self._loop:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                    ok, frame = self.cap.read()
                if not ok or frame is None:
                    return None
            return frame
        else:
            # Görsel modu: FPS'e göre döngü
            elapsed = time.monotonic() - self._start_time
            target_idx = int(elapsed * self.fps)
            if target_idx >= 1 and self._loop:
                return self._frame.copy()
            return self._frame.copy()

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened

    def actual_resolution(self) -> tuple[int, int]:
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (self.width, self.height)