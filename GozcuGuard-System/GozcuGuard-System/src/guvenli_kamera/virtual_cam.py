"""pyvirtualcam sarmalayıcı — sanal kameraya kare gönder."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# pyvirtualcam içe aktarma — yoksa graceful degradation
try:
    import pyvirtualcam
    PYVC_AVAILABLE = True
except ImportError:
    PYVC_AVAILABLE = False
    pyvirtualcam = None


class VirtualCamera:
    """OBS Virtual Camera / sanal kamera sürücüsü için yazıcı."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 fmt: int = 0):  # 0 = BGR (pyvirtualcam.PixelFormat.BGR)
        self.width = width
        self.height = height
        self.fps = fps
        self.fmt = fmt
        self._cam: Optional[object] = None
        self.error: Optional[str] = None
        self._opened = False
        self._lock = threading.Lock()

    def open(self) -> bool:
        """Sanal kamerayı aç. Driver yoksa False, hata loglar."""
        if not PYVC_AVAILABLE:
            logger.error("pyvirtualcam kurulu değil")
            self._opened = False
            return False
        try:
            fmt_enum = pyvirtualcam.PixelFormat.BGR if self.fmt == 0 else pyvirtualcam.PixelFormat.RGB
            self._cam = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=self.fps, fmt=fmt_enum
            )
            self._opened = True
            logger.info("Sanal kamera açıldı (%dx%d @ %dfps)", self.width, self.height, self.fps)
            return True
        except Exception as exc:
            self.error = ("Virtual camera could not be opened. Install OBS Studio and open it once "
                          "(the OBS Virtual Camera driver is required). Details: %s" % exc)
            logger.error(self.error)
            self._opened = False
            return False

    def send(self, frame_bgr: np.ndarray) -> bool:
        """BGR frame'i sanal kameraya gönder. Thread-safe."""
        if not self._opened or self._cam is None:
            return False
        with self._lock:
            try:
                # Frame boyutunu sanal kameraya uydur
                h, w = frame_bgr.shape[:2]
                if (w, h) != (self.width, self.height):
                    frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                self._cam.send(frame_bgr)
                self._cam.sleep_until_next_frame()
                return True
            except Exception as exc:
                logger.error("Sanal kamera gönderim hatası: %s", exc)
                return False

    def close(self) -> None:
        """Sanal kamerayı kapat."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
        self._opened = False
        logger.info("Sanal kamera kapatıldı")

    @property
    def is_opened(self) -> bool:
        return self._opened


class NullVirtualCamera:
    """Sanal kamera yokken kullanılan no-op writer (CLI/test modu)."""

    def __init__(self, *args, **kwargs):
        self._opened = True
        self._frame = None

    def open(self) -> bool:
        self._opened = True
        return True

    def send(self, frame_bgr: np.ndarray) -> bool:
        self._frame = frame_bgr
        return True

    def close(self) -> None:
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened