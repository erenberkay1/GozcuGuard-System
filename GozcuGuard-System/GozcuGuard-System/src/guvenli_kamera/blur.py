"""Small blur helpers (coordinate-based API). The live pipeline uses safety.py."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FrameBuffer:
    """Keeps the last N frames."""

    def __init__(self, capacity: int = 15):
        self.capacity = capacity
        self.frames: list = []

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            self.frames.pop(0)

    def get(self) -> Optional[np.ndarray]:
        return self.frames[-1] if self.frames else None


def apply_regional_blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        expand_ratio: float = 0.40, kernel_size: int = 51, sigma: float = 30.0) -> np.ndarray:
    """Expand the box and blur only that region."""
    h, w = frame.shape[:2]
    dx = int((x2 - x1) * expand_ratio / 2)
    dy = int((y2 - y1) * expand_ratio / 2)
    xa, ya = max(0, x1 - dx), max(0, y1 - dy)
    xb, yb = min(w, x2 + dx), min(h, y2 + dy)
    out = frame.copy()
    roi = out[ya:yb, xa:xb]
    if roi.size == 0:
        return out
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    out[ya:yb, xa:xb] = cv2.GaussianBlur(roi, (k, k), sigmaX=sigma, sigmaY=sigma)
    return out


def apply_solid_blur(frame: np.ndarray, kernel_size: int = 51, sigma: float = 30.0) -> np.ndarray:
    """Blur the whole frame."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)
