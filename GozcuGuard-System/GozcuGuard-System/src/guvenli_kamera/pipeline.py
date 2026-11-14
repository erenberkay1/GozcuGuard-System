"""Threaded main loop: camera -> (outfit) -> background blur -> nudity blur -> virtual camera.

Two independent, opt-in protections (the user chooses in the UI):
  * Nudity blur  - NudeNet detection; blurs the detected regions (or the whole frame, depending on the policy).
  * Background blur - blurs everything except the person.

Safety rule: while the nudity blur is ON, every frame that leaves the pipeline goes through
SafetyEngine.process(). If the detection model fails (or the panic button is active) the raw frame is
NOT sent; a protection screen is sent instead. If the user switched the nudity blur OFF, frames pass untouched.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .camera import Camera, FakeCamera
from .detector import Detector, DetectorError
from .safety import SafetyEngine, protection_screen
from .virtual_cam import VirtualCamera, NullVirtualCamera
from .pose import PoseEstimator
from .segmentation import Segmentation, blur_background, approx_person_mask, refine_person_mask
from .outfit.base import OutfitProvider
from .outfit.none import NoOpOutfitProvider
from .outfit.overlay import OverlayProvider
from .hotkeys import HotkeyManager
from ._credit import credit_ok

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, camera_source: str = "0", use_fake_camera: bool = False,
                 fake_source: Optional[str] = None, safety_policy: str = "regional",
                 outfit_provider: Optional[OutfitProvider] = None, enable_panik: bool = True,
                 use_virtual_camera: bool = True, detect_every: int = 3,
                 male_breast_exposed: bool = False,
                 enable_nudity_blur: bool = True, enable_background_blur: bool = False,
                 resolution: tuple = (1280, 720)):
        self.camera_source = str(camera_source)
        self.use_fake_camera = use_fake_camera
        self.fake_source = fake_source or ""
        self.use_virtual_camera = use_virtual_camera
        self.detect_every = max(1, int(detect_every))
        self.resolution = (int(resolution[0]), int(resolution[1]))

        self.camera = None
        self.virtual = NullVirtualCamera()
        self.detector = Detector(inference_resolution=320, include_male_breast=male_breast_exposed)
        self.safety = SafetyEngine({"policy": safety_policy})
        self.outfit = outfit_provider or NoOpOutfitProvider({"mode": "off"})
        self.pose = PoseEstimator()
        self.segmenter = Segmentation()

        self.hotkey_mgr: Optional[HotkeyManager] = None
        if enable_panik:
            self.hotkey_mgr = HotkeyManager(key_combo="ctrl+shift+b", on_toggle=self.set_panic)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self.fps = 0.0
        self.error: Optional[str] = None
        self._last_det = None  # (text, time)
        self._nudity_enabled = bool(enable_nudity_blur)
        self._bg_enabled = bool(enable_background_blur)
        self._det_failed = False
        self._force_detect = True
        self._mask_mode = ""  # "MediaPipe" / "approximate"
        self._det_worker = None
        self._det_warm = False      # becomes True after the first detection result
        self._analysis = None       # cached (pose, mask) from the last analysis
        self.proc_ms = 0.0
        self.wait_ms = 0.0

    # -- control ---------------------------------------------------------
    def start(self) -> bool:
        if self._running:
            return True
        if not credit_ok():
            self.error = "This copy has been modified (author credit removed). Get the original SafeCamera by Eren Berkay Gun."
            logger.error(self.error)
            return False
        if self.use_fake_camera:
            self.camera = FakeCamera(source=self.fake_source or "test.mp4")
        else:
            idx = int(self.camera_source) if self.camera_source.isdigit() else 0
            self.camera = Camera(device_index=idx, width=self.resolution[0], height=self.resolution[1])
        if not self.camera.open():
            self.error = ("Camera could not be opened (source %s). Another program may be using the camera "
                          "(Zoom, a browser tab). Check Windows Settings > Privacy > Camera and allow desktop "
                          "apps, or try source 1." % self.camera_source)
            logger.error(self.error)
            return False

        w, h = self.camera.actual_resolution()
        if self.use_virtual_camera:
            self.virtual = VirtualCamera(width=w, height=h, fps=30)
            if not self.virtual.open():
                self.error = self.virtual.error or "Virtual camera could not be opened"
                logger.warning("%s - preview only", self.error)
        else:
            self.virtual = NullVirtualCamera()

        if self.hotkey_mgr is not None:
            self.hotkey_mgr.start()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._det_worker is not None:
            self._det_worker.stop()
            self._det_worker = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.camera is not None:
            self.camera.close()
        self.virtual.close()
        if self.hotkey_mgr is not None:
            self.hotkey_mgr.stop()

    def set_panic(self, on: bool) -> None:
        self.safety.set_panic(on)
        logger.info("Panic: %s", "ON" if on else "off")

    def set_outfit(self, provider) -> None:
        with self._frame_lock:
            self.outfit = provider
        logger.info("Outfit provider updated: %s", getattr(provider, "get_status", lambda: "?")())

    def set_nudity_blur_enabled(self, enabled: bool) -> None:
        self._nudity_enabled = bool(enabled)
        if not enabled:
            self.safety.reset()
        else:
            self._force_detect = True
            self._det_warm = False  # wait for a fresh detection before showing the camera
        logger.info("Nudity blur: %s", "ON" if enabled else "OFF")

    set_blur_regions_enabled = set_nudity_blur_enabled  # backwards-compatible alias

    def set_background_blur_enabled(self, enabled: bool) -> None:
        self._bg_enabled = bool(enabled)
        logger.info("Background blur: %s", "ON" if enabled else "OFF")

    def set_include_male_breast(self, include: bool) -> None:
        self.detector.set_include_male_breast(bool(include))

    def status(self) -> dict:
        ld = self._last_det
        last_text = ld[0] if ld and time.monotonic() - ld[1] < 10 else ""
        return {"last_detection": last_text, "fps": round(self.fps, 1),
                "nudity_blur": self._nudity_enabled, "background_blur": self._bg_enabled,
                "detector": self.detector.is_loaded(), "failed": self.safety.failed,
                "panic": self.safety.panic, "events": self.safety.event_count,
                "error": self.error, "mask_mode": self._mask_mode,
                "proc_ms": round(self.proc_ms, 1), "wait_ms": round(self.wait_ms, 1),
                "size": (self.camera.actual_resolution() if self.camera is not None else (0, 0)),
                "virtual": getattr(self.virtual, "is_opened", False)}

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    # -- internal ----------------------------------------------------------
    def _publish(self, safe: np.ndarray) -> None:
        try:
            if self.virtual.is_opened:
                self.virtual.send(safe)
        except Exception as exc:
            logger.error("Virtual camera send error: %s", exc)
        with self._frame_lock:
            self._last_frame = safe.copy()

    # worker callbacks (called from the detection thread)
    def _on_det_result(self, dets: list) -> None:
        self.safety.report(dets)
        self._det_warm = True
        if dets:
            best = max(dets, key=lambda x: x.confidence)
            self._last_det = ("%s %.2f" % (best.class_name, best.confidence), time.monotonic())

    def _on_det_error(self, exc) -> None:
        logger.error("Detection error: %s", exc)
        self.safety.report_failure()

    def _on_det_load_failed(self) -> None:
        self._det_failed = True
        self.error = "Nudity detection model could not be loaded: %s" % self.detector.load_error()
        logger.error(self.error)
        self.safety.report_failure()

    def _ensure_worker(self) -> None:
        if self._det_worker is None:
            self._det_worker = _DetectWorker(self.detector, lambda: self.safety.threshold, self._on_det_result,
                                             self._on_det_error, self._on_det_load_failed)
            self._det_worker.start()

    @staticmethod
    def _scale_pose(pose: Optional[dict], f: float) -> Optional[dict]:
        if not pose:
            return pose
        out = {}
        for k, v in pose.items():
            if isinstance(v, tuple) and v and all(isinstance(c, (int, float)) for c in v):
                out[k] = tuple(int(c * f) for c in v)
            else:
                out[k] = v
        return out

    def _analyze(self, frame: np.ndarray):
        """Pose + person mask on a small copy of the frame (much cheaper than full resolution)."""
        h, w = frame.shape[:2]
        small, f = frame, 1.0
        if w > ANALYSIS_WIDTH:
            f = ANALYSIS_WIDTH / float(w)
            small = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        pose_s = self.pose.estimate(small)
        mask_s = self._person_mask(small, pose_s)
        return (self._scale_pose(pose_s, 1.0 / f) if pose_s else None), mask_s

    def _compose(self, frame: np.ndarray, n: int) -> np.ndarray:
        outfit = self.outfit
        outfit_active = isinstance(outfit, OverlayProvider) and outfit._ready
        bg = self._bg_enabled
        if not (outfit_active or bg):
            self._analysis = None
            return frame
        if self._analysis is None or n % 2 == 1:  # analyse every 2nd frame, reuse in between
            self._analysis = self._analyze(frame)
        pose, mask = self._analysis
        composed = frame
        if outfit_active:
            try:
                # the rough (approximate) mask would clip the garment: only use the real MediaPipe mask
                composed = outfit.process(frame, pose, mask if self._mask_mode == "MediaPipe" else None)
            except Exception as exc:
                logger.error("Outfit error (using real image): %s", exc)
                composed = frame
        if bg and mask is not None:
            try:
                composed = blur_background(composed, mask)
            except Exception as exc:
                logger.error("Background blur error: %s", exc)
        return composed

    def _person_mask(self, frame: np.ndarray, pose) -> Optional[np.ndarray]:
        mask = self.segmenter.get_mask(frame)
        if mask is not None:
            self._mask_mode = "MediaPipe"
            return refine_person_mask(mask, pose)
        self._mask_mode = "approximate"
        return approx_person_mask(frame.shape, pose)

    def _run_loop(self) -> None:
        w, h = self.camera.actual_resolution()
        self._publish(protection_screen((h, w), "Loading..."))
        paced_by_virtual = isinstance(self.virtual, VirtualCamera) and self.virtual.is_opened
        n = 0
        last = time.monotonic()
        while self._running:
            t_wait = time.monotonic()
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.monotonic()
            self.wait_ms = 0.9 * self.wait_ms + 0.1 * (t0 - t_wait) * 1000.0
            nudity = self._nudity_enabled
            try:
                n += 1
                if nudity:
                    self._ensure_worker()
                    self.safety.tick()
                    if self._det_failed:
                        self.safety.report_failure()
                    elif self._force_detect or n % self.detect_every == 0:
                        self._force_detect = False
                        self._det_worker.submit(frame)  # detection runs in its own thread

                if nudity and not self._det_warm and not self._det_failed:
                    # never show the raw camera before the first detection result is in
                    safe = protection_screen(frame.shape, "Starting protection...")
                else:
                    composed = self._compose(frame, n)
                    # nudity blur is always the LAST step (and panic always applies)
                    safe = self.safety.process(composed) if (nudity or self.safety.panic) else composed
            except Exception as exc:
                logger.exception("Loop error: %s", exc)
                if nudity:  # never send a raw frame while the protection is on
                    self.safety.report_failure()
                    safe = protection_screen(frame.shape, "Protection error - camera disabled")
                else:
                    safe = frame
            self._publish(safe)

            now = time.monotonic()
            self.proc_ms = 0.9 * self.proc_ms + 0.1 * (now - t0) * 1000.0
            inst = 1.0 / max(now - last, 1e-3)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
            last = now
            if not paced_by_virtual:
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t0)))


ANALYSIS_WIDTH = 640


class _DetectWorker(threading.Thread):
    """Runs the (slow) nudity detector off the main loop. Only the newest submitted frame is processed."""

    def __init__(self, detector, get_threshold, on_result, on_error, on_load_failed):
        super().__init__(daemon=True)
        self.detector = detector
        self.get_threshold = get_threshold
        self.on_result = on_result
        self.on_error = on_error
        self.on_load_failed = on_load_failed
        self._cond = threading.Condition()
        self._frame = None
        self._halt = False

    def submit(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._halt = True
            self._cond.notify()

    def run(self) -> None:
        if not self.detector.load():
            self.on_load_failed()
            return
        while True:
            with self._cond:
                while self._frame is None and not self._halt:
                    self._cond.wait()
                if self._halt:
                    return
                frame, self._frame = self._frame, None
            try:
                self.on_result(self.detector.detect(frame, threshold=self.get_threshold()))
            except DetectorError as exc:
                self.on_error(exc)
            except Exception as exc:  # never let the thread die silently
                self.on_error(exc)
