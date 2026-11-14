"""Giriş noktası — GUI (varsayılan) veya CLI --cli ile test modu."""
from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def cli_mode(args) -> int:
    """Headless test mode: fake camera üzerinden pipeline çalışır, FPS/CPU raporu."""
    from .camera import FakeCamera
    import time
    import os

    # Windows'da CPU ölçümü için psutil yoksa geçici ölçüm
    cpu_available = False
    try:
        import psutil
        cpu_available = True
    except ImportError:
        pass

    outfit = None
    if getattr(args, "garment", None):
        from .outfit.overlay import OverlayProvider
        outfit = OverlayProvider({"mode": "overlay"})
        if not outfit.load_garment(args.garment):
            logger.error("Could not load garment image: %s", args.garment)
            return 1
    pipeline = Pipeline(
        camera_source=args.source,
        use_fake_camera=True,
        fake_source=args.source,
        safety_policy=args.policy or "regional",
        outfit_provider=outfit,
        enable_panik=False,
        use_virtual_camera=not getattr(args, "no_virtualcam", False),
    )
    if not pipeline.start():
        logger.error("Could not start pipeline")
        return 1

    try:
        frame_count = 0
        t_start = time.monotonic()
        while frame_count < args.frames:
            frame = pipeline.get_last_frame()
            if frame is not None:
                frame_count += 1
            time.sleep(0.01)
            if frame_count % 30 == 0:
                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info("Frames processed: %d, FPS: %.1f", frame_count, fps)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    t_end = time.monotonic()
    elapsed = t_end - t_start
    logger.info("Processed %d frames in %.1f seconds, %.1f FPS", frame_count, elapsed,
                frame_count / elapsed if elapsed > 0 else 0)
    return 0


def gui_mode() -> int:
    """GUI modu."""
    from .gui import run_gui
    return run_gui()


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeCamera - webcam protection")
    sub = parser.add_subparsers(dest="command")

    p_cli = sub.add_parser("cli", help="Headless test mode")
    p_cli.add_argument("--source", type=str, default="test.mp4", help="Video/image file")
    p_cli.add_argument("--policy", type=str, choices=["regional", "solid", "cooldown", "overlay"],
                       default="regional", help="Safety policy")
    p_cli.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    p_cli.add_argument("--no-virtualcam", action="store_true", help="Do not open the virtual camera")
    p_cli.add_argument("--garment", type=str, default=None, help="Garment image (PNG/JPG)")
    p_cli.set_defaults(func=cli_mode)

    p_gui = sub.add_parser("gui", help="GUI mode (default)")
    p_gui.set_defaults(func=lambda args: gui_mode())

    args = parser.parse_args()
    if not hasattr(args, "func"):
        sys.exit(gui_mode())  # komut verilmezse arayüz açılır

    ret = args.func(args)
    sys.exit(ret if isinstance(ret, int) else 0)


if __name__ == "__main__":
    main()