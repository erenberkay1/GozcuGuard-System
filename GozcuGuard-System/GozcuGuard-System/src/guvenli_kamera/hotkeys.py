"""Global panik tuşu — `keyboard` ile (fallback: Windows API)."""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# keyboard kütüphanesini dene; Windows'da çalışır
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None


class HotkeyManager:
    """Global kısayol dinleyici (Ctrl+Shift+B gibi)."""

    def __init__(self, key_combo: str = "ctrl+shift+b",
                 on_toggle: Optional[Callable[[bool], None]] = None):
        self.key_combo = key_combo
        self.on_toggle = on_toggle
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._panik_active = False
        self._hook = None

    def start(self) -> bool:
        """Dinlemeyi başlat."""
        if not KEYBOARD_AVAILABLE:
            logger.warning("keyboard library missing, panic hotkey disabled")
            return False
        if self._running:
            return True
        try:
            keyboard.add_hotkey(self.key_combo, self._on_hotkey, suppress=False)
            self._running = True
            logger.info("Listening for panic hotkey: %s", self.key_combo)
            return True
        except Exception as exc:
            logger.error("Could not register panic hotkey: %s", exc)
            return False

    def _on_hotkey(self) -> None:
        """Hotkey tetiklendiğinde."""
        self._panik_active = not self._panik_active
        logger.info("Panic hotkey: %s", "ON" if self._panik_active else "OFF")
        if self.on_toggle:
            try:
                self.on_toggle(self._panik_active)
            except Exception as exc:
                logger.error("Panic callback error: %s", exc)

    def is_panic_active(self) -> bool:
        return self._panik_active

    def stop(self) -> None:
        """Dinlemeyi durdur."""
        if not self._running:
            return
        try:
            keyboard.remove_hotkey(self.key_combo)
        except Exception:
            pass
        self._running = False
        logger.info("Panic hotkey listener stopped")

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled