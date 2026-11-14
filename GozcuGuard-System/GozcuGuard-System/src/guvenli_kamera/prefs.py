"""User preferences (which protections are enabled) stored as a small JSON file."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULTS = {
    "onboarding_done": False,   # welcome dialog already shown?
    "nudity_blur": True,        # recommended default, the user is asked on first start
    "background_blur": False,
    "male_chest": False,
}


def prefs_path() -> str:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "SafeCamera", "prefs.json")


def load_prefs(path: Optional[str] = None) -> dict:
    data = dict(DEFAULTS)
    try:
        with open(path or prefs_path(), "r", encoding="utf-8") as f:
            stored = json.load(f)
        for k in DEFAULTS:
            if k in stored and isinstance(stored[k], bool):
                data[k] = stored[k]
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not read preferences: %s", exc)
    return data


def save_prefs(prefs: dict, path: Optional[str] = None) -> bool:
    target = path or prefs_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump({k: bool(prefs.get(k, v)) for k, v in DEFAULTS.items()}, f, indent=2)
        return True
    except Exception as exc:
        logger.warning("Could not save preferences: %s", exc)
        return False
