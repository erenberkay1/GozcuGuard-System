"""Kaynak (assets) yolları — geliştirme ve PyInstaller exe için."""
import os
import sys


def assets_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "assets")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


def model_path(name: str) -> str:
    return os.path.join(assets_dir(), "models", name)
