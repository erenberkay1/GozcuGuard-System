"""Yapılandırma yönetimi: ayarlar, profiller, JSON kayıt."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

CONFIG_DIR = Path(os.getenv("APPDATA", Path.home())) / "GuvenliKamera"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class DetectorConfig:
    """Çıplaklık algılama ayarları."""
    enabled: bool = True
    threshold: float = 0.35
    classes: list[str] = field(default_factory=lambda: [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    ])
    inference_resolution: int = 320
    detection_interval: int = 2  # her N karede bir


@dataclass
class BlurConfig:
    """Blur/pikselate ayarları."""
    mode: str = "regional"  # regional, solid, cooldown, overlay
    kernel_size: int = 51
    sigma: float = 30.0
    box_expand_ratio: float = 0.40
    temporal_hold_frames: int = 15
    cooldown_seconds: float = 3.0
    overlay_text: str = "Camera temporarily disabled"
    male_breast_exposed: bool = False  # Üstsüz erkek sınıfını da blurla


@dataclass
class OutfitConfig:
    """Kıyafet sistemi ayarları."""
    mode: str = "off"  # off, light, realistic
    garment_path: str = ""
    anchor_points: dict[str, list[float]] = field(default_factory=dict)
    profile_name: str = "varsayılan"
    realistic_prompt: str = "Substitute the current top with a navy blazer and white shirt"
    realistic_api_key: str = ""


@dataclass
class BackgroundConfig:
    """Arka plan ayarları."""
    enabled: bool = False
    mode: str = "blur"  # blur, color, image
    blur_strength: int = 51
    color: list[int] = field(default_factory=lambda: [0, 0, 0])
    image_path: str = ""


@dataclass
class PerformanceConfig:
    """Performans profili ayarları."""
    profile: str = "Dengeli"  # Hafif, Dengeli, Kaliteli
    auto_adjust: bool = True
    target_fps: int = 20
    detection_resolution: int = 320
    process_every_n_frames: int = 1
    use_gpu: bool = False


@dataclass
class CameraConfig:
    """Kamera ayarları."""
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror_preview: bool = True


@dataclass
class HotkeyConfig:
    """Panik tuşu ayarları."""
    enabled: bool = True
    key_combo: str = "ctrl+shift+b"


@dataclass
class UIConfig:
    """Arayüz ayarları."""
    theme: str = "dark"
    window_width: int = 900
    window_height: int = 700
    debug_mode: bool = False


@dataclass
class AppConfig:
    """Ana uygulama yapılandırması."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    outfit: OutfitConfig = field(default_factory=OutfitConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        config = cls()
        for key, value in data.items():
            if hasattr(config, key) and isinstance(value, dict):
                dataclass_type = type(getattr(config, key))
                setattr(config, key, dataclass_type.from_dict(value) if hasattr(dataclass_type, 'from_dict') else dataclass_type(**value))
            elif hasattr(config, key):
                setattr(config, key, value)
        return config


def load_config() -> AppConfig:
    """Yapılandırmayı dosyadan yükle, yoksa varsayılanları döndür."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig.from_dict(data)
        except Exception:
            pass
    return AppConfig()


def save_config(config: AppConfig) -> None:
    """Yapılandırmayı dosyaya kaydet."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_performance_preset(profile: str) -> dict[str, Any]:
    """Performans profili için önceden tanımlı ayarları döndür."""
    presets = {
        "Hafif": {
            "detection_resolution": 256,
            "detection_interval": 3,
            "process_every_n_frames": 2,
            "blur_kernel_size": 31,
            "target_fps": 15,
        },
        "Dengeli": {
            "detection_resolution": 320,
            "detection_interval": 2,
            "process_every_n_frames": 1,
            "blur_kernel_size": 51,
            "target_fps": 20,
        },
        "Kaliteli": {
            "detection_resolution": 640,
            "detection_interval": 1,
            "process_every_n_frames": 1,
            "blur_kernel_size": 71,
            "target_fps": 30,
        },
    }
    return presets.get(profile, presets["Dengeli"])