"""Configuration management for Linux Face Unlock.

Supports reading from /etc/faceunlock/config.ini or ~/.config/faceunlock/config.ini.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH_SYSTEM = "/etc/faceunlock/config.ini"
DEFAULT_CONFIG_PATH_USER = os.path.expanduser("~/.config/faceunlock/config.ini")


@dataclass
class FaceUnlockConfig:
    # Camera settings
    camera_device: str = "auto"
    fallback_to_webcam: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30

    # Security settings
    cosine_threshold: float = 0.33
    min_liveness_score: float = 0.55
    require_liveness: bool = True
    max_timeout: float = 3.5
    warmup_frames: int = 4
    check_lid: bool = True

    # PAM settings
    pam_enabled: bool = True
    show_feedback: bool = True

    # Paths
    models_dir: str = ""
    data_dir: str = ""
    config_path: str = ""

    @classmethod
    def load(cls, custom_path: Optional[str] = None) -> FaceUnlockConfig:
        cfg = cls()

        search_paths = []
        if custom_path:
            search_paths.append(custom_path)
        if os.geteuid() != 0:
            search_paths.extend([
                DEFAULT_CONFIG_PATH_USER,
                DEFAULT_CONFIG_PATH_SYSTEM,
                str(Path(__file__).resolve().parent.parent / "config.ini"),
            ])
        else:
            search_paths.extend([
                DEFAULT_CONFIG_PATH_SYSTEM,
                DEFAULT_CONFIG_PATH_USER,
                str(Path(__file__).resolve().parent.parent / "config.ini"),
            ])

        found_path = ""
        parser = configparser.ConfigParser()
        for p in search_paths:
            if os.path.exists(p):
                try:
                    parser.read(p)
                    found_path = p
                    break
                except Exception:
                    pass

        cfg.config_path = found_path

        if parser.has_section("camera"):
            cam = parser["camera"]
            cfg.camera_device = cam.get("device", cfg.camera_device)
            cfg.fallback_to_webcam = cam.getboolean("fallback_to_webcam", cfg.fallback_to_webcam)
            cfg.width = cam.getint("width", cfg.width)
            cfg.height = cam.getint("height", cfg.height)
            cfg.fps = cam.getint("fps", cfg.fps)

        if parser.has_section("security"):
            sec = parser["security"]
            cfg.cosine_threshold = sec.getfloat("cosine_threshold", cfg.cosine_threshold)
            cfg.min_liveness_score = sec.getfloat("min_liveness_score", cfg.min_liveness_score)
            cfg.require_liveness = sec.getboolean("require_liveness", cfg.require_liveness)
            cfg.max_timeout = sec.getfloat("max_timeout", cfg.max_timeout)
            cfg.warmup_frames = sec.getint("warmup_frames", cfg.warmup_frames)
            cfg.check_lid = sec.getboolean("check_lid", cfg.check_lid)

        if parser.has_section("pam"):
            pam = parser["pam"]
            cfg.pam_enabled = pam.getboolean("enabled", cfg.pam_enabled)
            cfg.show_feedback = pam.getboolean("show_feedback", cfg.show_feedback)

        if parser.has_section("paths"):
            paths = parser["paths"]
            cfg.models_dir = paths.get("models_dir", cfg.models_dir)
            cfg.data_dir = paths.get("data_dir", cfg.data_dir)

        return cfg

    def save(self, path: Optional[str] = None) -> str:
        """Save current configuration to INI file."""
        target = path or self.config_path or (
            DEFAULT_CONFIG_PATH_SYSTEM if os.geteuid() == 0 else DEFAULT_CONFIG_PATH_USER
        )
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

        parser = configparser.ConfigParser()
        parser["camera"] = {
            "device": self.camera_device,
            "fallback_to_webcam": str(self.fallback_to_webcam).lower(),
            "width": str(self.width),
            "height": str(self.height),
            "fps": str(self.fps),
        }
        parser["security"] = {
            "cosine_threshold": f"{self.cosine_threshold:.3f}",
            "min_liveness_score": f"{self.min_liveness_score:.2f}",
            "require_liveness": str(self.require_liveness).lower(),
            "max_timeout": f"{self.max_timeout:.1f}",
            "warmup_frames": str(self.warmup_frames),
            "check_lid": str(self.check_lid).lower(),
        }
        parser["pam"] = {
            "enabled": str(self.pam_enabled).lower(),
            "show_feedback": str(self.show_feedback).lower(),
        }
        parser["paths"] = {
            "models_dir": self.models_dir,
            "data_dir": self.data_dir,
        }

        with open(target, "w", encoding="utf-8") as f:
            parser.write(f)

        self.config_path = target
        return target
