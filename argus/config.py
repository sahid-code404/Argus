"""Configuration management for Argus Biometric Authentication.

Supports reading from /etc/argus/config.ini or ~/.config/argus/config.ini (with fallback to legacy paths).
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH_SYSTEM = "/etc/argus/config.ini"
DEFAULT_CONFIG_PATH_USER = os.path.expanduser("~/.config/argus/config.ini")
LEGACY_CONFIG_PATH_SYSTEM = "/etc/faceunlock/config.ini"
LEGACY_CONFIG_PATH_USER = os.path.expanduser("~/.config/faceunlock/config.ini")


@dataclass
class ArgusConfig:
    # Camera settings
    camera_device: str = "auto"
    fallback_to_webcam: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30

    # Security settings
    cosine_threshold: float = 0.38
    min_liveness_score: float = 0.65
    require_liveness: bool = True
    max_timeout: float = 3.5
    warmup_frames: int = 4
    check_lid: bool = True
    use_tpm: bool = False
    security_mode: str = "balanced"
    min_consensus_frames: int = 3

    # Policies
    policy_sudo: bool = True
    policy_polkit: bool = True
    policy_lockscreen: bool = True
    policy_sshd: bool = False

    @property
    def policy_gdm(self) -> bool:
        """Deprecated alias for policy_lockscreen; login managers are disabled."""
        return self.policy_lockscreen

    @policy_gdm.setter
    def policy_gdm(self, value: bool) -> None:
        self.policy_lockscreen = bool(value)

    @property
    def policy_lockers(self) -> bool:
        return self.policy_lockscreen

    @policy_lockers.setter
    def policy_lockers(self, value: bool) -> None:
        self.policy_lockscreen = bool(value)

    @property
    def policy_login(self) -> bool:
        return False

    @policy_login.setter
    def policy_login(self, value: bool) -> None:
        pass

    # Performance
    cache_camera_params: bool = True

    # PAM settings
    pam_enabled: bool = True
    show_feedback: bool = True
    feedback_style: str = "animated"

    # Paths
    models_dir: str = ""
    data_dir: str = ""
    config_path: str = ""

    def apply_security_mode(self, mode: str) -> None:
        """Apply pre-calibrated security mode profile."""
        m = mode.lower()
        if m == "convenient":
            self.security_mode = "convenient"
            self.cosine_threshold = 0.35
            self.min_liveness_score = 0.55
            self.max_timeout = 2.5
            self.min_consensus_frames = 2
        elif m == "strict":
            self.security_mode = "strict"
            self.cosine_threshold = 0.44
            self.min_liveness_score = 0.75
            self.max_timeout = 4.0
            self.min_consensus_frames = 4
        else:
            self.security_mode = "balanced"
            self.cosine_threshold = 0.38
            self.min_liveness_score = 0.65
            self.max_timeout = 3.5
            self.min_consensus_frames = 3

    def is_service_enabled(self, service_name: str) -> bool:
        """Check if Face Unlock is permitted for the given PAM service name."""
        if not self.pam_enabled:
            return False

        name = service_name.lower().strip()
        if "sudo" in name:
            return self.policy_sudo
        elif "polkit" in name:
            return self.policy_polkit
        elif any(k in name for k in ("lock", "screensaver")):
            return self.policy_lockscreen
        elif "ssh" in name:
            return self.policy_sshd
        elif any(k in name for k in ("gdm", "sddm", "lightdm", "login")):
            # Session login managers are explicitly disabled to prevent GNOME Keyring lockouts
            return False
        return False

    @classmethod
    def load(cls, custom_path: Optional[str] = None) -> ArgusConfig:
        cfg = cls()

        search_paths = []
        if custom_path:
            search_paths.append(custom_path)
        if os.geteuid() != 0:
            search_paths.extend([
                DEFAULT_CONFIG_PATH_USER,
                DEFAULT_CONFIG_PATH_SYSTEM,
                LEGACY_CONFIG_PATH_USER,
                LEGACY_CONFIG_PATH_SYSTEM,
                str(Path(__file__).resolve().parent.parent / "config.ini"),
            ])
        else:
            search_paths.extend([
                DEFAULT_CONFIG_PATH_SYSTEM,
                DEFAULT_CONFIG_PATH_USER,
                LEGACY_CONFIG_PATH_SYSTEM,
                LEGACY_CONFIG_PATH_USER,
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
            cfg.use_tpm = sec.getboolean("use_tpm", cfg.use_tpm)
            cfg.security_mode = sec.get("security_mode", cfg.security_mode)
            cfg.min_consensus_frames = sec.getint("min_consensus_frames", cfg.min_consensus_frames)

        if parser.has_section("policies"):
            pol = parser["policies"]
            cfg.policy_sudo = pol.getboolean("sudo", cfg.policy_sudo)
            cfg.policy_polkit = pol.getboolean("polkit", cfg.policy_polkit)
            if "lockscreen" in pol:
                cfg.policy_lockscreen = pol.getboolean("lockscreen", cfg.policy_lockscreen)
            elif "lockers" in pol:
                cfg.policy_lockscreen = pol.getboolean("lockers", cfg.policy_lockscreen)
            elif "gdm" in pol:
                cfg.policy_lockscreen = pol.getboolean("gdm", cfg.policy_lockscreen)
            cfg.policy_sshd = pol.getboolean("sshd", cfg.policy_sshd)

        if parser.has_section("performance"):
            perf = parser["performance"]
            cfg.cache_camera_params = perf.getboolean("cache_camera_params", cfg.cache_camera_params)

        if parser.has_section("pam"):
            pam = parser["pam"]
            cfg.pam_enabled = pam.getboolean("enabled", cfg.pam_enabled)
            cfg.show_feedback = pam.getboolean("show_feedback", cfg.show_feedback)
            cfg.feedback_style = pam.get("feedback_style", cfg.feedback_style)

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
            "use_tpm": str(self.use_tpm).lower(),
            "security_mode": self.security_mode,
            "min_consensus_frames": str(self.min_consensus_frames),
        }
        parser["policies"] = {
            "sudo": str(self.policy_sudo).lower(),
            "polkit": str(self.policy_polkit).lower(),
            "lockscreen": str(self.policy_lockscreen).lower(),
            "sshd": str(self.policy_sshd).lower(),
        }
        parser["performance"] = {
            "cache_camera_params": str(self.cache_camera_params).lower(),
        }
        parser["pam"] = {
            "enabled": str(self.pam_enabled).lower(),
            "show_feedback": str(self.show_feedback).lower(),
            "feedback_style": self.feedback_style,
        }
        parser["paths"] = {
            "models_dir": self.models_dir,
            "data_dir": self.data_dir,
        }

        with open(target, "w", encoding="utf-8") as f:
            parser.write(f)

        self.config_path = target
        return target


# Compatibility alias
FaceUnlockConfig = ArgusConfig
