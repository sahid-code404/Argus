"""PAM authentication handler for Linux Face Unlock.

Executes during PAM authentication requests (e.g. sudo, login, gdm, swaylock).
Handles:
- Fast camera access & auto-exposure warmup
- Hardware selection (Depth / IR / universal RGB fallback)
- Lid-closed and screen-inhibit detection
- Multi-frame liveness verification
- Clean exit codes for PAM conversation (0 = SUCCESS, 1 = AUTH_ERR / FALLBACK)
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

from argus.config import ArgusConfig, FaceUnlockConfig
from argus.engine import FaceEngine, VerificationResult
from argus.hardware import CameraDevice, CameraType, select_camera


def is_laptop_lid_closed() -> bool:
    """Check if the laptop lid is physically closed via ACPI."""
    for lid_state_path in glob.glob("/proc/acpi/button/lid/*/state"):
        try:
            with open(lid_state_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                if "closed" in content:
                    return True
        except OSError:
            pass
    return False


class PamAuthenticator:
    """Manages the real-time biometric verification flow for PAM."""

    def __init__(self, config: Optional[FaceUnlockConfig] = None):
        self.config = config or FaceUnlockConfig.load()
        self.engine = FaceEngine(
            model_dir=self.config.models_dir or None,
            detection_threshold=0.45,
            cosine_threshold=self.config.cosine_threshold,
            data_dir=self.config.data_dir or None,
            min_liveness_threshold=self.config.min_liveness_score,
            use_tpm=self.config.use_tpm,
        )

    def authenticate_user(self, user: str, service: Optional[str] = None) -> Tuple[bool, str]:
        """Perform simultaneous biometric authentication for user.

        Returns (success: bool, status_message: str).
        """
        service_name = service or os.environ.get("PAM_SERVICE", "sudo")
        if not self.config.is_service_enabled(service_name):
            return False, f"Face unlock disabled for PAM service: {service_name}"

        if not self.engine.is_enrolled(user):
            return False, f"User '{user}' has no enrolled face vectors; skipping to next PAM module."

        if self.config.check_lid and is_laptop_lid_closed():
            return False, "Laptop lid is closed; skipping face unlock."

        from argus.simultaneous_biometrics import SimultaneousBiometrics
        sim_auth = SimultaneousBiometrics(self.engine, self.config)
        res = sim_auth.authenticate(user)
        return res.success, res.message


def pam_main(argv: Optional[list[str]] = None) -> int:
    """CLI / PAM executable entrypoint.

    Exit codes:
    0 = PAM_SUCCESS (User authenticated)
    1 = PAM_AUTH_ERR (Authentication failed or fallback needed)
    """
    try:
        args = argv or sys.argv[1:]
        user = args[0] if args else (os.environ.get("PAM_USER") or "")
        if not user or user == "root":
            user = os.environ.get("SUDO_USER") or user or os.environ.get("USER") or os.getlogin()
        user = user.strip() if user else "user"

        authenticator = PamAuthenticator()
        success, message = authenticator.authenticate_user(user)

        return 0 if success else 1
    except (KeyboardInterrupt, EOFError):
        return 1
    except Exception as e:
        sys.stderr.write(f"\033[91m[Argus PAM Error]\033[0m {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pam_main())
