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

from faceunlock.config import FaceUnlockConfig
from faceunlock.engine import FaceEngine, VerificationResult
from faceunlock.hardware import CameraDevice, CameraType, select_camera


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
            detection_threshold=0.6,
            cosine_threshold=self.config.cosine_threshold,
            data_dir=self.config.data_dir or None,
            min_liveness_threshold=self.config.min_liveness_score,
        )

    def authenticate_user(self, user: str) -> Tuple[bool, str]:
        """Perform face authentication for user.

        Returns (success: bool, status_message: str).
        """
        if not self.config.pam_enabled:
            return False, "Face unlock is disabled in configuration."

        if not self.engine.is_enrolled(user):
            return False, f"User '{user}' has no enrolled face vectors; skipping to next PAM module."

        if self.config.check_lid and is_laptop_lid_closed():
            return False, "Laptop lid is closed; skipping face unlock."

        # Select camera device
        preferred = None if self.config.camera_device == "auto" else self.config.camera_device
        try:
            camera_dev, selection_reason = select_camera(preferred)
        except Exception as e:
            return False, f"Camera selection error: {e}"

        # Open video stream
        # Try V4L2 backend
        cap = cv2.VideoCapture(camera_dev.path, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Try default backend
            cap = cv2.VideoCapture(camera_dev.path)
            if not cap.isOpened():
                return False, f"Failed to open video capture device {camera_dev.path} (camera may be in use)."

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Warmup camera auto-exposure and white-balance
            for _ in range(self.config.warmup_frames):
                ret, _ = cap.read()
                if not ret:
                    break

            start_time = time.monotonic()
            frames_processed = 0
            is_ir = camera_dev.camera_type == CameraType.INFRARED

            self.engine.liveness_detector.reset()

            while (time.monotonic() - start_time) < self.config.max_timeout:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frames_processed += 1
                result: VerificationResult = self.engine.verify(
                    user=user,
                    frame=frame,
                    is_ir_mode=is_ir,
                    require_liveness=self.config.require_liveness,
                )

                if result.matched:
                    elapsed = time.monotonic() - start_time
                    msg = (
                        f"Face authentication verified for {user} in {elapsed:.2f}s "
                        f"(confidence={result.confidence:.3f}, liveness={result.liveness.verdict}, "
                        f"device={camera_dev.type_label})"
                    )
                    return True, msg

                # Short sleep to prevent busy waiting
                time.sleep(0.01)

            elapsed = time.monotonic() - start_time
            return False, f"Face authentication timed out after {elapsed:.2f}s ({frames_processed} frames checked)."

        finally:
            cap.release()


def pam_main(argv: Optional[list[str]] = None) -> int:
    """CLI / PAM executable entrypoint.

    Exit codes:
    0 = PAM_SUCCESS (User authenticated)
    1 = PAM_AUTH_ERR (Authentication failed or fallback needed)
    """
    args = argv or sys.argv[1:]
    user = args[0] if args else os.environ.get("PAM_USER") or os.getlogin()

    authenticator = PamAuthenticator()
    success, message = authenticator.authenticate_user(user)

    if authenticator.config.show_feedback:
        if success:
            sys.stderr.write(f"\033[92m[FaceUnlock]\033[0m {message}\n")
        else:
            sys.stderr.write(f"\033[93m[FaceUnlock]\033[0m {message}\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(pam_main())
