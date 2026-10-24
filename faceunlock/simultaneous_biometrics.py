"""Simultaneous multi-modal biometric authentication for Linux Face Unlock.

Runs Face Unlock and Fingerprint reader concurrently in parallel threads:
- Whichever modality succeeds first immediately grants authentication.
- Non-blocking, real-time D-Bus / fprintd verification.
- Displays compact, colorful ASCII animations while scanning.
- Cleanly cancels and releases hardware resources upon verification or timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

from faceunlock.config import FaceUnlockConfig
from faceunlock.engine import FaceEngine, VerificationResult
from faceunlock.hardware import CameraDevice, CameraType, select_camera
from faceunlock.ui import BiometricSpinner


@dataclass
class BiometricAuthResult:
    success: bool
    method: Optional[str]  # "FACE", "FINGERPRINT", or None
    message: str
    details: Optional[dict] = None


class FingerprintVerifier:
    """Manages asynchronous fingerprint verification via fprintd."""

    def __init__(self, user: str):
        self.user = user
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.verified = False
        self.error: Optional[str] = None

    @staticmethod
    def is_available_for_user(user: str) -> bool:
        """Check if fprintd is installed and the user has enrolled fingerprints."""
        fprint_list_bin = shutil.which("fprintd-list")
        if not fprint_list_bin:
            return False
        try:
            res = subprocess.run(
                [fprint_list_bin, user],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
            )
            # Check if enrolled fingers were found
            if res.returncode == 0 and "has no fingers enrolled" not in res.stdout:
                return True
        except Exception:
            pass
        return False

    def start(self, on_match_callback=None) -> bool:
        """Start non-blocking fingerprint verification in a worker thread."""
        fprint_verify_bin = shutil.which("fprintd-verify")
        if not fprint_verify_bin:
            return False

        def _worker():
            try:
                self.proc = subprocess.Popen(
                    [fprint_verify_bin, self.user],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                if not self.proc.stdout:
                    return

                for line in iter(self.proc.stdout.readline, ""):
                    if self._stop_event.is_set():
                        break
                    line_lower = line.lower()
                    if "verify-match (done)" in line_lower:
                        self.verified = True
                        if on_match_callback:
                            on_match_callback("FINGERPRINT")
                        break
                    elif "verify-no-match" in line_lower:
                        # Mismatched finger touched; sensor can still retry
                        pass
            except Exception as e:
                self.error = str(e)
            finally:
                self.stop()

        self._thread = threading.Thread(target=_worker, name="FingerprintVerifierThread", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop verification and release fingerprint device."""
        self._stop_event.set()
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            except Exception:
                pass
            self.proc = None


class SimultaneousBiometrics:
    """Orchestrates simultaneous Face and Fingerprint biometric authentication."""

    def __init__(self, engine: FaceEngine, config: FaceUnlockConfig):
        self.engine = engine
        self.config = config

    def authenticate(self, user: str) -> BiometricAuthResult:
        """Run simultaneous face unlock and fingerprint reader verification."""
        auth_event = threading.Event()
        stop_event = threading.Event()
        result_holder = {"method": None, "message": "Verification timed out", "success": False}

        def on_auth_success(method: str):
            if not auth_event.is_set():
                result_holder["method"] = method
                result_holder["success"] = True
                result_holder["message"] = f"Access Granted ({method.capitalize()})"
                auth_event.set()
                stop_event.set()

        # 1. Initialize Fingerprint Verifier if available
        fp_verifier: Optional[FingerprintVerifier] = None
        has_fp = FingerprintVerifier.is_available_for_user(user)
        if has_fp:
            fp_verifier = FingerprintVerifier(user)
            fp_verifier.start(on_match_callback=on_auth_success)

        # 2. Initialize Face Verification Worker
        face_failure_reason = "No face detected"

        def _face_worker():
            nonlocal face_failure_reason
            preferred = None if self.config.camera_device == "auto" else self.config.camera_device
            try:
                camera_dev, _ = select_camera(preferred)
            except Exception as e:
                face_failure_reason = f"Camera error: {e}"
                return

            cap = cv2.VideoCapture(camera_dev.path, cv2.CAP_V4L2)
            if not cap.isOpened():
                if camera_dev.path.startswith("/dev/video"):
                    try:
                        dev_num = int(camera_dev.path.replace("/dev/video", ""))
                        cap = cv2.VideoCapture(dev_num)
                    except Exception:
                        pass
                if not cap.isOpened():
                    cap = cv2.VideoCapture(camera_dev.path)
                    if not cap.isOpened():
                        face_failure_reason = f"Failed to open video capture device {camera_dev.path} (camera may be in use)"
                        return

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                cap.set(cv2.CAP_PROP_FPS, self.config.fps)

                for _ in range(self.config.warmup_frames):
                    if stop_event.is_set():
                        return
                    ret, _ = cap.read()
                    if not ret:
                        break

                is_ir = camera_dev.camera_type == CameraType.INFRARED
                self.engine.liveness_detector.reset()
                match_score = 0
                min_required_matches = 1 if is_ir else 2

                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue

                    res: VerificationResult = self.engine.verify(
                        user=user,
                        frame=frame,
                        is_ir_mode=is_ir,
                        require_liveness=self.config.require_liveness,
                    )

                    if res.face_detected:
                        if not res.liveness.is_live:
                            face_failure_reason = "Spoof attack detected (presentation rejected)"
                        elif not res.matched:
                            face_failure_reason = "Unrecognized face"

                    if res.matched and res.liveness.is_live:
                        match_score += 1
                        if match_score >= min_required_matches:
                            on_auth_success("FACE")
                            return
                    else:
                        match_score = max(0, match_score - 1)

                    time.sleep(0.01)
            finally:
                cap.release()

        face_thread = threading.Thread(target=_face_worker, name="FaceWorkerThread", daemon=True)
        face_thread.start()

        # 3. Main Loop: Spinner Animation & Timeout
        spinner_msg = (
            "Scanning face or touch sensor..."
            if has_fp
            else "Looking for face..."
        )
        spinner = BiometricSpinner(message=spinner_msg)

        start_time = time.monotonic()
        timeout = self.config.max_timeout

        try:
            while not auth_event.is_set() and (time.monotonic() - start_time) < timeout:
                if self.config.show_feedback:
                    spinner.tick()
                time.sleep(0.08)

            if auth_event.is_set():
                if self.config.show_feedback:
                    method_label = "Face Verified" if result_holder["method"] == "FACE" else "Fingerprint Verified"
                    spinner.success(f"Access Granted ({method_label})")
                return BiometricAuthResult(
                    success=True,
                    method=result_holder["method"],
                    message="Access Granted",
                )
            else:
                stop_event.set()
                if self.config.show_feedback:
                    spinner.failure(face_failure_reason)
                return BiometricAuthResult(
                    success=False,
                    method=None,
                    message=face_failure_reason,
                )
        finally:
            stop_event.set()
            if fp_verifier:
                fp_verifier.stop()
            face_thread.join(timeout=0.5)
