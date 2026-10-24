"""Simultaneous multi-modal biometric authentication for Linux Face Unlock.

Runs Face Unlock and Fingerprint reader concurrently in parallel threads:
- Whichever modality succeeds first immediately grants authentication.
- Real-time PTY-based fprintd verification without I/O buffering.
- Runs fprintd with target user privileges and '-f any' to accept any enrolled finger.
- Directly renders live ASCII animation on /dev/tty so PAM output displays in real time.
- Shows clean prompt: 'Looking for face (or use fingerprint / password)...'
- If no fingerprint reader: 'Looking for face (or use password)...'
- On success: Direct work! Shows nothing!
- On failure: Shows only '[FaceUnlock] Face unlock failed' and nothing else!
"""

from __future__ import annotations

import os
import pty
import select
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

from faceunlock.config import FaceUnlockConfig
from faceunlock.engine import FaceEngine, VerificationResult
from faceunlock.hardware import CameraDevice, CameraType, select_camera


@dataclass
class BiometricAuthResult:
    success: bool
    method: Optional[str]  # "FACE", "FINGERPRINT", or None
    message: str
    details: Optional[dict] = None


class FingerprintVerifier:
    """Manages real-time asynchronous fingerprint verification via fprintd."""

    def __init__(self, user: str):
        self.user = user
        self.proc: Optional[subprocess.Popen] = None
        self._master: Optional[int] = None
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

        # Drop privileges to target user if running as root
        preexec = None
        try:
            import pwd
            pw = pwd.getpwnam(user)
            target_uid = pw.pw_uid
            target_gid = pw.pw_gid
            if os.geteuid() == 0 and target_uid != 0:
                def _drop():
                    os.setgid(target_gid)
                    os.setuid(target_uid)
                preexec = _drop
        except Exception:
            pass

        try:
            cmd = [fprint_list_bin] if preexec else [fprint_list_bin, user]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
                preexec_fn=preexec,
            )
            if res.returncode == 0 and "has no fingers enrolled" not in res.stdout and "- #" in res.stdout:
                return True
        except Exception:
            pass
        return False

    def start(self, on_match_callback=None) -> bool:
        """Start non-blocking fingerprint verification using a pseudo-terminal."""
        fprint_verify_bin = shutil.which("fprintd-verify")
        if not fprint_verify_bin:
            return False

        # Drop privileges to target user if running as root so polkit allows verification
        preexec = None
        try:
            import pwd
            pw = pwd.getpwnam(self.user)
            target_uid = pw.pw_uid
            target_gid = pw.pw_gid
            if os.geteuid() == 0 and target_uid != 0:
                def _drop():
                    os.setgid(target_gid)
                    os.setuid(target_uid)
                preexec = _drop
        except Exception:
            pass

        try:
            master, slave = pty.openpty()
            self._master = master
            # Pass -f any to accept any enrolled finger (index, thumb, left, right)
            cmd = [fprint_verify_bin, "-f", "any"]
            self.proc = subprocess.Popen(
                cmd,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                close_fds=True,
                preexec_fn=preexec,
            )
            os.close(slave)
        except Exception as e:
            self.error = str(e)
            return False

        def _worker():
            buffer = ""
            try:
                while not self._stop_event.is_set():
                    r, _, _ = select.select([self._master], [], [], 0.05)
                    if not r:
                        if self.proc and self.proc.poll() is not None:
                            if self.proc.poll() == 0:
                                self.verified = True
                                if on_match_callback:
                                    on_match_callback("FINGERPRINT")
                            break
                        continue

                    try:
                        chunk = os.read(self._master, 1024).decode("utf-8", errors="ignore")
                        if not chunk:
                            break
                        buffer += chunk.lower()
                        if "verify-match" in buffer:
                            self.verified = True
                            if on_match_callback:
                                on_match_callback("FINGERPRINT")
                            break
                    except OSError:
                        break
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
                        self.proc.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            except Exception:
                pass
            self.proc = None
        if self._master is not None:
            try:
                os.close(self._master)
            except OSError:
                pass
            self._master = None


def _open_tty_stream():
    """Attempt to open /dev/tty directly so output bypasses PAM pipe buffering."""
    try:
        return open("/dev/tty", "w")
    except OSError:
        return sys.stderr


class SimultaneousBiometrics:
    """Orchestrates simultaneous Face and Fingerprint biometric authentication."""

    def __init__(self, engine: FaceEngine, config: FaceUnlockConfig):
        self.engine = engine
        self.config = config

    def authenticate(self, user: str) -> BiometricAuthResult:
        """Run simultaneous face unlock and fingerprint reader verification."""
        auth_event = threading.Event()
        stop_event = threading.Event()
        result_holder = {"method": None, "message": "Face unlock failed", "success": False}

        def on_auth_success(method: str):
            if not auth_event.is_set():
                result_holder["method"] = method
                result_holder["success"] = True
                result_holder["message"] = "Access Granted"
                auth_event.set()
                stop_event.set()

        # 1. Initialize Fingerprint Verifier if available
        fp_verifier: Optional[FingerprintVerifier] = None
        has_fp = FingerprintVerifier.is_available_for_user(user)
        if has_fp:
            fp_verifier = FingerprintVerifier(user)
            fp_verifier.start(on_match_callback=on_auth_success)

        # 2. Setup TTY Stream for real-time terminal display
        tty = _open_tty_stream() if self.config.show_feedback else None
        is_tty = tty is not None and hasattr(tty, "isatty") and tty.isatty()

        if has_fp:
            prompt_str = "[FaceUnlock] Looking for face (or use fingerprint / password)..."
        else:
            prompt_str = "[FaceUnlock] Looking for face (or use password)..."

        if tty and not is_tty:
            tty.write(prompt_str + "\n")
            tty.flush()

        # 3. Initialize Face Verification Worker
        face_failure_reason = "No face detected"

        def _face_worker():
            nonlocal face_failure_reason
            preferred = None if self.config.camera_device == "auto" else self.config.camera_device
            try:
                camera_dev, _ = select_camera(preferred)
            except Exception as e:
                face_failure_reason = f"Failed to open video capture device: {e}"
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

        # 4. Main Loop: Real-Time ASCII Animation on TTY
        frames = ["◖ ◉ ◗", "◖ ◈ ◗", "◖ ● ◗", "◖ ◎ ◗"]
        colors = ["\033[1;36m", "\033[1;35m", "\033[1;33m", "\033[1;32m"]
        frame_idx = 0
        start_time = time.monotonic()
        timeout = self.config.max_timeout

        try:
            while not auth_event.is_set() and (time.monotonic() - start_time) < timeout:
                if tty and is_tty:
                    f = frames[frame_idx % len(frames)]
                    c = colors[frame_idx % len(colors)]
                    frame_idx += 1
                    tty.write(f"\r\033[K{prompt_str} {c}[ {f} ]\033[0m")
                    tty.flush()
                time.sleep(0.08)

            if auth_event.is_set():
                if tty:
                    if is_tty:
                        tty.write("\r\033[K")
                    tty.flush()
                # On success: show NOTHING! Direct work!
                return BiometricAuthResult(
                    success=True,
                    method=result_holder["method"],
                    message="Access Granted",
                )
            else:
                stop_event.set()
                if tty:
                    if is_tty:
                        tty.write("\r\033[K[FaceUnlock] Face unlock failed\n")
                    else:
                        tty.write("[FaceUnlock] Face unlock failed\n")
                    tty.flush()

                err_msg = (
                    face_failure_reason
                    if "failed to open" in face_failure_reason.lower()
                    else "Face unlock failed"
                )
                return BiometricAuthResult(
                    success=False,
                    method=None,
                    message=err_msg,
                )
        finally:
            stop_event.set()
            if fp_verifier:
                fp_verifier.stop()
            face_thread.join(timeout=0.5)
            if tty and tty != sys.stderr:
                try:
                    tty.close()
                except OSError:
                    pass
