"""Simultaneous multi-modal biometric authentication for Argus.

Runs Face Unlock and Fingerprint reader concurrently in parallel threads:
- Whichever modality succeeds first immediately grants authentication.
- Real-time PTY-based fprintd verification without I/O buffering.
- Runs fprintd with target user privileges and '-f any' to accept any enrolled finger.
- Directly renders live ASCII animation on /dev/tty so PAM output displays in real time.
- Shows clean prompt: '[Argus] Looking for face or fingerprint...'
- If no fingerprint reader: '[Argus] Looking for face (or use password)...'
- On success: Direct work! Shows nothing!
- On failure: Shows only '[Argus] Biometric verification failed' and nothing else!
"""

from __future__ import annotations

import os
from pathlib import Path
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

from argus.config import ArgusConfig, FaceUnlockConfig
from argus.engine import FaceEngine, VerificationResult
from argus.hardware import CameraDevice, CameraType, select_camera


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
        self._lock = threading.RLock()
        self.verified = False
        self.error: Optional[str] = None

    @staticmethod
    def is_available_for_user(user: str) -> bool:
        """Check if fprintd is installed and the user has enrolled fingerprints."""
        # Check /var/lib/fprint/<user> directory directly if readable (instant & reliable)
        try:
            fprint_dir = Path(f"/var/lib/fprint/{user}")
            if fprint_dir.exists() and any(fprint_dir.iterdir()):
                return True
        except Exception:
            pass

        # Check /var/lib/fprint for any user subfolder if running as root
        try:
            for p in Path("/var/lib/fprint").iterdir():
                if p.is_dir() and (p.name == user or user in ("root", "")):
                    if any(p.iterdir()):
                        return True
        except Exception:
            pass

        # Fallback to fprintd-list <user>
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
        """Stop verification and release fingerprint device cleanly."""
        self._stop_event.set()
        with self._lock:
            proc = self.proc
            self.proc = None
            if proc:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=0.2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception:
                    pass
            master = self._master
            self._master = None
            if master is not None:
                try:
                    os.close(master)
                except OSError:
                    pass


def _get_output_stream():
    """Locate and open the user's true interactive terminal screen to bypass PAM line buffering.

    If running under a GUI display manager (GDM, LightDM, SDDM) or without an interactive TTY,
    returns None to ensure NO raw ANSI escapes are emitted into GUI login dialogues.
    """
    # 1. Try parent process (sudo) standard descriptors
    try:
        ppid = os.getppid()
        for fd in (2, 1, 0):
            target = os.readlink(f"/proc/{ppid}/fd/{fd}")
            if target.startswith("/dev/pts/") or target.startswith("/dev/tty"):
                return open(f"/proc/{ppid}/fd/{fd}", "w")
    except Exception:
        pass

    # 2. Try /dev/tty directly if accessible
    try:
        if os.path.exists("/dev/tty"):
            fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
            return os.fdopen(fd, "w")
    except Exception:
        pass

    # 3. Only if sys.stdout is an actual interactive TTY
    try:
        if sys.stdout and sys.stdout.isatty():
            return sys.stdout
    except Exception:
        pass

    # Non-interactive / piped / GUI session (GDM, Polkit): suppress ANSI animations
    return None


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

        # 2. Setup output stream for live terminal display
        out = _get_output_stream() if self.config.show_feedback else None

        if has_fp:
            prompt_str = "\033[1;36m[Argus]\033[0m Looking for face or fingerprint..."
        else:
            prompt_str = "\033[1;36m[Argus]\033[0m Looking for face (or use password)..."

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
                cap.release()
                if camera_dev.path.startswith("/dev/video"):
                    try:
                        dev_num = int(camera_dev.path.replace("/dev/video", ""))
                        cap = cv2.VideoCapture(dev_num)
                    except Exception:
                        pass
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(camera_dev.path)
                    if not cap.isOpened():
                        cap.release()
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
                consecutive_matches = 0
                min_required_matches = getattr(self.config, "min_consensus_frames", 3)
                if is_ir:
                    min_required_matches = max(1, min_required_matches - 1)

                read_errors = 0
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        read_errors += 1
                        if read_errors > 25:
                            face_failure_reason = "Camera video stream disconnected or unreadable"
                            return
                        time.sleep(0.01)
                        continue
                    read_errors = 0

                    res: VerificationResult = self.engine.verify(
                        user=user,
                        frame=frame,
                        is_ir_mode=is_ir,
                        require_liveness=self.config.require_liveness,
                    )

                    if res.matched and res.liveness.is_live:
                        consecutive_matches += 1
                        if consecutive_matches >= min_required_matches:
                            on_auth_success("FACE")
                            return
                    else:
                        consecutive_matches = 0
                        if res.face_detected and not res.liveness.is_live:
                            face_failure_reason = f"Presentation attack detected ({res.liveness.verdict})"
                        elif res.face_detected and not res.matched:
                            face_failure_reason = "Unrecognized face"

                    time.sleep(0.01)
            finally:
                if cap is not None:
                    cap.release()

        face_thread = threading.Thread(target=_face_worker, name="FaceWorkerThread", daemon=True)
        face_thread.start()

        # 4. Main Loop: Real-Time Colorful ASCII Animation
        frames = [
            ("\033[1;36m", "◖ ◉ ◗"),
            ("\033[1;35m", "◖ ◈ ◗"),
            ("\033[1;33m", "◖ ● ◗"),
            ("\033[1;32m", "◖ ◎ ◗"),
            ("\033[1;34m", "◖ ◐ ◗"),
            ("\033[1;36m", "◖ ◑ ◗"),
        ]
        frame_idx = 0
        start_time = time.monotonic()
        timeout = self.config.max_timeout

        style = getattr(self.config, "feedback_style", "animated").lower()
        if not self.config.show_feedback or style == "quiet":
            out = None
        elif style == "clean" and out:
            out.write(f"{prompt_str}...\n")
            out.flush()

        try:
            while not auth_event.is_set() and (time.monotonic() - start_time) < timeout:
                if out and style == "animated":
                    color, symbol = frames[frame_idx % len(frames)]
                    frame_idx += 1
                    out.write(f"\r\033[K{prompt_str} {color}[ {symbol} ]\033[0m")
                    out.flush()

                # If face worker has finished and no fingerprint is running, abort immediately
                if not face_thread.is_alive() and (not fp_verifier or not fp_verifier._thread or not fp_verifier._thread.is_alive()):
                    break

                time.sleep(0.08)

            if auth_event.is_set():
                if out:
                    if style == "animated":
                        # On success: erase prompt line completely, show NOTHING! Direct work!
                        out.write("\r\033[K")
                        out.flush()
                return BiometricAuthResult(
                    success=True,
                    method=result_holder["method"],
                    message="Access Granted",
                )
            else:
                stop_event.set()
                if out:
                    if style == "animated":
                        out.write("\r\033[K\033[1;33m[Argus] Biometric verification failed\033[0m\n")
                    else:
                        out.write("\033[1;33m[Argus] Biometric verification failed\033[0m\n")
                    out.flush()

                err_msg = (
                    face_failure_reason
                    if any(k in face_failure_reason.lower() for k in ("failed to open", "attack detected", "unrecognized", "disconnected"))
                    else "Biometric verification failed"
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
            if out and out not in (sys.stdout, sys.stderr):
                try:
                    out.close()
                except OSError:
                    pass
