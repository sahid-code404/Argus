"""Command Line Interface (CLI) for Linux Face Unlock.

Provides administrative tools for:
- Hardware probing and sensor capability diagnosis
- Multi-sample enrollment with anti-spoofing validation
- Interactive testing and biometric confidence visualization
- User management and configuration editing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np

from faceunlock.config import FaceUnlockConfig
from faceunlock.engine import FaceEngine, VerificationResult
from faceunlock.hardware import CameraType, probe_cameras, select_camera
from faceunlock.liveness import LivenessResult
from faceunlock.ui import (
    BiometricSpinner,
    animate_countdown,
    render_banner,
    render_step_progress,
)


def enforce_sudo_privilege(command_name: str) -> None:
    """Enforce root/sudo privileges. Automatically re-executes via sudo if running interactively."""
    if os.geteuid() == 0:
        return

    # If running interactively, re-exec via sudo
    if sys.stdin.isatty():
        try:
            if sys.argv[0].endswith("faceunlock"):
                cmd = ["sudo", sys.argv[0]] + sys.argv[1:]
            else:
                cmd = ["sudo", sys.executable] + sys.argv
            os.execvp("sudo", cmd)
        except Exception as e:
            sys.stderr.write(f"\033[91m[!] Privilege Error: Failed to run with sudo: {e}\033[0m\n")
            sys.exit(1)
    else:
        sys.stderr.write(
            f"\033[91m[!] Permission denied: 'faceunlock {command_name}' requires administrator privileges.\033[0m\n"
            f"Please run: sudo faceunlock {command_name}\n"
        )
        sys.exit(1)


def cmd_probe(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Probe all cameras and show classification & fallback status."""
    print("=" * 65)
    print(" Linux Face Unlock - Universal Camera & Sensor Prober")
    print("=" * 65)

    cameras = probe_cameras(include_all=args.all)
    if not cameras:
        print("\033[91m[!] No video capture devices found in /dev/video*.\033[0m")
        return 1

    print(f"\nDiscovered {len(cameras)} video device(s):")
    for idx, cam in enumerate(cameras, 1):
        color = "\033[92m" if cam.is_depth or cam.is_ir else "\033[96m"
        print(f" {idx}. {color}{cam.type_label}\033[0m")
        print(f"    Path:     {cam.path}")
        print(f"    Name:     {cam.name}")
        print(f"    Formats:  {', '.join(cam.formats) if cam.formats else 'Metadata/Unknown'}")
        print(f"    Priority: {cam.priority} (1=Depth, 2=Infrared, 3=RGB Webcam)")
        if cam.vendor_id or cam.model_id:
            print(f"    USB ID:   {cam.vendor_id}:{cam.model_id}")
        print()

    selected, reason = select_camera(None if cfg.camera_device == "auto" else cfg.camera_device)
    print("-" * 65)
    print(f"\033[1mActive Selection:\033[0m {selected.path} ({selected.name})")
    print(f"\033[1mResolution Strategy:\033[0m {reason}")
    print("=" * 65)
    return 0


def cmd_list(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """List enrolled users."""
    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
    )
    users = engine.get_enrolled_users()
    print("=" * 45)
    print(" Linux Face Unlock - Enrolled Users")
    print("=" * 45)
    if not users:
        print("No users currently enrolled. Use 'faceunlock enroll' to add a user.")
    else:
        for u in users:
            vectors = engine.load_user_embeddings(u)
            print(f" • \033[1m{u}\033[0m: {len(vectors)} biometric sample(s) registered")
    print("=" * 45)
    return 0


def cmd_enroll(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Interactive enrollment wizard capturing varied angles."""
    enforce_sudo_privilege("enroll")
    user = args.user or os.environ.get("SUDO_USER") or os.environ.get("USER") or "user"
    target_samples = max(3, min(10, args.samples))

    render_banner("BIOMETRIC FACE ENROLLMENT", f"Registering {target_samples} samples for '{user}'")
    print("\033[1mInstructions:\033[0m")
    print(f" We will capture {target_samples} face samples under varied head angles.")
    print(" Please follow each prompt on screen.")
    print("-" * 54)

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
    )

    selected_cam, reason = select_camera(None if cfg.camera_device == "auto" else cfg.camera_device)
    print(f"Using camera: \033[1;36m{selected_cam.path}\033[0m ({selected_cam.type_label})")

    cap = cv2.VideoCapture(selected_cam.path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(selected_cam.path)
        if not cap.isOpened():
            print(f"\033[91m[!] Cannot open camera {selected_cam.path}.\033[0m")
            return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    # Warmup
    for _ in range(cfg.warmup_frames):
        cap.read()

    embeddings: List[np.ndarray] = []
    instructions = [
        "Look directly at the camera with a neutral expression",
        "Tilt head slightly up and smile",
        "Turn head slightly to the left",
        "Turn head slightly to the right",
        "Tilt head slightly down with eyes open",
    ]

    try:
        sample_idx = 0
        while sample_idx < target_samples:
            inst = instructions[sample_idx % len(instructions)]
            print()
            render_step_progress(sample_idx + 1, target_samples, inst)
            animate_countdown(seconds=2, message="Position face")

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\033[93m[!] Frame capture dropped. Retrying...\033[0m")
                continue

            faces = engine.detect_faces(frame)
            if not faces:
                print("\033[93m[!] No face detected. Please ensure your face is well-lit and centered.\033[0m")
                continue
            if len(faces) > 1:
                print("\033[93m[!] Multiple faces detected! Only one face may be enrolled at a time.\033[0m")
                continue

            primary_face = faces[0]
            confidence = float(primary_face[14])
            if confidence < 0.50:
                print("\033[93m[!] Face not clear enough. Retrying...\033[0m")
                continue

            # Check liveness
            liveness = engine.liveness_detector.evaluate_frame(
                frame, primary_face, is_ir_mode=(selected_cam.camera_type == CameraType.INFRARED)
            )

            feat = engine.extract_embedding(frame, primary_face)
            embeddings.append(feat)
            sample_idx += 1
            print(f"\033[1;32m✔ Sample {sample_idx}/{target_samples} encrypted and registered!\033[0m")

        success = engine.enroll_user(user, embeddings, camera_type=selected_cam.camera_type.value)
        if success:
            render_banner("ENROLLMENT COMPLETE", f"User '{user}' secured with {len(embeddings)} biometric vectors")
            print("Biometric vault encrypted with hardware-bound AES-256 + cancelable projection.")
            print("Test anytime with: \033[1;36mfaceunlock test\033[0m\n")
            return 0
        else:
            print("\033[91m[!] Failed to save enrollment data.\033[0m")
            return 1
    finally:
        cap.release()


def cmd_test(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Test live face verification with interactive feedback."""
    enforce_sudo_privilege("test")
    user = args.user or os.environ.get("SUDO_USER") or os.environ.get("USER") or "user"

    render_banner("BIOMETRIC VERIFICATION TEST", f"Testing authentication for '{user}'")

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        cosine_threshold=cfg.cosine_threshold,
        min_liveness_threshold=cfg.min_liveness_score,
    )

    if not engine.is_enrolled(user):
        print(f"\033[91m[!] User '{user}' is not enrolled.\033[0m Run 'faceunlock enroll {user}' first.\n")
        return 1

    selected_cam, reason = select_camera(None if cfg.camera_device == "auto" else cfg.camera_device)
    print(f"Active Camera: \033[1;36m{selected_cam.path}\033[0m ({selected_cam.type_label})")
    print(f"Strategy:      {reason}")
    print("-" * 54)

    cap = cv2.VideoCapture(selected_cam.path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(selected_cam.path)
        if not cap.isOpened():
            print(f"\033[91m[!] Cannot open camera {selected_cam.path}.\033[0m")
            return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    # Warmup
    for _ in range(cfg.warmup_frames):
        cap.read()

    spinner = BiometricSpinner(message="Scanning face in front of camera...", stream=sys.stdout)
    start_time = time.monotonic()
    engine.liveness_detector.reset()
    is_ir = (selected_cam.camera_type == CameraType.INFRARED)

    passed = False
    fail_reason = "No face detected"
    match_score = 0
    min_required_matches = 1 if is_ir else 2

    try:
        while time.monotonic() - start_time < cfg.max_timeout:
            spinner.tick()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            result = engine.verify(
                user=user,
                frame=frame,
                is_ir_mode=is_ir,
                require_liveness=cfg.require_liveness,
            )

            if result.face_detected:
                if not result.liveness.is_live:
                    fail_reason = "Presentation attack detected (photo or screen spoof rejected)"
                elif not result.matched:
                    fail_reason = "Unrecognized face"

            if result.matched and result.liveness.is_live:
                match_score += 1
            else:
                match_score = max(0, match_score - 1)

            if match_score >= min_required_matches:
                passed = True
                break
            time.sleep(0.02)

        if passed:
            spinner.success("Face Verified: Access Granted")
            print(f"\n\033[1;32m✔ Biometric Verification Succeeded!\033[0m")
            print(f"  User:     \033[1m{user}\033[0m")
            print(f"  Identity: Confirmed")
            print(f"  Liveness: Genuine human present")
            print(f"  Sensor:   {selected_cam.type_label} ({selected_cam.path})\n")
            return 0
        else:
            spinner.failure(f"Verification Failed: {fail_reason}")
            print(f"\n\033[1;31m✗ Biometric Verification Failed\033[0m")
            print(f"  Reason:   {fail_reason}\n")
            return 1
    finally:
        cap.release()


def cmd_remove(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Remove enrolled user."""
    enforce_sudo_privilege("remove")
    engine = FaceEngine(data_dir=cfg.data_dir or None)
    if engine.remove_user(args.user):
        print(f"\033[1;32m✔ Enrolled data for user '{args.user}' removed successfully.\033[0m")
        return 0
    else:
        print(f"\033[1;33m✗ User '{args.user}' was not enrolled.\033[0m")
        return 1


def cmd_config(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """View or edit configuration."""
    if args.action == "set":
        enforce_sudo_privilege("config set")
        if not args.key or not args.value:
            print("Usage: faceunlock config set <key> <value>")
            return 1
        key = args.key.lower()
        val = args.value
        if key in ("device", "camera.device"):
            cfg.camera_device = val
        elif key in ("cosine_threshold", "security.cosine_threshold"):
            cfg.cosine_threshold = float(val)
        elif key in ("min_liveness_score", "security.min_liveness_score"):
            cfg.min_liveness_score = float(val)
        elif key in ("require_liveness", "security.require_liveness"):
            cfg.require_liveness = val.lower() in ("true", "1", "yes")
        elif key in ("max_timeout", "security.max_timeout"):
            cfg.max_timeout = float(val)
        elif key in ("pam_enabled", "pam.enabled"):
            cfg.pam_enabled = val.lower() in ("true", "1", "yes")
        else:
            print(f"Unknown configuration key: {key}")
            return 1
        saved_path = cfg.save()
        print(f"\033[92m[✓] Updated {key} = {val} in {saved_path}\033[0m")
        return 0
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faceunlock",
        description="Universal, High-Security Face Unlock System for Linux",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # probe
    p_probe = subparsers.add_parser("probe", help="Scan and diagnose cameras (IR, Depth, Webcam)")
    p_probe.add_argument("-a", "--all", action="store_true", help="Include non-capture/metadata nodes")

    # list
    subparsers.add_parser("list", help="List enrolled users")

    # enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a user's face")
    p_enroll.add_argument("user", nargs="?", help="Username (defaults to current user)")
    p_enroll.add_argument("-s", "--samples", type=int, default=5, help="Number of face samples (default: 5)")

    # test
    p_test = subparsers.add_parser("test", help="Test live face verification and anti-spoofing")
    p_test.add_argument("user", nargs="?", help="Username (defaults to current user)")

    # remove
    p_rem = subparsers.add_parser("remove", help="Remove enrolled user data")
    p_rem.add_argument("user", help="Username to remove")

    # config
    p_cfg = subparsers.add_parser("config", help="View or modify configuration")
    p_cfg.add_argument("action", choices=["show", "set"], default="show", nargs="?")
    p_cfg.add_argument("key", nargs="?", help="Config key")
    p_cfg.add_argument("value", nargs="?", help="Config value")

    args = parser.parse_args(argv or sys.argv[1:])
    cfg = FaceUnlockConfig.load()

    if not args.command or args.command == "probe":
        return cmd_probe(args if args.command else argparse.Namespace(all=False), cfg)
    elif args.command == "list":
        return cmd_list(args, cfg)
    elif args.command == "enroll":
        return cmd_enroll(args, cfg)
    elif args.command == "test":
        return cmd_test(args, cfg)
    elif args.command == "remove":
        return cmd_remove(args, cfg)
    elif args.command == "config":
        return cmd_config(args, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
