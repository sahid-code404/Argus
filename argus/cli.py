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
from pathlib import Path
from typing import List, Optional
import shutil

# Ensure base directory is in sys.path even when re-exec'd via sudo without PYTHONPATH
_BASE_DIR = str(Path(__file__).resolve().parent.parent)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np

from argus.config import ArgusConfig, FaceUnlockConfig
from argus.engine import FaceEngine, VerificationResult
from argus.hardware import CameraType, probe_cameras, safe_release_camera, select_camera
from argus.liveness import LivenessResult
from argus.ui import (
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
            bin_path = shutil.which("argus") or shutil.which("faceunlock") or "/usr/local/bin/argus"
            if os.path.exists(bin_path) and os.access(bin_path, os.X_OK):
                cmd = ["sudo", bin_path] + sys.argv[1:]
            else:
                cmd = ["sudo", "env", f"PYTHONPATH={_BASE_DIR}", sys.executable, "-m", "argus.cli"] + sys.argv[1:]
            os.execvp("sudo", cmd)
        except Exception as e:
            sys.stderr.write(f"\033[91m[!] Privilege Error: Failed to run with sudo: {e}\033[0m\n")
            sys.exit(1)
    else:
        sys.stderr.write(
            f"\033[91m[!] Permission denied: 'argus {command_name}' requires administrator privileges.\033[0m\n"
            f"Please run: sudo argus {command_name}\n"
        )
        sys.exit(1)


def cmd_probe(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Probe all cameras and show classification & fallback status."""
    print("=" * 65)
    print(" Argus - Universal Camera & Sensor Prober")
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
    """List enrolled biometric face models."""
    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        use_tpm=cfg.use_tpm,
    )

    targets = getattr(args, "target", [])
    if isinstance(targets, str):
        targets = [targets]
    elif targets is None:
        targets = []

    # Filter out descriptor tokens like 'face', 'faces', 'model', 'models'
    filter_user = None
    for t in targets:
        clean = t.strip()
        if clean.lower() not in ("face", "faces", "model", "models"):
            filter_user = clean
            break

    enrolled_users = engine.get_enrolled_users()
    if filter_user:
        if filter_user in enrolled_users:
            users_to_list = [filter_user]
        else:
            print(f"\033[93m[!] User '{filter_user}' has no enrolled face models.\033[0m")
            return 0
    else:
        users_to_list = enrolled_users

    total_models = 0
    rows = []
    for u in users_to_list:
        models = engine.list_user_models(u)
        if not models:
            vecs = engine.load_user_embeddings(u)
            if vecs:
                models = [{
                    "id": 1,
                    "name": "Face1",
                    "camera_type": "rgb_webcam",
                    "count": len(vecs),
                    "created_at": "Earlier",
                }]

        for m in models:
            total_models += 1
            rows.append({
                "user": u,
                "id": f"#{m.get('id', 1)}",
                "name": str(m.get("name", "Face1"))[:20],
                "samples": str(m.get("count", 5)),
                "sensor": str(m.get("camera_type", "rgb_webcam"))[:14],
                "date": str(m.get("created_at", "Earlier")),
            })

    print("=" * 96)
    print("                    Argus - Registered Biometric Face Models")
    print("=" * 96)

    if not rows:
        print(" No face models currently enrolled.")
        print(" Use 'sudo argus add' to register a new face model.")
        print("=" * 96)
        return 0

    header = f" {'User':<14} | {'ID':<6} | {'Model Name':<20} | {'Samples':<8} | {'Sensor':<14} | {'Registered Date (UTC)':<24}"
    print(header)
    print(" " + "-" * 14 + "-+-" + "-" * 6 + "-+-" + "-" * 20 + "-+-" + "-" * 8 + "-+-" + "-" * 14 + "-+-" + "-" * 24)

    for r in rows:
        print(f" {r['user']:<14} | {r['id']:<6} | {r['name']:<20} | {r['samples']:<8} | {r['sensor']:<14} | {r['date']:<24}")

    print("=" * 96)
    print(f" Total: {total_models} face model(s) enrolled across {len(users_to_list)} user(s).")
    print("=" * 96)
    return 0


def cmd_enroll(args: argparse.Namespace, cfg: FaceUnlockConfig, model_name: str = "Face1") -> int:
    """Interactive terminal enrollment wizard capturing varied angles."""
    enforce_sudo_privilege("enroll")
    user = getattr(args, "user", None) or os.environ.get("SUDO_USER") or os.environ.get("USER") or "user"
    target_samples = max(3, min(10, getattr(args, "samples", 5)))

    render_banner("BIOMETRIC FACE ENROLLMENT", f"Registering {target_samples} samples for '{user}' (Model: {model_name})")
    print("\033[1mInstructions:\033[0m")
    print(f" We will capture {target_samples} face samples under varied head angles.")
    print(" Please follow each prompt on screen.")
    print("-" * 54)

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        use_tpm=cfg.use_tpm,
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

            # Anti-spoofing check: prohibit screens, photos, videos
            liveness = engine.liveness_detector.evaluate_frame(
                frame, primary_face, is_ir_mode=(selected_cam.camera_type == CameraType.INFRARED)
            )
            if not liveness.is_live:
                print(f"\033[91m[!] Spoof detected ({liveness.verdict})! Screen / photo / video attack rejected.\033[0m")
                continue

            feat = engine.extract_embedding(frame, primary_face)
            embeddings.append(feat)
            sample_idx += 1
            print(f"\033[1;32m✔ Sample {sample_idx}/{target_samples} encrypted and registered!\033[0m")

        # Verify intra-model consistency: all captured angles must belong to the same person
        is_consistent, min_sim = engine.verify_intra_model_consistency(embeddings)
        if not is_consistent:
            print(f"\n\033[91m[!] Enrollment rejected: Inconsistent faces detected across angles (min similarity: {min_sim:.3f} < 0.35).\033[0m")
            print("\033[91m[!] All angles must belong to the same individual.\033[0m\n")
            return 1

        # Duplicate face check
        is_dup, dup_model, dup_sim = engine.check_duplicate_face(embeddings, user=user)
        if is_dup:
            print(f"\n\033[91m[!] Duplicate face rejected: This face is already enrolled as '{dup_model}' (similarity: {dup_sim:.3f}).\033[0m")
            print("\033[91m[!] Argus prevents enrolling the same face multiple times.\033[0m\n")
            return 1

        success = engine.enroll_user(
            user,
            embeddings,
            camera_type=selected_cam.camera_type.value,
            model_name=model_name,
            check_duplicate=False,
            check_intra_consistency=False,
        )
        if success:
            render_banner("ENROLLMENT COMPLETE", f"Face model '{model_name}' registered for '{user}'")
            print("Biometric vault encrypted with hardware-bound AES-256 + cancelable projection.")
            print("Test anytime with: \033[1;36margus test\033[0m\n")
            return 0
        else:
            print("\033[91m[!] Failed to save enrollment data.\033[0m")
            return 1
    finally:
        safe_release_camera(cap)


def _terminal_test(user: str, engine: FaceEngine, cfg: FaceUnlockConfig) -> int:
    """Run console-based live face verification with terminal spinner."""
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
    min_required_matches = getattr(cfg, "min_consensus_frames", 3) if not is_ir else 2

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
                    fail_reason = f"Presentation attack detected ({result.liveness.verdict})"
                elif not result.matched:
                    fail_reason = "Unrecognized face"

            if result.matched and result.liveness.is_live:
                match_score += 1
            else:
                match_score = 0

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
        safe_release_camera(cap)


def cmd_add(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Register a new face model with iOS-inspired circular reticle animation."""
    enforce_sudo_privilege("add")
    user = getattr(args, "user", None) or os.environ.get("SUDO_USER") or os.environ.get("USER") or "user"

    print("=" * 65)
    print("             BIOMETRIC FACE MODEL REGISTRATION")
    print("=" * 65)
    print(f"Adding face model for the current user ({user})")

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        use_tpm=cfg.use_tpm,
    )

    existing_models = engine.list_user_models(user)
    next_id = max([m.get("id", 0) for m in existing_models] or [0]) + 1
    auto_default_name = f"Face{next_id}"

    try:
        raw_name = input(
            f"Enter face model (model index #{next_id}) name (max 24 characters) [#{next_id} {auto_default_name}]: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\033[1;31m✗ Face enrollment failed or cancelled. Try again.\033[0m")
        print("Try running: sudo argus add\n")
        return 1

    model_name = raw_name if raw_name else auto_default_name
    # Strip leading "#<id> " if user entered or copied that format
    if model_name.startswith(f"#{next_id} "):
        model_name = model_name[len(f"#{next_id} "):].strip()
    elif model_name.startswith(f"#{next_id}"):
        model_name = model_name[len(f"#{next_id}"):].strip() or auto_default_name

    if len(model_name) > 24:
        model_name = model_name[:24]
        print(f"Notice: Model name truncated to 24 characters: '{model_name}'")

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        print("\033[93m[!] No graphical display detected. Falling back to terminal enrollment wizard...\033[0m\n")
        return cmd_enroll(args, cfg, model_name=model_name)

    try:
        from argus.gui_enroll import run_gui_enrollment
        print(f"\nOpening webcam preview window... Please follow the guided head animations.")
        success = run_gui_enrollment(user=user, model_name=model_name, engine=engine, config=cfg)
        if success:
            print(f"\n\033[1;32m✔ Face model '{model_name}' successfully enrolled and encrypted for user '{user}'.\033[0m\n")
            return 0
        else:
            print(f"\n\033[1;31m✗ Face enrollment failed or cancelled. Try again.\033[0m")
            print("Try running: sudo argus add\n")
            return 1
    except Exception as e:
        sys.stderr.write(f"\033[91m[!] Graphical enrollment error: {e}\033[0m\n")
        print("Falling back to terminal enrollment wizard...\n")
        return cmd_enroll(args, cfg, model_name=model_name)


def cmd_test(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Test live face verification with camera preview window or terminal fallback."""
    enforce_sudo_privilege("test")
    user = getattr(args, "user", None) or os.environ.get("SUDO_USER") or os.environ.get("USER") or "user"

    render_banner("BIOMETRIC VERIFICATION TEST", f"Testing authentication for '{user}'")

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        cosine_threshold=cfg.cosine_threshold,
        min_liveness_threshold=cfg.min_liveness_score,
        use_tpm=cfg.use_tpm,
    )

    if not engine.is_enrolled(user):
        print(f"\033[91m[!] User '{user}' is not enrolled.\033[0m Run 'sudo argus add {user}' first.\n")
        return 1

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        try:
            from argus.gui_test import run_gui_test
            run_gui_test(user=user, engine=engine, config=cfg)
            return 0
        except Exception as e:
            print(f"\033[93m[!] Graphical preview encountered an issue ({e}). Falling back to terminal...\033[0m\n")

    return _terminal_test(user, engine, cfg)


def cmd_remove(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Remove enrolled face model or user."""
    enforce_sudo_privilege("remove")
    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        use_tpm=cfg.use_tpm,
    )

    targets = getattr(args, "target", [])
    if isinstance(targets, str):
        targets = [targets]
    elif targets is None:
        targets = []

    # Strip descriptor prefixes like 'facemodel', 'face', 'model', 'facemodels', 'models'
    tokens = []
    for t in targets:
        clean = t.strip()
        if not tokens and clean.lower() in ("facemodel", "face", "model", "facemodels", "models"):
            continue
        if clean:
            tokens.append(clean)

    enrolled_users = engine.get_enrolled_users()

    # Case 1: Interactive mode if no target provided
    if not tokens:
        all_models_flat = []
        for u in enrolled_users:
            for m in engine.list_user_models(u):
                all_models_flat.append((u, m))

        if not all_models_flat and not enrolled_users:
            print("\033[93m[!] No enrolled face models found to remove.\033[0m")
            return 0

        print("=" * 65)
        print("          Argus - Remove Enrolled Face Model")
        print("=" * 65)
        print("Select a face model to remove:\n")
        idx_map = {}
        counter = 1
        for u, m in all_models_flat:
            print(f"  [{counter}] {u} - #{m.get('id', 1)} '{m.get('name', 'Face1')}' ({m.get('count', 5)} samples, {m.get('created_at', 'Registered')})")
            idx_map[str(counter)] = (u, m)
            counter += 1

        print("  [A] Remove ALL biometric data for current user")
        print("  [Q] Cancel / Quit")
        print("-" * 65)

        try:
            choice = input(f"Enter selection [1-{len(all_models_flat)}, A, Q]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 0

        if not choice or choice.lower() in ("q", "quit", "exit"):
            print("Operation cancelled.")
            return 0

        if choice.lower() == "a":
            target_user = os.environ.get("SUDO_USER") or os.environ.get("USER") or (enrolled_users[0] if enrolled_users else "user")
            if engine.remove_user(target_user):
                print(f"\033[1;32m✔ All enrolled data for user '{target_user}' removed successfully.\033[0m")
                return 0
            else:
                print(f"\033[93m[!] No data found for user '{target_user}'.\033[0m")
                return 1

        if choice in idx_map:
            u, m = idx_map[choice]
            ok, removed_name = engine.remove_model(u, str(m.get("id")))
            if ok:
                print(f"\033[1;32m✔ Successfully removed face model '{removed_name}' for user '{u}'.\033[0m")
                return 0
            else:
                print(f"\033[91m[!] Failed to remove model '{m.get('name')}'.\033[0m")
                return 1
        else:
            print("\033[91m[!] Invalid selection.\033[0m")
            return 1

    # Case 2: Target argument provided (e.g. "facemodel Face1", "Face1", "1", "#1", "shaque")
    target_str = " ".join(tokens).strip()

    # Check if target is a user directly
    if target_str in enrolled_users:
        if engine.remove_user(target_str):
            print(f"\033[1;32m✔ All enrolled biometric data for user '{target_str}' removed successfully.\033[0m")
            return 0

    # Search for model by ID or name across enrolled users
    current_user = os.environ.get("SUDO_USER") or os.environ.get("USER")
    users_priority = []
    if current_user and current_user in enrolled_users:
        users_priority.append(current_user)
    for u in enrolled_users:
        if u not in users_priority:
            users_priority.append(u)

    for u in users_priority:
        models = engine.list_user_models(u)
        for m in models:
            m_id = str(m.get("id", ""))
            m_name = str(m.get("name", ""))
            m_id_clean = m_id.lstrip("#").strip().lower()
            target_clean = target_str.lstrip("#").strip().lower()

            if (
                m_id == target_str
                or m_id_clean == target_clean
                or m_name.lower() == target_str.lower()
                or m_name.lower().lstrip("#").strip() == target_clean
            ):
                ok, removed_name = engine.remove_model(u, m_id)
                if ok:
                    print(f"\033[1;32m✔ Successfully removed face model '{removed_name}' (#{m_id}) for user '{u}'.\033[0m")
                    remaining = engine.list_user_models(u)
                    if not remaining:
                        print(f"Notice: No face models remain for user '{u}'. Biometric vault cleared.")
                    return 0

    # Fallback: check if target_str was a user with unmigrated data
    if engine.remove_user(target_str):
        print(f"\033[1;32m✔ Enrolled data for user '{target_str}' removed successfully.\033[0m")
        return 0

    print(f"\033[91m[!] No face model or user found matching '{target_str}'.\033[0m")
    print("Use 'sudo argus list face' to view all registered biometric models.")
    return 1


def cmd_status(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Show comprehensive security and runtime status."""
    from argus.pam_integrator import PamIntegrator
    from argus.tpm import TPMManager

    integrator = PamIntegrator()
    services = integrator.get_status_overview()

    tpm_supported = TPMManager.is_hardware_present()
    data_path = Path(cfg.data_dir or ("/etc/argus/faces" if Path("/etc/argus").exists() else "/etc/faceunlock/faces"))
    tpm_sealed = (data_path / ".tpm_blob" / "tpm_pub.bin").exists() or (data_path / "tpm_pub.bin").exists()

    engine = FaceEngine(
        model_dir=cfg.models_dir or None,
        data_dir=cfg.data_dir or None,
        use_tpm=cfg.use_tpm,
    )
    users = engine.get_enrolled_users()

    print("=" * 65)
    print(" Argus - System Diagnostic & Security Status")
    print("=" * 65)

    state_str = "\033[92m● ACTIVE\033[0m" if cfg.pam_enabled else "\033[91m○ DISABLED (Kill-switch active)\033[0m"
    print(f" Master Service Switch: {state_str}")
    print(f" Security Profile:     \033[1m{cfg.security_mode.upper()}\033[0m (cosine: {cfg.cosine_threshold:.2f}, liveness: {cfg.min_liveness_score:.2f})")

    selected_cam, cam_reason = select_camera(None if cfg.camera_device == "auto" else cfg.camera_device)
    cam_color = "\033[92m" if selected_cam.is_depth or selected_cam.is_ir else "\033[96m"
    print(f" Camera Device:        {cam_color}{selected_cam.name} ({selected_cam.path})\033[0m [{selected_cam.type_label}]")

    if tpm_supported and tpm_sealed:
        tpm_str = "\033[92m✔ TPM 2.0 Active (Hardware Master Key Sealed)\033[0m"
    elif tpm_supported and cfg.use_tpm:
        tpm_str = "\033[93m⚠ TPM 2.0 Supported (Key will seal on next enrollment)\033[0m"
    elif tpm_supported:
        tpm_str = "\033[96m• TPM 2.0 Supported (Configured for Software Storage)\033[0m"
    else:
        tpm_str = "\033[90m• Software AES-256-GCM / PBKDF2 (No TPM hardware)\033[0m"
    print(f" Hardware Encryption:  {tpm_str}")

    if users:
        print(f" Enrolled Users:       \033[1m{', '.join(users)}\033[0m ({len(users)} user(s))")
    else:
        print(" Enrolled Users:       \033[93mNone enrolled\033[0m")

    print("-" * 65)
    print(" Configured PAM Authentication Endpoints:")
    for srv_key, info in services.items():
        if info["available"]:
            status_tag = "\033[92mENABLED\033[0m" if info["enabled"] else "\033[90mDISABLED\033[0m"
            print(f"   • {srv_key:<16} [{status_tag}]  {info['description']}")
    print("=" * 65)
    return 0


def cmd_enable(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Enable Face Unlock system-wide."""
    enforce_sudo_privilege("enable")
    cfg.pam_enabled = True
    cfg.save()
    print("\033[92m[✓] Face Unlock has been ENABLED system-wide.\033[0m")
    return 0


def cmd_disable(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Disable Face Unlock system-wide (privacy kill-switch)."""
    enforce_sudo_privilege("disable")
    cfg.pam_enabled = False
    cfg.save()
    print("\033[93m[✓] Face Unlock has been DISABLED system-wide (privacy kill-switch active).\033[0m")
    return 0


def cmd_profile(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """View or switch security profile."""
    if args.profile:
        enforce_sudo_privilege("profile")
        mode = args.profile.lower()
        if mode not in ("convenient", "balanced", "strict"):
            print(f"Error: Unknown profile '{args.profile}'. Options: convenient, balanced, strict")
            return 1
        cfg.apply_security_mode(mode)
        cfg.save()
        print(f"\033[92m[✓] Security profile set to: \033[1m{mode.upper()}\033[0m")
    print(f"Active Profile: \033[1m{cfg.security_mode.upper()}\033[0m")
    print(f"  • Cosine Match Threshold:  {cfg.cosine_threshold:.2f}")
    print(f"  • Minimum Liveness Score:  {cfg.min_liveness_score:.2f}")
    print(f"  • Maximum Timeout:         {cfg.max_timeout:.1f}s")
    return 0


def cmd_service(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """Manage PAM service integrations."""
    from argus.pam_integrator import PamIntegrator
    integrator = PamIntegrator()

    if args.action == "list" or not args.action:
        print("=" * 65)
        print(" Available PAM Services for Argus")
        print("=" * 65)
        for srv, info in integrator.get_status_overview().items():
            avail_tag = "Available" if info["available"] else "Not Found"
            status_tag = "\033[92mENABLED\033[0m" if info["enabled"] else "\033[90mDISABLED\033[0m"
            print(f" • {srv:<15} [{status_tag}] ({avail_tag}) - {info['description']}")
        print("=" * 65)
        return 0

    if not args.service:
        print("Usage: argus service <enable|disable> <service-name>")
        return 1

    enforce_sudo_privilege(f"service {args.action}")
    srv = args.service.lower().strip()

    if any(k in srv for k in ("gdm", "sddm", "lightdm", "login")):
        print("\033[93m[!] Session login managers (GDM, SDDM, login) are disabled by design to prevent GNOME Keyring lockouts.\033[0m")
        print("To unlock your desktop screen with biometrics, use: \033[1;36msudo argus service enable lockscreen\033[0m")
        return 1

    if args.action == "enable":
        if not integrator.is_service_available(srv):
            print(f"\033[91m[!] Service '{srv}' is not available on this system.\033[0m")
            return 1
        if integrator.enable_service(srv):
            if srv == "sudo":
                cfg.policy_sudo = True
            elif "polkit" in srv:
                cfg.policy_polkit = True
            elif any(k in srv for k in ("lock", "screensaver")):
                cfg.policy_lockscreen = True
            elif "ssh" in srv:
                cfg.policy_sshd = True
            cfg.save()
            print(f"\033[92m[✓] Argus successfully enabled for '{srv}'.\033[0m")
            return 0
        else:
            print(f"\033[91m[!] Failed to enable Argus for '{srv}'. Check /etc/pam.d/ permissions.\033[0m")
            return 1

    elif args.action == "disable":
        if integrator.disable_service(srv):
            if srv == "sudo":
                cfg.policy_sudo = False
            elif "polkit" in srv:
                cfg.policy_polkit = False
            elif any(k in srv for k in ("lock", "screensaver")):
                cfg.policy_lockscreen = False
            elif "ssh" in srv:
                cfg.policy_sshd = False
            cfg.save()
            print(f"\033[92m[✓] Argus disabled for '{srv}'.\033[0m")
            return 0
        else:
            print(f"\033[91m[!] Failed to disable Argus for '{srv}'.\033[0m")
            return 1

    return 0


def _interactive_config_menu(cfg: FaceUnlockConfig) -> int:
    """Terminal interactive menu to view and toggle options."""
    enforce_sudo_privilege("config")
    from argus.pam_integrator import PamIntegrator
    from argus.tpm import TPMManager

    integrator = PamIntegrator()
    tpm_supported = TPMManager.is_hardware_present()

    while True:
        pam_status = integrator.get_status_overview()

        print("\n" + "=" * 65)
        print("               ARGUS - INTERACTIVE CONFIGURATION")
        print("=" * 65)

        def _fmt_state(val: bool) -> str:
            return "\033[92m[ENABLED]\033[0m" if val else "\033[90m[DISABLED]\033[0m"

        sudo_en = cfg.policy_sudo and pam_status.get("sudo", {}).get("enabled", False)
        polkit_en = cfg.policy_polkit and pam_status.get("polkit-1", {}).get("enabled", False)
        lock_en = cfg.policy_lockscreen and pam_status.get("lockscreen", {}).get("enabled", False)
        ssh_en = cfg.policy_sshd and pam_status.get("sshd", {}).get("enabled", False)

        tpm_state_str = (_fmt_state(cfg.use_tpm) if tpm_supported else "\033[90m[NOT SUPPORTED]\033[0m")
        feedback_str = _fmt_state(cfg.show_feedback)

        print(" [PAM Services Integration]")
        print(f"  1. Sudo Privilege Elevation:       {_fmt_state(sudo_en)}")
        print(f"  2. Polkit GUI Dialogs:             {_fmt_state(polkit_en)}")
        print(f"  3. Desktop Lock Screen:            {_fmt_state(lock_en)}")
        print(f"  4. SSH Remote Login:               {_fmt_state(ssh_en)}")
        print()
        print(" [Authentication & Terminal Feedback]")
        print(f"  5. Terminal Scanning Feedback:     {feedback_str}")
        print(f"  6. Feedback Display Mode:          \033[1m{cfg.feedback_style.upper()}\033[0m (animated / clean / quiet)")
        print(f"  7. Recognition Timeout:            \033[1m{cfg.max_timeout:.1f}s\033[0m")
        print()
        print(" [Security Profile & Hardware]")
        print(f"  8. Security Mode:                  \033[1m{cfg.security_mode.upper()}\033[0m (Threshold: {cfg.cosine_threshold:.2f}, Liveness: {cfg.min_liveness_score:.2f})")
        print(f"  9. Hardware TPM 2.0 Key Sealing:   {tpm_state_str}")
        print("-" * 65)
        print(" Commands: [1-9] Toggle/Modify | [S] Save & Apply | [Q] Quit without saving")
        print("=" * 65)

        try:
            choice = input("Select an option [1-9, S, Q]: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 0

        if choice == "1":
            cfg.policy_sudo = not cfg.policy_sudo
            print(f"Sudo set to: {'ENABLED' if cfg.policy_sudo else 'DISABLED'}")
        elif choice == "2":
            cfg.policy_polkit = not cfg.policy_polkit
            print(f"Polkit set to: {'ENABLED' if cfg.policy_polkit else 'DISABLED'}")
        elif choice == "3":
            cfg.policy_lockscreen = not cfg.policy_lockscreen
            print(f"Desktop Lock Screen set to: {'ENABLED' if cfg.policy_lockscreen else 'DISABLED'}")
        elif choice == "4":
            if not cfg.policy_sshd:
                print("\nNotice: Enabling Face Unlock for SSH will attempt camera access on the local host machine during remote logins.")
                confirm = input("Enable Face Unlock for SSH? [y/N]: ").strip().lower()
                if confirm in ("y", "yes"):
                    cfg.policy_sshd = True
            else:
                cfg.policy_sshd = False
            print(f"SSH set to: {'ENABLED' if cfg.policy_sshd else 'DISABLED'}")
        elif choice == "5":
            cfg.show_feedback = not cfg.show_feedback
            print(f"Terminal feedback: {'ENABLED' if cfg.show_feedback else 'DISABLED'}")
        elif choice == "6":
            cycle = {"animated": "clean", "clean": "quiet", "quiet": "animated"}
            cfg.feedback_style = cycle.get(cfg.feedback_style.lower(), "animated")
            print(f"Feedback style set to: {cfg.feedback_style.upper()}")
        elif choice == "7":
            try:
                raw_t = input(f"Enter new timeout in seconds (1.0 - 10.0) [{cfg.max_timeout}]: ").strip()
                if raw_t:
                    t = float(raw_t)
                    if 1.0 <= t <= 10.0:
                        cfg.max_timeout = t
                        print(f"Timeout set to {cfg.max_timeout}s")
                    else:
                        print("Value must be between 1.0 and 10.0")
            except ValueError:
                print("Invalid number.")
        elif choice == "8":
            print("\nSecurity Profiles:")
            print("  [L] Low / Convenient   (Threshold: 0.35, Liveness: 0.55, Timeout: 2.5s)")
            print("  [M] Medium / Balanced  (Threshold: 0.38, Liveness: 0.65, Timeout: 3.5s)")
            print("  [H] High / Strict      (Threshold: 0.44, Liveness: 0.75, Timeout: 4.0s)")
            print("  [C] Custom Tweaks")
            sub = input("Select profile [L/M/H/C]: ").strip().upper()
            if sub == "L":
                cfg.apply_security_mode("convenient")
            elif sub == "M":
                cfg.apply_security_mode("balanced")
            elif sub == "H":
                cfg.apply_security_mode("strict")
            elif sub == "C":
                try:
                    c_th = float(input(f"Enter Cosine Match Threshold (0.25 - 0.50) [{cfg.cosine_threshold}]: ").strip() or cfg.cosine_threshold)
                    l_th = float(input(f"Enter Min Liveness Score (0.30 - 0.90) [{cfg.min_liveness_score}]: ").strip() or cfg.min_liveness_score)
                    cfg.cosine_threshold = c_th
                    cfg.min_liveness_score = l_th
                    cfg.security_mode = "custom"
                except ValueError:
                    print("Invalid numeric values.")
            print(f"Security mode set to: {cfg.security_mode.upper()}")
        elif choice == "9":
            if tpm_supported:
                cfg.use_tpm = not cfg.use_tpm
                print(f"TPM 2.0 sealing set to: {'ENABLED' if cfg.use_tpm else 'DISABLED'}")
            else:
                print("TPM 2.0 hardware chip not found on this system.")
        elif choice == "S":
            # Apply PAM changes
            if cfg.policy_sudo:
                integrator.enable_service("sudo")
            else:
                integrator.disable_service("sudo")

            if cfg.policy_polkit:
                integrator.enable_service("polkit-1")
            else:
                integrator.disable_service("polkit-1")

            if cfg.policy_lockscreen:
                integrator.enable_service("lockscreen")
            else:
                integrator.disable_service("lockscreen")

            # Always ensure session login managers are disabled/cleaned up to prevent keyring issues
            integrator.disable_service("gdm-password")
            integrator.disable_service("login")

            if cfg.policy_sshd and integrator.is_service_available("sshd"):
                integrator.enable_service("sshd")
            elif integrator.is_service_available("sshd"):
                integrator.disable_service("sshd")

            saved_path = cfg.save()
            print(f"\n\033[1;32m✔ Configuration saved to {saved_path} and PAM services successfully synced!\033[0m")
            return 0
        elif choice == "Q":
            print("\nExited without saving.")
            return 0
        else:
            print(f"Invalid option: '{choice}'")


def cmd_config(args: argparse.Namespace, cfg: FaceUnlockConfig) -> int:
    """View or edit configuration."""
    if args.action == "set":
        enforce_sudo_privilege("config set")
        if not args.key or not args.value:
            print("Usage: argus config set <key> <value>")
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
        elif key in ("use_tpm", "security.use_tpm"):
            cfg.use_tpm = val.lower() in ("true", "1", "yes")
        elif key in ("security_mode", "security.security_mode"):
            cfg.apply_security_mode(val)
        elif key in ("feedback_style", "pam.feedback_style"):
            cfg.feedback_style = val.lower()
        elif key in ("show_feedback", "pam.show_feedback"):
            cfg.show_feedback = val.lower() in ("true", "1", "yes")
        elif key in ("lockscreen", "policy_lockscreen", "policies.lockscreen", "lockers"):
            cfg.policy_lockscreen = val.lower() in ("true", "1", "yes")
        elif key in ("sudo", "policies.sudo"):
            cfg.policy_sudo = val.lower() in ("true", "1", "yes")
        elif key in ("polkit", "policies.polkit"):
            cfg.policy_polkit = val.lower() in ("true", "1", "yes")
        elif key in ("ssh", "sshd", "policies.sshd"):
            cfg.policy_sshd = val.lower() in ("true", "1", "yes")
        elif any(k in key for k in ("gdm", "login", "sddm", "lightdm")):
            print("\033[93m[!] Session login managers (GDM, SDDM, login) are disabled by design to protect user keyrings.\033[0m")
            return 1
        else:
            print(f"Unknown configuration key: {key}")
            return 1
        saved_path = cfg.save()
        print(f"\033[92m[✓] Updated {key} = {val} in {saved_path}\033[0m")
        return 0

    if sys.stdin.isatty() and (not args.action or args.action == "show"):
        return _interactive_config_menu(cfg)

    # Non-interactive summary
    print("=" * 65)
    print(" Argus Configuration Summary")
    print("=" * 65)
    print(f" Camera Device:        {cfg.camera_device}")
    print(f" Security Mode:        {cfg.security_mode.upper()} (threshold: {cfg.cosine_threshold:.2f}, liveness: {cfg.min_liveness_score:.2f})")
    print(f" Max Timeout:          {cfg.max_timeout:.1f}s")
    print(f" TPM 2.0 Sealing:      {'Active' if cfg.use_tpm else 'Disabled'}")
    print(f" Terminal Feedback:    {'Enabled' if cfg.show_feedback else 'Disabled'} ({cfg.feedback_style})")
    print(f" Policies: Sudo={cfg.policy_sudo}, Polkit={cfg.policy_polkit}, LockScreen={cfg.policy_lockscreen}, SSH={cfg.policy_sshd}")
    print("=" * 65)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus - Universal, High-Security Biometric Authentication for Linux",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # probe
    p_probe = subparsers.add_parser("probe", help="Scan and diagnose cameras (IR, Depth, Webcam)")
    p_probe.add_argument("-a", "--all", action="store_true", help="Include non-capture/metadata nodes")

    # list
    p_list = subparsers.add_parser("list", aliases=["list-face", "list-faces"], help="List enrolled biometric face models")
    p_list.add_argument("target", nargs="*", help="Optional filter: 'face', 'models', or username")

    # add
    p_add = subparsers.add_parser("add", help="Register a new face model with iOS-like guided animation")
    p_add.add_argument("user", nargs="?", help="Username (defaults to current user)")

    # enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a user's face (alias for add)")
    p_enroll.add_argument("user", nargs="?", help="Username (defaults to current user)")
    p_enroll.add_argument("-s", "--samples", type=int, default=5, help="Number of face samples (default: 5)")

    # test
    p_test = subparsers.add_parser("test", help="Test live face verification and anti-spoofing")
    p_test.add_argument("user", nargs="?", help="Username (defaults to current user)")

    # remove
    p_rem = subparsers.add_parser("remove", aliases=["rm", "remove-face"], help="Remove enrolled biometric face model or user")
    p_rem.add_argument("target", nargs="*", help="Model name, index (#1), or username to remove")

    # status
    subparsers.add_parser("status", help="Show system status, TPM hardware state, and PAM services")

    # enable / disable
    subparsers.add_parser("enable", help="Enable Face Unlock system-wide")
    subparsers.add_parser("disable", help="Disable Face Unlock system-wide (privacy kill-switch)")

    # profile
    p_prof = subparsers.add_parser("profile", help="View or set security profile (convenient, balanced, strict)")
    p_prof.add_argument("profile", nargs="?", choices=["convenient", "balanced", "strict"], help="Security profile")

    # service
    p_srv = subparsers.add_parser("service", help="Manage PAM service integrations (lockscreen, polkit-1, sudo, etc.)")
    p_srv.add_argument("action", choices=["list", "enable", "disable"], default="list", nargs="?")
    p_srv.add_argument("service", nargs="?", help="Service name (e.g. lockscreen, polkit-1, sudo)")

    # config
    p_cfg = subparsers.add_parser("config", help="View or modify configuration")
    p_cfg.add_argument("action", choices=["show", "set"], default="show", nargs="?")
    p_cfg.add_argument("key", nargs="?", help="Config key")
    p_cfg.add_argument("value", nargs="?", help="Config value")

    args = parser.parse_args(argv or sys.argv[1:])
    cfg = FaceUnlockConfig.load()

    if not args.command or args.command == "status":
        return cmd_status(args, cfg)
    elif args.command == "probe":
        return cmd_probe(args, cfg)
    elif args.command == "enable":
        return cmd_enable(args, cfg)
    elif args.command == "disable":
        return cmd_disable(args, cfg)
    elif args.command == "profile":
        return cmd_profile(args, cfg)
    elif args.command == "service":
        return cmd_service(args, cfg)
    elif args.command in ("list", "list-face", "list-faces"):
        return cmd_list(args, cfg)
    elif args.command in ("add", "enroll"):
        return cmd_add(args, cfg)
    elif args.command == "test":
        return cmd_test(args, cfg)
    elif args.command in ("remove", "rm", "remove-face"):
        return cmd_remove(args, cfg)
    elif args.command == "config":
        return cmd_config(args, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
