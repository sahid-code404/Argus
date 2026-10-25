"""Graphical Face Enrollment Interface with Guided iOS-style Reticle Animations.

Provides:
- Real-time RGB webcam preview (never IR/depth raw data)
- Apple iOS Face ID inspired circular segmented progress ticks
- Step-by-step guidance across multiple head angles (center, right, left, up, down)
- Automatic saving on completion and auto-close on failure/cancel
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# Force Qt/Wayland compatibility under modern GNOME
os.environ.setdefault("QT_QPA_PLATFORM", "xcb;wayland")

import cv2
import numpy as np

from argus.config import ArgusConfig, FaceUnlockConfig
from argus.engine import FaceEngine
from argus.hardware import CameraDevice, CameraType, probe_cameras, select_camera


# Guidance stages for varied biometric sample capture
ENROLL_STAGES = [
    {"label": "Look straight into camera", "hint": "Position your face in the center ring"},
    {"label": "Turn head slightly RIGHT", "hint": "Turn slowly to the right"},
    {"label": "Turn head slightly LEFT", "hint": "Turn slowly to the left"},
    {"label": "Tilt head slightly UP", "hint": "Tilt your chin slightly up"},
    {"label": "Tilt head slightly DOWN", "hint": "Tilt your chin slightly down"},
]


def _ensure_gui_environment() -> None:
    """Ensure DISPLAY and XAUTHORITY are inherited when running under sudo."""
    if os.geteuid() == 0:
        sudo_user = os.environ.get("SUDO_USER")
        if sudo_user:
            import pwd
            try:
                pw = pwd.getpwnam(sudo_user)
                uid = pw.pw_uid
                # Check user runtime dir for Wayland / Xwayland auth
                user_run = f"/run/user/{uid}"
                if "WAYLAND_DISPLAY" not in os.environ and os.path.exists(f"{user_run}/wayland-0"):
                    os.environ["WAYLAND_DISPLAY"] = "wayland-0"
                if "XDG_RUNTIME_DIR" not in os.environ and os.path.exists(user_run):
                    os.environ["XDG_RUNTIME_DIR"] = user_run
                if "DISPLAY" not in os.environ:
                    os.environ["DISPLAY"] = ":0"
            except Exception:
                pass


def _find_webcam_device(config: FaceUnlockConfig) -> str:
    """Find the best RGB webcam device for user enrollment preview (never IR/depth)."""
    cameras = probe_cameras(include_all=False)
    for cam in cameras:
        if cam.camera_type == CameraType.RGB_WEBCAM and os.path.exists(cam.path):
            return cam.path
    # Fallback to config or /dev/video0
    if os.path.exists("/dev/video0"):
        return "/dev/video0"
    return "/dev/video0"


def _draw_ios_reticle(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    completed_ticks: int,
    total_ticks: int = 40,
    face_in_bounds: bool = False,
    is_success: bool = False,
) -> None:
    """Draw circular progress ticks and facial oval guide inspired by iOS Face ID."""
    overlay = img.copy()

    # Draw dark translucent vignette outside the central interaction area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius + 25, 255, -1)
    dark_bg = (overlay * 0.45).astype(np.uint8)
    img[:] = np.where(mask[:, :, None] == 255, img, dark_bg)

    # 1. Outer Circular Ticks
    for i in range(total_ticks):
        angle_deg = i * (360.0 / total_ticks) - 90.0
        angle_rad = math.radians(angle_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        r_inner = radius - 8
        r_outer = radius + 8

        p1 = (int(cx + r_inner * cos_a), int(cy + r_inner * sin_a))
        p2 = (int(cx + r_outer * cos_a), int(cy + r_outer * sin_a))

        if is_success or i < completed_ticks:
            color = (50, 245, 90)  # Apple Neon Green
            thickness = 3
        else:
            color = (130, 130, 130)  # Subtle gray
            thickness = 1

        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # 2. Central Oval Guidance
    oval_w = int(radius * 0.72)
    oval_h = int(radius * 0.95)
    if is_success:
        guide_color = (50, 245, 90)
    elif face_in_bounds:
        guide_color = (255, 220, 40)  # Cyan / Apple blue
    else:
        guide_color = (200, 200, 200)

    cv2.ellipse(img, (cx, cy), (oval_w, oval_h), 0, 0, 360, guide_color, 2, cv2.LINE_AA)


def _draw_hud_banner(img: np.ndarray, top_text: str, sub_text: str, bottom_instruction: str, hint: str) -> None:
    """Draw sleek modern HUD banners for stage instructions and feedback."""
    h, w = img.shape[:2]

    # Top banner pill
    top_w, top_h = 420, 60
    tx1, ty1 = (w - top_w) // 2, 20
    tx2, ty2 = tx1 + top_w, ty1 + top_h

    sub_top = img[ty1:ty2, tx1:tx2]
    if sub_top.shape[0] == top_h and sub_top.shape[1] == top_w:
        rect = np.zeros_like(sub_top, dtype=np.uint8)
        cv2.rectangle(rect, (0, 0), (top_w, top_h), (25, 25, 25), -1)
        cv2.addWeighted(sub_top, 0.35, rect, 0.65, 0, sub_top)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (70, 70, 70), 1, cv2.LINE_AA)

    cv2.putText(img, top_text, (tx1 + 20, ty1 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, sub_text, (tx1 + 20, ty1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 220, 255), 1, cv2.LINE_AA)

    # Bottom instruction pill
    bot_w, bot_h = 480, 68
    bx1, by1 = (w - bot_w) // 2, h - 88
    bx2, by2 = bx1 + bot_w, by1 + bot_h

    sub_bot = img[by1:by2, bx1:bx2]
    if sub_bot.shape[0] == bot_h and sub_bot.shape[1] == bot_w:
        rect = np.zeros_like(sub_bot, dtype=np.uint8)
        cv2.rectangle(rect, (0, 0), (bot_w, bot_h), (20, 20, 20), -1)
        cv2.addWeighted(sub_bot, 0.35, rect, 0.65, 0, sub_bot)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (60, 60, 60), 1, cv2.LINE_AA)

    cv2.putText(img, bottom_instruction, (bx1 + 25, by1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (50, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(img, hint, (bx1 + 25, by1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)


def run_gui_enrollment(
    user: str,
    model_name: str,
    engine: FaceEngine,
    config: Optional[FaceUnlockConfig] = None,
) -> bool:
    """Run interactive graphical camera enrollment window with iOS Face ID style animations.

    Returns True if enrollment succeeded and embeddings were saved; False otherwise.
    """
    _ensure_gui_environment()
    cfg = config or FaceUnlockConfig.load()
    cam_path = _find_webcam_device(cfg)

    cap = cv2.VideoCapture(cam_path)
    if not cap.isOpened():
        sys.stderr.write(f"\033[91m[!] Error: Unable to open RGB webcam at {cam_path}\033[0m\n")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Argus - Biometric Face Enrollment"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except Exception as e:
        sys.stderr.write(f"\033[91m[!] GUI display error: {e}\033[0m\n")
        cap.release()
        return False

    captured_embeddings: List[np.ndarray] = []
    total_stages = len(ENROLL_STAGES)
    current_stage = 0
    stage_hold_frames = 0
    needed_hold_frames = 6

    total_ticks = 40
    ticks_per_stage = total_ticks // total_stages
    completed_ticks = 0

    success = False
    start_time = time.monotonic()
    timeout = 60.0  # 60s total allowed for user to follow guided steps

    try:
        while time.monotonic() - start_time < timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            reticle_radius = int(min(w, h) * 0.35)

            # Detect faces
            faces = engine.detect_faces(frame)
            face_in_bounds = False
            best_face = None

            if faces:
                # Find largest face
                faces.sort(key=lambda f: f[2] * f[3], reverse=True)
                candidate = faces[0]
                fx, fy, fw, fh = candidate[:4]
                face_center_x = fx + fw / 2.0
                face_center_y = fy + fh / 2.0

                # Check if face is centered inside the reticle
                dist_from_center = math.dist((face_center_x, face_center_y), (cx, cy))
                min_face_size = reticle_radius * 0.70
                max_face_size = reticle_radius * 1.60

                if dist_from_center < (reticle_radius * 0.45) and (min_face_size <= fh <= max_face_size):
                    face_in_bounds = True
                    best_face = candidate

            # Guidance logic
            stage_info = ENROLL_STAGES[current_stage]
            bottom_instr = stage_info["label"]
            hint_str = stage_info["hint"]

            if not faces:
                hint_str = "No face detected. Look directly at the screen."
            elif not face_in_bounds:
                hint_str = "Center your face inside the circle."
            else:
                stage_hold_frames += 1
                if stage_hold_frames >= needed_hold_frames:
                    # Extract high-accuracy normalized embedding
                    emb = engine.extract_embedding(frame, best_face)
                    captured_embeddings.append(emb)

                    current_stage += 1
                    completed_ticks = min(total_ticks, current_stage * ticks_per_stage)
                    stage_hold_frames = 0

                    if current_stage >= total_stages:
                        completed_ticks = total_ticks
                        success = True
                        break

            # Render reticle and overlays
            _draw_ios_reticle(frame, cx, cy, reticle_radius, completed_ticks, total_ticks, face_in_bounds)
            _draw_hud_banner(
                frame,
                top_text=f"Enrolling User: {user}",
                sub_text=f"Model: {model_name[:24]} ({current_stage}/{total_stages} Complete)",
                bottom_instruction=bottom_instr,
                hint=hint_str,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                success = False
                break

            # Check if user clicked window close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                success = False
                break

        # If all 5 angles captured successfully, show completion splash for 1 second
        if success and len(captured_embeddings) >= total_stages:
            for _ in range(25):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)
                    _draw_ios_reticle(frame, cx, cy, reticle_radius, total_ticks, total_ticks, is_success=True)
                    _draw_hud_banner(
                        frame,
                        top_text="Biometric Registration Complete",
                        sub_text=f"Saved {len(captured_embeddings)} angles for '{user}'",
                        bottom_instruction="Success! Encrypting Vault...",
                        hint="Face model registered and hardware-secured",
                    )
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(40)

            # Save embeddings to encrypted biometric vault
            saved = engine.enroll_user(user, captured_embeddings, camera_type="rgb_webcam", model_name=model_name)
            return saved

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Drain events
        for _ in range(5):
            cv2.waitKey(1)

    return False
