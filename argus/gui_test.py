"""Graphical Face Detection and Verification Live Preview Window.

Provides:
- Real-time webcam preview with face bounding box
- 5-point facial landmark tracking
- Real-time anti-spoofing and match score HUD overlay
- Instant visual feedback on verification pass/fail
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb;wayland")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ["QT_LOGGING_RULES"] = "*.warning=false;qt.*=false;QFontDatabase=false"

import cv2
import numpy as np

from argus.config import ArgusConfig, FaceUnlockConfig
from argus.engine import FaceEngine
from argus.gui_enroll import _ensure_gui_environment, _find_webcam_device
from argus.hardware import select_camera


def run_gui_test(user: str, engine: FaceEngine, config: Optional[FaceUnlockConfig] = None) -> None:
    """Run interactive graphical camera preview displaying face detection and verification HUD."""
    _ensure_gui_environment()
    cfg = config or ArgusConfig.load()
    cam_path = _find_webcam_device(cfg)

    cap = cv2.VideoCapture(cam_path)
    if not cap.isOpened():
        sys.stderr.write(f"\033[91m[!] Error: Unable to open RGB webcam at {cam_path}\033[0m\n")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Argus - Live Verification Preview"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except Exception as e:
        sys.stderr.write(f"\033[91m[!] GUI display error: {e}\033[0m\n")
        cap.release()
        return

    is_enrolled = engine.is_enrolled(user)
    enrolled_embeddings = engine.load_user_embeddings(user) if is_enrolled else []

    fps_count = 0
    fps_start = time.monotonic()
    current_fps = 0.0

    print("=" * 60)
    print(f" Live Camera Preview Started for user: '{user}'")
    print(" Close the camera window or press 'q' / ESC to exit.")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            fps_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.monotonic()

            h, w = frame.shape[:2]
            faces = engine.detect_faces(frame)

            best_match_score = 0.0
            is_verified = False
            liveness_label = "PENDING"
            liveness_color = (200, 200, 200)

            for face in faces:
                fx, fy, fw, fh = [int(v) for v in face[:4]]
                landmarks = face[4:14].reshape(-1, 2)

                # Liveness check
                liveness_res = engine.liveness_detector.evaluate_frame(frame, face)
                if liveness_res.is_live:
                    liveness_label = f"LIVE ({liveness_res.confidence:.2f})"
                    liveness_color = (50, 255, 120)
                else:
                    liveness_label = f"SPOOF ({liveness_res.verdict})"
                    liveness_color = (60, 60, 255)

                # Feature matching
                if is_enrolled and enrolled_embeddings:
                    emb = engine.extract_embedding(frame, face)
                    match_score, _, _ = engine.find_best_match(emb, enrolled_embeddings, user=user)
                    best_match_score = max(best_match_score, match_score)
                    if match_score >= cfg.cosine_threshold and liveness_res.is_live:
                        is_verified = True
                        box_color = (50, 255, 100)  # Green
                        status_str = f"VERIFIED: {user} ({match_score:.3f})"
                    else:
                        box_color = (50, 180, 255)  # Orange
                        status_str = f"UNKNOWN ({match_score:.3f})"
                else:
                    box_color = (255, 200, 40)
                    status_str = "FACE DETECTED"

                # Draw bounding box
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2, cv2.LINE_AA)

                # Draw 5 landmarks
                for lx, ly in landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 255), -1, cv2.LINE_AA)

                # Draw tag above face
                tag_y = max(25, fy - 10)
                cv2.putText(frame, status_str, (fx, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, box_color, 1, cv2.LINE_AA)

            # Top info HUD pill
            hud_bg = frame[10:90, 10:360]
            if hud_bg.shape[0] == 80 and hud_bg.shape[1] == 350:
                rect = np.zeros_like(hud_bg, dtype=np.uint8)
                cv2.rectangle(rect, (0, 0), (350, 80), (20, 20, 20), -1)
                cv2.addWeighted(hud_bg, 0.35, rect, 0.65, 0, hud_bg)
                cv2.rectangle(frame, (10, 10), (360, 90), (60, 60, 60), 1, cv2.LINE_AA)

            cv2.putText(frame, f"User: {user} ({'Enrolled' if is_enrolled else 'Not Enrolled'})", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Liveness: {liveness_label}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, liveness_color, 1, cv2.LINE_AA)
            threshold_str = f"Threshold: {cfg.cosine_threshold:.2f} | FPS: {current_fps:.1f}"
            cv2.putText(frame, threshold_str, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            # Bottom exit prompt
            cv2.putText(frame, "Press 'q' or ESC to exit preview", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass
        print("\033[92m[✓] Live camera preview closed.\033[0m")
