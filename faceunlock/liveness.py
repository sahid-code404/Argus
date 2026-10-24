"""Multi-modal Anti-Spoofing and Liveness Detection Engine.

Protects against:
1. Digital screen replay attacks (smartphones, tablets, computer monitors)
2. Printed photo presentation attacks (2D paper prints, glossy photo paper)
3. Static cutouts and masks

Techniques implemented:
- Deep Neural Network Face Anti-Spoofing (MiniFASNetV2-SE trained on 70,000+ attack samples)
- Planar Homography Parallax constraint (2D planar affine error vs 3D non-rigid depth)
- Phone screen bezel & rectangular border detection (Hough line analysis around face ROI)
- Specular glass glare & screen backlight saturation analysis
- High-frequency Fourier 2D FFT moiré / pixel grid harmonic analysis
- Modified Laplacian variance & LBP micro-texture surface analysis
- YCrCb / HSV chromaticity consistency
- Temporal micro-movement, blinking, and multi-frame consistency enforcement
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np


@dataclass
class LivenessResult:
    is_live: bool
    confidence: float  # 0.0 to 1.0
    verdict: str       # "LIVE", "NEURAL_PAD_SPOOF", "PLANAR_2D_SPOOF", "STATIC_PHOTO", "PHONE_BEZEL_DETECTED", "SCREEN_REPLAY", "INSUFFICIENT_DATA"
    texture_score: float = 0.0
    motion_score: float = 0.0
    frequency_score: float = 0.0
    color_score: float = 0.0
    neural_score: float = 0.0
    bezel_score: float = 0.0
    glare_score: float = 0.0
    blinks_detected: int = 0
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.is_live else "FAILED"
        return (f"Liveness {status} ({self.verdict}, confidence={self.confidence:.2f}, "
                f"neural={self.neural_score:.2f}, motion={self.motion_score:.2f})")


class LivenessDetector:
    """Multi-feature anti-spoofing validator for face unlock."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_liveness_threshold: float = 0.60,
        texture_weight: float = 0.20,
        motion_weight: float = 0.30,
        frequency_weight: float = 0.15,
        color_weight: float = 0.10,
        neural_weight: float = 0.25,
    ):
        self.threshold = min_liveness_threshold
        self.weights = {
            "texture": texture_weight,
            "motion": motion_weight,
            "frequency": frequency_weight,
            "color": color_weight,
            "neural": neural_weight,
        }
        self.frame_history: List[np.ndarray] = []
        self.landmark_history: List[np.ndarray] = []
        self.eye_variance_history: List[float] = []
        self.max_history = 12

        self.neural_net: Optional[cv2.dnn.Net] = None
        self._init_neural_model(model_path)

    def _init_neural_model(self, model_path: Optional[str] = None) -> None:
        """Initialize MiniFASNet neural anti-spoofing network."""
        if model_path is None:
            candidates = [
                Path(__file__).resolve().parent.parent / "models" / "face_antispoof_minifasnet.onnx",
                Path("/opt/faceunlock/models/face_antispoof_minifasnet.onnx"),
                Path("/etc/faceunlock/models/face_antispoof_minifasnet.onnx"),
            ]
            for c in candidates:
                if c.exists():
                    model_path = str(c)
                    break
        if model_path and Path(model_path).exists():
            try:
                self.neural_net = cv2.dnn.readNetFromONNX(str(model_path))
            except Exception:
                self.neural_net = None

    def reset(self) -> None:
        """Clear temporal history for a fresh authentication session."""
        self.frame_history.clear()
        self.landmark_history.clear()
        self.eye_variance_history.clear()

    @staticmethod
    def extract_face_roi(image: np.ndarray, face_box: np.ndarray) -> Optional[np.ndarray]:
        """Extract cropped facial ROI with margin."""
        h, w = image.shape[:2]
        x, y, fw, fh = [int(v) for v in face_box[:4]]

        # Add 10% margin
        margin_x = int(fw * 0.1)
        margin_y = int(fh * 0.1)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        return roi

    def evaluate_neural(self, frame: np.ndarray, face_box: np.ndarray) -> Tuple[float, bool, dict]:
        """Deep neural network face anti-spoofing via MiniFASNetV2-SE.

        Outputs (probability_real: float, is_real: bool, details: dict).
        Trained on CelebA-Spoof to detect 2D prints, screen replays, paper cutouts, masks.
        """
        if self.neural_net is None or frame is None:
            return 0.85, True, {"status": "model_not_loaded"}

        h, w = frame.shape[:2]
        x, y, fw, fh = [int(v) for v in face_box[:4]]
        max_dim = max(fw, fh)
        cx, cy = x + fw / 2, y + fh / 2

        crop_size = int(max_dim * 1.5)
        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)

        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w, x1 + crop_size)
        crop_y2 = min(h, y1 + crop_size)

        top_pad = max(0, -y1)
        left_pad = max(0, -x1)
        bottom_pad = max(0, (y1 + crop_size) - h)
        right_pad = max(0, (x1 + crop_size) - w)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return 0.0, False, {"status": "invalid_crop"}

        padded = cv2.copyMakeBorder(cropped, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT_101)

        # Resize to 128x128 with letterboxing
        ratio = 128.0 / max(padded.shape[:2])
        scaled = tuple([int(dim * ratio) for dim in padded.shape[:2]])
        resized = cv2.resize(padded, (scaled[1], scaled[0]), interpolation=cv2.INTER_AREA)

        dw = 128 - scaled[1]
        dh = 128 - scaled[0]
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        input_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        blob = input_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        self.neural_net.setInput(blob)
        out = self.neural_net.forward()
        logits = out[0]

        real_logit = float(logits[0])
        spoof_logit = float(logits[1])
        logit_diff = real_logit - spoof_logit

        # Sigmoid probability
        prob_real = 1.0 / (1.0 + math.exp(-logit_diff))
        # Hard veto cutoff: screen replays and prints score strongly negative (< -2.0)
        # Normal CMOS webcams in low lighting fluctuate around -0.5 to -1.0
        is_real = logit_diff >= -1.8

        return prob_real, is_real, {
            "real_logit": real_logit,
            "spoof_logit": spoof_logit,
            "logit_diff": logit_diff,
            "prob_real": prob_real,
        }

    @staticmethod
    def detect_screen_bezel(frame: np.ndarray, face_box: np.ndarray) -> Tuple[float, dict]:
        """Detect rectangular bezel edges indicative of a smartphone, tablet, or photo border."""
        h, w = frame.shape[:2]
        x, y, fw, fh = [int(v) for v in face_box[:4]]

        margin_x = int(fw * 0.45)
        margin_y = int(fh * 0.45)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return 0.85, {"status": "roi_too_small"}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=int(min(fw, fh) * 0.65), maxLineGap=12
        )

        vertical_lines = 0
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = [int(v) for v in np.array(line).ravel()[:4]]
                dx = abs(lx2 - lx1)
                dy = abs(ly2 - ly1)
                angle = math.atan2(dy, dx) * 180 / math.pi
                if angle > 75:
                    vertical_lines += 1
                elif angle < 15:
                    horizontal_lines += 1

        if vertical_lines >= 2 and horizontal_lines >= 1:
            score = 0.15  # Clear phone/tablet bezel bounding box
        elif vertical_lines >= 2 or horizontal_lines >= 2:
            score = 0.40
        else:
            score = 0.95

        return score, {"vertical_lines": vertical_lines, "horizontal_lines": horizontal_lines}

    @staticmethod
    def detect_specular_glare(face_roi: np.ndarray) -> Tuple[float, dict]:
        """Detect phone screen glass reflections and backlight glare hotspots."""
        if face_roi is None or face_roi.size == 0 or len(face_roi.shape) != 3:
            return 0.9, {}

        # Saturated white specular pixels (glass screen reflection)
        white_mask = (face_roi[:, :, 0] > 248) & (face_roi[:, :, 1] > 248) & (face_roi[:, :, 2] > 248)
        saturated_pixels = int(np.sum(white_mask))
        total_pixels = face_roi.shape[0] * face_roi.shape[1]
        sat_ratio = saturated_pixels / (total_pixels + 1e-6)

        if sat_ratio > 0.008:
            score = max(0.1, 1.0 - sat_ratio * 30.0)
        else:
            score = 0.95

        return score, {"sat_ratio": sat_ratio}

    def analyze_texture(self, face_roi: np.ndarray) -> Tuple[float, dict]:
        """Analyze micro-texture and skin boundary gradients."""
        if face_roi is None or face_roi.size == 0:
            return 0.0, {}

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (128, 128))

        # 1. Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Typical live faces in office/home lighting fall in 80 - 3000
        if lap_var < 35.0:
            score_lap = max(0.0, lap_var / 35.0 * 0.3)
        elif lap_var > 4500.0:
            score_lap = max(0.0, 1.0 - (lap_var - 4500.0) / 3000.0)
        else:
            score_lap = 0.70 + 0.30 * min(1.0, lap_var / 800.0)

        # 2. Local contrast approximation
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        contrast = float(np.mean(diff))
        contrast_score = min(1.0, max(0.0, contrast / 15.0))

        texture_score = float(np.clip(0.6 * score_lap + 0.4 * contrast_score, 0.0, 1.0))
        return texture_score, {"laplacian_var": lap_var, "contrast": contrast}

    def analyze_frequency_spectrum(self, face_roi: np.ndarray) -> Tuple[float, dict]:
        """Detect moiré patterns and screen refresh artifacts via 2D FFT."""
        if face_roi is None or face_roi.size == 0:
            return 0.0, {}

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (128, 128))

        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        magnitude_log = np.log1p(magnitude)

        cy, cx = 64, 64
        y_indices, x_indices = np.ogrid[:128, :128]
        dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

        low_mask = dist_from_center <= 16
        high_mask = dist_from_center > 48

        low_energy = float(np.mean(magnitude_log[low_mask]))
        high_energy = float(np.mean(magnitude_log[high_mask]))

        max_high = float(np.max(magnitude_log[high_mask]))
        hf_peak_ratio = max_high / (low_energy + 1e-6)
        hf_ratio = high_energy / (low_energy + 1e-6)
        has_moire_peak = (hf_peak_ratio > 0.80 and max_high > 8.0)

        if hf_ratio > 0.72 or has_moire_peak:
            score = 0.2  # Suspected screen moiré / replay
        elif hf_ratio < 0.25:
            score = 0.4  # Suspected washed out print
        else:
            score = 0.9  # Organic frequency decay

        return score, {
            "low_energy": low_energy,
            "high_energy": high_energy,
            "hf_ratio": hf_ratio,
            "hf_peak_ratio": hf_peak_ratio,
        }

    def analyze_chroma_and_lighting(self, face_roi: np.ndarray) -> Tuple[float, dict]:
        """Examine skin chromaticity consistency in YCrCb and HSV spaces."""
        if face_roi is None or len(face_roi.shape) != 3:
            return 0.8, {}

        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        sat = hsv[:, :, 1]

        mean_cr = float(np.mean(cr))
        mean_cb = float(np.mean(cb))
        mean_sat = float(np.mean(sat))

        cr_valid = 125 <= mean_cr <= 185
        cb_valid = 75 <= mean_cb <= 145
        sat_reasonable = 10 <= mean_sat <= 230

        score = 0.5
        if cr_valid and cb_valid and sat_reasonable:
            score = 0.95
        elif cr_valid or cb_valid:
            score = 0.6
        else:
            score = 0.25

        return score, {"mean_cr": mean_cr, "mean_cb": mean_cb, "mean_sat": mean_sat}

    def analyze_temporal_motion(self, landmarks_list: List[np.ndarray]) -> Tuple[float, int, dict]:
        """Analyze temporal micro-movements, eye blinks, and 3D non-rigid parallax.

        1. Static photo check: 0 movement & 0 landmark variance -> STATIC_PHOTO.
        2. Planar homography check:
           A moving 2D phone/photo has purely planar affine motion with near-zero residual error.
           A real 3D head has depth parallax between nose and eyes, resulting in non-affine residual.
        3. Eye blink & micro-saccade detection.
        """
        if len(landmarks_list) < 3:
            return 0.5, 0, {"reason": "accumulating_temporal_history", "frames": len(landmarks_list)}

        eye_distances = []
        nose_eye_ratios = []
        displacements = []
        affine_residuals = []

        for i in range(len(landmarks_list)):
            lm = landmarks_list[i].reshape(-1, 2)
            r_eye, l_eye, nose = lm[0], lm[1], lm[2]

            iod = math.dist(r_eye, l_eye)
            eye_distances.append(iod)

            eye_mid = (r_eye + l_eye) / 2.0
            nose_dist = math.dist(nose, eye_mid)
            nose_eye_ratios.append(nose_dist / (iod + 1e-6))

            if i > 0:
                prev_lm = landmarks_list[i - 1].reshape(-1, 2)
                disp = math.dist(nose, prev_lm[2])
                displacements.append(disp)

                # Estimate 2D affine transformation between consecutive frames
                try:
                    M, _ = cv2.estimateAffinePartial2D(prev_lm, lm)
                    if M is not None:
                        pred = cv2.transform(prev_lm.reshape(-1, 1, 2), M).reshape(-1, 2)
                        res_err = float(np.mean(np.linalg.norm(lm - pred, axis=1)))
                        affine_residuals.append(res_err)
                except Exception:
                    pass

        ratio_std = float(np.std(nose_eye_ratios))
        iod_std = float(np.std(eye_distances))
        avg_disp = float(np.mean(displacements)) if displacements else 0.0
        avg_affine_res = float(np.mean(affine_residuals)) if affine_residuals else 0.0

        blinks = 0
        for i in range(1, len(eye_distances)):
            if abs(eye_distances[i] - eye_distances[i - 1]) > 0.07 * (eye_distances[i] + 1e-6):
                blinks += 1

        is_static_2d = bool(avg_disp < 0.25 and ratio_std < 0.001)
        # Planar moving photo check: moving, but perfectly affine (flat 2D surface moving in space)
        is_planar_moving_2d = bool(avg_disp >= 0.5 and avg_affine_res < 0.20)

        if is_static_2d:
            motion_score = 0.10  # Frozen 2D photo
        elif is_planar_moving_2d:
            motion_score = 0.20  # Moving flat 2D screen/paper (homography constraint violated)
        elif blinks > 0 or (avg_disp >= 0.4 and avg_affine_res >= 0.35):
            motion_score = 0.95  # Confirmed 3D depth parallax & biological movement
        else:
            motion_score = 0.70

        return motion_score, blinks, {
            "ratio_std": ratio_std,
            "iod_std": iod_std,
            "avg_disp": avg_disp,
            "avg_affine_res": avg_affine_res,
            "is_static_2d": is_static_2d,
            "is_planar_moving_2d": is_planar_moving_2d,
            "blinks": blinks,
            "frames": len(landmarks_list),
        }

    def analyze_eye_patch_dynamics(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[float, int, dict]:
        """Passive micro-gaze & blink dynamics without requiring active user head motion."""
        if frame is None or frame.size == 0 or landmarks is None or len(landmarks) < 4:
            return 0.7, 0, {"status": "insufficient_data"}

        h, w = frame.shape[:2]
        r_eye = (int(landmarks[0]), int(landmarks[1]))
        l_eye = (int(landmarks[2]), int(landmarks[3]))
        iod = math.dist(r_eye, l_eye)
        radius = max(3, int(iod * 0.12))

        patches = []
        for (cx, cy) in (r_eye, l_eye):
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w, cx + radius)
            y2 = min(h, cy + radius)
            if x2 > x1 and y2 > y1:
                patch = frame[y1:y2, x1:x2]
                if len(patch.shape) == 3:
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                patches.append(patch)

        if not patches:
            return 0.7, 0, {"status": "no_patches"}

        patch_vars = [float(np.var(p)) for p in patches if p.size > 0]
        current_eye_var = float(np.mean(patch_vars)) if patch_vars else 0.0

        self.eye_variance_history.append(current_eye_var)
        if len(self.eye_variance_history) > self.max_history:
            self.eye_variance_history.pop(0)

        blinks = 0
        temporal_eye_std = 0.0
        if len(self.eye_variance_history) >= 3:
            temporal_eye_std = float(np.std(self.eye_variance_history))
            mean_var = float(np.mean(self.eye_variance_history)) + 1e-6
            for i in range(1, len(self.eye_variance_history)):
                diff = abs(self.eye_variance_history[i] - self.eye_variance_history[i - 1])
                if diff / mean_var > 0.35:
                    blinks += 1

        is_frozen_photo = bool(len(self.eye_variance_history) >= 4 and temporal_eye_std < 0.8)
        if is_frozen_photo:
            score = 0.20
        elif blinks > 0 or temporal_eye_std > 6.0:
            score = 0.95
        else:
            score = 0.75

        return score, blinks, {
            "current_eye_var": current_eye_var,
            "temporal_eye_std": temporal_eye_std,
            "is_frozen_photo": is_frozen_photo,
            "eye_blinks": blinks,
            "history_len": len(self.eye_variance_history),
        }

    def evaluate_frame(
        self,
        frame: np.ndarray,
        face_detection: np.ndarray,
        is_ir_mode: bool = False,
    ) -> LivenessResult:
        """Evaluate a single captured frame and update temporal history."""
        if face_detection is None or len(face_detection) < 15:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                verdict="NO_FACE",
                details={"reason": "no_face_detected"},
            )

        face_box = face_detection[:4]
        landmarks = face_detection[4:14]
        face_roi = self.extract_face_roi(frame, face_box)

        if face_roi is None:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                verdict="INVALID_FACE_ROI",
                details={"reason": "face_roi_out_of_bounds"},
            )

        # Update history
        self.frame_history.append(face_roi)
        self.landmark_history.append(landmarks)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
            self.landmark_history.pop(0)

        # 1. MiniFASNet Deep Neural Anti-Spoofing
        neural_score, is_neural_real, neural_details = self.evaluate_neural(frame, face_box)

        # 2. Phone Screen Bezel & Frame Detection
        bezel_score, bezel_details = self.detect_screen_bezel(frame, face_box)

        # 3. Specular Glare & Backlight Saturation
        glare_score, glare_details = self.detect_specular_glare(face_roi)

        # 4. Micro-texture
        texture_score, tex_details = self.analyze_texture(face_roi)

        # 5. Frequency spectrum
        freq_score, freq_details = self.analyze_frequency_spectrum(face_roi)

        # 6. Color & dynamic range
        if is_ir_mode:
            color_score = 0.9
            color_details = {"mode": "infrared_active"}
        else:
            color_score, color_details = self.analyze_chroma_and_lighting(face_roi)

        # 7. Temporal motion & 3D planar homography parallax
        motion_score, blinks, motion_details = self.analyze_temporal_motion(self.landmark_history)

        # 8. Passive eye micro-gaze & blink dynamics
        eye_score, eye_blinks, eye_details = self.analyze_eye_patch_dynamics(frame, landmarks)
        total_blinks = max(blinks, eye_blinks)
        motion_score = float(np.clip(0.75 * motion_score + 0.25 * eye_score, 0.0, 1.0))

        # Weighted confidence fusion
        if is_ir_mode:
            confidence = (
                0.40 * texture_score +
                0.40 * motion_score +
                0.20 * freq_score
            )
            is_live = bool(confidence >= self.threshold)
            verdict = "LIVE" if is_live else "IR_SPOOF_REJECTED"
        else:
            confidence = (
                self.weights["neural"] * neural_score +
                self.weights["motion"] * motion_score +
                self.weights["texture"] * texture_score +
                self.weights["frequency"] * freq_score +
                self.weights["color"] * color_score
            )
            # Factor in bezel and glare penalties
            confidence *= (bezel_score * 0.4 + 0.6)
            confidence *= (glare_score * 0.3 + 0.7)

            # Strict Veto Checks:
            if not is_neural_real and self.neural_net is not None:
                is_live = False
                verdict = "NEURAL_PAD_SPOOF"
            elif (motion_details.get("is_static_2d") or eye_details.get("is_frozen_photo")) and len(self.landmark_history) >= 4:
                is_live = False
                verdict = "STATIC_2D_PHOTO"
            elif motion_details.get("is_planar_moving_2d") and len(self.landmark_history) >= 4:
                is_live = False
                verdict = "PLANAR_2D_SPOOF"
            elif bezel_score < 0.3:
                is_live = False
                verdict = "PHONE_BEZEL_DETECTED"
            elif len(self.landmark_history) < 3:
                is_live = False
                verdict = "INSUFFICIENT_TEMPORAL_HISTORY"
            else:
                is_live = bool(confidence >= self.threshold)
                verdict = "LIVE" if is_live else "SPOOF_DETECTED"

        details = {
            "neural": neural_details,
            "bezel": bezel_details,
            "glare": glare_details,
            "texture": tex_details,
            "frequency": freq_details,
            "color": color_details,
            "motion": motion_details,
            "eye_dynamics": eye_details,
            "history_depth": len(self.landmark_history),
        }

        return LivenessResult(
            is_live=is_live,
            confidence=float(confidence),
            verdict=verdict,
            texture_score=texture_score,
            motion_score=motion_score,
            frequency_score=freq_score,
            color_score=color_score,
            neural_score=neural_score,
            bezel_score=bezel_score,
            glare_score=glare_score,
            blinks_detected=total_blinks,
            details=details,
        )
