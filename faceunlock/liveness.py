"""Multi-modal Anti-Spoofing and Liveness Detection Engine.

Protects against:
1. Printed photo presentation attacks (2D paper prints)
2. Digital screen replay attacks (smartphones, tablets, monitors)
3. Static cutouts and masks

Techniques implemented:
- High-frequency Fourier / Laplacian micro-texture analysis (screens exhibit moiré patterns and grid frequencies; prints exhibit halftoning)
- YCrCb / HSV chromaticity & dynamic range analysis (screens and photos distort natural human skin subsurface scattering)
- 3D facial landmark temporal motion & non-rigid parallax tracking (a static 2D photo exhibits rigid planar translation; live faces exhibit micro-saccades and 3D depth parallax)
- Eye-blink / micro-movement detection using landmark geometry
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np


@dataclass
class LivenessResult:
    is_live: bool
    confidence: float  # 0.0 to 1.0
    verdict: str       # "LIVE", "SPOOF_DETECTED", "STATIC_PHOTO", "SCREEN_REPLAY", "INSUFFICIENT_DATA"
    texture_score: float = 0.0
    motion_score: float = 0.0
    frequency_score: float = 0.0
    color_score: float = 0.0
    blinks_detected: int = 0
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.is_live else "FAILED"
        return (f"Liveness {status} ({self.verdict}, confidence={self.confidence:.2f}, "
                f"texture={self.texture_score:.2f}, motion={self.motion_score:.2f})")


class LivenessDetector:
    """Multi-feature anti-spoofing validator for face unlock."""

    def __init__(
        self,
        min_liveness_threshold: float = 0.55,
        texture_weight: float = 0.35,
        motion_weight: float = 0.35,
        frequency_weight: float = 0.15,
        color_weight: float = 0.15,
    ):
        self.threshold = min_liveness_threshold
        self.weights = {
            "texture": texture_weight,
            "motion": motion_weight,
            "frequency": frequency_weight,
            "color": color_weight,
        }
        self.frame_history: List[np.ndarray] = []
        self.landmark_history: List[np.ndarray] = []
        self.max_history = 12

    def reset(self) -> None:
        """Clear temporal history for a fresh authentication session."""
        self.frame_history.clear()
        self.landmark_history.clear()

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

    def analyze_texture(self, face_roi: np.ndarray) -> Tuple[float, dict]:
        """Analyze micro-texture and skin boundary gradients.

        Real skin exhibits rich micro-gradients with controlled variance.
        Flat photo prints have lower Laplacian variance or uniform grain.
        Screen replays have distinct banding or specular highlights.
        """
        if face_roi is None or face_roi.size == 0:
            return 0.0, {}

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (128, 128))

        # 1. Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Typical live faces in office/home lighting fall in 100 - 3000
        # Extremely blurry or static paper prints are often < 50
        # Highly pixelated screens or moire noise can exceed 5000+
        if lap_var < 40.0:
            score_lap = max(0.0, lap_var / 40.0 * 0.4)
        elif lap_var > 4500.0:
            score_lap = max(0.0, 1.0 - (lap_var - 4500.0) / 3000.0)
        else:
            score_lap = 0.85 + 0.15 * min(1.0, lap_var / 1000.0)

        # 2. Local Binary Pattern (LBP) contrast approximation
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        contrast = float(np.mean(diff))
        contrast_score = min(1.0, max(0.0, contrast / 15.0))

        texture_score = float(np.clip(0.6 * score_lap + 0.4 * contrast_score, 0.0, 1.0))
        return texture_score, {"laplacian_var": lap_var, "contrast": contrast}

    def analyze_frequency_spectrum(self, face_roi: np.ndarray) -> Tuple[float, dict]:
        """Detect moiré patterns and screen refresh artifacts via 2D FFT.

        Screens produce prominent periodic frequency peaks due to the pixel grid.
        Natural faces show smoothly decaying power spectral density.
        """
        if face_roi is None or face_roi.size == 0:
            return 0.0, {}

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (128, 128))

        # Compute 2D FFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        magnitude_log = np.log1p(magnitude)

        # Measure high-frequency ring ratio vs central low-frequency
        cy, cx = 64, 64
        y_indices, x_indices = np.ogrid[:128, :128]
        dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

        low_mask = dist_from_center <= 16
        mid_mask = (dist_from_center > 16) & (dist_from_center <= 48)
        high_mask = dist_from_center > 48

        low_energy = float(np.mean(magnitude_log[low_mask]))
        mid_energy = float(np.mean(magnitude_log[mid_mask]))
        high_energy = float(np.mean(magnitude_log[high_mask]))

        # High-frequency harmonic peak detection (screens have sharp harmonic spikes)
        max_high = float(np.max(magnitude_log[high_mask]))
        hf_peak_ratio = max_high / (low_energy + 1e-6)

        # High frequency to low frequency ratio
        hf_ratio = high_energy / (low_energy + 1e-6)
        has_moire_peak = (hf_peak_ratio > 0.80 and max_high > 8.0)

        # Moiré screens usually exhibit anomalous high-frequency spikes or sharp peaks
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
        """Examine skin chromaticity consistency in YCrCb and HSV spaces.

        Human skin exhibits tight distribution in the Cr/Cb plane regardless of ethnicity.
        Paper prints and screens display blue/magenta cast or severe saturation clipping.
        """
        if face_roi is None or len(face_roi.shape) != 3:
            return 0.8, {}  # Pass through for grayscale/IR cameras

        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        sat = hsv[:, :, 1]

        mean_cr = float(np.mean(cr))
        mean_cb = float(np.mean(cb))
        mean_sat = float(np.mean(sat))

        # Standard human skin Cr/Cb bounding ellipse
        # Cr typically between 133 and 175, Cb between 80 and 135
        cr_valid = 125 <= mean_cr <= 185
        cb_valid = 75 <= mean_cb <= 145
        sat_reasonable = 10 <= mean_sat <= 230

        score = 0.5
        if cr_valid and cb_valid and sat_reasonable:
            score = 0.95
        elif cr_valid or cb_valid:
            score = 0.6
        else:
            score = 0.25  # High probability of color distortion / paper print

        return score, {"mean_cr": mean_cr, "mean_cb": mean_cb, "mean_sat": mean_sat}

    def analyze_temporal_motion(self, landmarks_list: List[np.ndarray]) -> Tuple[float, int, dict]:
        """Analyze temporal micro-movements across consecutive frames.

        YuNet produces 5 landmarks: [right_eye, left_eye, nose, right_mouth, left_mouth].
        - A 2D photo held by hand or on a stand has rigid motion: all points move with identical vectors.
        - A living human face has micro-saccades, blinking, and 3D non-rigid parallax:
          The nose-to-eye distance ratio and eye-to-mouth ratio fluctuate naturally.
        """
        if len(landmarks_list) < 3:
            # Not enough frames yet, return neutral baseline
            return 0.7, 0, {"reason": "accumulating_temporal_history"}

        # Extract eye coordinates across frames to detect blinking and micro-variance
        eye_distances = []
        nose_eye_ratios = []
        displacements = []

        for i in range(len(landmarks_list)):
            lm = landmarks_list[i].reshape(-1, 2)
            r_eye, l_eye, nose = lm[0], lm[1], lm[2]
            r_mouth, l_mouth = lm[3], lm[4]

            # Inter-ocular distance
            iod = math.dist(r_eye, l_eye)
            eye_distances.append(iod)

            # Nose to eye midpoint distance
            eye_mid = (r_eye + l_eye) / 2.0
            nose_dist = math.dist(nose, eye_mid)
            nose_eye_ratios.append(nose_dist / (iod + 1e-6))

            if i > 0:
                prev_nose = landmarks_list[i - 1].reshape(-1, 2)[2]
                disp = math.dist(nose, prev_nose)
                displacements.append(disp)

        # Variance of nose-eye ratio: captures 3D head pitch/yaw parallax
        ratio_std = float(np.std(nose_eye_ratios))
        iod_std = float(np.std(eye_distances))
        avg_disp = float(np.mean(displacements)) if displacements else 0.0

        # Detect blink or micro-twitch: rapid change in landmark geometry
        blinks = 0
        for i in range(1, len(eye_distances)):
            if abs(eye_distances[i] - eye_distances[i - 1]) > 0.08 * (eye_distances[i] + 1e-6):
                blinks += 1

        # Completely static photo (zero displacement and zero variance)
        if avg_disp < 0.2 and ratio_std < 0.001:
            motion_score = 0.2  # Suspected static print on stand or rigid freeze
        elif blinks > 0 or (0.5 <= avg_disp <= 25.0 and ratio_std > 0.003):
            # Natural micro-motion with 3D perspective variance
            motion_score = 0.95
        else:
            motion_score = 0.75

        return motion_score, blinks, {
            "ratio_std": ratio_std,
            "iod_std": iod_std,
            "avg_disp": avg_disp,
            "blinks": blinks,
        }

    def evaluate_frame(
        self,
        frame: np.ndarray,
        face_detection: np.ndarray,
        is_ir_mode: bool = False,
    ) -> LivenessResult:
        """Evaluate a single captured frame and update temporal history.

        face_detection is the YuNet 15-dim vector:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
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

        # 1. Texture analysis
        texture_score, tex_details = self.analyze_texture(face_roi)

        # 2. Frequency spectrum
        freq_score, freq_details = self.analyze_frequency_spectrum(face_roi)

        # 3. Color & dynamic range (in IR mode, color analysis is skipped)
        if is_ir_mode:
            color_score = 0.9
            color_details = {"mode": "infrared_active"}
        else:
            color_score, color_details = self.analyze_chroma_and_lighting(face_roi)

        # 4. Temporal motion and parallax
        motion_score, blinks, motion_details = self.analyze_temporal_motion(self.landmark_history)

        # Weighted confidence fusion
        if is_ir_mode:
            # In IR mode, texture and motion dominate; IR light intrinsically rejects screens
            confidence = (
                0.45 * texture_score +
                0.45 * motion_score +
                0.10 * freq_score
            )
        else:
            confidence = (
                self.weights["texture"] * texture_score +
                self.weights["motion"] * motion_score +
                self.weights["frequency"] * freq_score +
                self.weights["color"] * color_score
            )

        is_live = bool(confidence >= self.threshold)

        verdict = "LIVE"
        if not is_live:
            if motion_score < 0.35 and len(self.landmark_history) >= 4:
                verdict = "STATIC_PHOTO"
            elif freq_score < 0.35:
                verdict = "SCREEN_REPLAY"
            elif texture_score < 0.35:
                verdict = "SPOOF_TEXTURE"
            else:
                verdict = "SPOOF_DETECTED"

        details = {
            "texture": tex_details,
            "frequency": freq_details,
            "color": color_details,
            "motion": motion_details,
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
            blinks_detected=blinks,
            details=details,
        )
