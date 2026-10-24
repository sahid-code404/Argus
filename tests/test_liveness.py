"""Unit tests for the multi-feature anti-spoofing and liveness detector."""

import cv2
import numpy as np
import pytest
from faceunlock.liveness import LivenessDetector, LivenessResult


def test_liveness_empty_detection():
    """An empty or invalid detection vector should return NO_FACE."""
    detector = LivenessDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.evaluate_frame(frame, np.array([]))
    assert not res.is_live
    assert res.verdict == "NO_FACE"


def test_texture_analysis_blurry_vs_natural():
    """Flat/blurry surfaces (resembling low-res paper prints) score lower than rich texture."""
    detector = LivenessDetector()

    # Create a flat blurred surface
    flat_roi = np.full((128, 128, 3), 120, dtype=np.uint8)
    flat_roi = cv2.GaussianBlur(flat_roi, (15, 15), 0)
    score_flat, details_flat = detector.analyze_texture(flat_roi)

    # Create a patterned surface with natural skin-like gradient variance
    np.random.seed(42)
    natural_roi = np.random.normal(128, 15, (128, 128, 3)).clip(0, 255).astype(np.uint8)
    natural_roi = cv2.GaussianBlur(natural_roi, (3, 3), 0)
    score_nat, details_nat = detector.analyze_texture(natural_roi)

    assert score_nat > score_flat


def test_temporal_motion_static_photo():
    """A completely static face over multiple frames should be flagged as STATIC_PHOTO."""
    detector = LivenessDetector()

    # Identical landmarks across 5 frames (zero movement, zero 3D parallax)
    static_landmarks = np.array([100.0, 100.0, 150.0, 100.0, 125.0, 130.0, 110.0, 160.0, 140.0, 160.0])
    history = [static_landmarks for _ in range(5)]

    motion_score, blinks, details = detector.analyze_temporal_motion(history)
    assert motion_score <= 0.35
    assert details["avg_disp"] < 0.2


def test_temporal_motion_live_micro_movements():
    """Organic micro-movements and 3D parallax changes should receive high motion scores."""
    detector = LivenessDetector()

    history = []
    base = np.array([100.0, 100.0, 150.0, 100.0, 125.0, 130.0, 110.0, 160.0, 140.0, 160.0])
    for i in range(5):
        # Add subtle natural 3D head pitch shift to nose
        lm = base.copy()
        lm[5] += float(i * 1.5)  # Nose vertical shift
        lm[0] += float(np.sin(i) * 0.8)
        history.append(lm)

    motion_score, blinks, details = detector.analyze_temporal_motion(history)
    assert motion_score >= 0.85
    assert details["avg_disp"] > 0.4


def test_frequency_screen_moire_detection():
    """Synthetic periodic screen moiré stripes should be detected by frequency spectrum."""
    detector = LivenessDetector()

    # Generate high-frequency stripe pattern simulating screen refresh moiré
    moire = np.zeros((128, 128), dtype=np.uint8)
    moire[::2, :] = 255
    score, details = detector.analyze_frequency_spectrum(moire)
    assert score <= 0.4
    assert details["hf_peak_ratio"] > 0.8
