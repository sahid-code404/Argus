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


def test_moving_planar_photo_homography_rejection():
    """A moving flat 2D photo obeys pure 2D affine transform with zero depth residual."""
    detector = LivenessDetector()

    history = []
    base = np.array([100.0, 100.0, 150.0, 100.0, 125.0, 130.0, 110.0, 160.0, 140.0, 160.0])
    for i in range(5):
        # Apply a pure 2D rigid planar shift + rotation (flat surface moving in hand)
        theta = float(i) * 0.02
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts = base.reshape(-1, 2)
        pts_shifted = (pts @ R.T) + np.array([float(i * 3.0), float(i * 1.5)])
        history.append(pts_shifted.flatten())

    motion_score, blinks, details = detector.analyze_temporal_motion(history)
    # Must flag as planar moving 2D attack
    assert details["is_planar_moving_2d"] is True
    assert motion_score <= 0.25


def test_screen_bezel_detection():
    """Rectangular high-contrast borders around face (smartphone / tablet) should trigger bezel penalty."""
    detector = LivenessDetector()

    # Create image with phone-like rectangular borders
    frame = np.full((300, 300, 3), 40, dtype=np.uint8)
    # Draw a prominent rectangular smartphone bezel
    cv2.rectangle(frame, (60, 40), (240, 260), (220, 220, 220), 4)

    # Face box inside the phone screen
    face_box = np.array([90, 80, 100, 120])
    score, details = detector.detect_screen_bezel(frame, face_box)

    assert score <= 0.45
    assert details["vertical_lines"] >= 2 or details["horizontal_lines"] >= 2


def test_specular_glare_detection():
    """Sharp specular reflection hotspots on glass screens should be detected."""
    detector = LivenessDetector()

    # Glass surface with white glare spots (screen reflection)
    glare_roi = np.full((128, 128, 3), 120, dtype=np.uint8)
    glare_roi[40:55, 40:55] = 255  # Saturated white hotspot

    score, details = detector.detect_specular_glare(glare_roi)
    assert score < 0.95
    assert details["sat_ratio"] > 0.01


def test_insufficient_temporal_history_on_single_frame():
    """A single frame cannot authenticate in RGB fallback mode without temporal history or passes spoof detection."""
    detector = LivenessDetector()
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Valid YuNet face box
    face = np.array([200, 150, 120, 140, 230, 180, 290, 180, 260, 210, 240, 250, 280, 250, 0.9])

    res = detector.evaluate_frame(frame, face, is_ir_mode=False)
    assert not res.is_live
    assert res.verdict in ("INSUFFICIENT_TEMPORAL_HISTORY", "NEURAL_PAD_SPOOF")


def test_photo_attack_rejected_after_3_frames():
    """A static photo presented over 3+ frames must be strictly rejected."""
    detector = LivenessDetector()
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    face = np.array([200, 150, 120, 140, 230, 180, 290, 180, 260, 210, 240, 250, 280, 250, 0.9])

    # Feed 3 identical frames (simulating static printed photo held in front of webcam)
    res1 = detector.evaluate_frame(frame, face)
    res2 = detector.evaluate_frame(frame, face)
    res3 = detector.evaluate_frame(frame, face)

    assert not res3.is_live
    assert res3.verdict in ("STATIC_2D_PHOTO", "NEURAL_PAD_SPOOF")


def test_phone_bezel_rejection_veto():
    """Detecting smartphone / screen borders must strictly veto liveness."""
    detector = LivenessDetector()
    frame = np.full((300, 300, 3), 40, dtype=np.uint8)
    # Draw prominent rectangular smartphone bezel
    cv2.rectangle(frame, (60, 40), (240, 260), (220, 220, 220), 4)
    face_box = np.array([90, 80, 100, 120, 110, 100, 150, 100, 130, 120, 115, 140, 145, 140, 0.9])

    score, details = detector.detect_screen_bezel(frame, face_box[:4])
    assert score < 0.50


