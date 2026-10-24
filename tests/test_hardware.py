"""Unit and Integration tests for hardware sensor probing and camera classification."""

import pytest
from faceunlock.hardware import (
    CameraDevice,
    CameraType,
    classify_camera,
    probe_cameras,
    select_camera,
)


def test_classify_camera_depth():
    """Depth sensor formats or names should be classified as DEPTH with priority 1."""
    cam_type, priority = classify_camera("Intel RealSense Depth Camera", ["Z16 "], "8086", "0b07")
    assert cam_type == CameraType.DEPTH
    assert priority == 1

    cam_type, priority = classify_camera("Astra Embedded Depth Sensor", ["YUYV"], "2bc5", "0403")
    assert cam_type == CameraType.DEPTH
    assert priority == 1


def test_classify_camera_infrared():
    """IR formats, names, or known USB IDs should be classified as INFRARED with priority 2."""
    cam_type, priority = classify_camera("ThinkPad Integrated IR Camera", ["GREY"], "04f2", "b614")
    assert cam_type == CameraType.INFRARED
    assert priority == 2

    # Standard name but GREY format
    cam_type, priority = classify_camera("USB Camera", ["GREY", "Y8  "], "1bcf", "2cb0")
    assert cam_type == CameraType.INFRARED
    assert priority == 2

    # Known ThinkPad IR USB ID even with generic name
    cam_type, priority = classify_camera("Camera", ["YUYV"], "04f2", "b614")
    assert cam_type == CameraType.INFRARED
    assert priority == 2


def test_classify_camera_rgb_webcam():
    """Standard webcams should be classified as RGB_WEBCAM with priority 3."""
    cam_type, priority = classify_camera("XiaoMi USB 2.0 Webcam", ["MJPG", "YUYV"], "1bcf", "2cdd")
    assert cam_type == CameraType.RGB_WEBCAM
    assert priority == 3


def test_select_camera_fallback_logic(monkeypatch):
    """When no IR or Depth sensor is present, select_camera must fall back to RGB webcam."""
    mock_cameras = [
        CameraDevice(
            path="/dev/video0",
            name="XiaoMi USB 2.0 Webcam",
            camera_type=CameraType.RGB_WEBCAM,
            formats=["MJPG", "YUYV"],
            priority=3,
        ),
        CameraDevice(
            path="/dev/video1",
            name="XiaoMi USB 2.0 Webcam (Metadata)",
            camera_type=CameraType.RGB_WEBCAM,
            formats=[],
            priority=3,
        ),
    ]
    monkeypatch.setattr("faceunlock.hardware.probe_cameras", lambda include_all=False: mock_cameras)

    selected, reason = select_camera()
    assert selected.path == "/dev/video0"
    assert selected.camera_type == CameraType.RGB_WEBCAM
    assert "falling back" in reason.lower()
    assert "XiaoMi USB 2.0 Webcam" in reason


def test_select_camera_prefers_ir_over_rgb(monkeypatch):
    """When an IR camera is present, it must be selected over RGB webcams."""
    mock_cameras = [
        CameraDevice(
            path="/dev/video2",
            name="ThinkPad IR Camera",
            camera_type=CameraType.INFRARED,
            formats=["GREY"],
            priority=2,
            is_ir=True,
        ),
        CameraDevice(
            path="/dev/video0",
            name="Integrated RGB Webcam",
            camera_type=CameraType.RGB_WEBCAM,
            formats=["YUYV"],
            priority=3,
        ),
    ]
    # probe_cameras returns list sorted by priority
    sorted_mock = sorted(mock_cameras, key=lambda c: c.priority)
    monkeypatch.setattr("faceunlock.hardware.probe_cameras", lambda include_all=False: sorted_mock)

    selected, reason = select_camera()
    assert selected.path == "/dev/video2"
    assert selected.camera_type == CameraType.INFRARED
    assert "Infrared" in reason


def test_real_system_hardware_probe():
    """Verify camera probing runs on the host system without crashing."""
    cams = probe_cameras(include_all=True)
    assert isinstance(cams, list)
    # The laptop has at least /dev/video0
    capture_cams = [c for c in cams if c.is_capture]
    assert len(capture_cams) >= 1
    selected, reason = select_camera()
    assert selected is not None
    assert selected.path.startswith("/dev/video")
