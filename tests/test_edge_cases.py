"""Edge cases and fault tolerance tests for FaceUnlock."""

import json
from pathlib import Path
import numpy as np
import pytest
from faceunlock.config import FaceUnlockConfig
from faceunlock.engine import FaceEngine
from faceunlock.hardware import CameraDevice, CameraType, select_camera
from faceunlock.pam_handler import PamAuthenticator


def test_corrupted_face_json_handled_gracefully(tmp_path):
    """If an enrolled JSON file is corrupted or invalid, load_user_embeddings returns empty list."""
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    corrupt_file = faces_dir / "bob.json"
    corrupt_file.write_text("{ this is not valid json : 123 }")

    models_dir = Path(__file__).resolve().parent.parent / "models"
    engine = FaceEngine(model_dir=str(models_dir), data_dir=str(faces_dir))

    assert engine.is_enrolled("bob")
    loaded = engine.load_user_embeddings("bob")
    assert loaded == []


def test_grayscale_frame_input_detection(tmp_path):
    """Engine must gracefully handle 2D single-channel grayscale frames."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    engine = FaceEngine(model_dir=str(models_dir), data_dir=str(tmp_path))

    gray_frame = np.zeros((320, 320), dtype=np.uint8)
    faces = engine.detect_faces(gray_frame)
    assert isinstance(faces, list)


def test_nonexistent_custom_config_fallback(monkeypatch):
    """Loading a non-existent config path safely falls back to defaults."""
    monkeypatch.setattr("os.path.exists", lambda p: False)
    cfg = FaceUnlockConfig.load("/nonexistent/path/config.ini")
    assert cfg.camera_device == "auto"
    assert cfg.cosine_threshold == 0.38


def test_pam_authenticator_camera_open_failure(tmp_path, monkeypatch):
    """If the camera fails to open (e.g. device locked by another process), returns False cleanly."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()

    cfg = FaceUnlockConfig(
        models_dir=str(models_dir),
        data_dir=str(faces_dir),
        camera_device="/dev/video999_fake",
        pam_enabled=True,
        show_feedback=False,
    )
    engine = FaceEngine(model_dir=str(models_dir), data_dir=str(faces_dir))
    # Enroll dummy
    dummy_feat = np.ones((1, 128), dtype=np.float32)
    engine.enroll_user("alice", [dummy_feat])

    # Mock cv2.VideoCapture to simulate a camera locked by another process
    class FakeLockedCapture:
        def isOpened(self):
            return False
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: FakeLockedCapture())

    auth = PamAuthenticator(config=cfg)
    success, msg = auth.authenticate_user("alice")
    assert not success
    assert "failed to open" in msg.lower()


def test_stranger_face_strictly_rejected_at_threshold(tmp_path):
    """An unenrolled person with similarity below calibrated threshold must never verify."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()

    engine = FaceEngine(
        model_dir=str(models_dir),
        data_dir=str(faces_dir),
        cosine_threshold=0.38,
    )

    # Enroll user 'alice'
    rng = np.random.RandomState(42)
    v1 = rng.randn(1, 128).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)
    engine.enroll_user("alice", [v1])

    # Stranger with similarity ~0.30 (borderline but below 0.38)
    stranger = rng.randn(1, 128).astype(np.float32)
    stranger = stranger / np.linalg.norm(stranger)

    enrolled = engine.load_user_embeddings("alice")
    sim, idx, _ = engine.find_best_match(stranger, enrolled, user="alice")

    assert sim < 0.38

