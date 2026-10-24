"""Unit tests for the FaceEngine (YuNet face detector + SFace recognizer)."""

import os
from pathlib import Path
import numpy as np
import pytest
from faceunlock.engine import FaceEngine


@pytest.fixture
def temp_engine(tmp_path):
    models_dir = Path(__file__).resolve().parent.parent / "models"
    data_dir = tmp_path / "faces"
    engine = FaceEngine(
        model_dir=str(models_dir),
        data_dir=str(data_dir),
        cosine_threshold=0.363,
    )
    return engine


def test_engine_initialization(temp_engine):
    """FaceEngine initializes models and creates data directory with proper permissions."""
    assert temp_engine._detector is not None
    assert temp_engine._recognizer is not None
    assert temp_engine.data_dir.exists()


def test_cosine_similarity_identity_and_orthogonality(temp_engine):
    """Cosine similarity of vector with itself is 1.0; orthogonal vectors is ~0.0."""
    v1 = np.zeros((1, 128), dtype=np.float32)
    v1[0, 0] = 1.0

    v2 = np.zeros((1, 128), dtype=np.float32)
    v2[0, 1] = 1.0

    sim_self = temp_engine.compute_similarity(v1, v1)
    assert pytest.approx(sim_self, abs=1e-4) == 1.0

    sim_ortho = temp_engine.compute_similarity(v1, v2)
    assert pytest.approx(sim_ortho, abs=1e-4) == 0.0


def test_user_enrollment_and_persistence(temp_engine):
    """Enrolling a user writes a 0600 file and can be reloaded and matched."""
    user = "alice"
    assert not temp_engine.is_enrolled(user)

    v1 = np.random.normal(0, 1, (1, 128)).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    v2 = v1 + np.random.normal(0, 0.05, (1, 128)).astype(np.float32)
    v2 = v2 / np.linalg.norm(v2)

    success = temp_engine.enroll_user(user, [v1, v2], camera_type="rgb_webcam")
    assert success
    assert temp_engine.is_enrolled(user)
    assert user in temp_engine.get_enrolled_users()

    # Verify file permissions (0600)
    face_file = temp_engine._user_face_path(user)
    mode = oct(face_file.stat().st_mode & 0o777)
    assert mode == "0o600"

    # Reload vectors
    loaded = temp_engine.load_user_embeddings(user)
    assert len(loaded) == 2
    assert loaded[0].shape == (1, 128)

    # Check similarity of reloaded vector
    sim = temp_engine.compute_similarity(v1, loaded[0])
    assert sim > 0.999

    # Remove user
    removed = temp_engine.remove_user(user)
    assert removed
    assert not temp_engine.is_enrolled(user)
