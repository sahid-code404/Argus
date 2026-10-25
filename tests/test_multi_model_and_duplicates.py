"""Comprehensive tests for Argus multi-model biometric support, duplicate face rejection,
auto-indexed model naming, and CLI list/remove commands.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from argus.config import FaceUnlockConfig
from argus.crypto import BiometricVault, project_embedding
from argus.engine import FaceEngine
import argus.cli as cli


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="argus_test_multi_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def create_normalized_face_vector(seed: int, dim: int = 128) -> np.ndarray:
    """Generate deterministic unit-normalized 128D embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(1, dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


def test_auto_indexed_naming_logic(temp_dir):
    """Verify auto-indexed default naming: Face1 for 0 models, Face2 for 1 model, etc."""
    engine = FaceEngine(data_dir=temp_dir)
    user = "testuser"

    # 1. No models enrolled yet
    models = engine.list_user_models(user)
    assert len(models) == 0
    next_id = max([m.get("id", 0) for m in models] or [0]) + 1
    assert next_id == 1
    default_name = f"Face{next_id}"
    assert default_name == "Face1"

    # Enroll first face model (Face1)
    face1_vecs = [create_normalized_face_vector(seed=42) for _ in range(5)]
    ok = engine.enroll_user(user, face1_vecs, model_name=default_name)
    assert ok is True

    # 2. One model enrolled -> next must be #2 Face2
    models = engine.list_user_models(user)
    assert len(models) == 1
    assert models[0]["id"] == 1
    assert models[0]["name"] == "Face1"

    next_id = max([m.get("id", 0) for m in models] or [0]) + 1
    assert next_id == 2
    default_name = f"Face{next_id}"
    assert default_name == "Face2"

    # Enroll second face model with custom name (e.g. 'Mom')
    face2_vecs = [create_normalized_face_vector(seed=999) for _ in range(5)]
    ok = engine.enroll_user(user, face2_vecs, model_name="Mom")
    assert ok is True

    # 3. Two models enrolled -> next must be #3 Face3
    models = engine.list_user_models(user)
    assert len(models) == 2
    assert models[1]["id"] == 2
    assert models[1]["name"] == "Mom"

    next_id = max([m.get("id", 0) for m in models] or [0]) + 1
    assert next_id == 3
    default_name = f"Face{next_id}"
    assert default_name == "Face3"


def test_duplicate_face_prevention_same_user(temp_dir):
    """Verify exact or very similar same face cannot be re-enrolled under a new model name."""
    engine = FaceEngine(data_dir=temp_dir, cosine_threshold=0.33)
    user = "alice"

    # Enroll Alice's primary face
    primary_vec = create_normalized_face_vector(seed=123)
    alice_samples = [primary_vec + np.random.randn(1, 128) * 0.01 for _ in range(5)]
    # Normalize
    alice_samples = [v / np.linalg.norm(v) for v in alice_samples]

    saved = engine.enroll_user(user, alice_samples, model_name="Face1")
    assert saved is True

    # Attempt to enroll Alice's same face again under 'WorkFace'
    # Same person with slight lighting variation (cosine similarity ~0.95)
    same_face_samples = [primary_vec + np.random.randn(1, 128) * 0.02 for _ in range(5)]
    same_face_samples = [v / np.linalg.norm(v) for v in same_face_samples]

    is_dup, dup_model, sim = engine.check_duplicate_face(same_face_samples, user=user)
    assert is_dup is True
    assert "Face1" in dup_model
    assert sim > 0.80

    # enroll_user should reject duplicate enrollment
    rejected = engine.enroll_user(user, same_face_samples, model_name="WorkFace", check_duplicate=True)
    assert rejected is False

    # Check that only 1 model exists
    models = engine.list_user_models(user)
    assert len(models) == 1
    assert models[0]["name"] == "Face1"


def test_duplicate_face_prevention_cross_user(temp_dir):
    """Verify same person cannot enroll under a different user account."""
    engine = FaceEngine(data_dir=temp_dir, cosine_threshold=0.33)

    # Bob enrolls
    bob_vec = create_normalized_face_vector(seed=777)
    bob_samples = [bob_vec for _ in range(5)]
    assert engine.enroll_user("bob", bob_samples, model_name="Face1") is True

    # Eve tries to enroll the exact same face under user 'eve'
    is_dup, dup_model, sim = engine.check_duplicate_face(bob_samples, user="eve")
    assert is_dup is True
    assert "bob" in dup_model

    # enroll_user should reject
    rejected = engine.enroll_user("eve", bob_samples, model_name="Face1", check_duplicate=True)
    assert rejected is False


def test_multi_model_authentication(temp_dir):
    """Verify multiple distinct faces (e.g. user + family/friend) can both unlock the user account."""
    engine = FaceEngine(data_dir=temp_dir, cosine_threshold=0.33)
    user = "shaque"

    # Enrolled face 1: User
    user_vec = create_normalized_face_vector(seed=101)
    user_samples = [user_vec for _ in range(5)]
    assert engine.enroll_user(user, user_samples, model_name="Face1") is True

    # Enrolled face 2: Friend (orthogonal/different face)
    friend_vec = create_normalized_face_vector(seed=202)
    friend_samples = [friend_vec for _ in range(5)]
    # Duplicate check should pass because it's a completely different person
    is_dup, _, sim = engine.check_duplicate_face(friend_samples, user=user)
    assert is_dup is False
    assert engine.enroll_user(user, friend_samples, model_name="Friend", check_duplicate=True) is True

    # Verify both faces match against user's account
    enrolled_vectors = engine.load_user_embeddings(user)
    assert len(enrolled_vectors) == 10  # 5 samples each

    # Test user verification
    best_score_user, _, _ = engine.find_best_match(user_vec, enrolled_vectors, user=user)
    assert best_score_user >= 0.95

    # Test friend verification
    best_score_friend, _, _ = engine.find_best_match(friend_vec, enrolled_vectors, user=user)
    assert best_score_friend >= 0.95

    # Test unknown person verification (should fail)
    stranger_vec = create_normalized_face_vector(seed=9999)
    best_score_stranger, _, _ = engine.find_best_match(stranger_vec, enrolled_vectors, user=user)
    assert best_score_stranger < 0.33


def test_cmd_list_and_remove(temp_dir, capsys):
    """Verify 'argus list face' and 'argus remove facemodel <name>' CLI behaviors."""
    engine = FaceEngine(data_dir=temp_dir)
    user = "shaque"

    # Enroll Model 1: Face1
    v1 = [create_normalized_face_vector(seed=10) for _ in range(5)]
    assert engine.enroll_user(user, v1, model_name="Face1") is True

    # Enroll Model 2: Mom
    v2 = [create_normalized_face_vector(seed=20) for _ in range(5)]
    assert engine.enroll_user(user, v2, model_name="Mom") is True

    cfg = FaceUnlockConfig()
    cfg.data_dir = str(temp_dir)

    # 1. Test 'argus list face'
    args = MagicMock()
    args.target = ["face"]
    ret = cli.cmd_list(args, cfg)
    assert ret == 0
    out = capsys.readouterr().out
    assert "Registered Biometric Face Models" in out
    assert "Face1" in out
    assert "Mom" in out
    assert "#1" in out
    assert "#2" in out

    # 2. Test 'argus remove facemodel Mom'
    args_rem = MagicMock()
    args_rem.target = ["facemodel", "Mom"]
    with patch("argus.cli.enforce_sudo_privilege"):
        ret_rem = cli.cmd_remove(args_rem, cfg)
    assert ret_rem == 0
    out_rem = capsys.readouterr().out
    assert "Successfully removed face model 'Mom'" in out_rem

    # Model 1 should remain
    models = engine.list_user_models(user)
    assert len(models) == 1
    assert models[0]["name"] == "Face1"

    # 3. Test removing by ID: 'argus remove facemodel 1'
    args_rem_id = MagicMock()
    args_rem_id.target = ["facemodel", "1"]
    with patch("argus.cli.enforce_sudo_privilege"):
        ret_rem_id = cli.cmd_remove(args_rem_id, cfg)
    assert ret_rem_id == 0

    # Vault should now be completely cleared
    models_empty = engine.list_user_models(user)
    assert len(models_empty) == 0
