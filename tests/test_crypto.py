"""Unit tests for AES-256-GCM Biometric Vault and Cancelable Hashing."""

import numpy as np
import pytest
from pathlib import Path

from faceunlock.crypto import (
    BiometricVault,
    get_hardware_identity,
    get_orthonormal_projection,
    project_embedding,
    secure_wipe_file,
)


def test_hardware_identity_stability():
    """Hardware identity must be deterministic and non-empty."""
    id1 = get_hardware_identity()
    id2 = get_hardware_identity()
    assert id1 == id2
    assert len(id1) == 32


def test_orthonormal_projection_properties():
    """Cancelable random projection Q must be orthonormal (Q^T Q = I) and preserve cosine similarity."""
    q = get_orthonormal_projection("testuser", dim=128)
    assert q.shape == (128, 128)

    # Check Q^T Q == I
    identity = q.T @ q
    eye = np.eye(128, dtype=np.float32)
    assert np.allclose(identity, eye, atol=1e-5)

    # Cosine similarity preservation
    np.random.seed(123)
    v1 = np.random.randn(1, 128).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    v2 = v1 + np.random.randn(1, 128).astype(np.float32) * 0.1
    v2 /= np.linalg.norm(v2)

    raw_sim = float((v1 @ v2.T)[0, 0])

    proj_v1 = project_embedding(v1, "testuser")
    proj_v2 = project_embedding(v2, "testuser")

    proj_sim = float((proj_v1 @ proj_v2.T)[0, 0])

    assert abs(raw_sim - proj_sim) < 1e-5


def test_vault_encrypt_and_decrypt(tmp_path):
    """Encrypted vault must decrypt cleanly and match original projected vectors."""
    vault_file = tmp_path / "testuser.vault"

    np.random.seed(999)
    raw_vectors = [np.random.randn(1, 128).astype(np.float32) for _ in range(5)]
    for v in raw_vectors:
        v /= np.linalg.norm(v)

    success = BiometricVault.save_vault("testuser", raw_vectors, vault_file)
    assert success is True
    assert vault_file.exists()

    # Permissions must be 0600
    mode = oct(vault_file.stat().st_mode & 0o777)
    assert mode == "0o600"

    decrypted, header = BiometricVault.load_vault("testuser", vault_file)
    assert len(decrypted) == 5
    assert header["user"] == "testuser"
    assert header["count"] == 5

    # Check vector values match projected input
    expected_first = project_embedding(raw_vectors[0], "testuser")
    assert np.allclose(decrypted[0], expected_first, atol=1e-5)


def test_vault_tamper_proofing(tmp_path):
    """Tampering with ciphertext must cause decryption to fail closed."""
    vault_file = tmp_path / "tamper.vault"
    vectors = [np.random.randn(1, 128).astype(np.float32)]
    BiometricVault.save_vault("tamperuser", vectors, vault_file)

    # Tamper with the file
    content = vault_file.read_text()
    tampered = content.replace('"user": "tamperuser"', '"user": "attacker"')
    vault_file.write_text(tampered)

    # Attempt decryption
    decrypted, header = BiometricVault.load_vault("tamperuser", vault_file)
    assert len(decrypted) == 0
    assert "error" in header


def test_secure_wipe_file(tmp_path):
    """File wiping overwrites with zeros and deletes."""
    f = tmp_path / "secret.dat"
    f.write_text("very sensitive biometric template")
    assert f.exists()
    secure_wipe_file(f)
    assert not f.exists()
