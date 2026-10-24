from pathlib import Path
import os
import pytest

from faceunlock.tpm import TPMManager


def test_tpm_manager_is_supported():
    supported = TPMManager.is_supported()
    assert isinstance(supported, bool)


def test_tpm_seal_unseal_flow(tmp_path):
    if not TPMManager.is_supported():
        pytest.skip("TPM 2.0 hardware or tpm2-tools not available on this host")

    secret = b"test_biometric_master_secret_32b"
    blob_dir = tmp_path / "tpm_blobs"

    # Seal secret
    success = TPMManager.seal_secret(secret, blob_dir)
    assert success is True
    assert (blob_dir / "tpm_pub.bin").exists()
    assert (blob_dir / "tpm_priv.bin").exists()

    # Unseal secret
    unsealed = TPMManager.unseal_secret(blob_dir)
    assert unsealed == secret


def test_tpm_unsupported_graceful_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(TPMManager, "is_supported", lambda: False)
    assert not TPMManager.is_supported()

    secret = b"dummy_secret_32_bytes_long_12345"
    assert TPMManager.seal_secret(secret, tmp_path) is False
    assert TPMManager.unseal_secret(tmp_path) is None
