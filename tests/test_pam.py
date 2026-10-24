"""Unit tests for the PAM authentication handler and exit codes."""

import pytest
from faceunlock.config import FaceUnlockConfig
from faceunlock.pam_handler import PamAuthenticator, is_laptop_lid_closed, pam_main


def test_lid_detection():
    """Lid detection should return a boolean without crashing."""
    result = is_laptop_lid_closed()
    assert isinstance(result, bool)


def test_pam_authenticator_unenrolled_user(tmp_path):
    """An unenrolled user must immediately fail with a clean message and exit code 1."""
    cfg = FaceUnlockConfig(
        data_dir=str(tmp_path / "empty_faces"),
        pam_enabled=True,
    )
    auth = PamAuthenticator(config=cfg)
    success, msg = auth.authenticate_user("nonexistent_user")
    assert not success
    assert "no enrolled" in msg


def test_pam_authenticator_disabled(tmp_path):
    """When PAM is disabled in config, authentication must immediately fail."""
    cfg = FaceUnlockConfig(
        data_dir=str(tmp_path / "empty_faces"),
        pam_enabled=False,
    )
    auth = PamAuthenticator(config=cfg)
    success, msg = auth.authenticate_user("any_user")
    assert not success
    assert "disabled" in msg.lower()


def test_pam_main_exit_code_unenrolled(tmp_path, monkeypatch):
    """pam_main must exit with status 1 when authentication fails."""
    cfg = FaceUnlockConfig(
        data_dir=str(tmp_path / "empty_faces"),
        pam_enabled=True,
        show_feedback=False,
    )
    monkeypatch.setattr("faceunlock.config.FaceUnlockConfig.load", lambda: cfg)
    rc = pam_main(["nobody"])
    assert rc == 1
