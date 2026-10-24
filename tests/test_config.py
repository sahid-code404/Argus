"""Unit tests for configuration loading, updating, and saving."""

from faceunlock.config import FaceUnlockConfig


def test_default_config():
    cfg = FaceUnlockConfig()
    assert cfg.camera_device == "auto"
    assert cfg.fallback_to_webcam is True
    assert cfg.cosine_threshold == 0.363
    assert cfg.min_liveness_score == 0.55
    assert cfg.require_liveness is True
    assert cfg.max_timeout == 3.5


def test_config_save_and_load(tmp_path):
    cfg_file = tmp_path / "test_config.ini"
    cfg = FaceUnlockConfig(
        camera_device="/dev/video2",
        cosine_threshold=0.45,
        max_timeout=5.0,
        pam_enabled=True,
    )
    cfg.save(str(cfg_file))
    assert cfg_file.exists()

    loaded = FaceUnlockConfig.load(str(cfg_file))
    assert loaded.camera_device == "/dev/video2"
    assert loaded.cosine_threshold == 0.45
    assert loaded.max_timeout == 5.0
    assert loaded.pam_enabled is True
