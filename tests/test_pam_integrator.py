from pathlib import Path
import tempfile
import pytest

from faceunlock.pam_integrator import PamIntegrator, KNOWN_SERVICES


def test_pam_integrator_known_services():
    integrator = PamIntegrator()
    overview = integrator.get_status_overview()
    for srv in ("sudo", "polkit-1", "lockscreen", "sshd"):
        assert srv in overview
        assert "description" in overview[srv]
        assert "available" in overview[srv]
        assert "enabled" in overview[srv]


def test_pam_integrator_enable_disable(monkeypatch, tmp_path):
    etc_dir = tmp_path / "etc_pam"
    lib_dir = tmp_path / "lib_pam"
    etc_dir.mkdir()
    lib_dir.mkdir()

    # Create dummy swaylock vendor file in lib_pam
    dummy_vendor = lib_dir / "swaylock"
    dummy_vendor.write_text(
        "#%PAM-1.0\nauth substack password-auth\naccount include password-auth\n"
    )

    integrator = PamIntegrator(install_prefix="/opt/argus")
    monkeypatch.setattr("argus.pam_integrator.PAM_DIR_ETC", etc_dir)
    monkeypatch.setattr("argus.pam_integrator.PAM_DIR_LIB", lib_dir)
    monkeypatch.setattr("faceunlock.pam_integrator.PAM_DIR_ETC", etc_dir)
    monkeypatch.setattr("faceunlock.pam_integrator.PAM_DIR_LIB", lib_dir)

    assert integrator.is_service_available("lockscreen")
    assert not integrator.is_service_enabled("lockscreen")

    # Enable service
    success = integrator.enable_service("lockscreen")
    assert success
    assert (etc_dir / "swaylock").exists()
    assert integrator.is_service_enabled("lockscreen")

    content = (etc_dir / "swaylock").read_text()
    assert ("pam_argus.py" in content) or ("pam_faceunlock.py" in content)

    # Disable service
    disabled = integrator.disable_service("lockscreen")
    assert disabled
    assert not integrator.is_service_enabled("lockscreen")
    clean_content = (etc_dir / "swaylock").read_text()
    assert "pam_argus.py" not in clean_content
    assert "pam_faceunlock.py" not in clean_content

    # Verify session login managers are blocked
    assert integrator.enable_service("gdm-password") is False
    assert integrator.enable_service("login") is False
