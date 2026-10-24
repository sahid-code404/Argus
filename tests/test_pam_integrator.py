from pathlib import Path
import tempfile
import pytest

from faceunlock.pam_integrator import PamIntegrator, KNOWN_SERVICES


def test_pam_integrator_known_services():
    integrator = PamIntegrator()
    overview = integrator.get_status_overview()
    for srv in ("sudo", "gdm-password", "polkit-1", "login"):
        assert srv in overview
        assert "description" in overview[srv]
        assert "available" in overview[srv]
        assert "enabled" in overview[srv]


def test_pam_integrator_enable_disable(monkeypatch, tmp_path):
    etc_dir = tmp_path / "etc_pam"
    lib_dir = tmp_path / "lib_pam"
    etc_dir.mkdir()
    lib_dir.mkdir()

    # Create dummy gdm-password vendor file in lib_pam
    dummy_vendor = lib_dir / "gdm-password"
    dummy_vendor.write_text(
        "#%PAM-1.0\nauth substack password-auth\naccount include password-auth\n"
    )

    integrator = PamIntegrator(install_prefix="/opt/faceunlock")
    monkeypatch.setattr("faceunlock.pam_integrator.PAM_DIR_ETC", etc_dir)
    monkeypatch.setattr("faceunlock.pam_integrator.PAM_DIR_LIB", lib_dir)

    assert integrator.is_service_available("gdm-password")
    assert not integrator.is_service_enabled("gdm-password")

    # Enable service
    success = integrator.enable_service("gdm-password")
    assert success
    assert (etc_dir / "gdm-password").exists()
    assert (etc_dir / "gdm-password.faceunlock.bak").exists()
    assert integrator.is_service_enabled("gdm-password")

    content = (etc_dir / "gdm-password").read_text()
    assert "pam_faceunlock.py" in content

    # Disable service
    disabled = integrator.disable_service("gdm-password")
    assert disabled
    assert not integrator.is_service_enabled("gdm-password")
    clean_content = (etc_dir / "gdm-password").read_text()
    assert "pam_faceunlock.py" not in clean_content
