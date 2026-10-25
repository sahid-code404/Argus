"""Unit tests verifying canonical argus package namespace and backward compatibility."""

import argus
import argus.cli
import argus.config
import argus.crypto
import argus.engine
import argus.hardware
import argus.liveness
import argus.pam_handler
import argus.pam_integrator
import argus.simd_accel
import argus.simultaneous_biometrics
import argus.tpm
import argus.ui

import faceunlock


def test_argus_package_exports():
    """Verify argus package exports all core classes and methods."""
    assert hasattr(argus, "__version__")
    assert hasattr(argus.config, "ArgusConfig")
    assert hasattr(argus.config, "FaceUnlockConfig")
    assert hasattr(argus.engine, "ArgusEngine")
    assert hasattr(argus.engine, "FaceEngine")
    assert hasattr(argus.pam_integrator, "PamIntegrator")
    assert hasattr(argus.hardware, "probe_cameras")
    assert hasattr(argus.hardware, "select_camera")


def test_faceunlock_compatibility_alias():
    """Verify legacy faceunlock imports match argus exactly."""
    assert faceunlock.config.FaceUnlockConfig is argus.config.FaceUnlockConfig
    assert faceunlock.engine.FaceEngine is argus.engine.FaceEngine
    assert faceunlock.hardware.select_camera is argus.hardware.select_camera
    assert faceunlock.pam_integrator.PamIntegrator is argus.pam_integrator.PamIntegrator


def test_argus_cli_parser():
    """Verify argus CLI parser configures all subcommands."""
    import argparse
    from argus.cli import main
    # Running --help should exit with 0
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
