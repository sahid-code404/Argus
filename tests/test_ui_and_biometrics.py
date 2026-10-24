"""Tests for UI animations, CLI privilege gating, and simultaneous biometric modules."""

import io
import os
import sys
import pytest
from faceunlock.ui import BiometricSpinner, render_banner, render_step_progress
from faceunlock.simultaneous_biometrics import FingerprintVerifier, BiometricAuthResult
from faceunlock.cli import enforce_sudo_privilege


def test_ui_spinner_output():
    """Spinner should write clean frames without throwing exceptions."""
    stream = io.StringIO()
    # Mock isatty
    stream.isatty = lambda: True

    spinner = BiometricSpinner("Testing spinner", stream=stream)
    for _ in range(10):
        spinner.tick()
    spinner.success("Test Success")
    spinner.failure("Test Failure")

    output = stream.getvalue()
    assert "Testing spinner" in output
    assert "Test Success" in output
    assert "Test Failure" in output


def test_ui_banner_and_progress(capsys):
    """Banners and progress bars should render cleanly."""
    render_banner("TEST TITLE", "Subtext description")
    render_step_progress(2, 5, "Turn head left")
    captured = capsys.readouterr().out
    assert "TEST TITLE" in captured
    assert "Step 2/5:" in captured
    assert "Turn head left" in captured


def test_fingerprint_verifier_availability():
    """Fingerprint detection returns boolean."""
    avail = FingerprintVerifier.is_available_for_user("shaque")
    assert isinstance(avail, bool)


def test_sudo_enforcement_non_interactive(monkeypatch):
    """Non-interactive non-root execution should cleanly raise SystemExit with code 1."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        enforce_sudo_privilege("enroll")
    assert exc_info.value.code == 1
