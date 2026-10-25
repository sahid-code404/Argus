import sys
import argus
import argus.cli
import argus.config
import argus.crypto
import argus.engine
import argus.gui_enroll
import argus.gui_test
import argus.hardware
import argus.liveness
import argus.pam_handler
import argus.pam_integrator
import argus.simd_accel
import argus.simultaneous_biometrics
import argus.tpm
import argus.ui

sys.modules["faceunlock"] = argus
for _sub in [
    "cli", "config", "crypto", "engine", "gui_enroll", "gui_test",
    "hardware", "liveness", "pam_handler", "pam_integrator",
    "simd_accel", "simultaneous_biometrics", "tpm", "ui"
]:
    sys.modules[f"faceunlock.{_sub}"] = getattr(argus, _sub)
