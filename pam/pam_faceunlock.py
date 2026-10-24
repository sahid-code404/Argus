#!/usr/bin/env python3
"""Standalone executable script invoked by Linux pam_exec.so.

Usage in PAM config:
auth sufficient pam_exec.so stdout /path/to/pam_faceunlock.py
"""

import os
import sys
from pathlib import Path

# Re-exec with isolated venv python if present to ensure matching C ABI
BASE_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"
if VENV_PYTHON.exists() and os.path.realpath(sys.executable) != os.path.realpath(str(VENV_PYTHON)):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from faceunlock.pam_handler import pam_main

if __name__ == "__main__":
    sys.exit(pam_main())
