#!/usr/bin/env python3
"""Backward-compatibility PAM executable redirecting to Argus."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"
if VENV_PYTHON.exists() and sys.prefix != str(BASE_DIR / "venv"):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from argus.pam_handler import pam_main

if __name__ == "__main__":
    sys.exit(pam_main())
