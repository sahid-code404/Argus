"""Argus - Universal, High-Security Biometric Authentication for Linux.

Supports Depth, Infrared (IR) with Windows Hello emitter controls,
and seamless fallback to standard RGB webcams with multi-modal anti-spoofing liveness detection.
"""

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

__version__ = "1.0.0"
__author__ = "Argus Project"
