"""Linux Face Unlock - Universal, High-Security Biometric Authentication for Linux.

Supports Depth, Infrared (IR) with Windows Hello emitter controls,
and seamless fallback to standard RGB webcams with multi-modal anti-spoofing liveness detection.
"""

import os

# Quiet internal OpenCV DNN graph engine warnings on OpenCV >= 4.11
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

__version__ = "1.0.0"
__author__ = "Linux Face Unlock Project"
