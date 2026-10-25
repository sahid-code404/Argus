"""Hardware sensor probing and discovery module for Linux Face Unlock.

Universally detects and classifies:
1. Depth Sensors (Intel RealSense, Orbbec Astra, ToF, V4L2 Z16) - Priority 1
2. Infrared (IR) Cameras (Windows Hello IR cameras, V4L2 GREY/Y8/Y16) - Priority 2
3. Standard RGB Webcams (V4L2 YUYV, MJPG, NV12, RGB24) - Priority 3 (Universal Fallback)
"""

from __future__ import annotations

import enum
import fcntl
import glob
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class CameraType(enum.Enum):
    DEPTH = "depth"
    INFRARED = "infrared"
    RGB_WEBCAM = "rgb_webcam"


# V4L2 ioctl constants (Linux videodev2.h)
# VIDIOC_QUERYCAP: _IOR('V', 0, struct v4l2_capability) = 0x80685600
VIDIOC_QUERYCAP = 0x80685600
# VIDIOC_ENUM_FMT: _IOWR('V', 2, struct v4l2_fmtdesc) = 0xc0405602
VIDIOC_ENUM_FMT = 0xC0405602
# V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
# V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
# V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_CAP_DEVICE_CAPS = 0x80000000

# Pixel formats associated with IR and Depth cameras
IR_PIXEL_FORMATS = frozenset({
    "GREY", "Y8  ", "Y8", "Y16 ", "Y16", "Y10 ", "Y10", "Y12 ", "Y12",
    "INVI", "INVR", "INVO", "RELI", "L8  ", "L16 "
})
DEPTH_PIXEL_FORMATS = frozenset({
    "Z16 ", "Z16", "INVZ", "CNF4", "INZI"
})

# Name keywords indicating IR or Depth capability
IR_NAME_KEYWORDS = ("ir", "infrared", "hello", "windows hello", "realsense ir")
DEPTH_NAME_KEYWORDS = ("depth", "realsense depth", "astra depth", "tof", "time of flight")

# Known USB Vendor/Product IDs for IR cameras (hex strings lowercase)
KNOWN_IR_USB_IDS = {
    # Intel RealSense
    ("8086", "0b07"), ("8086", "0b3a"), ("8086", "0b5c"), ("8086", "0a66"),
    # Lenovo ThinkPad IR cameras (Chicony, Sunplus, Realtek, Bison)
    ("04f2", "b614"), ("04f2", "b615"), ("04f2", "b64f"), ("04f2", "b6d9"),
    ("13d3", "56a2"), ("13d3", "56e5"), ("13d3", "56ff"),
    ("0bda", "58f4"), ("0bda", "58f5"), ("0bda", "58b0"),
    # Dell / HP / Microsoft Surface IR cameras
    ("045e", "090c"), ("045e", "0934"), ("045e", "0946"),
    ("1bcf", "2cb0"), ("1bcf", "2b95"), ("0c45", "6723")
}


@dataclass
class CameraDevice:
    path: str
    name: str
    camera_type: CameraType
    formats: List[str] = field(default_factory=list)
    priority: int = 3
    is_ir: bool = False
    is_depth: bool = False
    vendor_id: str = ""
    model_id: str = ""
    bus_info: str = ""
    is_capture: bool = True

    @property
    def type_label(self) -> str:
        if self.camera_type == CameraType.DEPTH:
            return "Depth 3D Sensor"
        if self.camera_type == CameraType.INFRARED:
            return "Infrared (IR) Camera"
        return "Standard RGB Webcam"

    def __str__(self) -> str:
        fmt_str = ", ".join(self.formats[:4]) if self.formats else "Unknown"
        return (f"[{self.type_label}] {self.path} - '{self.name}' "
                f"(formats: {fmt_str}, priority={self.priority})")


def query_v4l2_device_details(dev_path: str) -> Tuple[bool, str, str, List[str]]:
    """Query V4L2 device for capture capability, card name, bus_info, and supported pixel formats.

    Returns (is_capture, card_name, bus_info, formats).
    """
    if not os.path.exists(dev_path):
        return False, "", "", []

    is_capture = False
    card_name = ""
    bus_info = ""
    formats: List[str] = []

    try:
        fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        try:
            fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False, "", "", []

    try:
        # struct v4l2_capability (104 bytes on x86_64)
        # driver[16], card[32], bus_info[32], version[u32], capabilities[u32], device_caps[u32], reserved[3*u32]
        cap_buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap_buf)
        driver, card, bus, ver, caps, dev_caps = struct.unpack("16s32s32sIII", cap_buf[:92])

        card_name = card.split(b"\x00")[0].decode("utf-8", "replace").strip()
        bus_info = bus.split(b"\x00")[0].decode("utf-8", "replace").strip()

        # Check capture capability
        effective_caps = dev_caps if (caps & V4L2_CAP_DEVICE_CAPS) else caps
        is_capture = bool(effective_caps & V4L2_CAP_VIDEO_CAPTURE)

        if not is_capture:
            return False, card_name, bus_info, []

        # Enumerate formats: struct v4l2_fmtdesc
        # index(u32), type(u32), flags(u32), description[32], pixelformat(u32), mbtype(u32), reserved[4*u32]
        fmt_desc_size = 64
        for idx in range(64):
            fmt_buf = bytearray(fmt_desc_size)
            # pack index and type
            struct.pack_into("II", fmt_buf, 0, idx, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, fmt_buf)
                pixfmt = struct.unpack_from("4s", fmt_buf, 44)[0].decode("ascii", "replace")
                formats.append(pixfmt)
            except OSError:
                break
    except Exception:
        pass
    finally:
        os.close(fd)

    return is_capture, card_name, bus_info, formats


def query_udev_properties(dev_node_name: str) -> dict[str, str]:
    """Query udev properties for a video device (e.g. video0)."""
    props = {}
    syspath = f"/sys/class/video4linux/{dev_node_name}"
    if not os.path.exists(syspath):
        return props

    # Read sysfs name directly
    name_file = os.path.join(syspath, "name")
    if os.path.exists(name_file):
        try:
            with open(name_file, "r", encoding="utf-8", errors="replace") as f:
                props["NAME"] = f.read().strip()
        except OSError:
            pass

    # Read udev properties via udevadm
    try:
        res = subprocess.run(
            ["udevadm", "info", "-q", "property", "-p", syspath],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    props[k.strip()] = v.strip()
    except Exception:
        pass

    return props


def classify_camera(
    name: str,
    formats: List[str],
    vendor_id: str = "",
    model_id: str = "",
) -> Tuple[CameraType, int]:
    """Classify camera into DEPTH, INFRARED, or RGB_WEBCAM with assigned priority.

    Priority:
    1 - Depth (highest fidelity for 3D anti-spoofing)
    2 - Infrared (high security, dark lighting immunity, Windows Hello grade)
    3 - RGB Webcam (universal fallback with software anti-spoofing)
    """
    name_lower = name.lower()
    vid_pid = (vendor_id.lower(), model_id.lower())

    # Check Depth
    has_depth_fmt = any(f in DEPTH_PIXEL_FORMATS for f in formats)
    has_depth_name = any(kw in name_lower for kw in DEPTH_NAME_KEYWORDS)
    if has_depth_fmt or has_depth_name:
        return CameraType.DEPTH, 1

    # Check Infrared
    has_ir_fmt = any(f in IR_PIXEL_FORMATS for f in formats)
    has_ir_name = any(kw in name_lower for kw in IR_NAME_KEYWORDS)
    is_known_ir_device = vid_pid in KNOWN_IR_USB_IDS

    if has_ir_fmt or has_ir_name or is_known_ir_device:
        return CameraType.INFRARED, 2

    # Standard RGB Webcam
    return CameraType.RGB_WEBCAM, 3


def probe_cameras(include_all: bool = False) -> List[CameraDevice]:
    """Probe all video devices on the system and return sorted list by priority."""
    video_nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.search(r"\d+", p).group()) if re.search(r"\d+", p) else 999,
    )

    devices: List[CameraDevice] = []

    for dev_path in video_nodes:
        node_name = os.path.basename(dev_path)
        udev_props = query_udev_properties(node_name)
        is_capture, card_name, bus_info, formats = query_v4l2_device_details(dev_path)

        display_name = card_name or udev_props.get("NAME") or udev_props.get("ID_MODEL") or node_name
        vendor_id = udev_props.get("ID_VENDOR_ID", "")
        model_id = udev_props.get("ID_MODEL_ID", "")

        # Skip devices without capture capabilities unless requested
        if not is_capture and not include_all:
            continue

        cam_type, priority = classify_camera(display_name, formats, vendor_id, model_id)

        dev = CameraDevice(
            path=dev_path,
            name=display_name,
            camera_type=cam_type,
            formats=formats,
            priority=priority,
            is_ir=(cam_type == CameraType.INFRARED),
            is_depth=(cam_type == CameraType.DEPTH),
            vendor_id=vendor_id,
            model_id=model_id,
            bus_info=bus_info or udev_props.get("ID_PATH", ""),
            is_capture=is_capture,
        )
        devices.append(dev)

    # Sort primarily by priority (1 -> 2 -> 3), then by node path (/dev/video0 before /dev/video1)
    devices.sort(key=lambda d: (d.priority, d.path))
    return devices


def select_camera(preferred_path: Optional[str] = None) -> Tuple[CameraDevice, str]:
    """Select the best available camera with explicit fallback rationale.

    Returns (selected_device, rationale_message).
    """
    cameras = probe_cameras(include_all=False)

    if not cameras:
        # Fallback to /dev/video0 if exists even if unqueried
        if os.path.exists("/dev/video0"):
            dev = CameraDevice(
                path="/dev/video0",
                name="Fallback Video Device",
                camera_type=CameraType.RGB_WEBCAM,
                priority=3,
            )
            return dev, "No standard V4L2 capture devices enumerated; defaulting to /dev/video0."
        raise RuntimeError("No usable video camera devices found in /dev/video*.")

    # If preferred path is specified and valid
    if preferred_path:
        for cam in cameras:
            if cam.path == preferred_path:
                return cam, f"Selected configured camera device: {cam.path} ({cam.name})"
        # If preferred not found, log warning and continue to best selection
        fallback_msg_prefix = f"Preferred camera '{preferred_path}' not found. "
    else:
        fallback_msg_prefix = ""

    best_cam = cameras[0]

    if best_cam.camera_type == CameraType.DEPTH:
        reason = f"{fallback_msg_prefix}Selected high-fidelity Depth Sensor: {best_cam.path} ({best_cam.name})"
    elif best_cam.camera_type == CameraType.INFRARED:
        reason = f"{fallback_msg_prefix}Selected Infrared (IR) Camera: {best_cam.path} ({best_cam.name})"
    else:
        reason = (
            f"{fallback_msg_prefix}No dedicated IR or Depth sensor detected. "
            f"Seamlessly falling back to universal RGB webcam: {best_cam.path} ({best_cam.name})"
        )

    return best_cam, reason
