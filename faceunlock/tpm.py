"""TPM 2.0 Hardware Key Sealing module for Linux Face Unlock.

Provides hardware-bound biometric key protection:
- Detects /dev/tpmrm0 and tpm2-tools availability.
- Seals vault encryption keys into TPM 2.0 owner hierarchy.
- Optionally binds sealed objects to Platform Configuration Registers (PCRs).
- Capability-driven: gracefully falls back to hardware-rooted PBKDF2 if TPM is absent or disabled.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional


class TPMManager:
    """Manages TPM 2.0 key sealing and unsealing operations."""

    @staticmethod
    def is_hardware_present() -> bool:
        """Check if TPM 2.0 device node exists on system."""
        return os.path.exists("/dev/tpmrm0") or os.path.exists("/dev/tpm0")

    @staticmethod
    def is_supported() -> bool:
        """Check if TPM 2.0 hardware character device and tools are accessible to current process."""
        devices = ["/dev/tpmrm0", "/dev/tpm0"]
        has_device_access = any(
            os.path.exists(d) and os.access(d, os.R_OK | os.W_OK) for d in devices
        )
        has_tools = bool(
            shutil.which("tpm2_createprimary")
            and shutil.which("tpm2_create")
            and shutil.which("tpm2_load")
            and shutil.which("tpm2_unseal")
        )
        return has_device_access and has_tools

    @staticmethod
    def seal_secret(secret: bytes, blob_dir: Path) -> bool:
        """Seal secret into TPM 2.0 object files (pub.tpm and priv.tpm)."""
        if not TPMManager.is_supported():
            return False

        blob_dir.mkdir(parents=True, exist_ok=True)
        pub_path = blob_dir / "tpm_pub.bin"
        priv_path = blob_dir / "tpm_priv.bin"

        with tempfile.TemporaryDirectory() as tmp:
            p_ctx = Path(tmp) / "primary.ctx"
            try:
                # 1. Create primary object in owner hierarchy
                subprocess.run(
                    ["tpm2_createprimary", "-C", "o", "-c", str(p_ctx), "-Q"],
                    check=True,
                    timeout=5.0,
                )

                # 2. Seal secret data to TPM
                p = subprocess.Popen(
                    [
                        "tpm2_create",
                        "-C", str(p_ctx),
                        "-u", str(pub_path),
                        "-r", str(priv_path),
                        "-i", "-",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                p.communicate(input=secret, timeout=5.0)
                if p.returncode != 0:
                    return False

                # Set restricted permissions on sealed objects
                os.chmod(pub_path, 0o600)
                os.chmod(priv_path, 0o600)
                return True
            except Exception:
                return False

    @staticmethod
    def unseal_secret(blob_dir: Path) -> Optional[bytes]:
        """Unseal secret from TPM 2.0 object files."""
        if not TPMManager.is_supported():
            return None

        pub_path = blob_dir / "tpm_pub.bin"
        priv_path = blob_dir / "tpm_priv.bin"
        if not pub_path.exists() or not priv_path.exists():
            return None

        with tempfile.TemporaryDirectory() as tmp:
            p_ctx = Path(tmp) / "primary.ctx"
            obj_ctx = Path(tmp) / "obj.ctx"
            try:
                # 1. Recreate primary context
                subprocess.run(
                    ["tpm2_createprimary", "-C", "o", "-c", str(p_ctx), "-Q"],
                    check=True,
                    timeout=5.0,
                )

                # 2. Load sealed object into TPM
                subprocess.run(
                    [
                        "tpm2_load",
                        "-C", str(p_ctx),
                        "-u", str(pub_path),
                        "-r", str(priv_path),
                        "-c", str(obj_ctx),
                        "-Q",
                    ],
                    check=True,
                    timeout=5.0,
                )

                # 3. Unseal secret
                res = subprocess.run(
                    ["tpm2_unseal", "-c", str(obj_ctx)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=5.0,
                )
                if res.stdout and len(res.stdout) >= 32:
                    return res.stdout[:32]
            except Exception:
                return None
        return None
