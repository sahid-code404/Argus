"""PAM Integration Manager for Linux Face Unlock.

Supports automated, safe PAM injection and removal for:
- Sudo (sudo elevation)
- GDM (GNOME Display Manager - gdm-password)
- Polkit (GUI authentication dialogs - polkit-1)
- Console Login (login)
- SDDM (KDE Plasma Display Manager - sddm)
- LightDM (XFCE / Cinnamon / MATE - lightdm)
- Screen Lockers (swaylock, hyprlock)
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional


KNOWN_SERVICES = {
    "sudo": "Sudo Privilege Elevation",
    "gdm-password": "GDM (GNOME Display Manager / Lock Screen)",
    "polkit-1": "Polkit (GUI Administration Dialogs)",
    "login": "Console / Virtual Terminal Login",
    "sddm": "SDDM (KDE Plasma Display Manager)",
    "lightdm": "LightDM (XFCE / Cinnamon / MATE)",
    "swaylock": "Swaylock Screen Locker",
    "hyprlock": "Hyprlock Screen Locker",
}

PAM_DIR_ETC = Path("/etc/pam.d")
PAM_DIR_LIB = Path("/usr/lib/pam.d")


class PamIntegrator:
    """Manages system PAM modules and automated service integration."""

    def __init__(self, install_prefix: str = "/opt/faceunlock"):
        self.install_prefix = Path(install_prefix)
        self.pam_script = self.install_prefix / "pam" / "pam_faceunlock.py"
        self.pam_rule = f"auth       sufficient   pam_exec.so quiet {self.pam_script}\n"

    def get_service_path(self, service: str) -> Optional[Path]:
        """Find the PAM configuration file in /etc/pam.d or /usr/lib/pam.d."""
        etc_path = PAM_DIR_ETC / service
        if etc_path.exists():
            return etc_path
        lib_path = PAM_DIR_LIB / service
        if lib_path.exists():
            return lib_path
        return None

    def is_service_available(self, service: str) -> bool:
        """Check if the service exists on this Linux installation."""
        return self.get_service_path(service) is not None

    def is_service_enabled(self, service: str) -> bool:
        """Check if Face Unlock is enabled in /etc/pam.d/<service>."""
        etc_path = PAM_DIR_ETC / service
        if not etc_path.exists():
            return False
        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "pam_faceunlock.py" in content
        except Exception:
            return False

    def enable_service(self, service: str) -> bool:
        """Enable Face Unlock for target PAM service with automatic backup."""
        if not self.is_service_available(service):
            return False

        etc_path = PAM_DIR_ETC / service
        # If not present in /etc/pam.d, copy vendor template from /usr/lib/pam.d
        if not etc_path.exists():
            lib_path = PAM_DIR_LIB / service
            if lib_path.exists():
                try:
                    shutil.copy(lib_path, etc_path)
                except Exception:
                    return False
            else:
                return False

        # Create backup if not already present
        bak_path = etc_path.with_suffix(".faceunlock.bak")
        if not bak_path.exists():
            try:
                shutil.copy(etc_path, bak_path)
            except Exception:
                pass

        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove existing faceunlock lines
            clean_lines = [l for l in lines if "pam_faceunlock.py" not in l]

            # Find the best insertion point in auth section
            insert_idx = 0
            for idx, line in enumerate(clean_lines):
                # Skip comments and selinux lines
                stripped = line.strip()
                if stripped.startswith("#") or "pam_selinux" in stripped or not stripped:
                    insert_idx = idx + 1
                elif stripped.startswith("auth"):
                    insert_idx = idx
                    break

            clean_lines.insert(insert_idx, self.pam_rule)

            with open(etc_path, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

            return True
        except Exception:
            return False

    def disable_service(self, service: str) -> bool:
        """Remove Face Unlock from target PAM service."""
        etc_path = PAM_DIR_ETC / service
        if not etc_path.exists():
            return False

        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            clean_lines = [l for l in lines if "pam_faceunlock.py" not in l]
            with open(etc_path, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

            return True
        except Exception:
            return False

    def get_status_overview(self) -> Dict[str, Dict[str, object]]:
        """Return full diagnostic status of all known PAM services."""
        status = {}
        for srv, desc in KNOWN_SERVICES.items():
            avail = self.is_service_available(srv)
            enabled = self.is_service_enabled(srv)
            status[srv] = {
                "description": desc,
                "available": avail,
                "enabled": enabled,
            }
        return status


def main() -> int:
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 -m faceunlock.pam_integrator <enable|disable|status> [service]")
        return 1

    action = sys.argv[1].lower()
    integrator = PamIntegrator()

    if action == "status":
        overview = integrator.get_status_overview()
        for srv, info in overview.items():
            st = "ENABLED" if info["enabled"] else "DISABLED"
            av = "available" if info["available"] else "not present"
            print(f"{srv}: {st} ({av})")
        return 0

    if len(sys.argv) < 3:
        print("Error: service name required (e.g., sudo, gdm-password, polkit-1)")
        return 1

    service = sys.argv[2]
    if action in ("enable", "install"):
        success = integrator.enable_service(service)
        if success:
            print(f"Successfully enabled Face Unlock for {service}")
            return 0
        else:
            print(f"Failed to enable Face Unlock for {service}")
            return 1
    elif action in ("disable", "uninstall", "remove"):
        success = integrator.disable_service(service)
        if success:
            print(f"Successfully disabled Face Unlock for {service}")
            return 0
        else:
            print(f"Failed to disable Face Unlock for {service}")
            return 1
    else:
        print(f"Unknown action: {action}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
