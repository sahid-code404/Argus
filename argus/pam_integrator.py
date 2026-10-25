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
    "polkit-1": "Polkit (GUI Administration Dialogs)",
    "lockscreen": "Desktop Lock Screen (screensavers & screen lockers)",
    "sshd": "OpenSSH Daemon (SSH remote logins)",
}

LOCKSCREEN_SERVICES = [
    "gnome-screensaver",
    "swaylock",
    "hyprlock",
    "waylock",
    "i3lock",
    "cinnamon-screensaver",
    "mate-screensaver",
    "xfce4-screensaver",
    "light-locker",
    "vlock",
]

PAM_DIR_ETC = Path("/etc/pam.d")
PAM_DIR_LIB = Path("/usr/lib/pam.d")


class PamIntegrator:
    """Manages system PAM modules and automated service integration for Argus."""

    def __init__(self, install_prefix: Optional[str] = None):
        if install_prefix is not None:
            self.install_prefix = Path(install_prefix)
        elif Path("/opt/argus").exists():
            self.install_prefix = Path("/opt/argus")
        else:
            self.install_prefix = Path("/opt/faceunlock")

        pam_argus = self.install_prefix / "pam" / "pam_argus.py"
        pam_faceunlock = self.install_prefix / "pam" / "pam_faceunlock.py"
        self.pam_script = pam_argus if pam_argus.exists() or not pam_faceunlock.exists() else pam_faceunlock
        self.pam_rule = f"auth       sufficient   pam_exec.so quiet {self.pam_script}\n"

    def get_service_path(self, service: str) -> Optional[Path]:
        """Find the PAM configuration file in /etc/pam.d or /usr/lib/pam.d."""
        if service in ("lockscreen", "lockers", "screensaver"):
            for srv in LOCKSCREEN_SERVICES:
                p = self.get_service_path(srv)
                if p is not None:
                    return p
            return PAM_DIR_ETC / "gnome-screensaver"

        etc_path = PAM_DIR_ETC / service
        if etc_path.exists():
            return etc_path
        lib_path = PAM_DIR_LIB / service
        if lib_path.exists():
            return lib_path
        return None

    def is_service_available(self, service: str) -> bool:
        """Check if the service exists on this Linux installation."""
        if service in ("lockscreen", "lockers", "screensaver"):
            return True
        return self.get_service_path(service) is not None

    def is_service_enabled(self, service: str) -> bool:
        """Check if Argus is enabled in /etc/pam.d/<service>."""
        if service in ("lockscreen", "lockers", "screensaver"):
            for srv in LOCKSCREEN_SERVICES:
                if self._is_single_service_enabled(srv):
                    return True
            return False
        return self._is_single_service_enabled(service)

    def _is_single_service_enabled(self, service: str) -> bool:
        etc_path = PAM_DIR_ETC / service
        if not etc_path.exists():
            return False
        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "pam_argus.py" in content or "pam_faceunlock.py" in content
        except Exception:
            return False

    def enable_service(self, service: str) -> bool:
        """Enable Argus for target PAM service with automatic backup."""
        if any(k in service for k in ("gdm", "sddm", "lightdm", "login")):
            # Explicitly disallow session login managers to protect user keyrings
            return False

        if service in ("lockscreen", "lockers", "screensaver"):
            enabled_any = False
            for srv in LOCKSCREEN_SERVICES:
                if self.is_service_available(srv):
                    if self._enable_single_service(srv):
                        enabled_any = True
            if not enabled_any:
                gnome_path = PAM_DIR_ETC / "gnome-screensaver"
                try:
                    content = (
                        f"#%PAM-1.0\n"
                        f"{self.pam_rule}"
                        f"auth       include      password-auth\n"
                        f"account    include      password-auth\n"
                    )
                    gnome_path.write_text(content, encoding="utf-8")
                    enabled_any = True
                except Exception:
                    pass
            # Always ensure gdm-password is not hooked
            self._disable_single_service("gdm-password")
            return enabled_any

        return self._enable_single_service(service)

    def _enable_single_service(self, service: str) -> bool:
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
        bak_path = etc_path.with_suffix(".argus.bak")
        legacy_bak_path = etc_path.with_suffix(".faceunlock.bak")
        if not bak_path.exists():
            try:
                shutil.copy(etc_path, bak_path)
            except Exception:
                pass
        if not legacy_bak_path.exists():
            try:
                shutil.copy(etc_path, legacy_bak_path)
            except Exception:
                pass

        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove existing argus and legacy faceunlock lines
            clean_lines = [l for l in lines if "pam_argus.py" not in l and "pam_faceunlock.py" not in l]

            # Find the best insertion point in auth section
            insert_idx = 0
            for idx, line in enumerate(clean_lines):
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
        """Remove Argus from target PAM service."""
        if service in ("lockscreen", "lockers", "screensaver"):
            for srv in LOCKSCREEN_SERVICES:
                self._disable_single_service(srv)
            return True
        return self._disable_single_service(service)

    def _disable_single_service(self, service: str) -> bool:
        etc_path = PAM_DIR_ETC / service
        if not etc_path.exists():
            return False

        try:
            with open(etc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            clean_lines = [l for l in lines if "pam_argus.py" not in l and "pam_faceunlock.py" not in l]
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
        print("Usage: python3 -m argus.pam_integrator <enable|disable|status> [service]")
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
        print("Error: service name required (e.g., sudo, polkit-1, lockscreen)")
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
