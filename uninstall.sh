#!/bin/bash
# =================================================================
#  Argus - Complete Uninstallation & System Cleanup Wizard
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="/opt/argus"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/argus"
FACES_DIR="/etc/argus/faces"

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "${USER:-}")}"
[ "$TARGET_USER" = "root" ] && TARGET_USER=""
TARGET_HOME=""
if [ -n "$TARGET_USER" ]; then
    TARGET_HOME=$(getent passwd "$TARGET_USER" 2>/dev/null | cut -d: -f6 || echo "/home/$TARGET_USER")
fi

INTERACTIVE=0
[ -t 0 ] && INTERACTIVE=1

echo "╭──────────────────────────────────────────────────────────╮"
echo "│ Argus - Complete Biometric Uninstaller                   │"
echo "╰──────────────────────────────────────────────────────────╯"

# 1. Require root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "[*] Administrator privileges required. Elevating via sudo..."
    exec sudo "$0" "$@"
fi

# 2. Revert PAM services cleanly
echo "--> Step 1/4: Reverting PAM authentication hooks..."
for f in /etc/pam.d/*; do
    if [ -f "$f" ]; then
        sed -i '/pam_argus.py/d' "$f" 2>/dev/null || true
        sed -i '/pam_faceunlock.py/d' "$f" 2>/dev/null || true
    fi
done

# If .argus.bak files exist, restore clean backups
for bak in /etc/pam.d/*.argus.bak; do
    if [ -f "$bak" ]; then
        orig="${bak%.argus.bak}"
        if [ "$orig" = "/etc/pam.d/gnome-screensaver" ] && ! [ -f "/usr/lib/pam.d/gnome-screensaver" ]; then
            rm -f "$orig"
        fi
        rm -f "$bak"
    fi
done
rm -f /etc/pam.d/*.faceunlock.bak 2>/dev/null || true
echo "  ✔ PAM authentication hooks safely removed"

# 3. Handle dependency uninstallation (ONLY packages installed by Argus)
echo -e "\n--> Step 2/4: Checking Argus-installed system dependencies..."
INSTALLED_PKGS_FILE=""
if [ -f "$INSTALL_PREFIX/.installed_packages" ]; then
    INSTALLED_PKGS_FILE="$INSTALL_PREFIX/.installed_packages"
elif [ -f "$CONFIG_DIR/.installed_packages" ]; then
    INSTALLED_PKGS_FILE="$CONFIG_DIR/.installed_packages"
fi

if [ -n "$INSTALLED_PKGS_FILE" ] && [ -f "$INSTALLED_PKGS_FILE" ]; then
    PACKAGES_TO_REMOVE=$(cat "$INSTALLED_PKGS_FILE" | tr '\n' ' ' | xargs || true)
    if [ -n "$PACKAGES_TO_REMOVE" ]; then
        echo "  The following packages were newly downloaded by Argus during install:"
        echo "    $PACKAGES_TO_REMOVE"
        echo "  (Note: Packages that were already on your system before Argus will NOT be removed)."
        REM_DEPS="N"
        if [ "$INTERACTIVE" -eq 1 ]; then
            read -rp "  Remove only these Argus-installed dependencies? [y/N]: " rem_ans
            REM_DEPS="${rem_ans:-N}"
        fi
        if [[ "$REM_DEPS" =~ ^[Yy]$ ]]; then
            echo "  [*] Removing Argus-installed dependencies..."
            if command -v dnf >/dev/null 2>&1; then
                dnf remove -y $PACKAGES_TO_REMOVE 2>/dev/null || true
            elif command -v apt-get >/dev/null 2>&1; then
                apt-get remove -y $PACKAGES_TO_REMOVE 2>/dev/null || true
            elif command -v pacman >/dev/null 2>&1; then
                pacman -R --noconfirm $PACKAGES_TO_REMOVE 2>/dev/null || true
            elif command -v zypper >/dev/null 2>&1; then
                zypper remove -y $PACKAGES_TO_REMOVE 2>/dev/null || true
            fi
            echo "  ✔ Argus-installed dependencies removed"
        else
            echo "  • Preserving system packages"
        fi
    else
        echo "  • No additional packages were installed by Argus"
    fi
else
    echo "  • No Argus package record found"
fi

# 4. Remove system root files (/opt/argus, /etc/argus, /usr/local/bin, udev, selinux)
echo -e "\n--> Step 3/4: Removing system files in root directories..."
rm -rf "$INSTALL_PREFIX" /opt/faceunlock
rm -rf "$CONFIG_DIR" /etc/faceunlock
rm -f "$BIN_DIR/argus" "$BIN_DIR/faceunlock"
rm -f /etc/udev/rules.d/99-argus-camera.rules /etc/udev/rules.d/99-faceunlock-camera.rules
if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --subsystem-match=video4linux 2>/dev/null || true
fi

if command -v semodule >/dev/null 2>&1; then
    semodule -r argus_gdm 2>/dev/null || true
    semodule -r faceunlock_gdm 2>/dev/null || true
fi
echo "  ✔ System root files in /opt, /etc, /usr/local/bin cleaned"

# 5. Clean compiled artifacts from home directory and source repo
echo -e "\n--> Step 4/4: Cleaning compiled artifacts and ensuring flexible home directory..."
# Clean any compiled libraries or python bytecode in repo
rm -f "$SCRIPT_DIR/argus/libargus_simd.so" "$SCRIPT_DIR/faceunlock/libfaceunlock_simd.so" "$SCRIPT_DIR/faceunlock/libargus_simd.so"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean home directory configs
if [ -n "$TARGET_HOME" ] && [ -d "$TARGET_HOME" ]; then
    rm -rf "$TARGET_HOME/.config/argus" "$TARGET_HOME/.config/faceunlock" 2>/dev/null || true
fi

# Restore repo ownership to user
if [ -n "$TARGET_USER" ]; then
    TARGET_GROUP=$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")
    chown -R "$TARGET_USER:$TARGET_GROUP" "$SCRIPT_DIR" 2>/dev/null || true
fi
echo "  ✔ Source directory and user home folder cleanly restored"

echo -e "\n✔ Argus has been completely and cleanly uninstalled from the system."
