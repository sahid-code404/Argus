#!/bin/bash
# Uninstaller for Argus
set -euo pipefail

INSTALL_PREFIX="/opt/argus"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/argus"

echo "Uninstalling Argus..."

if [ "$(id -u)" -eq 0 ]; then
    for f in /etc/pam.d/*; do
        if [ -f "$f" ]; then
            sed -i '/pam_argus.py/d' "$f" 2>/dev/null || true
            sed -i '/pam_faceunlock.py/d' "$f" 2>/dev/null || true
        fi
    done
    rm -rf "$INSTALL_PREFIX" /opt/faceunlock
    rm -f "$BIN_DIR/argus" "$BIN_DIR/faceunlock"
    rm -f "/etc/udev/rules.d/99-argus-camera.rules" "/etc/udev/rules.d/99-faceunlock-camera.rules"
    echo "[!] Preserving $CONFIG_DIR and /etc/faceunlock (remove manually if desired: rm -rf $CONFIG_DIR)"
fi

echo "[✓] Argus successfully uninstalled."
