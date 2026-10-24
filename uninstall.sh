#!/bin/bash
# Uninstaller for Linux Face Unlock
set -euo pipefail

INSTALL_PREFIX="/opt/faceunlock"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/faceunlock"
USER_BIN="$HOME/.local/bin/faceunlock"
USER_CONFIG="$HOME/.config/faceunlock"

echo "Uninstalling Linux Face Unlock..."

if [ "$(id -u)" -eq 0 ]; then
    rm -rf "$INSTALL_PREFIX"
    rm -f "$BIN_DIR/faceunlock"
    rm -f "/etc/udev/rules.d/99-faceunlock-camera.rules"
    echo "[!] Preserving $CONFIG_DIR (remove manually if desired: rm -rf $CONFIG_DIR)"
fi

rm -f "$USER_BIN"
echo "[✓] Face Unlock successfully uninstalled."
