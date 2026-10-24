#!/bin/bash
# Universal Installer for Linux Face Unlock
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="/opt/faceunlock"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/faceunlock"
FACES_DIR="/etc/faceunlock/faces"
MODELS_DIR="$INSTALL_PREFIX/models"

echo "================================================================="
echo " Installing Linux Face Unlock (Universal Biometrics)"
echo "================================================================="

# 1. Check root or non-root
IS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=1
fi

if [ "$IS_ROOT" -eq 1 ]; then
    echo "[*] Running as root. Performing system-wide installation..."
    mkdir -p "$INSTALL_PREFIX" "$CONFIG_DIR" "$FACES_DIR" "$MODELS_DIR"

    # Copy files
    cp -r "$SCRIPT_DIR/faceunlock" "$INSTALL_PREFIX/"
    cp -r "$SCRIPT_DIR/pam" "$INSTALL_PREFIX/"
    cp -r "$SCRIPT_DIR/models/"* "$MODELS_DIR/"

    # Setup Python environment in /opt/faceunlock/venv if uv or python3 is available
    if command -v uv >/dev/null 2>&1; then
        echo "[*] Setting up isolated venv using uv..."
        uv venv "$INSTALL_PREFIX/venv"
        uv pip install --python "$INSTALL_PREFIX/venv/bin/python" numpy opencv-python-headless
    elif [ -d "$SCRIPT_DIR/venv" ]; then
        echo "[*] Copying pre-built virtual environment..."
        cp -r "$SCRIPT_DIR/venv" "$INSTALL_PREFIX/"
    fi

    # Create default config if not exists
    if [ ! -f "$CONFIG_DIR/config.ini" ]; then
        cat <<EOF > "$CONFIG_DIR/config.ini"
[camera]
device = auto
fallback_to_webcam = true
width = 640
height = 480
fps = 30

[security]
cosine_threshold = 0.363
min_liveness_score = 0.55
require_liveness = true
max_timeout = 3.5
warmup_frames = 4
check_lid = true

[pam]
enabled = true
show_feedback = true

[paths]
models_dir = $MODELS_DIR
data_dir = $FACES_DIR
EOF
        chmod 644 "$CONFIG_DIR/config.ini"
    fi

    # Restrict faces directory permissions
    chmod 700 "$FACES_DIR"

    # Create CLI symlink in /usr/local/bin
    cat <<EOF > "$BIN_DIR/faceunlock"
#!/bin/bash
BASE_DIR="$INSTALL_PREFIX"
if [ -f "\$BASE_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="\$BASE_DIR/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi
PYTHONPATH="\$BASE_DIR:\$PYTHONPATH" exec "\$PYTHON_EXEC" -m faceunlock.cli "\$@"
EOF
    chmod +x "$BIN_DIR/faceunlock"
    chmod +x "$INSTALL_PREFIX/pam/pam_faceunlock.py"

    # Install udev rule for camera access
    if [ -d "/etc/udev/rules.d" ]; then
        cat <<EOF > "/etc/udev/rules.d/99-faceunlock-camera.rules"
# Camera access for faceunlock biometric service
SUBSYSTEM=="video4linux", TAG+="uaccess", MODE="0660", GROUP="video"
EOF
        udevadm control --reload-rules || true
        udevadm trigger --subsystem-match=video4linux || true
    fi

    echo "================================================================="
    echo " System-wide installation complete!"
    echo " - Binary: $BIN_DIR/faceunlock"
    echo " - PAM Helper: $INSTALL_PREFIX/pam/pam_faceunlock.py"
    echo " - Config: $CONFIG_DIR/config.ini"
    echo ""
    echo "Next steps:"
    echo " 1. Probe your camera:       faceunlock probe"
    echo " 2. Enroll your face:        faceunlock enroll <username>"
    echo " 3. Test face verification:  faceunlock test <username>"
    echo " 4. Review PAM instructions: cat $INSTALL_PREFIX/pam/faceunlock-pam.conf"
    echo "================================================================="
else
    echo "[*] Non-root detected. Performing local user installation..."
    USER_BIN="$HOME/.local/bin"
    USER_CONFIG="$HOME/.config/faceunlock"
    USER_FACES="$USER_CONFIG/faces"

    mkdir -p "$USER_BIN" "$USER_CONFIG" "$USER_FACES"
    chmod 700 "$USER_FACES"

    # Create local wrapper
    cat <<EOF > "$USER_BIN/faceunlock"
#!/bin/bash
BASE_DIR="$SCRIPT_DIR"
if [ -f "\$BASE_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="\$BASE_DIR/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi
PYTHONPATH="\$BASE_DIR:\$PYTHONPATH" exec "\$PYTHON_EXEC" -m faceunlock.cli "\$@"
EOF
    chmod +x "$USER_BIN/faceunlock"

    # Local config
    if [ ! -f "$USER_CONFIG/config.ini" ]; then
        cat <<EOF > "$USER_CONFIG/config.ini"
[camera]
device = auto
fallback_to_webcam = true
width = 640
height = 480
fps = 30

[security]
cosine_threshold = 0.363
min_liveness_score = 0.55
require_liveness = true
max_timeout = 3.5
warmup_frames = 4
check_lid = true

[pam]
enabled = true
show_feedback = true

[paths]
models_dir = $SCRIPT_DIR/models
data_dir = $USER_FACES
EOF
    fi

    echo "================================================================="
    echo " User-space installation complete!"
    echo " - Binary: $USER_BIN/faceunlock"
    echo " - Config: $USER_CONFIG/config.ini"
    echo ""
    echo "Run '$USER_BIN/faceunlock probe' to get started."
    echo "================================================================="
fi
