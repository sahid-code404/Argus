#!/bin/bash
# =================================================================
#  Universal Linux Face Unlock - Interactive Installer Wizard
#  Simultaneous Face + Fingerprint PAM Biometric Authentication
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="/opt/faceunlock"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/faceunlock"
FACES_DIR="/etc/faceunlock/faces"
MODELS_DIR="$INSTALL_PREFIX/models"

# ANSI Colors
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[1;36m"
C_GREEN="\033[1;32m"
C_YELLOW="\033[1;33m"
C_RED="\033[1;31m"
C_DIM="\033[2m"

# 1. Automatic Elevation via sudo
if [ "$(id -u)" -ne 0 ]; then
    if [ -t 0 ]; then
        echo -e "${C_CYAN}[*] Administrator privileges required. Elevating via sudo...${C_RESET}"
        exec sudo "$0" "$@"
    else
        echo -e "${C_RED}[!] Please run install.sh with administrator privileges: sudo ./install.sh${C_RESET}" >&2
        exit 1
    fi
fi

# Detect calling user
TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "$USER")}"
TARGET_HOME=$(getent passwd "$TARGET_USER" 2>/dev/null | cut -d: -f6 || echo "/home/$TARGET_USER")

echo -e "\n${C_CYAN}╭──────────────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_CYAN}│${C_RESET} ${C_BOLD}Linux Face Unlock - Universal Biometric Setup Wizard${C_RESET}     ${C_CYAN}│${C_RESET}"
echo -e "${C_CYAN}│${C_RESET} ${C_DIM}Simultaneous Multi-Modal PAM Biometric Authentication${C_RESET}    ${C_CYAN}│${C_RESET}"
echo -e "${C_CYAN}╰──────────────────────────────────────────────────────────╯${C_RESET}\n"

# 2. Hardware Discovery
echo -e "${C_BOLD}--> Step 1/5: Detecting Biometric Hardware...${C_RESET}"

CAMERA_COUNT=$(find /dev -name "video*" 2>/dev/null | wc -l)
if [ "$CAMERA_COUNT" -gt 0 ]; then
    echo -e "  ${C_GREEN}✔${C_RESET} Camera Sensor: Found $CAMERA_COUNT video device(s) (/dev/video0 ready)"
else
    echo -e "  ${C_YELLOW}⚠${C_RESET} No camera devices detected in /dev/video*. Face unlock may require external webcam."
fi

FPRINT_DEVICE=""
if command -v fprintd-list >/dev/null 2>&1; then
    FPRINT_OUT=$(fprintd-list "$TARGET_USER" 2>&1 || true)
    if echo "$FPRINT_OUT" | grep -qi "Sensor"; then
        FPRINT_DEVICE="Active"
        ENROLLED_FINGERS=$(echo "$FPRINT_OUT" | grep -c " - #" || true)
        echo -e "  ${C_GREEN}✔${C_RESET} Fingerprint Reader: Hardware detected ($ENROLLED_FINGERS enrolled print(s) for $TARGET_USER)"
    else
        echo -e "  ${C_DIM}• Fingerprint Reader: No sensor or daemon inactive${C_RESET}"
    fi
else
    echo -e "  ${C_DIM}• Fingerprint Reader: fprintd not installed (Face unlock only)${C_RESET}"
fi

# 3. Native SIMD Compilation
echo -e "\n${C_BOLD}--> Step 2/5: Building Native SIMD Accelerator...${C_RESET}"
if command -v gcc >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/src/fast_simd.c" ]; then
    echo -e "  [*] Compiling native AVX2/FMA vector acceleration in C..."
    gcc -O3 -mavx2 -mfma -shared -fPIC -o "$SCRIPT_DIR/faceunlock/libfaceunlock_simd.so" "$SCRIPT_DIR/src/fast_simd.c"
    echo -e "  ${C_GREEN}✔${C_RESET} Native C SIMD library compiled successfully (libfaceunlock_simd.so)"
elif [ -f "$SCRIPT_DIR/faceunlock/libfaceunlock_simd.so" ]; then
    echo -e "  ${C_GREEN}✔${C_RESET} Using pre-compiled AVX2 SIMD library"
else
    echo -e "  ${C_YELLOW}⚠${C_RESET} GCC not found; using optimized NumPy fallback"
fi

# 4. Deploy Files & Environment
echo -e "\n${C_BOLD}--> Step 3/5: Deploying System Package to /opt/faceunlock...${C_RESET}"
mkdir -p "$INSTALL_PREFIX" "$CONFIG_DIR" "$FACES_DIR" "$MODELS_DIR"

cp -rf "$SCRIPT_DIR/faceunlock" "$INSTALL_PREFIX/"
cp -rf "$SCRIPT_DIR/pam" "$INSTALL_PREFIX/"
[ -d "$SCRIPT_DIR/src" ] && cp -rf "$SCRIPT_DIR/src" "$INSTALL_PREFIX/"
cp -rf "$SCRIPT_DIR/models/"* "$MODELS_DIR/"

chmod 755 "$INSTALL_PREFIX/pam/pam_faceunlock.py"
[ -f "$INSTALL_PREFIX/faceunlock/libfaceunlock_simd.so" ] && chmod 755 "$INSTALL_PREFIX/faceunlock/libfaceunlock_simd.so"

# Virtual environment setup
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [ -z "$UV_BIN" ] && [ -x "$TARGET_HOME/.local/bin/uv" ]; then
    UV_BIN="$TARGET_HOME/.local/bin/uv"
fi

if [ -n "$UV_BIN" ]; then
    echo -e "  [*] Creating isolated Python runtime via $UV_BIN..."
    "$UV_BIN" venv --allow-existing "$INSTALL_PREFIX/venv"
    "$UV_BIN" pip install --python "$INSTALL_PREFIX/venv/bin/python" numpy opencv-python-headless cryptography
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "  [*] Copying tested virtual environment..."
    rm -rf "$INSTALL_PREFIX/venv"
    cp -r "$SCRIPT_DIR/venv" "$INSTALL_PREFIX/"
fi

# Ensure system-site-packages is enabled so D-Bus / system libraries are accessible
if [ -f "$INSTALL_PREFIX/venv/pyvenv.cfg" ]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/g' "$INSTALL_PREFIX/venv/pyvenv.cfg"
fi

# System configuration
if [ ! -f "$CONFIG_DIR/config.ini" ]; then
    cat <<EOF > "$CONFIG_DIR/config.ini"
[camera]
device = auto
fallback_to_webcam = true
width = 640
height = 480
fps = 30

[security]
cosine_threshold = 0.33
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

chmod 700 "$FACES_DIR"

# Install /usr/local/bin/faceunlock wrapper
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

# Udev camera access rule
if [ -d "/etc/udev/rules.d" ]; then
    cat <<EOF > "/etc/udev/rules.d/99-faceunlock-camera.rules"
SUBSYSTEM=="video4linux", TAG+="uaccess", MODE="0660", GROUP="video"
EOF
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --subsystem-match=video4linux 2>/dev/null || true
fi

# Sync existing enrolled user faces
if [ -d "$TARGET_HOME/.config/faceunlock/faces" ]; then
    echo -e "  [*] Synchronizing biometric vaults from $TARGET_USER..."
    cp -f "$TARGET_HOME/.config/faceunlock/faces/"*.vault "$FACES_DIR/" 2>/dev/null || true
    cp -f "$TARGET_HOME/.config/faceunlock/faces/"*.json "$FACES_DIR/" 2>/dev/null || true
    cp -f "$TARGET_HOME/.config/faceunlock/faces/.vault_secret" "$FACES_DIR/" 2>/dev/null || true
    chmod 600 "$FACES_DIR/"* "$FACES_DIR/.vault_secret" 2>/dev/null || true
fi
echo -e "  ${C_GREEN}✔${C_RESET} System deployment complete (/opt/faceunlock)"

# 5. PAM Integration
echo -e "\n${C_BOLD}--> Step 4/5: PAM Authentication Configuration...${C_RESET}"
INTERACTIVE=0
[ -t 0 ] && INTERACTIVE=1

configure_pam() {
    if [ -f "/etc/pam.d/sudo" ]; then
        [ ! -f "/etc/pam.d/sudo.faceunlock.bak" ] && cp /etc/pam.d/sudo /etc/pam.d/sudo.faceunlock.bak
        sed -i '/pam_faceunlock.py/d' /etc/pam.d/sudo
        sed -i '1s|^|auth       sufficient   pam_exec.so quiet stdout '"$INSTALL_PREFIX"'/pam/pam_faceunlock.py\n|' /etc/pam.d/sudo
        echo -e "  ${C_GREEN}✔${C_RESET} Configured /etc/pam.d/sudo with quiet stdout"
    fi
}

if [ "$INTERACTIVE" -eq 1 ]; then
    read -rp "Enable simultaneous Face + Fingerprint unlock for sudo? [Y/n]: " pam_ans
    pam_ans="${pam_ans:-Y}"
    if [[ "$pam_ans" =~ ^[Yy]$ ]]; then
        configure_pam
    else
        echo -e "  ${C_DIM}Skipped PAM configuration.${C_RESET}"
    fi
else
    configure_pam
fi

# 6. Biometric Enrollment
echo -e "\n${C_BOLD}--> Step 5/5: Biometric Face Enrollment...${C_RESET}"
IS_ENROLLED=0
if [ -f "$FACES_DIR/$TARGET_USER.vault" ] || [ -f "$TARGET_HOME/.config/faceunlock/faces/$TARGET_USER.vault" ]; then
    IS_ENROLLED=1
fi

if [ "$INTERACTIVE" -eq 1 ]; then
    if [ "$IS_ENROLLED" -eq 1 ]; then
        read -rp "User '$TARGET_USER' is already enrolled. Re-enroll face now? [y/N]: " enroll_ans
        enroll_ans="${enroll_ans:-N}"
    else
        read -rp "No face enrolled for '$TARGET_USER'. Enroll face now? [Y/n]: " enroll_ans
        enroll_ans="${enroll_ans:-Y}"
    fi

    if [[ "$enroll_ans" =~ ^[Yy]$ ]]; then
        echo ""
        "$BIN_DIR/faceunlock" enroll "$TARGET_USER"

        # Sync to user home as well
        mkdir -p "$TARGET_HOME/.config/faceunlock/faces"
        cp -f "$FACES_DIR/$TARGET_USER.vault" "$TARGET_HOME/.config/faceunlock/faces/" 2>/dev/null || true
        cp -f "$FACES_DIR/.vault_secret" "$TARGET_HOME/.config/faceunlock/faces/" 2>/dev/null || true
        chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.config/faceunlock" 2>/dev/null || true

        echo ""
        read -rp "Run live verification test now? [Y/n]: " test_ans
        test_ans="${test_ans:-Y}"
        if [[ "$test_ans" =~ ^[Yy]$ ]]; then
            echo ""
            "$BIN_DIR/faceunlock" test "$TARGET_USER"
        fi
    fi
fi

echo -e "\n${C_GREEN}╭──────────────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_GREEN}│${C_RESET} ${C_BOLD}Installation Successful! Linux Face Unlock is Ready.${C_RESET}   ${C_GREEN}│${C_RESET}"
echo -e "${C_GREEN}╰──────────────────────────────────────────────────────────╯${C_RESET}"
echo -e " Quick Commands:"
echo -e "  • Test verification:  ${C_CYAN}faceunlock test${C_RESET}"
echo -e "  • Re-enroll face:     ${C_CYAN}faceunlock enroll${C_RESET}"
echo -e "  • Camera probe:       ${C_CYAN}faceunlock probe${C_RESET}"
echo -e "  • Sudo test:          ${C_CYAN}sudo true${C_RESET}\n"
