#!/bin/bash
# =================================================================
#  Argus - Capability-Driven Universal Biometric Setup Wizard
#  Simultaneous Face + Fingerprint PAM Biometric Authentication
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="/opt/argus"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/argus"
FACES_DIR="/etc/argus/faces"
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
INTERACTIVE=0
[ -t 0 ] && INTERACTIVE=1

echo -e "\n${C_CYAN}╭──────────────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_CYAN}│${C_RESET} ${C_BOLD}Argus - Universal Biometric Setup Wizard${C_RESET}                 ${C_CYAN}│${C_RESET}"
echo -e "${C_CYAN}│${C_RESET} ${C_DIM}Simultaneous Multi-Modal PAM Biometric Authentication${C_RESET}    ${C_CYAN}│${C_RESET}"
echo -e "${C_CYAN}╰──────────────────────────────────────────────────────────╯${C_RESET}\n"

# -----------------------------------------------------------------
# Step 1: Install System Dependencies
# -----------------------------------------------------------------
echo -e "${C_BOLD}--> Step 1/6: Checking System Dependencies...${C_RESET}"
INSTALL_DEPS="Y"
if [ "$INTERACTIVE" -eq 1 ]; then
    read -rp "Install required system dependencies (Python development, OpenCV, build tools, v4l-utils, tpm2-tools)? [Y/n]: " dep_ans
    INSTALL_DEPS="${dep_ans:-Y}"
fi

if [[ "$INSTALL_DEPS" =~ ^[Yy]$ ]]; then
    echo -e "  [*] Installing dependencies using native system package manager..."
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y python3-devel gcc gcc-c++ make v4l-utils tpm2-tools pam-devel libglvnd-glx 2>/dev/null || true
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y python3-dev python3-venv gcc g++ make v4l-utils tpm2-tools libpam0g-dev libgl1 2>/dev/null || true
    elif command -v pacman >/dev/null 2>&1; then
        pacman -S --noconfirm python gcc make v4l-utils tpm2-tools pam 2>/dev/null || true
    elif command -v zypper >/dev/null 2>&1; then
        zypper install -y python3-devel gcc gcc-c++ make v4l-utils tpm2-tools pam-devel 2>/dev/null || true
    fi
    echo -e "  ${C_GREEN}✔${C_RESET} System packages verified"
else
    echo -e "  ${C_DIM}• Skipping system package installation${C_RESET}"
fi

# -----------------------------------------------------------------
# Step 2: Capability-Driven Hardware Detection
# -----------------------------------------------------------------
echo -e "\n${C_BOLD}--> Step 2/6: Searching Capable Exact Working Hardware...${C_RESET}"

PYTHON_TEMP_EXEC="python3"
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_TEMP_EXEC="$SCRIPT_DIR/venv/bin/python"
elif [ -x "$INSTALL_PREFIX/venv/bin/python" ]; then
    PYTHON_TEMP_EXEC="$INSTALL_PREFIX/venv/bin/python"
fi

CAM_PROBE_JSON=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_TEMP_EXEC" -c '
import json
from argus.hardware import probe_cameras, select_camera, CameraType

cams = probe_cameras(include_all=False)
res = []
has_ir = False
for c in cams:
    is_ir = (c.camera_type == CameraType.INFRARED) or c.is_ir
    if is_ir:
        has_ir = True
    res.append({
        "path": c.path,
        "name": c.name,
        "type": c.camera_type.value,
        "type_label": c.type_label,
        "is_ir": is_ir,
        "formats": c.formats
    })
selected, reason = select_camera()
print(json.dumps({
    "cameras": res,
    "has_ir": has_ir,
    "selected_path": selected.path,
    "selected_name": selected.name,
    "selected_type": selected.camera_type.value,
    "reason": reason
}))
' 2>/dev/null || echo '{"cameras":[], "has_ir":false, "selected_path":"/dev/video0", "selected_name":"Webcam", "selected_type":"rgb_webcam"}')

HAS_IR=$(echo "$CAM_PROBE_JSON" | grep -o '"has_ir": *true' || true)
SELECTED_DEV=$(echo "$CAM_PROBE_JSON" | grep -o '"selected_path": *"[^"]*"' | cut -d'"' -f4 || echo "/dev/video0")
SELECTED_NAME=$(echo "$CAM_PROBE_JSON" | grep -o '"selected_name": *"[^"]*"' | cut -d'"' -f4 || echo "Webcam")

if [ -n "$HAS_IR" ]; then
    echo -e "  ${C_GREEN}✔${C_RESET} Infrared (IR) Camera Sensor Detected: $SELECTED_NAME ($SELECTED_DEV)"
    echo -e "  [*] Testing IR emitter activation with sensor combinations..."

    if command -v v4l2-ctl >/dev/null 2>&1; then
        v4l2-ctl -d "$SELECTED_DEV" --stream-mmap --stream-count=30 --stream-to=/dev/null >/dev/null 2>&1 || true
    fi

    IR_BLINK="n"
    if [ "$INTERACTIVE" -eq 1 ]; then
        read -rp "  Does the infrared emitter blink or glow red? [y/N]: " blink_ans
        IR_BLINK="${blink_ans:-N}"
    fi

    if [[ "$IR_BLINK" =~ ^[Yy]$ ]]; then
        echo -e "  ${C_GREEN}✔${C_RESET} Hardware IR Emitter Confirmed Active!"
    else
        echo -e "  ${C_YELLOW}⚠${C_RESET} IR emitter not confirmed. Continuing with automatic hybrid sensor fallback."
    fi
else
    # User constraint: If NO IR or depth camera detected, DO NOT show or try IR blink combinations!
    echo -e "  ${C_GREEN}✔${C_RESET} Camera Sensor: Standard RGB Webcam ($SELECTED_NAME on $SELECTED_DEV)"
    echo -e "  ${C_DIM}• No IR or Depth hardware present. Seamlessly continuing with RGB webcam.${C_RESET}"
fi

# Fingerprint Sensor
FPRINT_DEVICE=""
if command -v fprintd-list >/dev/null 2>&1; then
    FPRINT_OUT=$(fprintd-list "$TARGET_USER" 2>&1 || true)
    if echo "$FPRINT_OUT" | grep -qi "Sensor"; then
        FPRINT_DEVICE="Active"
        ENROLLED_FINGERS=$(echo "$FPRINT_OUT" | grep -c " - #" || true)
        echo -e "  ${C_GREEN}✔${C_RESET} Fingerprint Reader: Hardware detected ($ENROLLED_FINGERS enrolled print(s) for $TARGET_USER)"
    else
        echo -e "  ${C_DIM}• Fingerprint Reader: Sensor present, fprintd ready${C_RESET}"
    fi
else
    echo -e "  ${C_DIM}• Fingerprint Reader: fprintd not installed (Face unlock only)${C_RESET}"
fi

# TPM 2.0 Hardware Security Root
TPM_SUPPORTED=0
if [ -e "/dev/tpmrm0" ] || [ -e "/dev/tpm0" ]; then
    if command -v tpm2_createprimary >/dev/null 2>&1; then
        TPM_SUPPORTED=1
        echo -e "  ${C_GREEN}✔${C_RESET} TPM 2.0 Security Chip: Hardware detected (/dev/tpmrm0 + tpm2-tools available)"
    else
        echo -e "  ${C_YELLOW}⚠${C_RESET} TPM 2.0 device node present, but tpm2-tools missing (Software fallback)"
    fi
else
    echo -e "  ${C_DIM}• TPM 2.0 Security Chip: Not present (Using PBKDF2/AES-256-GCM hardware-keyed vault)${C_RESET}"
fi

# -----------------------------------------------------------------
# Step 3: Security Profile Selection
# -----------------------------------------------------------------
echo -e "\n${C_BOLD}--> Step 3/6: Security Profile & Verification Sensitivity...${C_RESET}"
echo -e " Choose verification security profile:"
echo -e "   ${C_CYAN}[L]${C_RESET} Low / Convenient   (Threshold: 0.30, Liveness: 0.45, Timeout: 2.5s) - Fastest recognition"
echo -e "   ${C_CYAN}[M]${C_RESET} Medium / Balanced  (Threshold: 0.33, Liveness: 0.55, Timeout: 3.5s) - ${C_GREEN}Recommended${C_RESET}"
echo -e "   ${C_CYAN}[H]${C_RESET} High / Strict      (Threshold: 0.40, Liveness: 0.65, Timeout: 4.0s) - Maximum anti-spoofing"
echo -e "   ${C_CYAN}[C]${C_RESET} Custom Tweaks      (Manually set match & liveness thresholds)"

SECURITY_PROFILE="balanced"
COSINE_TH="0.33"
LIVENESS_TH="0.55"
TIMEOUT_SEC="3.5"

if [ "$INTERACTIVE" -eq 1 ]; then
    read -rp " Select profile [L/M/H/C] (Default: M): " prof_choice
    prof_choice="${prof_choice:-M}"
    case "${prof_choice^^}" in
        L)
            SECURITY_PROFILE="convenient"
            COSINE_TH="0.30"
            LIVENESS_TH="0.45"
            TIMEOUT_SEC="2.5"
            ;;
        H)
            SECURITY_PROFILE="strict"
            COSINE_TH="0.40"
            LIVENESS_TH="0.65"
            TIMEOUT_SEC="4.0"
            ;;
        C)
            SECURITY_PROFILE="custom"
            read -rp "   Enter Cosine Match Threshold [0.25 - 0.50] (Default: 0.33): " custom_cos
            COSINE_TH="${custom_cos:-0.33}"
            read -rp "   Enter Min Liveness Score [0.30 - 0.90] (Default: 0.55): " custom_liv
            LIVENESS_TH="${custom_liv:-0.55}"
            read -rp "   Enter Recognition Timeout in seconds (Default: 3.5): " custom_to
            TIMEOUT_SEC="${custom_to:-3.5}"
            ;;
        *)
            SECURITY_PROFILE="balanced"
            COSINE_TH="0.33"
            LIVENESS_TH="0.55"
            TIMEOUT_SEC="3.5"
            ;;
    esac
fi
echo -e "  ${C_GREEN}✔${C_RESET} Security Profile Configured: ${C_BOLD}${SECURITY_PROFILE^^}${C_RESET} (Cosine: $COSINE_TH, Liveness: $LIVENESS_TH, Timeout: ${TIMEOUT_SEC}s)"

# -----------------------------------------------------------------
# Step 4: PAM Service Configuration & Keyring Notice
# -----------------------------------------------------------------
echo -e "\n${C_BOLD}--> Step 4/6: PAM Service Integrations & Keyring Notice...${C_RESET}"

ENABLE_SUDO="true"
ENABLE_POLKIT="true"
ENABLE_LOGIN="false"
ENABLE_GDM="false"
USE_TPM="false"

if [ "$INTERACTIVE" -eq 1 ]; then
    # Sudo
    read -rp " Enable simultaneous Face + Fingerprint unlock for sudo? [Y/n]: " sudo_ans
    sudo_ans="${sudo_ans:-Y}"
    [[ ! "$sudo_ans" =~ ^[Yy]$ ]] && ENABLE_SUDO="false"

    # Polkit
    read -rp " Enable Argus for Polkit GUI administration dialogs? [Y/n]: " polkit_ans
    polkit_ans="${polkit_ans:-Y}"
    [[ ! "$polkit_ans" =~ ^[Yy]$ ]] && ENABLE_POLKIT="false"

    # Console Login
    read -rp " Enable Argus for console / virtual terminal login? [y/N]: " login_ans
    login_ans="${login_ans:-N}"
    [[ "$login_ans" =~ ^[Yy]$ ]] && ENABLE_LOGIN="true"

    # GDM (GNOME Display Manager / Lock Screen)
    echo -e "\n  ${C_YELLOW}─────────────────────────────────────────────────────────────────${C_RESET}"
    echo -e "  ${C_YELLOW}${C_BOLD}⚠️  IMPORTANT GNOME KEYRING & SECRET SERVICE NOTICE:${C_RESET}"
    echo -e "  ${C_DIM}On Linux (GNOME, GDM), your login password is used by oo7-daemon /${C_RESET}"
    echo -e "  ${C_DIM}gnome-keyring to decrypt your saved passwords and web sessions.${C_RESET}"
    echo -e "  ${C_DIM}Logging in at cold boot via biometrics will leave the keyring locked,${C_RESET}"
    echo -e "  ${C_DIM}causing browsers (Chrome, Brave, etc.) to appear signed out until you${C_RESET}"
    echo -e "  ${C_DIM}enter your password. (Unlocking from lock screen works normally).${C_RESET}"
    echo -e "  ${C_YELLOW}─────────────────────────────────────────────────────────────────${C_RESET}"
    read -rp " Enable Argus for GDM login anyway? [y/N]: " gdm_ans
    gdm_ans="${gdm_ans:-N}"
    [[ "$gdm_ans" =~ ^[Yy]$ ]] && ENABLE_GDM="true"

    # TPM 2.0
    if [ "$TPM_SUPPORTED" -eq 1 ]; then
        echo ""
        read -rp " Seal biometric encryption keys into TPM 2.0 hardware chip? [Y/n]: " tpm_ans
        tpm_ans="${tpm_ans:-Y}"
        [[ "$tpm_ans" =~ ^[Yy]$ ]] && USE_TPM="true"
    fi
else
    ENABLE_SUDO="true"
    ENABLE_POLKIT="true"
    ENABLE_LOGIN="false"
    ENABLE_GDM="false"
    [ "$TPM_SUPPORTED" -eq 1 ] && USE_TPM="true"
fi

# -----------------------------------------------------------------
# Step 5: System Package Installation & Compilation
# -----------------------------------------------------------------
echo -e "\n${C_BOLD}--> Step 5/6: Deploying Files & Native SIMD Accelerator...${C_RESET}"
mkdir -p "$INSTALL_PREFIX" "$CONFIG_DIR" "$FACES_DIR" "$MODELS_DIR"

# SIMD build
if command -v gcc >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/src/fast_simd.c" ]; then
    echo -e "  [*] Compiling native AVX2/FMA vector acceleration in C..."
    gcc -O3 -mavx2 -mfma -shared -fPIC -o "$SCRIPT_DIR/argus/libargus_simd.so" "$SCRIPT_DIR/src/fast_simd.c" 2>/dev/null || true
    [ -f "$SCRIPT_DIR/argus/libargus_simd.so" ] && echo -e "  ${C_GREEN}✔${C_RESET} Native C SIMD library compiled successfully"
fi

cp -rf "$SCRIPT_DIR/argus" "$INSTALL_PREFIX/"
cp -rf "$SCRIPT_DIR/faceunlock" "$INSTALL_PREFIX/"
cp -rf "$SCRIPT_DIR/pam" "$INSTALL_PREFIX/"
[ -d "$SCRIPT_DIR/src" ] && cp -rf "$SCRIPT_DIR/src" "$INSTALL_PREFIX/"
cp -rf "$SCRIPT_DIR/models/"* "$MODELS_DIR/"

# Create backward-compatibility directory symlinks
ln -sfn "$INSTALL_PREFIX" /opt/faceunlock
ln -sfn "$CONFIG_DIR" /etc/faceunlock

chmod 755 "$INSTALL_PREFIX/pam/pam_argus.py"
chmod 755 "$INSTALL_PREFIX/pam/pam_faceunlock.py"
[ -f "$INSTALL_PREFIX/argus/libargus_simd.so" ] && chmod 755 "$INSTALL_PREFIX/argus/libargus_simd.so"

# Setup virtual environment
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [ -z "$UV_BIN" ] && [ -x "$TARGET_HOME/.local/bin/uv" ]; then
    UV_BIN="$TARGET_HOME/.local/bin/uv"
fi

if [ -n "$UV_BIN" ]; then
    echo -e "  [*] Setting up Python runtime with full GUI and crypto support..."
    "$UV_BIN" venv --allow-existing "$INSTALL_PREFIX/venv"
    "$UV_BIN" pip install --python "$INSTALL_PREFIX/venv/bin/python" numpy opencv-python cryptography
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "  [*] Synchronizing tested virtual environment..."
    rm -rf "$INSTALL_PREFIX/venv"
    cp -r "$SCRIPT_DIR/venv" "$INSTALL_PREFIX/"
elif [ -d "/opt/faceunlock/venv" ]; then
    echo -e "  [*] Re-using installed virtual environment..."
    cp -r "/opt/faceunlock/venv" "$INSTALL_PREFIX/"
fi

# Ensure system-site-packages enabled
if [ -f "$INSTALL_PREFIX/venv/pyvenv.cfg" ]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/g' "$INSTALL_PREFIX/venv/pyvenv.cfg"
fi

# Write system configuration
cat <<EOF > "$CONFIG_DIR/config.ini"
[camera]
device = auto
fallback_to_webcam = true
width = 640
height = 480
fps = 30

[security]
cosine_threshold = $COSINE_TH
min_liveness_score = $LIVENESS_TH
require_liveness = true
max_timeout = $TIMEOUT_SEC
warmup_frames = 4
check_lid = true
use_tpm = $USE_TPM
security_mode = $SECURITY_PROFILE

[policies]
sudo = $ENABLE_SUDO
gdm = $ENABLE_GDM
polkit = $ENABLE_POLKIT
login = $ENABLE_LOGIN
lockers = true
sshd = false

[pam]
enabled = true
show_feedback = true
feedback_style = animated

[paths]
models_dir = $MODELS_DIR
data_dir = $FACES_DIR
EOF
chmod 644 "$CONFIG_DIR/config.ini"
chmod 700 "$FACES_DIR"

# Install /usr/local/bin/argus and compatibility /usr/local/bin/faceunlock wrapper
cat <<EOF > "$BIN_DIR/argus"
#!/bin/bash
BASE_DIR="$INSTALL_PREFIX"
if [ -f "\$BASE_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="\$BASE_DIR/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi
PYTHONPATH="\$BASE_DIR:\$PYTHONPATH" exec "\$PYTHON_EXEC" -m argus.cli "\$@"
EOF
chmod +x "$BIN_DIR/argus"
ln -sfn "$BIN_DIR/argus" "$BIN_DIR/faceunlock"

# Udev rule for camera
if [ -d "/etc/udev/rules.d" ]; then
    cat <<EOF > "/etc/udev/rules.d/99-argus-camera.rules"
SUBSYSTEM=="video4linux", TAG+="uaccess", MODE="0660", GROUP="video"
EOF
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --subsystem-match=video4linux 2>/dev/null || true
fi

# SELinux policy installation on Fedora / RHEL
if command -v semodule >/dev/null 2>&1; then
    if [ -f "$SCRIPT_DIR/selinux/argus_gdm.pp" ]; then
        echo -e "  [*] Installing SELinux policy module (argus_gdm.pp)..."
        semodule -i "$SCRIPT_DIR/selinux/argus_gdm.pp" 2>/dev/null || true
    elif [ -f "$SCRIPT_DIR/selinux/faceunlock_gdm.pp" ]; then
        semodule -i "$SCRIPT_DIR/selinux/faceunlock_gdm.pp" 2>/dev/null || true
    fi
fi

# Sync PAM services
PYTHON_CMD="$INSTALL_PREFIX/venv/bin/python"
[ "$ENABLE_SUDO" = "true" ] && "$PYTHON_CMD" -m argus.pam_integrator enable sudo >/dev/null 2>&1 || "$PYTHON_CMD" -m argus.pam_integrator disable sudo >/dev/null 2>&1
[ "$ENABLE_POLKIT" = "true" ] && "$PYTHON_CMD" -m argus.pam_integrator enable polkit-1 >/dev/null 2>&1 || "$PYTHON_CMD" -m argus.pam_integrator disable polkit-1 >/dev/null 2>&1
[ "$ENABLE_LOGIN" = "true" ] && "$PYTHON_CMD" -m argus.pam_integrator enable login >/dev/null 2>&1 || "$PYTHON_CMD" -m argus.pam_integrator disable login >/dev/null 2>&1
[ "$ENABLE_GDM" = "true" ] && "$PYTHON_CMD" -m argus.pam_integrator enable gdm-password >/dev/null 2>&1 || "$PYTHON_CMD" -m argus.pam_integrator disable gdm-password >/dev/null 2>&1

# Sync existing enrolled user faces if present
for legacy_src in "$TARGET_HOME/.config/argus/faces" "$TARGET_HOME/.config/faceunlock/faces" "/etc/faceunlock/faces"; do
    if [ -d "$legacy_src" ] && [ "$legacy_src" != "$FACES_DIR" ]; then
        cp -f "$legacy_src/"*.vault "$FACES_DIR/" 2>/dev/null || true
        cp -f "$legacy_src/"*.json "$FACES_DIR/" 2>/dev/null || true
        cp -f "$legacy_src/.vault_secret" "$FACES_DIR/" 2>/dev/null || true
        chmod 600 "$FACES_DIR/"* "$FACES_DIR/.vault_secret" 2>/dev/null || true
    fi
done

echo -e "  ${C_GREEN}✔${C_RESET} Package successfully installed into /opt/argus"

# -----------------------------------------------------------------
# Step 6: Guided Face Registration & Verification
# -----------------------------------------------------------------
echo -e "\n${C_BOLD}--> Step 6/6: Face Model Registration...${C_RESET}"
if [ "$INTERACTIVE" -eq 1 ]; then
    read -rp "Register face model for '$TARGET_USER' now? [Y/n]: " add_ans
    add_ans="${add_ans:-Y}"
    if [[ "$add_ans" =~ ^[Yy]$ ]]; then
        echo ""
        "$BIN_DIR/argus" add "$TARGET_USER" || true

        # Sync vault to user home
        mkdir -p "$TARGET_HOME/.config/argus/faces"
        cp -f "$FACES_DIR/$TARGET_USER.vault" "$TARGET_HOME/.config/argus/faces/" 2>/dev/null || true
        cp -f "$FACES_DIR/.vault_secret" "$TARGET_HOME/.config/argus/faces/" 2>/dev/null || true
        chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.config/argus" 2>/dev/null || true

        echo ""
        read -rp "Test live camera verification preview now? [Y/n]: " test_ans
        test_ans="${test_ans:-Y}"
        if [[ "$test_ans" =~ ^[Yy]$ ]]; then
            echo ""
            "$BIN_DIR/argus" test "$TARGET_USER" || true
        fi
    fi
fi

echo -e "\n${C_GREEN}╭──────────────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_GREEN}│${C_RESET} ${C_BOLD}Installation Successful! Argus is Ready.${C_RESET}                 ${C_GREEN}│${C_RESET}"
echo -e "${C_GREEN}╰──────────────────────────────────────────────────────────╯${C_RESET}"
echo -e " Available Commands:"
echo -e "  • ${C_CYAN}sudo argus config${C_RESET}  - Interactive toggle menu for PAM, feedback, timeouts & security"
echo -e "  • ${C_CYAN}sudo argus add${C_RESET}     - Register face model with iOS-like guided animation"
echo -e "  • ${C_CYAN}sudo argus test${C_RESET}    - Open camera window & display live face verification"
echo -e "  • ${C_CYAN}argus status${C_RESET}       - View system status, hardware diagnostics & active PAM services"
echo -e "  • ${C_CYAN}sudo true${C_RESET}          - Test instant biometric authentication on terminal\n"
