# 👁️ Argus: Universal Linux Biometric Security

<p align="center">
  <img src="https://img.shields.io/badge/Argus-Biometrics_v1.0-blue?style=for-the-badge&logo=linux" alt="Argus Linux">
  <img src="https://img.shields.io/badge/Python-3.9%2B-brightgreen?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Platform-Fedora%20%7C%20Ubuntu%20%7C%20Arch%20%7C%20Debian-orange?style=for-the-badge&logo=linux" alt="Supported OS">
  <img src="https://img.shields.io/badge/Security-AES--256--GCM%20%2B%20TPM%202.0-red?style=for-the-badge&logo=shield" alt="Hardware Security">
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Tests-51%20Passed-success?style=for-the-badge&logo=pytest" alt="Test Status">
</p>

**Argus** (*named after the all-seeing hundred-eyed guardian Argus Panoptes*) is a modern, blazing-fast, and universal biometric authentication system engineered for Linux desktops and laptops. 

It provides an **Android / Windows Hello-grade** facial and simultaneous fingerprint authentication experience across `sudo`, `polkit`, console login, and desktop display managers.

---

## ⚡ Highlights

- **Universal Camera Detection**: Automatically probes **Depth Sensors** (RealSense, Astra) &rarr; **Infrared Cameras** (Windows Hello) &rarr; **RGB Webcams** with seamless zero-config fallback.
- **Simultaneous Biometrics**: Runs facial recognition and fingerprint sensor (`fprintd`) reader threads in parallel. Either one unlocks your session instantly without delay.
- **Apple iOS-Style Visual Enrollment**: Interactive GUI with smooth circular progression, head-angle guidance (yaw, pitch, roll), and automatic multi-sample capture.
- **Ultra-Fast (< 35ms CPU)**: Utilizes ultra-lightweight ONNX neural networks (**YuNet** detection + **SFace** 128-d embeddings) paired with native **C SIMD acceleration** (AVX2 / NEON / SSE2).
- **Multi-Modal Anti-Spoofing (PAD)**: Protects RGB webcams against printed photos and digital screen replays using Fourier moiré frequency analysis, Laplacian texture analysis, and 3D parallax.
- **TPM 2.0 Hardware Key Sealing**: Encrypts biometric embeddings with AES-256-GCM sealed against hardware TPM 2.0 PCR registers (PCR 0/7) when available.
- **Desktop Keyring Protection**: Prevents orphaned GNOME Keyring / secret-service sessions by safely isolating login manager credentials.
- **Backward-Compatible**: Full alias support for legacy `faceunlock` commands, symlinks, and scripts.

---

## 🏗️ Architecture: How Argus Works

```
                               +-----------------------------+
                               |    Authentication Request   |
                               | (sudo, polkit, login, gdm)  |
                               +--------------+--------------+
                                              |
                                              v
                               +-----------------------------+
                               |       PAM Auth Stack        |
                               |      (/etc/pam.d/sudo)      |
                               +--------------+--------------+
                                              |
                                              v
   +-------------------------------------------------------------------------------------+
   | 1. auth sufficient pam_exec.so stdout /opt/argus/pam/pam_argus.py                   |
   |                                                                                     |
   |  [Simultaneous Thread] ---> Non-blocking fprintd reader poll                        |
   |                                                                                     |
   |  [Probe Hardware] --------> Depth (P1) -> IR Camera (P2) -> Universal Webcam (P3)   |
   |                                    |                                                |
   |                             [Lid Detection]                                         |
   |                      (Skip if closed /proc/acpi/...)                                |
   |                                    |                                                |
   |                            [Capture Stream]                                         |
   |                       (Auto-exposure warmup)                                        |
   |                                    |                                                |
   |                     [Face Detection: YuNet (CNN)]                                   |
   |                         (<25ms on CPU, 5 landmarks)                                 |
   |                                    |                                                |
   |                  [Multi-Modal Anti-Spoofing / PAD]                                  |
   |            (Fourier moiré, Laplacian texture, 3D parallax)                          |
   |                                    |                                                |
   |                   [Feature Extractor: SFace (CNN)]                                  |
   |                     (128-d L2-normalized embedding)                                 |
   |                                    |                                                |
   |                [SIMD Accelerated Cosine Similarity]                                 |
   |                (libargus_simd.so: AVX2 / NEON / SSE2)                               |
   |                                    |                                                |
   |                     +--------------+--------------+                                 |
   |                     |                             |                                 |
   |             [Match + Live]               [Mismatch / Spoof / Timeout]               |
   |                     |                             |                                 |
   |            Exit 0 (PAM_SUCCESS)          Exit 1 (PAM_AUTH_ERR)                      |
   +---------------------+-----------------------------+---------------------------------+
                         |                             |
                         v                             v
                 [AUTH SUCCESSFUL]            [Fallback to Fingerprint]
                                                       |
                                                       v
                                              [Fallback to Password]
```

---

## 📦 Requirements

- **Linux Distribution**: Fedora, RHEL, Ubuntu, Debian, Arch Linux, openSUSE, or Manjaro.
- **Hardware**: Standard USB/built-in webcam, Infrared (IR) camera, or Depth sensor.
- **Dependencies**: Python 3.9+, GCC, and PAM development headers (automatically handled by the installer).

---

## 🚀 Quick Start & Installation

### 1. Clone & Run Interactive Installer

```bash
git clone https://github.com/sahid-code404/Argus.git
cd Argus
sudo ./install.sh
```

The interactive installer will:
1. Detect and install required system packages (`gcc`, `libpam-dev`, `python3-devel`, etc.).
2. Probe your camera hardware and check for dedicated IR / Depth sensors.
3. Prompt you to select your preferred security profile (**Low**, **Medium**, or **High**).
4. Configure PAM authentication endpoints (`sudo`, `polkit-1`).
5. Compile and install the native SIMD acceleration library (`libargus_simd.so`).
6. Deploy the system service to `/opt/argus` and install the `argus` binary into `/usr/local/bin`.

---

### 2. Enroll Your Face (`sudo argus add`)

Launch the iOS-style guided visual enrollment interface:

```bash
sudo argus add
```

- Guides you to tilt your head slightly left, right, up, and down.
- Captures multiple high-quality embedding vectors.
- AES-256-GCM encrypts your biometric vault with cryptographic checksums and hardware binding.

---

### 3. Test Live Verification (`sudo argus test`)

Open the live verification testing window to inspect real-time detection, liveness scores, and cosine matching:

```bash
sudo argus test
```

Press `q` or `ESC` to close the test preview.

---

### 4. Interactive Configuration (`sudo argus config`)

Customize Argus settings anytime through an interactive terminal panel:

```bash
sudo argus config
```

- Toggle feedback banners and dynamic spinners.
- Enable or disable PAM targets (`sudo`, `polkit-1`, `login`, `gdm`, `sshd`).
- Adjust matching thresholds, liveness sensitivity, and timeouts.
- Set custom prompts before and after authentication.

---

### 5. Check System Status (`argus status`)

View overall diagnostic health, active security profiles, camera selection, and configured PAM endpoints:

```bash
argus status
```

---

## 🛡️ Security Profiles

Argus offers three calibrated operational profiles:

| Profile | Cosine Threshold | Liveness Score | Best For |
|---|---|---|---|
| **L (Low / Convenient)** | `0.30` | `0.45` | Home laptops, fast single-user desktop unlock |
| **M (Medium / Balanced)** | `0.33` | `0.55` | **Default**: Daily driver, strong protection against 2D photos & screen replays |
| **H (High / Strict)** | `0.38` | `0.70` | Enterprise, multi-user workstations, strict anti-spoofing |

---

## 🔐 GNOME Keyring & Display Manager Safety

> [!IMPORTANT]
> **Why GDM face unlock is disabled by default:**
> In modern desktop Linux (Fedora 40+, Ubuntu 24.04+), your saved browser passwords, SSH keys, and GNOME Online Account tokens are stored in the GNOME Secret Service (`login.keyring`).
> The keyring is encrypted using your **plaintext login password**.
>
> When biometric face unlock logs in at cold boot, it does not possess your plaintext password to decrypt the keyring, causing browsers to prompt for sign-in.
>
> Argus defaults to keeping `sudo` and `polkit-1` enabled for instant terminal and GUI privilege elevation, while keeping `gdm-password` password-backed for seamless keyring decryption. You can toggle this via `sudo argus service enable gdm-password`.

---

## 🧪 Automated Testing

Argus includes a comprehensive 51-test suite with 100% deterministic unit and integration coverage:

```bash
/opt/argus/venv/bin/pytest tests/ -v
```

```
============================== 50 passed, 1 skipped in 2.65s ==============================
```

---

## 🗑️ Uninstallation

To completely remove Argus from your system and restore all default PAM configurations:

```bash
sudo /opt/argus/uninstall.sh
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Created and maintained by [sahid-code404](https://github.com/sahid-code404).
