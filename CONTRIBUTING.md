# Contributing to Argus

Thank you for your interest in improving **Argus**! We welcome contributions, bug reports, feature requests, and hardware compatibility feedback.

---

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sahid-code404/Argus.git
   cd Argus
   ```

2. **Create a development virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install pytest flake8
   ```

3. **Build the SIMD Acceleration Library**:
   ```bash
   gcc -O3 -shared -fPIC -mavx2 -mfma -o argus/libargus_simd.so src/fast_simd.c
   cp argus/libargus_simd.so faceunlock/libargus_simd.so
   cp argus/libargus_simd.so faceunlock/libfaceunlock_simd.so
   ```

4. **Run the Test Suite**:
   ```bash
   pytest tests/ -v
   ```

---

## 📋 Contribution Guidelines

- **Hardware Testing**: If you test Argus on laptops with dedicated IR cameras (Windows Hello), RealSense depth cameras, or unique fingerprint sensors, please share your `argus probe` diagnostics in a GitHub Issue!
- **Zero Cloud / Total Privacy**: Argus strictly runs locally on your machine. All embeddings and neural networks must execute locally on CPU/NPU without cloud dependencies.
- **Fail-Safe PAM Semantics**: Authentication fallbacks to password or fingerprint must always remain rock-solid. A biometric failure or camera timeout must never lock a user out of their system.
- **Tests**: Ensure any new features or bug fixes are accompanied by unit tests under `tests/`.

---

## 📄 Code of Conduct

Please maintain a welcoming, respectful, and collaborative environment. Be kind to fellow contributors and users.
