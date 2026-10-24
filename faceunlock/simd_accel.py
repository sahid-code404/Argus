"""Python ctypes binding for the native C SIMD accelerator."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

_LIB: Optional[ctypes.CDLL] = None

def _load_simd_lib() -> Optional[ctypes.CDLL]:
    global _LIB
    if _LIB is not None:
        return _LIB

    candidates = [
        Path(__file__).resolve().parent / "libfaceunlock_simd.so",
        Path("/opt/faceunlock/faceunlock/libfaceunlock_simd.so"),
        Path("/usr/lib64/libfaceunlock_simd.so"),
        Path("/usr/lib/libfaceunlock_simd.so"),
    ]
    for c in candidates:
        if c.exists():
            try:
                lib = ctypes.CDLL(str(c))
                # Function signatures
                lib.simd_dot_product.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                lib.simd_dot_product.restype = ctypes.c_float

                lib.simd_cosine_similarity.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                lib.simd_cosine_similarity.restype = ctypes.c_float

                lib.simd_batch_match.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                ]
                lib.simd_batch_match.restype = ctypes.c_int

                lib.secure_zero_memory.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                lib.secure_zero_memory.restype = None

                _LIB = lib
                return _LIB
            except Exception:
                pass
    return None


def native_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity using native AVX2 SIMD if available, falling back to NumPy."""
    lib = _load_simd_lib()
    if lib is not None:
        f1 = np.ascontiguousarray(v1.flatten(), dtype=np.float32)
        f2 = np.ascontiguousarray(v2.flatten(), dtype=np.float32)
        if len(f1) == len(f2) and len(f1) > 0:
            p1 = f1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            p2 = f2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return float(lib.simd_cosine_similarity(p1, p2, len(f1)))

    # Fallback
    f1 = v1.flatten()
    f2 = v2.flatten()
    n1 = np.linalg.norm(f1)
    n2 = np.linalg.norm(f2)
    if n1 > 1e-7 and n2 > 1e-7:
        return float(np.dot(f1 / n1, f2 / n2))
    return 0.0


def native_batch_match(enrolled_matrix: np.ndarray, live_vec: np.ndarray) -> Tuple[int, float]:
    """Batch match live vector against M enrolled vectors in AVX2 C.

    Returns (best_index, max_similarity).
    """
    lib = _load_simd_lib()
    m, d = enrolled_matrix.shape
    if lib is not None and m > 0 and d > 0:
        mat = np.ascontiguousarray(enrolled_matrix, dtype=np.float32)
        live = np.ascontiguousarray(live_vec.flatten(), dtype=np.float32)
        sims = np.zeros(m, dtype=np.float32)

        p_mat = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_live = live.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_sims = sims.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        best_score = ctypes.c_float(0.0)

        best_idx = lib.simd_batch_match(
            p_mat, p_live, p_sims, m, d, ctypes.byref(best_score)
        )
        return int(best_idx), float(best_score.value)

    # Fallback to NumPy SIMD dot
    dot_products = np.dot(enrolled_matrix, live_vec.flatten())
    best_idx = int(np.argmax(dot_products))
    return best_idx, float(dot_products[best_idx])


def native_zero_wipe(arr: np.ndarray) -> None:
    """Zero-fill array in memory with volatile memory fence."""
    lib = _load_simd_lib()
    if lib is not None and arr.size > 0:
        lib.secure_zero_memory(arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
    else:
        arr.fill(0)
