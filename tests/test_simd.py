"""Unit tests and benchmarks for native C SIMD accelerator."""

import numpy as np
import pytest
from faceunlock.simd_accel import (
    native_cosine_similarity,
    native_batch_match,
    native_zero_wipe,
    _load_simd_lib,
)


def test_simd_library_loaded():
    lib = _load_simd_lib()
    assert lib is not None, "Native SIMD library failed to load!"


def test_simd_cosine_similarity_accuracy():
    np.random.seed(42)
    for _ in range(50):
        v1 = np.random.randn(128).astype(np.float32)
        v2 = np.random.randn(128).astype(np.float32)

        c_sim = native_cosine_similarity(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        py_sim = float(np.dot(v1 / n1, v2 / n2))

        diff = abs(c_sim - py_sim)
        assert diff <= 0.0001


def test_simd_batch_match_accuracy():
    np.random.seed(1337)
    enrolled = np.random.randn(10, 128).astype(np.float32)
    norms = np.linalg.norm(enrolled, axis=1, keepdims=True)
    enrolled /= norms

    live = enrolled[4].copy() # Perfect match with candidate 4
    best_idx, best_score = native_batch_match(enrolled, live)
    assert best_idx == 4
    assert best_score >= 0.9999


def test_secure_zero_memory_wipe():
    arr = np.ones(256, dtype=np.float32)
    native_zero_wipe(arr)
    assert np.all(arr == 0.0)
