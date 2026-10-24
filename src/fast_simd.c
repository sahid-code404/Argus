/**
 * Linux Face Unlock - High Performance SIMD C Accelerator
 *
 * Implements AVX2/FMA vector dot products, cosine similarities,
 * and secure zero-wipe routines for biometric operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/**
 * Compute dot product of two float arrays of length n.
 * Uses AVX2 FMA (8 floats per cycle) when available.
 */
float simd_dot_product(const float* a, const float* b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    int i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }

    // Horizontal sum of 8 floats
    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);

    // Remaining elements
    for (; i < n; i++) {
        total += a[i] * b[i];
    }
    return total;
#else
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        total += a[i] * b[i];
    }
    return total;
#endif
}

/**
 * Compute cosine similarity between two float vectors of length n.
 * Calculates dot(a, b) / (norm(a) * norm(b)) in a single pass.
 */
float simd_cosine_similarity(const float* a, const float* b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    int i = 0;
    __m256 dot_acc = _mm256_setzero_ps();
    __m256 norm_a_acc = _mm256_setzero_ps();
    __m256 norm_b_acc = _mm256_setzero_ps();

    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    // Horizontal reductions
    __m128 d_lo = _mm256_castps256_ps128(dot_acc);
    __m128 d_hi = _mm256_extractf128_ps(dot_acc, 1);
    __m128 d_sum = _mm_add_ps(d_lo, d_hi);
    d_sum = _mm_hadd_ps(d_sum, d_sum);
    d_sum = _mm_hadd_ps(d_sum, d_sum);
    float dot = _mm_cvtss_f32(d_sum);

    __m128 na_lo = _mm256_castps256_ps128(norm_a_acc);
    __m128 na_hi = _mm256_extractf128_ps(norm_a_acc, 1);
    __m128 na_sum = _mm_add_ps(na_lo, na_hi);
    na_sum = _mm_hadd_ps(na_sum, na_sum);
    na_sum = _mm_hadd_ps(na_sum, na_sum);
    float norm_a_sq = _mm_cvtss_f32(na_sum);

    __m128 nb_lo = _mm256_castps256_ps128(norm_b_acc);
    __m128 nb_hi = _mm256_extractf128_ps(norm_b_acc, 1);
    __m128 nb_sum = _mm_add_ps(nb_lo, nb_hi);
    nb_sum = _mm_hadd_ps(nb_sum, nb_sum);
    nb_sum = _mm_hadd_ps(nb_sum, nb_sum);
    float norm_b_sq = _mm_cvtss_f32(nb_sum);

    for (; i < n; i++) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    float denom = sqrtf(norm_a_sq) * sqrtf(norm_b_sq);
    if (denom < 1e-7f) {
        return 0.0f;
    }
    return dot / denom;
#else
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 1e-7f) ? (dot / denom) : 0.0f;
#endif
}

/**
 * Batch match: compare 1 live vector against M enrolled vectors (each of length D).
 * Writes similarities to out_sims[M]. Returns index of maximum similarity.
 */
int simd_batch_match(const float* enrolled, const float* live, float* out_sims, int m, int d, float* out_best_score) {
    int best_idx = -1;
    float max_score = -1.0f;

    for (int i = 0; i < m; i++) {
        const float* cand = enrolled + (i * d);
        float sim = simd_cosine_similarity(cand, live, d);
        if (out_sims != NULL) {
            out_sims[i] = sim;
        }
        if (sim > max_score) {
            max_score = sim;
            best_idx = i;
        }
    }

    if (out_best_score != NULL) {
        *out_best_score = max_score;
    }
    return best_idx;
}

/**
 * Matrix multiplication: C = A (M x K) * B (K x N).
 */
void simd_matrix_multiply(const float* a, const float* b, float* c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/**
 * Cryptographic memory wipe: zero out memory buffer using volatile pointer
 * to prevent compiler dead-store elimination.
 */
void secure_zero_memory(void* ptr, size_t len) {
    if (ptr == NULL || len == 0) return;
#if defined(__STDC_LIB_EXT1__)
    memset_s(ptr, len, 0, len);
#else
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    while (len--) {
        *p++ = 0;
    }
#endif
}
