#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// Group size for residual quantization (matching HappyGuitar / SubSpec specification)
#define RESIDUAL_QUANT_GROUP_SIZE 128

/**
 * High-performance AVX2 + FMA3 CPU kernel for 8-bit residual matrix-vector multiplication.
 * 
 * Computes: Y = W_res * X  (accumulated across N tokens)
 * 
 * Parameters:
 *   W: int8 weight buffer in tile format [d_out / 16, d_in, 16]
 *   S: fp16 per-group scale buffer [d_out / 16, d_in / G, 16] (G = 128)
 *   d_out: output dimension (must be multiple of 16)
 *   d_in: input dimension (must be multiple of 128)
 *   x: input activations buffer [n_tokens, d_in]
 *   n_tokens: number of tokens (N = 1, 2, 4, 8, 16, 32...)
 *   y: output activations buffer [n_tokens, d_out]
 *   n_threads: number of CPU threads to utilize
 * 
 * Returns 0 on success, non-zero on error.
 */
int residual_gemv_i8_avx2(
    const int8_t * W,
    const uint16_t * S,
    int d_out,
    int d_in,
    const float * x,
    int n_tokens,
    float * y,
    int n_threads
);

/**
 * High-performance AVX2 + FMA3 CPU kernel for packed b-bit (2, 4, 6, 8 bit) residual GEMV.
 * 
 * Parameters:
 *   P: packed byte buffer [d_out / 16, d_in, 2 * b]
 *   S: fp16 per-group scale buffer [d_out / 16, d_in / G, 16]
 *   d_out: output dimension (multiple of 16)
 *   d_in: input dimension (multiple of 128)
 *   b: bit width (2, 4, 6, or 8)
 *   x: input activations [n_tokens, d_in]
 *   n_tokens: number of tokens
 *   y: output buffer [n_tokens, d_out]
 *   n_threads: thread count
 */
int residual_gemv_packed_avx2(
    const uint8_t * P,
    const uint16_t * S,
    int d_out,
    int d_in,
    int b,
    const float * x,
    int n_tokens,
    float * y,
    int n_threads
);

/**
 * Fast Walsh-Hadamard Transform (FWHT) in AVX2.
 * 
 * Performs orthogonal Hadamard rotation on input activations x.
 * Used to smooth outlier channels before residual decomposition:
 *   W_rot = H_out * W * H_in^T
 * 
 * Parameters:
 *   x: activation vector / matrix to rotate in-place [n_tokens, d]
 *   d: dimension of vector (must be factored into block_size chunks)
 *   n_tokens: number of tokens
 *   block_size: Hadamard block size (power of 2, typically 256, 512, 1024, or 2048)
 *   inverse: if true, applies inverse transform (same as forward with normalization)
 */
int fast_hadamard_transform_avx2(
    float * x,
    int d,
    int n_tokens,
    int block_size,
    bool inverse
);

/**
 * Memory bandwidth streaming test helper.
 * Measures sustained DRAM read bandwidth (GB/s) on the current CPU.
 */
double residual_measure_memory_bandwidth_gbs(size_t buffer_bytes, int n_threads);

#ifdef __cplusplus
}
#endif

// Performance benchmark result struct
struct residual_bench_result {
    int d_out;
    int d_in;
    int n_tokens;
    int bit_width;
    double time_ms;
    double gflops;
    double bandwidth_gbs;
    double latency_per_token_us;
};

// C++ API for running comprehensive kernel validation and profiling
std::vector<residual_bench_result> residual_benchmark_suite(
    const std::vector<std::pair<int, int>> & shapes,
    const std::vector<int> & token_batches,
    int bit_width,
    int n_threads
);
