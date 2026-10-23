#include "residual-cpu.h"

#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <future>
#include <iostream>
#include <algorithm>

#define G RESIDUAL_QUANT_GROUP_SIZE

// Helper: convert fp16 bits (uint16_t) to float32 using F16C or bit-level fallback
static inline float fp16_to_fp32_scalar(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x03ff);

    if (exp == 0) {
        if (mant == 0) {
            uint32_t f_bits = sign;
            float f;
            memcpy(&f, &f_bits, 4);
            return f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t f_bits = sign | 0x7f800000 | (mant << 13);
        float f;
        memcpy(&f, &f_bits, 4);
        return f;
    }

    exp = exp + (127 - 15);
    mant = mant << 13;
    uint32_t f_bits = sign | (exp << 23) | mant;
    float f;
    memcpy(&f, &f_bits, 4);
    return f;
}

// Convert 8 fp16 values into __m256 float vector
static inline __m256 load_fp16_to_f32_vec(const uint16_t * ptr) {
#if defined(__F16C__) || defined(_MSC_VER)
    __m128i h = _mm_loadu_si128((const __m128i *)ptr);
    return _mm256_cvtph_ps(h);
#else
    float tmp[8];
    for (int i = 0; i < 8; ++i) {
        tmp[i] = fp16_to_fp32_scalar(ptr[i]);
    }
    return _mm256_loadu_ps(tmp);
#endif
}

// Multi-thread dispatcher that works without requiring OpenMP (critical for GGML_OPENMP=OFF)
template <typename F>
static void parallel_for_tiles(int total_tiles, int n_threads, F && func) {
    if (n_threads <= 1 || total_tiles <= 1) {
        func(0, total_tiles);
        return;
    }

    n_threads = std::min(n_threads, total_tiles);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    int chunk = (total_tiles + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, total_tiles);
        if (start < end) {
            workers.emplace_back([&func, start, end]() {
                func(start, end);
            });
        }
    }

    for (auto & w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

// Micro-kernel for 8-bit residual GEMV on a tile slice [tile_start, tile_end)
// Unrolls over N tokens simultaneously to achieve memory-bandwidth saturation
template <int N>
static void kernel_i8_tile_range(
    const int8_t * W,
    const uint16_t * S,
    int d_out,
    int d_in,
    const float * x,
    float * y,
    int tile_start,
    int tile_end
) {
    const int ng = d_in / G;

    for (int tile = tile_start; tile < tile_end; ++tile) {
        __m256 acc0[N];
        __m256 acc1[N];

        for (int t = 0; t < N; ++t) {
            acc0[t] = _mm256_setzero_ps();
            acc1[t] = _mm256_setzero_ps();
        }

        const int8_t   * wt = W + (size_t)tile * d_in * 16;
        const uint16_t * st = S + (size_t)tile * ng * 16;

        for (int g = 0; g < ng; ++g) {
            const __m256 sv0 = load_fp16_to_f32_vec(st + (size_t)g * 16);
            const __m256 sv1 = load_fp16_to_f32_vec(st + (size_t)g * 16 + 8);

            const int i_start = g * G;
            const int i_end   = i_start + G;

            for (int i = i_start; i < i_end; ++i) {
                const int8_t * wt_i = wt + (size_t)i * 16;

                // Load 8 int8 values for rows 0..7 and 8..15
                __m128i raw0 = _mm_loadl_epi64((const __m128i *)wt_i);
                __m128i raw1 = _mm_loadl_epi64((const __m128i *)(wt_i + 8));

                // Sign-extend int8 to int32, convert to float, scale
                __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw0)), sv0);
                __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw1)), sv1);

                // Multiply-accumulate across all N tokens
                for (int t = 0; t < N; ++t) {
                    float xt = x[(size_t)t * d_in + i];
                    __m256 xv = _mm256_set1_ps(xt);
                    acc0[t] = _mm256_fmadd_ps(w0, xv, acc0[t]);
                    acc1[t] = _mm256_fmadd_ps(w1, xv, acc1[t]);
                }
            }
        }

        for (int t = 0; t < N; ++t) {
            _mm256_storeu_ps(y + (size_t)t * d_out + (size_t)tile * 16,     acc0[t]);
            _mm256_storeu_ps(y + (size_t)t * d_out + (size_t)tile * 16 + 8, acc1[t]);
        }
    }
}

// Dispatcher for N tokens in 8-bit residual GEMV
static void residual_i8_dispatch(
    const int8_t * W,
    const uint16_t * S,
    int d_out,
    int d_in,
    const float * x,
    int n,
    float * y,
    int n_threads
) {
    const int nt = d_out / 16;

    int t_offset = 0;
    while (t_offset < n) {
        int chunk = n - t_offset;
        if (chunk >= 16) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_i8_tile_range<16>(W, S, d_out, d_in, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 16;
        } else if (chunk >= 8) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_i8_tile_range<8>(W, S, d_out, d_in, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 8;
        } else if (chunk >= 4) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_i8_tile_range<4>(W, S, d_out, d_in, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 4;
        } else if (chunk >= 2) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_i8_tile_range<2>(W, S, d_out, d_in, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 2;
        } else {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_i8_tile_range<1>(W, S, d_out, d_in, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 1;
        }
    }
}

int residual_gemv_i8_avx2(
    const int8_t * W,
    const uint16_t * S,
    int d_out,
    int d_in,
    const float * x,
    int n_tokens,
    float * y,
    int n_threads
) {
    if (d_out % 16 != 0 || d_in % RESIDUAL_QUANT_GROUP_SIZE != 0) {
        return -1;
    }
    if (n_tokens <= 0 || n_threads <= 0) {
        return -2;
    }

    residual_i8_dispatch(W, S, d_out, d_in, x, n_tokens, y, n_threads);
    return 0;
}

// Unpack helper: unpack 16 b-bit values into int8_t buffer
static inline void unpack_16_bbit(const uint8_t * src, int b, int8_t * dst) {
    const int offset = 1 << (b - 1);
    const int mask = (1 << b) - 1;

    if (b == 4) {
        // 8 bytes contain 16 nibbles
        for (int j = 0; j < 8; ++j) {
            uint8_t byte = src[j];
            dst[2 * j]     = (int8_t)((byte & 0x0F) - offset);
            dst[2 * j + 1] = (int8_t)(((byte >> 4) & 0x0F) - offset);
        }
    } else if (b == 6) {
        // 12 bytes contain 16 6-bit values = 96 bits
        // Bit extraction
        uint32_t bit_pos = 0;
        for (int j = 0; j < 16; ++j) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_rem  = bit_pos % 8;
            uint32_t chunk = src[byte_idx] | ((uint32_t)src[byte_idx + 1] << 8);
            if (bit_rem + 6 > 16) {
                chunk |= ((uint32_t)src[byte_idx + 2] << 16);
            }
            uint32_t val = (chunk >> bit_rem) & mask;
            dst[j] = (int8_t)((int)val - offset);
            bit_pos += 6;
        }
    } else if (b == 2) {
        // 4 bytes contain 16 2-bit values
        for (int j = 0; j < 4; ++j) {
            uint8_t byte = src[j];
            dst[4 * j]     = (int8_t)((byte & 0x03) - offset);
            dst[4 * j + 1] = (int8_t)(((byte >> 2) & 0x03) - offset);
            dst[4 * j + 2] = (int8_t)(((byte >> 4) & 0x03) - offset);
            dst[4 * j + 3] = (int8_t)(((byte >> 6) & 0x03) - offset);
        }
    } else {
        // Generic fallback for any bit width
        uint32_t bit_pos = 0;
        for (int j = 0; j < 16; ++j) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_rem  = bit_pos % 8;
            uint32_t chunk = src[byte_idx] | ((uint32_t)src[byte_idx + 1] << 8) | ((uint32_t)src[byte_idx + 2] << 16);
            uint32_t val = (chunk >> bit_rem) & mask;
            dst[j] = (int8_t)((int)val - offset);
            bit_pos += b;
        }
    }
}

// Micro-kernel for packed b-bit residual GEMV on a tile slice
template <int N>
static void kernel_packed_tile_range(
    const uint8_t * P,
    const uint16_t * S,
    int d_out,
    int d_in,
    int b,
    const float * x,
    float * y,
    int tile_start,
    int tile_end
) {
    const int ng = d_in / G;
    const int cb = 2 * b; // bytes per 16-row column

    for (int tile = tile_start; tile < tile_end; ++tile) {
        __m256 acc0[N];
        __m256 acc1[N];

        for (int t = 0; t < N; ++t) {
            acc0[t] = _mm256_setzero_ps();
            acc1[t] = _mm256_setzero_ps();
        }

        const uint8_t  * pt = P + (size_t)tile * d_in * cb;
        const uint16_t * st = S + (size_t)tile * ng * 16;

        for (int g = 0; g < ng; ++g) {
            const __m256 sv0 = load_fp16_to_f32_vec(st + (size_t)g * 16);
            const __m256 sv1 = load_fp16_to_f32_vec(st + (size_t)g * 16 + 8);

            const int i_start = g * G;
            const int i_end   = i_start + G;

            for (int i = i_start; i < i_end; ++i) {
                const uint8_t * col_bytes = pt + (size_t)i * cb;

                int8_t unpacked[16];
                unpack_16_bbit(col_bytes, b, unpacked);

                __m128i raw0 = _mm_loadl_epi64((const __m128i *)unpacked);
                __m128i raw1 = _mm_loadl_epi64((const __m128i *)(unpacked + 8));

                __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw0)), sv0);
                __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw1)), sv1);

                for (int t = 0; t < N; ++t) {
                    float xt = x[(size_t)t * d_in + i];
                    __m256 xv = _mm256_set1_ps(xt);
                    acc0[t] = _mm256_fmadd_ps(w0, xv, acc0[t]);
                    acc1[t] = _mm256_fmadd_ps(w1, xv, acc1[t]);
                }
            }
        }

        for (int t = 0; t < N; ++t) {
            _mm256_storeu_ps(y + (size_t)t * d_out + (size_t)tile * 16,     acc0[t]);
            _mm256_storeu_ps(y + (size_t)t * d_out + (size_t)tile * 16 + 8, acc1[t]);
        }
    }
}

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
) {
    if (b < 2 || b > 8) {
        return -1;
    }
    if (d_out % 16 != 0 || d_in % G != 0) {
        return -2;
    }
    if (n_tokens <= 0 || n_threads <= 0) {
        return -3;
    }

    const int nt = d_out / 16;
    int t_offset = 0;

    while (t_offset < n_tokens) {
        int chunk = n_tokens - t_offset;
        if (chunk >= 16) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_packed_tile_range<16>(P, S, d_out, d_in, b, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 16;
        } else if (chunk >= 8) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_packed_tile_range<8>(P, S, d_out, d_in, b, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 8;
        } else if (chunk >= 4) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_packed_tile_range<4>(P, S, d_out, d_in, b, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 4;
        } else if (chunk >= 2) {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_packed_tile_range<2>(P, S, d_out, d_in, b, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 2;
        } else {
            parallel_for_tiles(nt, n_threads, [&](int t_start, int t_end) {
                kernel_packed_tile_range<1>(P, S, d_out, d_in, b, x + (size_t)t_offset * d_in, y + (size_t)t_offset * d_out, t_start, t_end);
            });
            t_offset += 1;
        }
    }

    return 0;
}

// In-place Fast Walsh-Hadamard Transform (FWHT) for a single block of size B (power of 2)
static void fwht_block_avx2(float * x, int B, float inv_sqrt2) {
    const __m256 norm = _mm256_set1_ps(inv_sqrt2);

    for (int h = 1; h < B; h *= 2) {
        int step = 2 * h;

        if (h >= 8) {
            for (int i = 0; i < B; i += step) {
                for (int j = 0; j < h; j += 8) {
                    __m256 u = _mm256_loadu_ps(x + i + j);
                    __m256 v = _mm256_loadu_ps(x + i + j + h);

                    __m256 add = _mm256_mul_ps(_mm256_add_ps(u, v), norm);
                    __m256 sub = _mm256_mul_ps(_mm256_sub_ps(u, v), norm);

                    _mm256_storeu_ps(x + i + j,     add);
                    _mm256_storeu_ps(x + i + j + h, sub);
                }
            }
        } else {
            // Scalar butterfly for small stages h < 8
            for (int i = 0; i < B; i += step) {
                for (int j = 0; j < h; ++j) {
                    float u = x[i + j];
                    float v = x[i + j + h];
                    x[i + j]     = (u + v) * inv_sqrt2;
                    x[i + j + h] = (u - v) * inv_sqrt2;
                }
            }
        }
    }
}

int fast_hadamard_transform_avx2(
    float * x,
    int d,
    int n_tokens,
    int block_size,
    bool /*inverse*/
) {
    if (d % block_size != 0 || (block_size & (block_size - 1)) != 0) {
        return -1; // block_size must be a power of 2 and divide d
    }

    const float inv_sqrt2 = 1.0f / 1.41421356237309504880f;
    const int n_blocks = d / block_size;

    for (int t = 0; t < n_tokens; ++t) {
        float * xt = x + (size_t)t * d;
        for (int b = 0; b < n_blocks; ++b) {
            fwht_block_avx2(xt + (size_t)b * block_size, block_size, inv_sqrt2);
        }
    }

    return 0;
}

double residual_measure_memory_bandwidth_gbs(size_t buffer_bytes, int n_threads) {
    std::vector<int8_t> buffer(buffer_bytes, 1);

    const size_t chunk_size = 4096;
    const size_t n_chunks = buffer_bytes / chunk_size;

    auto t0 = std::chrono::high_resolution_clock::now();

    volatile int64_t sum = 0;

    parallel_for_tiles((int)n_chunks, n_threads, [&](int c_start, int c_end) {
        int64_t local_sum = 0;
        for (int c = c_start; c < c_end; ++c) {
            const int8_t * p = buffer.data() + (size_t)c * chunk_size;
            for (size_t j = 0; j < chunk_size; j += 64) {
                __m256i v0 = _mm256_loadu_si256((const __m256i *)(p + j));
                __m256i v1 = _mm256_loadu_si256((const __m256i *)(p + j + 32));
                local_sum += _mm256_extract_epi32(v0, 0) + _mm256_extract_epi32(v1, 0);
            }
        }
        sum += local_sum;
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    double gb_transferred = (double)buffer_bytes / (1024.0 * 1024.0 * 1024.0);
    return gb_transferred / elapsed_sec;
}

std::vector<residual_bench_result> residual_benchmark_suite(
    const std::vector<std::pair<int, int>> & shapes,
    const std::vector<int> & token_batches,
    int bit_width,
    int n_threads
) {
    std::vector<residual_bench_result> results;

    for (const auto & shape : shapes) {
        int d_out = shape.first;
        int d_in  = shape.second;

        size_t w_bytes = (bit_width == 8) 
            ? ((size_t)d_out * d_in) 
            : ((size_t)d_out * d_in * bit_width / 8);
        size_t s_bytes = ((size_t)d_out * (d_in / G)) * sizeof(uint16_t);

        std::vector<uint8_t> W(w_bytes, 3);
        std::vector<uint16_t> S(d_out * (d_in / G), 0x3C00); // 1.0 in fp16

        for (int n_tok : token_batches) {
            std::vector<float> X((size_t)n_tok * d_in, 0.5f);
            std::vector<float> Y((size_t)n_tok * d_out, 0.0f);

            // Warmup
            if (bit_width == 8) {
                residual_gemv_i8_avx2((const int8_t *)W.data(), S.data(), d_out, d_in, X.data(), n_tok, Y.data(), n_threads);
            } else {
                residual_gemv_packed_avx2(W.data(), S.data(), d_out, d_in, bit_width, X.data(), n_tok, Y.data(), n_threads);
            }

            int n_iters = (n_tok >= 16) ? 5 : 15;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int it = 0; it < n_iters; ++it) {
                if (bit_width == 8) {
                    residual_gemv_i8_avx2((const int8_t *)W.data(), S.data(), d_out, d_in, X.data(), n_tok, Y.data(), n_threads);
                } else {
                    residual_gemv_packed_avx2(W.data(), S.data(), d_out, d_in, bit_width, X.data(), n_tok, Y.data(), n_threads);
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double avg_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iters;

            double flops = 2.0 * (double)d_out * (double)d_in * (double)n_tok;
            double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
            double bw_gbs = ((double)(w_bytes + s_bytes) / (avg_time_ms * 1e-3)) / (1024.0 * 1024.0 * 1024.0);
            double lat_per_tok = (avg_time_ms * 1000.0) / n_tok;

            results.push_back({
                d_out,
                d_in,
                n_tok,
                bit_width,
                avg_time_ms,
                gflops,
                bw_gbs,
                lat_per_tok
            });
        }
    }

    return results;
}
