#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

#define G 128

#define KERNEL_I8(N)                                                                                                  \
static void kernel_i8_##N(const int8_t* W, const uint16_t* S, int d_out, int d_in, const float* x, float* y, int nth) {  \
    const int nt = d_out / 16, ng = d_in / G;                                                                          \
    _Pragma("omp parallel for num_threads(nth) schedule(static)")                                                      \
    for (int tile = 0; tile < nt; tile++) {                                                                            \
        __m512 acc[N];                                                                                                 \
        for (int t = 0; t < N; t++) acc[t] = _mm512_setzero_ps();                                                      \
        const int8_t* wt = W + (size_t)tile * d_in * 16;                                                               \
        const uint16_t* st = S + (size_t)tile * ng * 16;                                                               \
        for (int g = 0; g < ng; g++) {                                                                                 \
            const __m512 sv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(st + g * 16)));                      \
            for (int i = g * G; i < (g + 1) * G; i++) {                                                                \
                const __m512 w = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(                                \
                                     _mm_loadu_si128((const __m128i*)(wt + (size_t)i * 16)))), sv);                    \
                for (int t = 0; t < N; t++) acc[t] = _mm512_fmadd_ps(w, _mm512_set1_ps(x[(size_t)t * d_in + i]), acc[t]); \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < N; t++) _mm512_storeu_ps(y + (size_t)t * d_out + tile * 16, acc[t]);                       \
    }                                                                                                                  \
}

KERNEL_I8(1) KERNEL_I8(2) KERNEL_I8(4) KERNEL_I8(8) KERNEL_I8(12) KERNEL_I8(16) KERNEL_I8(24) KERNEL_I8(32)

int residual_i8(const int8_t* W, const uint16_t* S, int d_out, int d_in, const float* x, int n, float* y, int nth) {
    switch (n) {
        case 1: kernel_i8_1(W, S, d_out, d_in, x, y, nth); return 0;
        case 2: kernel_i8_2(W, S, d_out, d_in, x, y, nth); return 0;
        case 4: kernel_i8_4(W, S, d_out, d_in, x, y, nth); return 0;
        case 8: kernel_i8_8(W, S, d_out, d_in, x, y, nth); return 0;
        case 12: kernel_i8_12(W, S, d_out, d_in, x, y, nth); return 0;
        case 16: kernel_i8_16(W, S, d_out, d_in, x, y, nth); return 0;
        case 24: kernel_i8_24(W, S, d_out, d_in, x, y, nth); return 0;
        case 32: kernel_i8_32(W, S, d_out, d_in, x, y, nth); return 0;
    }
    return -1;
}

int64_t stream_read(const int8_t* buf, size_t nbytes, int nth) {
    int64_t total = 0;
    _Pragma("omp parallel for num_threads(nth) reduction(+:total) schedule(static)")
    for (size_t off = 0; off < nbytes; off += 4096) {
        __m512i a = _mm512_setzero_si512();
        size_t end = off + 4096 < nbytes ? off + 4096 : nbytes;
        for (size_t j = off; j + 64 <= end; j += 64) a = _mm512_add_epi8(a, _mm512_loadu_si512((const void*)(buf + j)));
        total += (int64_t)_mm512_reduce_add_epi32(_mm512_and_si512(a, _mm512_set1_epi32(1)));
    }
    return total;
}

#define KERNEL_PACKED(N)                                                                                              \
static void kernel_packed_##N(const uint8_t* P, const uint16_t* S, int d_out, int d_in, int b, const float* x, float* y, int nth) { \
    const int nt = d_out / 16, ng = d_in / G, cb = 2 * b;                                                              \
    uint8_t perm_a[16], sh_a[16];                                                                                      \
    for (int j = 0; j < 8; j++) { perm_a[j] = (uint8_t)j; perm_a[8 + j] = (uint8_t)(b + j); sh_a[j] = sh_a[8 + j] = (uint8_t)(j * b); } \
    const __m128i perm = _mm_loadu_si128((const __m128i*)perm_a), shv = _mm_loadu_si128((const __m128i*)sh_a);        \
    const __m128i mask = _mm_set1_epi8((char)((1 << b) - 1)), off = _mm_set1_epi8((char)(1 << (b - 1)));               \
    _Pragma("omp parallel for num_threads(nth) schedule(static)")                                                      \
    for (int tile = 0; tile < nt; tile++) {                                                                            \
        __m512 acc[N];                                                                                                 \
        for (int t = 0; t < N; t++) acc[t] = _mm512_setzero_ps();                                                      \
        const uint8_t* pt = P + (size_t)tile * d_in * cb;                                                              \
        const uint16_t* st = S + (size_t)tile * ng * 16;                                                               \
        for (int g = 0; g < ng; g++) {                                                                                 \
            const __m512 sv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(st + g * 16)));                      \
            for (int i = g * G; i < (g + 1) * G; i++) {                                                                \
                __m128i raw = _mm_loadu_si128((const __m128i*)(pt + (size_t)i * cb));                                  \
                __m128i u = _mm_and_si128(_mm_multishift_epi64_epi8(shv, _mm_permutexvar_epi8(perm, raw)), mask);     \
                const __m512 w = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_sub_epi8(u, off))), sv);    \
                for (int t = 0; t < N; t++) acc[t] = _mm512_fmadd_ps(w, _mm512_set1_ps(x[(size_t)t * d_in + i]), acc[t]); \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < N; t++) _mm512_storeu_ps(y + (size_t)t * d_out + tile * 16, acc[t]);                       \
    }                                                                                                                  \
}

KERNEL_PACKED(1) KERNEL_PACKED(2) KERNEL_PACKED(4) KERNEL_PACKED(8) KERNEL_PACKED(12) KERNEL_PACKED(16) KERNEL_PACKED(24) KERNEL_PACKED(32)

int residual_packed(const uint8_t* P, const uint16_t* S, int d_out, int d_in, int b, const float* x, int n, float* y, int nth) {
    if (b < 2 || b > 8) return -2;
    switch (n) {
        case 1: kernel_packed_1(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 2: kernel_packed_2(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 4: kernel_packed_4(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 8: kernel_packed_8(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 12: kernel_packed_12(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 16: kernel_packed_16(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 24: kernel_packed_24(P, S, d_out, d_in, b, x, y, nth); return 0;
        case 32: kernel_packed_32(P, S, d_out, d_in, b, x, y, nth); return 0;
    }
    return -1;
}
