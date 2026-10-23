// Host-to-device bandwidth on this link: pageable vs pinned vs cudaHostRegister.
// Ceiling for hypothesis 6 of docs/MOE-VIABLE-HYPOTHESES.md: the expert weights
// live in a plain pageable CPU buffer, and prefill streams ~18 GiB of them per
// ubatch at 6.56 GB/s (nsys). This says how much pinning could buy at most.
//
// build: nvcc -O2 -arch=sm_86 h2d_bw.cu -o h2d_bw.exe   (from a VS x64 prompt)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static double run(void * dst, const void * src, size_t chunk, size_t total, cudaStream_t s) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, s));
    size_t done = 0;
    for (size_t off = 0; off + chunk <= total; off += chunk, done += chunk) {
        CK(cudaMemcpyAsync(dst, (const char *) src + off, chunk, cudaMemcpyHostToDevice, s));
    }
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    return done / (ms * 1e-3) / 1e9;
}

int main(int argc, char ** argv) {
    const size_t total = (size_t) 2048 << 20;           // 2 GiB of host data
    const size_t chunks[] = { (size_t) 1 << 20, (size_t) 6 << 20, (size_t) 64 << 20 };  // ~1 expert tensor, a run, big
    void * dev; CK(cudaMalloc(&dev, (size_t) 64 << 20));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    char * pageable = (char *) malloc(total); memset(pageable, 1, total);
    char * pinned;   CK(cudaMallocHost(&pinned, total)); memset(pinned, 1, total);
    char * reg = (char *) malloc(total); memset(reg, 1, total);
    CK(cudaHostRegister(reg, total, cudaHostRegisterDefault));

    printf("%10s %10s %10s %10s\n", "chunk MiB", "pageable", "pinned", "registered");
    for (size_t c : chunks) {
        run(dev, pinned, c, total, s);  // warm
        double p0 = 0, p1 = 0, p2 = 0;
        for (int r = 0; r < 3; r++) {
            p0 += run(dev, pageable, c, total, s);
            p1 += run(dev, pinned, c, total, s);
            p2 += run(dev, reg, c, total, s);
        }
        printf("%10zu %10.2f %10.2f %10.2f   GB/s\n", c >> 20, p0 / 3, p1 / 3, p2 / 3);
    }
    return 0;
}
