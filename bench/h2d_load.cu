// Hypothesis No. 20: background PCIe load like the stock expert upload, to see
// how much a pageable host-to-device stream slows the CPU experts that a
// HybriMoE split would run at the same time. The upload copies from pageable
// memory, so the driver memcpys through a pinned staging buffer: DDR4 is read
// twice and written once per byte on the bus.
//
// Usage: h2d_load.exe <seconds> [chunk_MiB=2] [pinned=0] [blocking=0] [buffer_MiB=4096]
// build: bench\build_h2d_load.bat
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 30.0;
    const size_t chunk = (size_t) (argc > 2 ? atof(argv[2]) : 2.0) * (1 << 20);
    const bool pin = argc > 3 && atoi(argv[3]) != 0;
    // blocking = 1: the host thread sleeps in cudaStreamSynchronize instead of
    // spinning, so the load is the copy itself and not a busy core
    const bool blocking = argc > 4 && atoi(argv[4]) != 0;
    if (blocking) CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    // source buffer, MiB: 4 GiB by default (far bigger than L3, like the experts);
    // smaller to keep the load from competing for RAM with the process it disturbs
    const size_t total = (size_t) (argc > 5 ? atoi(argv[5]) : 4096) << 20;
    void * dev; CK(cudaMalloc(&dev, (size_t) 64 << 20));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    char * src;
    if (pin) { CK(cudaMallocHost(&src, total)); } else { src = (char *) malloc(total); }
    memset(src, 1, total);

    auto t0 = std::chrono::steady_clock::now();
    double done = 0, el = 0, last_done = 0, last_el = 0;
    size_t off = 0;
    while (el < secs) {
        for (int i = 0; i < 64; i++) {
            CK(cudaMemcpyAsync(dev, src + off, chunk, cudaMemcpyHostToDevice, s));
            off += chunk; if (off + chunk > total) off = 0;
            done += chunk;
        }
        CK(cudaStreamSynchronize(s));
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el - last_el >= 5.0) {  // a line every 5 s: the process is usually killed, not finished
            printf("%7.1f s  %.2f GB/s\n", el, (done - last_done) / (el - last_el) / 1e9);
            fflush(stdout);
            last_done = done; last_el = el;
        }
    }
    printf("h2d %s%s chunk %.1f MiB: %.2f GB/s over %.1f s\n", pin ? "pinned" : "pageable",
           blocking ? " blocking" : "",
           chunk / 1048576.0, done / el / 1e9, el);
    return 0;
}
