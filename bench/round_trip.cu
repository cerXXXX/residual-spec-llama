// GPU->CPU->GPU round trip on this machine (WDDM): the filter for hypothesis No. 23
// of docs/MOE-NEW-HYPOTHESES.md. A Tiel decode token is 41 such rounds: a CUDA split
// (attention, router; a CUDA graph of ~34 kernels) -> its outputs go to the host ->
// the CPU computes the experts -> the result goes back -> next CUDA split.
//
// Modes, all with the same token shape (R rounds, NK kernels of GPU work per round,
// GPU_US of GPU work and CPU_US of CPU work per round):
//   A   stock path of ggml_backend_sched_compute_splits on one CUDA device:
//       cudaGraphLaunch(split) -> cudaStreamSynchronize -> per input
//       cudaMemcpyAsync D2H on cudaStreamPerThread + sync (buffer_get_tensor) ->
//       CPU work -> cudaStreamSynchronize -> H2D + sync (buffer_set_tensor)
//   A1  same with one D2H per round (variant a: pack the split inputs)
//   B   variant b: the whole token is ONE CUDA graph. A kernel writes the activation
//       into mapped pinned memory and raises a flag; the host spins on the flag,
//       does the CPU work, writes the result into mapped memory and raises its flag;
//       a spinner kernel waits for it and pulls the result in.
//   B2  as B, but the flags are cuStreamWriteValue32 / cuStreamWaitValue32 nodes
//       (stream memory operations) instead of kernels — if WDDM supports them.
//   C   variant c (KTransformers): one CUDA graph with host nodes doing the CPU work.
//
// Per-round overhead = (token time - G) / R - CPU_US, where G is the same R x NK
// work kernels as one graph with no host involvement (the work is serial in every
// mode by data dependence, so this is what the host round trip adds).
//
// build: bench\build_round_trip.bat
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <windows.h>
#include <intrin.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define CKD(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char * s = nullptr; cuGetErrorString(e, &s); \
    printf("%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, s ? s : "?"); exit(1); } } while (0)

static int R      = 41;    // rounds per token (Tiel: 40 expert layers + head)
static int NK     = 34;    // kernels per CUDA split (1394 kernels / 41 splits, S0-04)
static int GPU_US = 0;     // GPU work per round, us (Tiel: 13.06 ms / 41 = 318)
static int CPU_US = 0;     // CPU work per round, us (Tiel: 17.5 ms / 41 = 427)
static int N_EMBD = 2048;  // Tiel hidden size, f32 activation = 8 KiB
static int TOKENS = 200;
static int WARM   = 20;
static int CORE   = 0;     // logical CPU the host thread is pinned to (-1: not pinned)

static const int NMAX = 4096;

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void k_work(float * buf, unsigned ns) {
    if (ns && threadIdx.x == 0) { unsigned long long t0 = gtime(); while (gtime() - t0 < ns) {} }
    __syncthreads();
    buf[threadIdx.x] += 1.0f;
}

// activation -> mapped host memory, then (optionally) the flag
__global__ void k_publish(const float * act, float * hact, int n, volatile unsigned * gflag, const unsigned * tok) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) hact[i] = act[i];
    __threadfence_system();
    __syncthreads();
    if (gflag && threadIdx.x == 0) *gflag = *tok + 1;
}

// wait for the host flag (if any), then pull the host result into device memory;
// hres[0] carries a tag the host wrote, checked against what we expect
__global__ void k_consume(volatile const unsigned * cflag, volatile const float * hres, float * res, int n,
                          const unsigned * tok, int round, unsigned * err) {
    if (threadIdx.x == 0) {
        if (cflag) { const unsigned want = *tok + 1; while (*cflag != want) {} }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) res[i] = hres[i];
    if (threadIdx.x == 0 && hres[0] != (float) ((*tok % 1000) * 64 + round)) atomicAdd(err, 1u);
}

__global__ void k_tok(unsigned * tok) { (*tok)++; }

static double now_us() {
    static LARGE_INTEGER f = {}; if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER c; QueryPerformanceCounter(&c); return c.QuadPart * 1e6 / f.QuadPart;
}
static void cpu_work(int us) {
    if (us <= 0) return;
    const double t0 = now_us(); while (now_us() - t0 < us) {}
}

struct bufs {
    float * dact, * dres, * dwork;      // device
    float * hact, * hres;               // mapped pinned, R * n each
    float * pin_in, * pin_out;          // plain pinned (stock path: CUDA_Host compute buffer)
    volatile unsigned * gflag, * cflag; // mapped pinned, R each
    unsigned * dtok, * derr;            // device
    unsigned htok;                      // host mirror of dtok
};

static void add_work(cudaStream_t s, bufs & b) {
    for (int k = 0; k < NK; k++) k_work<<<1, 128, 0, s>>>(b.dwork, (unsigned) (GPU_US * 1000 / NK));
}

static cudaGraphExec_t capture(cudaStream_t s, void (*fill)(cudaStream_t, bufs &), bufs & b) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    fill(s, b);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    return ge;
}

// ---- A / A1: stock ----
static cudaGraphExec_t g_split;
static void fill_split(cudaStream_t s, bufs & b) { add_work(s, b); }

static double token_stock(cudaStream_t s, bufs & b, int n_d2h) {
    const double t0 = now_us();
    const size_t sz[2] = { (size_t) N_EMBD * 4, 32 };   // activation, ids
    for (int r = 0; r < R; r++) {
        CK(cudaGraphLaunch(g_split, s));
        CK(cudaStreamSynchronize(s));                    // sched: synchronize(input_backend)
        for (int i = 0; i < n_d2h; i++) {                // buffer_get_tensor per input
            CK(cudaMemcpyAsync((char *) b.pin_in + i * 16384, b.dact, sz[i], cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CK(cudaStreamSynchronize(cudaStreamPerThread));
        }
        cpu_work(CPU_US);
        b.pin_out[0] = b.pin_in[0];
        CK(cudaStreamSynchronize(s));                    // sched: synchronize(split_backend) before overwriting
        CK(cudaMemcpyAsync(b.dres, b.pin_out, (size_t) N_EMBD * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CK(cudaStreamSynchronize(cudaStreamPerThread));  // buffer_set_tensor
    }
    CK(cudaStreamSynchronize(s));
    return now_us() - t0;
}

// ---- B: one graph, flags written/polled by kernels ----
static void fill_flags(cudaStream_t s, bufs & b) {
    for (int r = 0; r < R; r++) {
        add_work(s, b);
        k_publish<<<1, 256, 0, s>>>(b.dact, b.hact + (size_t) r * N_EMBD, N_EMBD, b.gflag + r, b.dtok);
        k_consume<<<1, 256, 0, s>>>(b.cflag + r, b.hres + (size_t) r * N_EMBD, b.dres, N_EMBD, b.dtok, r, b.derr);
    }
    k_tok<<<1, 1, 0, s>>>(b.dtok);
}

static long long g_flushes = 0;
static cudaStream_t g_stream;

static void host_round(bufs & b, int r, unsigned want, bool spin_gflag) {
    if (spin_gflag) {
        double t0 = 0; int spins = 0;
        while (b.gflag[r] != want) {
            _mm_pause();
            if (++spins == 1 << 14) {                       // WDDM may sit on the submission: nudge it
                if (t0 == 0) t0 = now_us();
                cudaStreamQuery(g_stream); g_flushes++; spins = 0;
                if (now_us() - t0 > 2e6) { printf("deadlock: round %d flag %u want %u\n", r, b.gflag[r], want); exit(1); }
            }
        }
    }
    volatile float * a = b.hact + (size_t) r * N_EMBD;
    float x = a[0];
    cpu_work(CPU_US);
    volatile float * o = b.hres + (size_t) r * N_EMBD;
    o[0] = (float) ((b.htok % 1000) * 64 + r) + 0.0f * x;
    _ReadWriteBarrier();
}

static double token_flags(cudaStream_t s, cudaGraphExec_t ge, bufs & b) {
    const double t0 = now_us();
    CK(cudaGraphLaunch(ge, s));
    cudaStreamQuery(s);                                     // push the submission out
    const unsigned want = b.htok + 1;
    for (int r = 0; r < R; r++) {
        host_round(b, r, want, true);
        b.cflag[r] = want;
    }
    CK(cudaStreamSynchronize(s));
    b.htok++;
    return now_us() - t0;
}

// ---- B2: one graph, flags via stream memory operations ----
// fixed values: GPU writes gflag=1, host resets it to 0 after seeing it; GPU waits
// cflag==1 and writes it back to 0 itself.
static CUdeviceptr dptr(volatile unsigned * p) { void * d; CK(cudaHostGetDevicePointer(&d, (void *) p, 0)); return (CUdeviceptr) d; }

static void fill_memops(cudaStream_t s, bufs & b) {
    for (int r = 0; r < R; r++) {
        add_work(s, b);
        k_publish<<<1, 256, 0, s>>>(b.dact, b.hact + (size_t) r * N_EMBD, N_EMBD, nullptr, b.dtok);
        CKD(cuStreamWriteValue32((CUstream) s, dptr(b.gflag + r), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
        CKD(cuStreamWaitValue32((CUstream) s, dptr(b.cflag + r), 1, CU_STREAM_WAIT_VALUE_EQ));
        CKD(cuStreamWriteValue32((CUstream) s, dptr(b.cflag + r), 0, CU_STREAM_WRITE_VALUE_DEFAULT));
        k_consume<<<1, 256, 0, s>>>(nullptr, b.hres + (size_t) r * N_EMBD, b.dres, N_EMBD, b.dtok, r, b.derr);
    }
    k_tok<<<1, 1, 0, s>>>(b.dtok);
}

static double token_memops(cudaStream_t s, cudaGraphExec_t ge, bufs & b) {
    const double t0 = now_us();
    CK(cudaGraphLaunch(ge, s));
    cudaStreamQuery(s);
    for (int r = 0; r < R; r++) {
        host_round(b, r, 1, true);
        b.gflag[r] = 0;
        b.cflag[r] = 1;
    }
    CK(cudaStreamSynchronize(s));
    b.htok++;
    return now_us() - t0;
}

// ---- C: one graph with host nodes ----
struct hn_arg { bufs * b; int r; };
static std::vector<hn_arg> g_hn;
static void CUDART_CB host_node(void * p) {
    hn_arg * a = (hn_arg *) p;
    host_round(*a->b, a->r, 0, false);
}
static void fill_hostfn(cudaStream_t s, bufs & b) {
    g_hn.resize(R);
    for (int r = 0; r < R; r++) {
        g_hn[r] = { &b, r };
        add_work(s, b);
        k_publish<<<1, 256, 0, s>>>(b.dact, b.hact + (size_t) r * N_EMBD, N_EMBD, nullptr, b.dtok);
        CK(cudaLaunchHostFunc(s, host_node, &g_hn[r]));
        k_consume<<<1, 256, 0, s>>>(nullptr, b.hres + (size_t) r * N_EMBD, b.dres, N_EMBD, b.dtok, r, b.derr);
    }
    k_tok<<<1, 1, 0, s>>>(b.dtok);
}
static double token_hostfn(cudaStream_t s, cudaGraphExec_t ge, bufs & b) {
    const double t0 = now_us();
    CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    b.htok++;
    return now_us() - t0;
}

// ---- G: GPU floor, the same R x NK work kernels in one graph, no host at all ----
static void fill_floor(cudaStream_t s, bufs & b) { for (int r = 0; r < R; r++) add_work(s, b); }
static double g_floor = 0;

static void report(const char * name, std::vector<double> & t, unsigned err) {
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2], p10 = t[t.size() / 10], p90 = t[t.size() * 9 / 10];
    const double per = med / R, ovh = (med - g_floor) / R - CPU_US;
    if (!strcmp(name, "G")) g_floor = med;
    printf("%-4s token %8.1f us (p10 %8.1f, p90 %8.1f)  round %7.1f us  overhead %6.1f us/round = %5.2f ms/token%s\n",
           name, med, p10, p90, per, ovh, ovh * R / 1000, err ? "  DATA ERRORS" : "");
}

int main(int argc, char ** argv) {
    std::string modes = "A,A1,B,B2,C";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() { return atoi(argv[++i]); };
        if (a == "--rounds") R = nxt();
        else if (a == "--nk") NK = nxt();
        else if (a == "--gpu-us") GPU_US = nxt();
        else if (a == "--cpu-us") CPU_US = nxt();
        else if (a == "--tokens") TOKENS = nxt();
        else if (a == "--modes") modes = argv[++i];
        else if (a == "--core") CORE = nxt();
        else { printf("unknown arg %s\n", a.c_str()); return 1; }
    }
    if (CORE >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << CORE);   // one core, like a -C 555 worker
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);   // --prio 2

    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CK(cudaSetDevice(0));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    CKD(cuInit(0));
    CUdevice dev; CKD(cuDeviceGet(&dev, 0));
    int wv = 0, wv64 = 0, drv = 0;
    cuDeviceGetAttribute(&wv, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, dev);
    cuDeviceGetAttribute(&wv64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, dev);
    cudaDriverGetVersion(&drv);
    printf("%s, driver %d, TCC %d, stream mem ops: wait_value_nor %d, 64bit %d\n", p.name, drv, p.tccDriver, wv, wv64);
    printf("rounds %d, kernels/split %d, gpu %d us, cpu %d us per round, activation %d B, %d tokens\n",
           R, NK, GPU_US, CPU_US, N_EMBD * 4, TOKENS);

    bufs b = {};
    CK(cudaMalloc(&b.dact, NMAX * 4)); CK(cudaMemset(b.dact, 0, NMAX * 4));
    CK(cudaMalloc(&b.dres, NMAX * 4));
    CK(cudaMalloc(&b.dwork, 128 * 4)); CK(cudaMemset(b.dwork, 0, 128 * 4));
    CK(cudaMalloc(&b.dtok, 4)); CK(cudaMemset(b.dtok, 0, 4));
    CK(cudaMalloc(&b.derr, 4)); CK(cudaMemset(b.derr, 0, 4));
    CK(cudaHostAlloc(&b.hact, (size_t) R * N_EMBD * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc(&b.hres, (size_t) R * N_EMBD * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void **) &b.gflag, R * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void **) &b.cflag, R * 4, cudaHostAllocMapped));
    memset((void *) b.gflag, 0, R * 4); memset((void *) b.cflag, 0, R * 4);
    CK(cudaMallocHost(&b.pin_in, 65536));
    CK(cudaMallocHost(&b.pin_out, 65536));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    g_stream = s;
    g_split = capture(s, fill_split, b);

    auto run = [&](const char * name, auto fn) {
        // keep the device-side tag counter in step with the host mirror
        CK(cudaMemcpy(b.dtok, &b.htok, 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(b.derr, 0, 4));
        for (int i = 0; i < WARM; i++) fn();
        std::vector<double> t;
        for (int i = 0; i < TOKENS; i++) t.push_back(fn());
        unsigned err = 0; CK(cudaMemcpy(&err, b.derr, 4, cudaMemcpyDeviceToHost));
        report(name, t, err);
    };
    auto has = [&](const char * m) {
        std::string x = "," + modes + ","; return x.find(std::string(",") + m + ",") != std::string::npos;
    };

    {
        cudaGraphExec_t ge = capture(s, fill_floor, b);
        run("G", [&]() { const double t0 = now_us(); CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s)); return now_us() - t0; });
    }
    if (has("A"))  run("A",  [&]() { return token_stock(s, b, 2); });
    if (has("A1")) run("A1", [&]() { return token_stock(s, b, 1); });
    if (has("B")) {
        cudaGraphExec_t ge = capture(s, fill_flags, b);
        g_flushes = 0;
        run("B", [&]() { return token_flags(s, ge, b); });
        if (g_flushes) printf("     B needed %lld submission nudges\n", g_flushes);
    }
    if (has("B2")) {
        if (!wv) printf("B2   skipped: no stream wait-value support on this device/driver\n");
        else {
            memset((void *) b.gflag, 0, R * 4); memset((void *) b.cflag, 0, R * 4);
            cudaGraphExec_t ge = capture(s, fill_memops, b);
            g_flushes = 0;
            run("B2", [&]() { return token_memops(s, ge, b); });
            if (g_flushes) printf("     B2 needed %lld submission nudges\n", g_flushes);
        }
    }
    if (has("C")) {
        cudaGraphExec_t ge = capture(s, fill_hostfn, b);
        run("C", [&]() { return token_hostfn(s, ge, b); });
    }
    return 0;
}
