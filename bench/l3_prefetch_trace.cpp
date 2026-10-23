// Hypothesis No. 4, steps 2+3 together (docs/MOE-VIABLE-HYPOTHESES.md): replay
// the real expert choices of Tiel token by token, layer by layer, with the
// real early-router predictions, and measure what an L3 prefetch in the GPU
// window would save on the CPU expert matmul.
//
// Weights: the whole GGUF is memory-mapped, all 256 experts per layer, like
// the server's CPU_Mapped buffer. ids: bench/router_probe.py writes
// ids.i32 = [n_tokens][40][real top-8, prev_in ranked top-8].
//
// Per layer step, modes:
//   cold        compute with the real ids, nothing prefetched
//   pred<m>     prefetch the m best predicted experts (whole gate/up/down
//               slices), then compute with the real ids
//   oracle<m>   the same with the m best real experts (a perfect predictor)
//   router      the predictor itself: router_L (F32, 2 MiB) x input, alone
// Layer 0 has no prediction and is skipped in the prefetch modes' savings.
//
// Usage: l3_prefetch_trace model.gguf ids.i32 [n_tokens=256] [threads=6]

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#define NOMINMAX
#include <windows.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int N_LAYER = 40;
static constexpr int N_USED  = 8;

struct range { const uint8_t * p; size_t n; };
struct prefetch_job { std::vector<range> ranges; size_t total = 0; };

static void prefetch_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const prefetch_job * job = (const prefetch_job *) ud;
    const size_t per = (job->total / nth + 127) & ~(size_t) 127;
    const size_t beg = per * ith, end = std::min(job->total, beg + per);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    size_t off = 0;
    for (const range & r : job->ranges) {
        const size_t rb = off, re = off + r.n;
        off = re;
        if (re <= beg || rb >= end) continue;
        const uint8_t * p = r.p + ((std::max(rb, beg) - rb) & ~(size_t) 127);
        const uint8_t * q = r.p + (std::min(re, end) - rb);
        for (; p + 128 <= q; p += 128) {
            a0 = _mm256_or_si256(a0, _mm256_loadu_si256((const __m256i *) (p +  0)));
            a1 = _mm256_or_si256(a1, _mm256_loadu_si256((const __m256i *) (p + 32)));
            a2 = _mm256_or_si256(a2, _mm256_loadu_si256((const __m256i *) (p + 64)));
            a3 = _mm256_or_si256(a3, _mm256_loadu_si256((const __m256i *) (p + 96)));
        }
    }
    a0 = _mm256_or_si256(_mm256_or_si256(a0, a1), _mm256_or_si256(a2, a3));
    ((int32_t *) dst->data)[ith] = _mm256_extract_epi32(a0, 0) | _mm256_extract_epi32(a0, 5);
}

// Optional: prefetch on the SMT siblings of the compute cores (odd logical
// CPUs, which mask 555 leaves free), threads sleep between jobs on
// WaitOnAddress -- the layout a real patch would use, wake-up included.
struct sib_pool {
    std::vector<std::thread> th;
    volatile LONG gen = 0;
    std::atomic<int> done{0};
    std::atomic<bool> quit{false};
    prefetch_job * job = nullptr;
    int n = 0;
    void start(int nthreads, prefetch_job * j) {
        n = nthreads; job = j;
        for (int i = 0; i < n; ++i) th.emplace_back([this, i] {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (2 * i + 1));
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            LONG seen = 0;
            ggml_tensor fake = {}; int32_t sink[64]; fake.data = sink;
            for (;;) {
                while (gen == seen) { LONG s = seen; WaitOnAddress(&gen, &s, sizeof(LONG), INFINITE); }
                seen = gen;
                if (quit) return;
                prefetch_op(&fake, i, n, job);
                done.fetch_add(1);
            }
        });
    }
    void run() {
        done = 0;
        InterlockedIncrement(&gen);
        WakeByAddressAll((PVOID) &gen);
        while (done.load() < n) _mm_pause();
    }
    void stop() { quit = true; InterlockedIncrement(&gen); WakeByAddressAll((PVOID) &gen); for (auto & t : th) t.join(); }
};

struct layer {
    ggml_tensor * w[3]; // gate, up, down
    ggml_tensor * router;
    ggml_tensor * ids;
    ggml_cgraph * gf, * rgf;
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf ids.i32 [n_tokens] [threads]\n", argv[0]); return 1; }
    const int NT  = argc > 4 ? atoi(argv[4]) : 6;
    const int SIB = argc > 5 ? atoi(argv[5]) : 0; // >0: prefetch on that many SMT siblings

    // ids
    FILE * fi = fopen(argv[2], "rb");
    _fseeki64(fi, 0, SEEK_END); const long long isz = _ftelli64(fi); _fseeki64(fi, 0, SEEK_SET);
    const int n_all = (int) (isz / (N_LAYER * 2 * N_USED * 4));
    const int T = std::min(n_all, argc > 3 ? atoi(argv[3]) : 256);
    std::vector<int32_t> ids((size_t) n_all * N_LAYER * 2 * N_USED);
    fread(ids.data(), 4, ids.size(), fi); fclose(fi);
    auto real = [&](int t, int il) { return &ids[(((size_t) t * N_LAYER + il) * 2 + 0) * N_USED]; };
    auto pred = [&](int t, int il) { return &ids[(((size_t) t * N_LAYER + il) * 2 + 1) * N_USED]; };
    fprintf(stderr, "trace: %d of %d tokens\n", T, n_all);

    // mmap the model
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    HANDLE fh = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const uint8_t * base = (const uint8_t *) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!base) { fprintf(stderr, "mmap failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(g);

    ggml_init_params wp = { (4 * N_LAYER + 8) * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<layer> L(N_LAYER);
    const char * names[4] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp" };
    for (int il = 0; il < N_LAYER; ++il) for (int k = 0; k < 4; ++k) {
        std::string tn = "blk." + std::to_string(il) + "." + names[k] + ".weight";
        ggml_tensor * src = ggml_get_tensor(meta, tn.c_str());
        ggml_tensor * t = ggml_new_tensor(wctx, src->type, ggml_n_dims(src), src->ne);
        t->data = (void *) (base + data_off + gguf_get_tensor_offset(g, gguf_find_tensor(g, tn.c_str())));
        if (k < 3) L[il].w[k] = t; else L[il].router = t;
    }

    ggml_init_params cp = { 64ull << 20, nullptr, false };
    ggml_context * cctx = ggml_init(cp);
    ggml_tensor * x = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, 2048, 1);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < 2048; ++i) ((float *) x->data)[i] = nd(rng);
    for (int il = 0; il < N_LAYER; ++il) {
        layer & l = L[il];
        l.ids = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, N_USED, 1);
        ggml_tensor * x3 = ggml_reshape_3d(cctx, x, 2048, 1, 1);
        ggml_tensor * gt = ggml_mul_mat_id(cctx, l.w[0], x3, l.ids);
        ggml_tensor * ut = ggml_mul_mat_id(cctx, l.w[1], x3, l.ids);
        ggml_tensor * dn = ggml_mul_mat_id(cctx, l.w[2], ggml_swiglu_split(cctx, gt, ut), l.ids);
        l.gf = ggml_new_graph(cctx);
        ggml_build_forward_expand(l.gf, dn);
        l.rgf = ggml_new_graph(cctx);
        ggml_build_forward_expand(l.rgf, ggml_mul_mat(cctx, l.router, x));
    }
    prefetch_job job;
    ggml_cgraph * pgf = ggml_new_graph(cctx);
    ggml_build_forward_expand(pgf, ggml_custom_4d(cctx, GGML_TYPE_I32, 64, 1, 1, 1, nullptr, 0, prefetch_op, GGML_N_TASKS_MAX, &job));

    ggml_threadpool_params tpp = ggml_threadpool_params_default(NT);
    memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
    for (int i = 0; i < 12; i += 2) tpp.cpumask[i] = true;
    tpp.strict_cpu = true; tpp.prio = GGML_SCHED_PRIO_HIGH; tpp.poll = 50;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    std::vector<uint8_t> work;
    auto run = [&](ggml_cgraph * gr) {
        ggml_cplan plan = ggml_graph_plan(gr, NT, tp);
        if (plan.work_size > work.size()) work.resize(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gr, &plan);
    };
    auto set_ids = [&](int il, const int32_t * src) { memcpy(L[il].ids->data, src, N_USED * 4); };
    auto fill_job = [&](int il, const int32_t * which, int m) {
        job.ranges.clear(); job.total = 0;
        for (ggml_tensor * t : L[il].w) for (int i = 0; i < m; ++i) {
            job.ranges.push_back({ (const uint8_t *) t->data + t->nb[2] * which[i], (size_t) t->nb[2] });
            job.total += t->nb[2];
        }
    };

    sib_pool sp_pool;
    if (SIB > 0) sp_pool.start(SIB, &job);
    auto prefetch = [&] { if (SIB > 0) sp_pool.run(); else run(pgf); };
    printf("prefetch on %s\n", SIB > 0 ? "SMT siblings, sleeping threads" : "the compute threadpool");

    // warm-up: page the used experts in, spin the pool up
    for (int t = 0; t < T; ++t) for (int il = 0; il < N_LAYER; ++il) { set_ids(il, real(t, il)); run(L[il].gf); }

    // per-layer cold reference, layers 1..39
    auto cold = [&](const char * tag) {
        double s = 0; std::vector<double> per(N_LAYER, 0);
        for (int t = 0; t < T; ++t) for (int il = 0; il < N_LAYER; ++il) {
            set_ids(il, real(t, il));
            double t0 = now_ms(); run(L[il].gf); double d = now_ms() - t0;
            per[il] += d; if (il > 0) s += d;
        }
        printf("%-9s compute %.4f ms/layer (layers 1-39), token sum all 40 = %.2f ms\n", tag, s / T / 39,
               [&] { double a = 0; for (double v : per) a += v; return a / T; }());
        return s / T / 39;
    };
    const double c0 = cold("cold");

    {
        double s = 0;
        for (int t = 0; t < T; ++t) for (int il = 1; il < N_LAYER; ++il) { double t0 = now_ms(); run(L[il].rgf); s += now_ms() - t0; }
        printf("%-9s %.4f ms/layer (F32 2 MiB matvec, 6 threads)\n", "router", s / T / 39);
    }

    struct mode { const char * name; bool oracle; int m; };
    const mode modes[] = {
        { "pred3", false, 3 }, { "pred4", false, 4 }, { "pred5", false, 5 }, { "pred6", false, 6 }, { "pred8", false, 8 },
        { "oracle4", true, 4 }, { "oracle5", true, 5 }, { "oracle8", true, 8 },
    };
    for (const mode & md : modes) {
        double sp = 0, sc = 0; int over25 = 0, over33 = 0, n = 0;
        for (int t = 0; t < T; ++t) for (int il = 0; il < N_LAYER; ++il) {
            set_ids(il, real(t, il));
            if (il == 0) { run(L[il].gf); continue; }
            fill_job(il, md.oracle ? real(t, il) : pred(t, il), md.m);
            double t0 = now_ms(); prefetch(); double t1 = now_ms(); run(L[il].gf); double t2 = now_ms();
            sp += t1 - t0; sc += t2 - t1; ++n;
            over25 += (t1 - t0) > 0.25; over33 += (t1 - t0) > 0.33;
        }
        const double p = sp / n, c = sc / n;
        printf("%-9s prefetch %.4f ms (>0.25: %4.1f %%, >0.33: %4.1f %%) | compute %.4f ms = %3.0f %% of cold | saved %.4f ms/layer, x39 = %.2f ms/token\n",
               md.name, p, 100.0 * over25 / n, 100.0 * over33 / n, c, 100 * c / c0, c0 - c, (c0 - c) * 39);
    }
    cold("cold-end");

    if (SIB > 0) sp_pool.stop();
    ggml_threadpool_free(tp);
    return 0;
}
