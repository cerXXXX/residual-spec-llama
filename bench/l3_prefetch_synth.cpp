// Hypothesis No. 4, step 3 (docs/MOE-VIABLE-HYPOTHESES.md): does the CPU expert
// matmul of one Tiel layer get faster when its 8 experts were read into L3
// beforehand? Real expert bytes from the GGUF, real ggml CPU kernels, the
// working thread layout (6 threads, mask 555, strict, prio high, poll 50).
//
// Each "layer" here holds the first NE experts of the real layer (the expert
// dimension is outermost, so that is a contiguous prefix of the tensor), and
// ids pick 8 of them at random per step. 40 layers x NE experts is far beyond
// L3, so walking the layers in order is a cold read, like generation.
//
// Modes, all per layer step:
//   cold         compute only, layers in order                 -> DDR4 baseline
//   hot          same layer, same ids, repeated                -> L3 ceiling
//   pf<F>        prefetch fraction F of the selected experts with the same
//                threadpool, then compute; both timed separately
//
// Build: bench\build_l3_prefetch_synth.bat. Run from the fork's bin dir so the
// ggml DLLs resolve.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int N_LAYER = 40;
static constexpr int N_USED  = 8;

struct range { const uint8_t * p; size_t n; };

struct prefetch_job {
    std::vector<range> ranges;
    size_t             total = 0;
};

// Split the job's bytes evenly across threads; each thread loads its share.
static void prefetch_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const prefetch_job * job = (const prefetch_job *) ud;
    const size_t per = (job->total / nth + 63) & ~(size_t) 63;
    size_t beg = per * ith;
    size_t end = std::min(job->total, beg + per);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    size_t off = 0;
    for (const range & r : job->ranges) {
        const size_t rb = off, re = off + r.n;
        off = re;
        if (re <= beg || rb >= end) continue;
        const size_t a = std::max(rb, beg) - rb;
        const size_t b = std::min(re, end) - rb;
        const uint8_t * p = r.p + (a & ~(size_t) 127);
        const uint8_t * q = r.p + b;
        for (; p + 128 <= q; p += 128) {
            acc0 = _mm256_or_si256(acc0, _mm256_loadu_si256((const __m256i *) (p +  0)));
            acc1 = _mm256_or_si256(acc1, _mm256_loadu_si256((const __m256i *) (p + 32)));
            acc2 = _mm256_or_si256(acc2, _mm256_loadu_si256((const __m256i *) (p + 64)));
            acc3 = _mm256_or_si256(acc3, _mm256_loadu_si256((const __m256i *) (p + 96)));
        }
    }
    acc0 = _mm256_or_si256(_mm256_or_si256(acc0, acc1), _mm256_or_si256(acc2, acc3));
    ((int32_t *) dst->data)[ith] = _mm256_extract_epi32(acc0, 0) | _mm256_extract_epi32(acc0, 5);
}

struct layer {
    ggml_tensor * gate, * up, * down;
    ggml_cgraph * gf;
    ggml_tensor * ids;
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

static double mean(const std::vector<double> & v) {
    double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [n_experts_per_layer=16] [steps=30] [threads=6]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    const int NE      = argc > 2 ? atoi(argv[2]) : 16;
    const int STEPS   = argc > 3 ? atoi(argv[3]) : 30;
    const int NT      = argc > 4 ? atoi(argv[4]) : 6;

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    FILE * f = fopen(path, "rb");
    const size_t data_off = gguf_get_data_offset(g);

    // weights: N_LAYER x 3 tensors, NE experts each
    size_t wbytes = 0;
    const char * names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (int il = 0; il < N_LAYER; ++il) for (auto * nm : names) {
        std::string tn = "blk." + std::to_string(il) + "." + nm + ".weight";
        ggml_tensor * t = ggml_get_tensor(meta, tn.c_str());
        wbytes += t->nb[2] * NE + 256;
    }
    ggml_init_params wp = { wbytes + 3 * N_LAYER * ggml_tensor_overhead() + (1 << 20), nullptr, false };
    ggml_context * wctx = ggml_init(wp);

    std::vector<layer> L(N_LAYER);
    size_t layer_sel_bytes = 0; // bytes of 8 experts, averaged later
    for (int il = 0; il < N_LAYER; ++il) {
        ggml_tensor * dst[3];
        for (int k = 0; k < 3; ++k) {
            std::string tn = "blk." + std::to_string(il) + "." + names[k] + ".weight";
            ggml_tensor * src = ggml_get_tensor(meta, tn.c_str());
            ggml_tensor * t = ggml_new_tensor_3d(wctx, src->type, src->ne[0], src->ne[1], NE);
            const int64_t tid = gguf_find_tensor(g, tn.c_str());
            _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, tid)), SEEK_SET);
            if (fread(t->data, 1, ggml_nbytes(t), f) != ggml_nbytes(t)) { fprintf(stderr, "short read %s\n", tn.c_str()); return 1; }
            dst[k] = t;
            layer_sel_bytes += t->nb[2] * N_USED;
            if (il == 0 || il == 1) fprintf(stderr, "%s %s %.4f MiB/expert\n", tn.c_str(), ggml_type_name(t->type), t->nb[2] / 1048576.0);
        }
        L[il].gate = dst[0]; L[il].up = dst[1]; L[il].down = dst[2];
    }
    fclose(f);
    const double sel_mib = layer_sel_bytes / (double) N_LAYER / 1048576.0;
    fprintf(stderr, "8 experts per layer: %.2f MiB average, %d layers resident %.1f MiB\n",
            sel_mib, N_LAYER, wbytes / 1048576.0);

    // compute graphs: x -> gate/up -> swiglu -> down, n_tokens = 1
    ggml_init_params cp = { 64ull << 20, nullptr, false };
    ggml_context * cctx = ggml_init(cp);
    ggml_tensor * x = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, 2048, 1);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < 2048; ++i) ((float *) x->data)[i] = nd(rng);
    for (int il = 0; il < N_LAYER; ++il) {
        ggml_tensor * ids = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, N_USED, 1);
        ggml_tensor * x3  = ggml_reshape_3d(cctx, x, 2048, 1, 1);
        ggml_tensor * gt  = ggml_mul_mat_id(cctx, L[il].gate, x3, ids);
        ggml_tensor * ut  = ggml_mul_mat_id(cctx, L[il].up,   x3, ids);
        ggml_tensor * h   = ggml_swiglu_split(cctx, gt, ut);
        ggml_tensor * dn  = ggml_mul_mat_id(cctx, L[il].down, h, ids);
        L[il].gf  = ggml_new_graph(cctx);
        L[il].ids = ids;
        ggml_build_forward_expand(L[il].gf, dn);
    }
    prefetch_job job;
    ggml_tensor * pf = ggml_custom_4d(cctx, GGML_TYPE_I32, 64, 1, 1, 1, nullptr, 0, prefetch_op, GGML_N_TASKS_MAX, &job);
    ggml_cgraph * pgf = ggml_new_graph(cctx);
    ggml_build_forward_expand(pgf, pf);

    // threadpool like the server: -t NT -C 555 --cpu-strict 1 --prio 2 --poll 50
    ggml_threadpool_params tpp = ggml_threadpool_params_default(NT);
    memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
    for (int i = 0; i < 12; i += 2) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    tpp.prio       = GGML_SCHED_PRIO_HIGH;
    tpp.poll       = 50;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);

    std::vector<uint8_t> work;
    auto run = [&](ggml_cgraph * gr) {
        ggml_cplan plan = ggml_graph_plan(gr, NT, tp);
        if (plan.work_size > work.size()) work.resize(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gr, &plan);
    };

    std::vector<int> perm(NE);
    auto pick_ids = [&](int il) {
        for (int i = 0; i < NE; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < N_USED; ++i) ((int32_t *) L[il].ids->data)[i] = perm[i];
    };
    // spread: the first frac of every selected expert of every tensor.
    // seq:    the selected experts in compute order (gate, up, down; ascending
    //         expert id, as mul_mat_id walks them), cut at frac of all bytes.
    auto fill_job = [&](int il, double frac, bool seq) {
        job.ranges.clear(); job.total = 0;
        std::vector<int32_t> id((const int32_t *) L[il].ids->data, (const int32_t *) L[il].ids->data + N_USED);
        if (seq) std::sort(id.begin(), id.end());
        size_t all = 0;
        for (ggml_tensor * t : { L[il].gate, L[il].up, L[il].down }) all += t->nb[2] * N_USED;
        const size_t budget = (size_t) (all * frac);
        for (ggml_tensor * t : { L[il].gate, L[il].up, L[il].down }) {
            for (int i = 0; i < N_USED; ++i) {
                size_t n = seq ? std::min((size_t) t->nb[2], budget - job.total)
                               : (size_t) (t->nb[2] * frac);
                n &= ~(size_t) 127;
                if (n == 0) continue;
                job.ranges.push_back({ (const uint8_t *) t->data + t->nb[2] * id[i], n });
                job.total += n;
            }
        }
    };

    // warm-up: touch everything once, let the pool spin up
    for (int il = 0; il < N_LAYER; ++il) { pick_ids(il); run(L[il].gf); }

    auto cold = [&](const char * tag) {
        std::vector<double> t;
        for (int s = 0; s < STEPS; ++s) for (int il = 0; il < N_LAYER; ++il) {
            pick_ids(il);
            double t0 = now_ms(); run(L[il].gf); t.push_back(now_ms() - t0);
        }
        printf("%-10s compute  median %.4f  mean %.4f ms/layer  (x40 = %.2f ms)  %.1f MiB/ms\n",
               tag, median(t), mean(t), mean(t) * 40, sel_mib / mean(t));
        return mean(t);
    };

    const double c0 = cold("cold");

    {   // hot: same layer, same ids
        std::vector<double> t;
        for (int il = 0; il < N_LAYER; ++il) {
            pick_ids(il);
            run(L[il].gf);
            for (int s = 0; s < STEPS; ++s) { double t0 = now_ms(); run(L[il].gf); t.push_back(now_ms() - t0); }
        }
        printf("%-10s compute  median %.4f  mean %.4f ms/layer  -> %.0f %% of cold\n",
               "hot", median(t), mean(t), 100 * mean(t) / c0);
    }

    for (bool seq : { false, true }) for (double frac : { 1.0, 0.75, 0.6, 0.5, 0.4, 0.25 }) {
        std::vector<double> tp_, tc;
        for (int s = 0; s < STEPS; ++s) for (int il = 0; il < N_LAYER; ++il) {
            pick_ids(il);
            fill_job(il, frac, seq);
            double t0 = now_ms(); run(pgf); double t1 = now_ms(); run(L[il].gf); double t2 = now_ms();
            tp_.push_back(t1 - t0); tc.push_back(t2 - t1);
        }
        char tag[16]; snprintf(tag, sizeof tag, "%s%.2f", seq ? "seq" : "spr", frac);
        printf("%-10s prefetch median %.4f mean %.4f | compute median %.4f mean %.4f ms/layer  -> compute %.0f %% of cold, saved %.4f ms\n",
               tag, median(tp_), mean(tp_), median(tc), mean(tc), 100 * mean(tc) / c0, c0 - mean(tc));
    }

    cold("cold-end");

    ggml_threadpool_free(tp);
    ggml_free(cctx); ggml_free(wctx); ggml_free(meta); gguf_free(g);
    return 0;
}
