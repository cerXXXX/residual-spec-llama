"""Hypothesis No. 20 (HybriMoE), step 1: filter on router traces, no build.

A mid-size turn (150-1000 new tokens) on the stock fork path syncs for the
routed ids of each MoE layer, uploads every routed expert over PCIe and then
computes on the GPU; the CPU idles meanwhile. HybriMoE would keep the lightly
loaded experts of the layer on the CPU and run them while the bus uploads the
rest, so a layer costs max(bus(G), cpu(C)) instead of bus(all).

Inputs:
  * traces from bench/expert_trace.exe ([40][n_tokens][8] uint8), windows of n
    consecutive prompt tokens give the per-layer expert counts of a turn;
  * bench-logs/no20-crossover/results.json: the same fork, GPU path (threshold
    128) and CPU path (threshold 8192) by turn size, to calibrate
      T_gpu(n) = F_g + b_g * n + sum_l U_l * s_l / BW
      T_cpu(n) = F_c + k_c * n + a * sum_l U_l        (k_c = dense GPU work + 320 * b)
    U_l = experts layer l routes to, s_l = MiB of one expert slice of layer l.

Hybrid layer: experts sorted by count ascending; the CPU takes a prefix C, the
bus the rest G; layer = max(bus(G) + return of the CPU result, g * cpu(C)),
cpu(C) = sum_{e in C} (a + b * c_e); g >= 1 is a slowdown of the CPU while the
bus copies from pageable memory (driver memcpy into staging competes for DDR4).

Usage: python bench/no20_hybrid.py [--crossover bench-logs/no20-crossover/results.json]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_hist import N_EXPERT, N_LAYER, load, slice_mib  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / ".cache" / "expert-trace"
BW_MIB_MS = 6.54e9 / 2**20 / 1e3   # selective upload, H2-P3 (6.54 GB/s)
A_DECODE = 17.5 / 320               # ms per expert at one token, CPU, S0-03
ACT_KIB = 8                         # f32 activations of one token, n_embd 2048
SIZES = (150, 300, 600, 1000)


def window_counts(ids, n, max_windows=40):
    """Per-window [40][256] expert counts for non-overlapping windows of n tokens."""
    total = ids.shape[1]
    starts = list(range(0, total - n + 1, n))
    if len(starts) > max_windows:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    out = []
    for s in starts:
        w = ids[:, s:s + n]
        c = np.zeros((N_LAYER, N_EXPERT))
        for il in range(N_LAYER):
            c[il] = np.bincount(w[il].ravel(), minlength=N_EXPERT)
        out.append(c)
    return out


def bus_ms(c, mib):
    return float(((c > 0) * mib[:, None]).sum() / BW_MIB_MS)


def union(c):
    return float((c > 0).sum())


def hybrid_ms(c, mib, n, a, b, g):
    """Sum over layers of the best CPU/bus split; returns (ms, share of experts on CPU)."""
    tot, on_cpu, used = 0.0, 0, 0
    # activations to the CPU before it starts, its result back after: per layer
    xfer = n * ACT_KIB / 1024 / BW_MIB_MS
    for il in range(N_LAYER):
        cc = np.sort(c[il][c[il] > 0])
        m = len(cc)
        used += m
        bus_e = mib[il] / BW_MIB_MS
        on = np.arange(m + 1) > 0
        cpu_cum = np.concatenate([[0.0], np.cumsum(a + b * cc)]) * g + np.where(on, xfer, 0.0)  # first j on CPU
        bus_rest = (m - np.arange(m + 1)) * bus_e
        lay = np.maximum(bus_rest + np.where(on, xfer, 0.0), cpu_cum)
        j = int(np.argmin(lay))
        tot += lay[j]
        on_cpu += j
    return tot, on_cpu / max(used, 1)


PP_RE = re.compile(r"task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")


def log_points(logdir):
    """(n, ms) of every scored-or-not turn in the crossover server logs: both repeats
    of each size are fresh text, so both count; the warm-up requests (first two) don't."""
    pts = {"gpu": [], "cpu": []}
    for log in sorted(Path(logdir).glob("*.log")):
        tag = log.stem.split("-")[0]
        if tag not in pts:
            continue
        rows = [(int(m[3]), float(m[2])) for m in PP_RE.finditer(log.read_text(encoding="utf-8", errors="replace"))]
        pts[tag] += rows[2:]
    return pts


def robust_lstsq(X, y):
    """Least squares, then drop the worst point while it is > 4x the median residual."""
    keep = np.ones(len(y), bool)
    while True:
        p = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = np.abs(y - X @ p)
        r[~keep] = 0
        i = int(np.argmax(r))
        if r[i] <= 4 * max(np.median(r[keep]), 50):
            return p, keep
        keep[i] = False


def calibrate(logdir, real_ids, mib):
    pts = log_points(logdir)
    stats = {}

    def st(n):
        if n not in stats:
            ws = window_counts(real_ids, n, 30)
            stats[n] = (np.mean([bus_ms(c, mib) for c in ws]), np.mean([union(c) for c in ws]))
        return stats[n]

    # below the threshold (128) the "gpu" server runs the CPU path too; from 1028 on prefetch
    g = np.array([(n, t, *st(n)) for n, t in pts["gpu"] if 128 <= n - 4 < 1024])
    c = np.array([(n, t, *st(n)) for n, t in pts["cpu"]])
    # GPU: t - bus = F_g + b_g * n
    (fg, bg), kg = robust_lstsq(np.c_[np.ones(len(g)), g[:, 0]], g[:, 1] - g[:, 2])
    # CPU: t - a * U = F_c + k_c * n. A free a comes out negative (U is nearly a
    # function of n here), so a is pinned at the decode cost of one cold expert:
    # 1.83 MiB from DDR4 cannot be read for less.
    (fc, kc), kk = robust_lstsq(np.c_[np.ones(len(c)), c[:, 0]], c[:, 1] - A_DECODE * c[:, 3])
    print("== calibration on", logdir)
    print(f"   GPU path: F_g {fg:7.1f} ms, b_g {bg:.3f} ms/token (+ bus from traces)")
    for (n, t, bus, u), k in zip(g, kg):
        print(f"     n {n:5.0f}  meas {t:7.0f}  bus {bus:6.0f}  model {fg + bg * n + bus:7.0f}{'' if k else '  (dropped)'}")
    print(f"   CPU path: F_c {fc:7.1f} ms, k_c {kc:.3f} ms/token, a {A_DECODE:.4f} ms per routed expert")
    for (n, t, bus, u), k in zip(c, kk):
        print(f"     n {n:5.0f}  meas {t:7.0f}  U {u:6.0f}  model {fc + kc * n + A_DECODE * u:7.0f}{'' if k else '  (dropped)'}")
    return fg, bg, fc, kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossover", default=str(ROOT / "bench-logs" / "no20-crossover"))
    ap.add_argument("--contention", type=float, default=1.5,
                    help="CPU slowdown while the bus copies from pageable memory (measured 1.35-1.52x "
                         "with a 256 MiB load buffer, bench/no20_contention.ps1 -BufMiB 256; pinned 1.1-1.15)")
    a_ = ap.parse_args()
    mib = slice_mib()
    tr = {"real": load(TRACES / "real.prefill.u8"),
          "review": load(TRACES / "gen-review.prefill.u8"),
          "wiki": load(TRACES / "wiki.prefill.u8")}
    fg, bg, fc, kc = calibrate(a_.crossover, tr["real"], mib)
    # b: per token of an expert on the CPU; all of k_c is charged to the experts
    # (the dense GPU work of the CPU path is in it too: pessimistic for the hybrid)
    b = kc / (8 * N_LAYER)
    print(f"\n   CPU expert = a + b * tokens: a {A_DECODE:.4f} ms, b {b:.4f} ms; bus expert "
          f"{mib.mean() / BW_MIB_MS:.3f} ms")

    g = a_.contention
    cases = [("as calibrated", A_DECODE, b, 1.0), (f"CPU x{g} under bus load", A_DECODE, b, g),
             (f"a x2, CPU x{g}", 2 * A_DECODE, b, g)]
    print("   cases: " + "; ".join(f"{i + 1} = {nm}" for i, (nm, *_) in enumerate(cases)))
    print("\n== turn time, ms: stock = min(GPU path, CPU path); hybrid = F_g + b_g n + sum of layers")
    for name, ids in tr.items():
        print(f"-- {name} ({ids.shape[1]} tokens)")
        for n in SIZES:
            ws = window_counts(ids, n)
            if not ws:
                continue
            gpu = np.mean([fg + bg * n + bus_ms(c, mib) for c in ws])
            cpu = np.mean([fc + kc * n + A_DECODE * union(c) for c in ws])
            stock = min(gpu, cpu)
            cnt = np.concatenate([c[c > 0] for c in ws])
            line = (f"   n {n:5d} ({len(ws):2d} win)  U/layer {np.mean([union(c) for c in ws]) / N_LAYER:5.1f}"
                    f"  tokens/expert median {np.median(cnt):4.0f}, <=3: {100 * np.mean(cnt <= 3):4.1f} %"
                    f"  | GPU {gpu:6.0f}  CPU {cpu:6.0f}")
            for i, (cname, ca, cb, cg) in enumerate(cases):
                hs = [hybrid_ms(c, mib, n, ca, cb, cg) for c in ws]
                h = np.mean([fg + bg * n + x for x, _ in hs])
                line += f"  | {i + 1}: {h:6.0f} ({100 * (h / stock - 1):+5.1f} %, cpu {100 * np.mean([s for _, s in hs]):3.0f} %)"
            print(line)


if __name__ == "__main__":
    main()
