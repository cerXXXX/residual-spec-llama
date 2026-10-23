#!/usr/bin/env python3
"""Hypothesis 2 of docs/MOE-VIABLE-HYPOTHESES.md: what a hybrid prefill pays for.

H7 said prefill is GPU-bound because nvidia-smi showed >=95 % load. That metric
cannot tell a kernel computing from a copy engine streaming expert weights over
PCIe: with op offload (default) every MUL_MAT_ID of a batch >= 32 runs on the
GPU, and the scheduler first uploads the experts that ubatch uses
(ggml-backend.cpp, "copying only the experts that are used").

This traces one `llama.exe bench` prefill under Nsight Systems and splits each
prompt-processing run into

    kernels       -- union of kernel intervals (GPU computing)
    HtoD          -- union of host-to-device memcpy intervals, and their bytes
    overlap       -- time both were active at once
    neither       -- the GPU doing nothing (host work, CPU splits)

Runs are told apart by idle gaps longer than --gap-ms; the first --skip runs
(bench warmup / first repeat) are dropped.

    python bench/prefill_xfer_profile.py -p 4096 -ub 4096
    python bench/prefill_xfer_profile.py --report existing.nsys-rep
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_kernel_profile import MODELS, DEFAULT_BIN, find_nsys, export_sqlite  # noqa: E402


def profile(a, base):
    cmd = [find_nsys(), "profile", "-o", base, "--force-overwrite", "true",
           "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
           a.llama_bin, "bench", "-m", a.model, "-ngl", "99", "-ncmoe", str(a.ncmoe),
           "-t", "6", "-C", "555", "--cpu-strict", "1", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-b", str(a.ub), "-ub", str(a.ub),
           "-p", str(a.p), "-n", "0", "-r", str(a.reps), "-lm", a.load_mode,
           "-nopo", str(a.nopo)] + a.rest
    print("[+] Executing: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return base + ".nsys-rep"


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(u):
    return sum(e - s for s, e in u)


def intersect(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def analyse(db, gap_ms, skip):
    con = sqlite3.connect(db)
    ker = [(s, e) for s, e in con.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")]
    try:
        mem = list(con.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    except sqlite3.OperationalError:
        mem = []
    h2d = [(s, e, b) for s, e, b, k in mem if k == 1]
    d2h = [(s, e, b) for s, e, b, k in mem if k == 2]

    # runs = clusters of GPU activity separated by long idle gaps
    ev = sorted([(s, e) for s, e in ker] + [(s, e) for s, e, _ in h2d])
    runs, cut = [], gap_ms * 1e6
    for s, e in ev:
        if runs and s - runs[-1][1] < cut:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])

    rows = []
    for idx, (r0, r1) in enumerate(runs):
        k = union([(s, e) for s, e in ker if r0 <= s < r1])
        hv = [(s, e, b) for s, e, b in h2d if r0 <= s < r1]
        h = union([(s, e) for s, e, _ in hv])
        dv = [(s, e, b) for s, e, b in d2h if r0 <= s < r1]
        span = r1 - r0
        kt, ht, ov = total(k), total(h), intersect(k, h)
        hb = sum(b for _, _, b in hv)
        rows.append({
            "run": idx,
            "span_ms": span / 1e6,
            "kernels_ms": kt / 1e6,
            "h2d_ms": ht / 1e6,
            "overlap_ms": ov / 1e6,
            "neither_ms": (span - kt - ht + ov) / 1e6,
            "h2d_GiB": hb / 2**30,
            "h2d_GBps": hb / ht if ht else 0.0,  # bytes per ns == GB/s
            "h2d_n": len(hv),
            "d2h_MiB": sum(b for _, _, b in dv) / 2**20,
        })
    return rows[skip:], rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default="tiel")
    p.add_argument("--llama-bin", default=DEFAULT_BIN)
    p.add_argument("-p", type=int, default=4096)
    p.add_argument("-ub", type=int, default=4096)
    p.add_argument("--ncmoe", type=int, default=40)
    p.add_argument("--nopo", type=int, default=0)
    p.add_argument("--load-mode", default="none")
    p.add_argument("-r", "--reps", type=int, default=3)
    p.add_argument("--gap-ms", type=float, default=300.0)
    p.add_argument("--skip", type=int, default=1, help="leading runs to drop (bench warmup)")
    p.add_argument("--report", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("rest", nargs="*")
    a = p.parse_args()
    a.model = MODELS.get(a.model, a.model)

    rep = a.report or profile(a, os.path.join(os.environ.get("TEMP", "."),
                                              f"pfxfer-p{a.p}-ub{a.ub}-nopo{a.nopo}"))
    kept, allrows = analyse(export_sqlite(rep), a.gap_ms, a.skip)
    hdr = f"{'run':>3} {'span':>9} {'kernels':>9} {'HtoD':>9} {'overlap':>9} {'neither':>9} {'HtoD GiB':>9} {'GB/s':>6} {'n':>6}"
    print("\n" + hdr)
    for r in allrows:
        mark = " " if r in kept else "*"
        print(f"{r['run']:>3}{mark}{r['span_ms']:9.1f} {r['kernels_ms']:9.1f} {r['h2d_ms']:9.1f} "
              f"{r['overlap_ms']:9.1f} {r['neither_ms']:9.1f} {r['h2d_GiB']:9.2f} {r['h2d_GBps']:6.2f} {r['h2d_n']:6d}")
    print("(* = dropped as warmup; times in ms)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"report": rep, "runs": allrows, "kept": [r["run"] for r in kept]}, f, indent=2)
        print(f"[+] {a.out}")


if __name__ == "__main__":
    sys.exit(main())
