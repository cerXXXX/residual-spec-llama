#!/usr/bin/env python3
"""Where the GPU half of a token goes, kernel by kernel (Nsight Systems).

The scheduler profiler in bin/b10826-srclocality measures the *host* side: how
long the host waits for CUDA.  That says how much wall time the GPU costs, but
not whether the GPU was computing or idling inside that window.  This does the
other half: it traces every CUDA kernel, then reports, per decode step,

    kernel busy   -- sum of kernel durations, i.e. the GPU actually working
    small gaps    -- idle inside one CUDA graph / split: per-launch overhead
    big gaps      -- idle while the host computes a CPU split

so "the GPU is slow" and "the GPU is stopped and started" can be told apart.

Protocol notes, all of them learned the hard way:

  * the marker is a kernel that runs exactly once per decode step, and it
    delimits the token windows.  For Gemma 4 that is `softcap_f32` (the final
    logit softcap).  For a model without one, pass a kernel that runs a fixed
    number of times per token and give that number as --stride (8 attention
    layers -> --marker flash_attn_ext_f16 --stride 8).  Without a marker the
    script prints the instance counts so one can be picked.
  * the first quarter of the windows is dropped: CUDA graph capture and the
    first tokens are not steady state.
  * nsys inflates the *span* of a token (CUPTI callbacks on the host), so the
    t/s printed under the profiler is not the benchmark number.  Kernel
    durations come from GPU timestamps and are not affected -- they are what
    this script reports.
  * --cuda-graph-trace=node is mandatory.  With the default (`graph`) a whole
    CUDA graph launch shows up as one range and every per-kernel number here
    becomes meaningless.

Usage:

    python gpu_kernel_profile.py -m gemma --ncmoe 23
    python gpu_kernel_profile.py -m D:\\...\\model.gguf --marker flash_attn_ext_f16 --stride 8
    python gpu_kernel_profile.py --report existing.nsys-rep      # re-analyse
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

NSYS_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems\target-windows-x64\nsys.exe",
    "nsys",
]

MODELS = {
    "gemma": r"D:\localllamacpp\models\gemma-4-26B-A4B-it-qat\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "tiel": r"D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf",
    "ornith9b": r"D:\localllamacpp\models\Ornith-1.5-9B-Q4_K_M\Ornith-1.5-9B-Q4_K_M.gguf",
}

DEFAULT_BIN = r"D:\localllamacpp\bin\llama-b10826.exe"


def find_nsys():
    for c in NSYS_CANDIDATES:
        if os.path.exists(c):
            return c
    return "nsys"


def profile(args, report):
    cmd = [find_nsys(), "profile", "-o", report, "--force-overwrite", "true",
           "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
           args.llama_bin, "bench", "-m", args.model,
           "-ngl", str(args.ngl), "-t", str(args.threads), "-C", args.cpu_mask,
           "--cpu-strict", "1", "-fa", "on", "-p", "0",
           "-n", str(args.n_gen), "-r", str(args.reps), "--no-warmup"]
    if args.ncmoe is not None:
        cmd += ["-ncmoe", str(args.ncmoe)]
    if args.ctk:
        cmd += ["-ctk", args.ctk, "-ctv", args.ctv or args.ctk]
    cmd += args.rest
    print("[+] Executing: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return report + ".nsys-rep"


def export_sqlite(rep):
    db = os.path.splitext(rep)[0] + ".sqlite"
    subprocess.run([find_nsys(), "export", "--type", "sqlite", "--force-overwrite", "true",
                    "-o", db, rep], check=True, stdout=subprocess.DEVNULL)
    return db


def analyse(db, marker, stride, gap_us, top):
    import sqlite3
    con = sqlite3.connect(db)
    rows = list(con.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName order by k.start"))

    marks = [r[0] for r in rows if marker in r[2]][::stride]
    if len(marks) < 8:
        counts = defaultdict(int)
        for _, _, nm in rows:
            counts[nm] += 1
        print(f"marker {marker!r} gives only {len(marks)} windows. Instance counts:")
        for nm, c in sorted(counts.items(), key=lambda x: x[1]):
            print(f"  {c:8d}  {nm[:96]}")
        raise SystemExit(1)

    wins = list(zip(marks[:-1], marks[1:]))
    wins = wins[len(wins) // 4:]
    cut = gap_us * 1000

    tot_busy = tot_span = small = big = 0
    n_small = n_big = n_k = 0
    per_t, per_n = defaultdict(int), defaultdict(int)
    buckets = defaultdict(lambda: [0, 0])
    i = 0
    for w0, w1 in wins:
        while i < len(rows) and rows[i][0] < w0:
            i += 1
        j, last_end = i, None
        while j < len(rows) and rows[j][0] < w1:
            st, en, nm = rows[j]
            d = en - st
            tot_busy += d
            key = nm[:64]
            per_t[key] += d
            per_n[key] += 1
            b = 0 if d < 2000 else 1 if d < 5000 else 2 if d < 15000 else 3 if d < 50000 else 4
            buckets[b][0] += 1
            buckets[b][1] += d
            if last_end is not None and st > last_end:
                g = st - last_end
                if g < cut:
                    small += g
                    n_small += 1
                else:
                    big += g
                    n_big += 1
            last_end = en if last_end is None else max(last_end, en)
            j += 1
        n_k += j - i
        tot_span += w1 - w0

    n = len(wins)
    res = {
        "tokens": n,
        "kernels_per_token": n_k / n,
        "span_ms": tot_span / n / 1e6,
        "kernel_busy_ms": tot_busy / n / 1e6,
        "small_gaps_ms": small / n / 1e6,
        "small_gaps_n": n_small / n,
        "big_gaps_ms": big / n / 1e6,
        "big_gaps_n": n_big / n,
        "by_kernel": [
            {"name": nm, "n": per_n[nm] / n, "ms": t / n / 1e6, "us_avg": t / per_n[nm] / 1e3}
            for nm, t in sorted(per_t.items(), key=lambda x: -x[1])[:top]
        ],
        "histogram": {lbl: {"n": buckets[b][0] / n, "ms": buckets[b][1] / n / 1e6}
                      for b, lbl in enumerate(["<2us", "2-5us", "5-15us", "15-50us", ">50us"])
                      if b in buckets},
    }
    return res


def grid_table(db, pattern="%mul_mat_vec%", limit=25):
    """Kernels grouped by launch geometry.

    For mul_mat_vec_q gridX is the number of output rows, so a row here maps
    onto a named tensor in the GGUF (inspect_gguf.py --tensors) and the MiB/ms
    of one matvec can be worked out exactly.
    """
    import sqlite3
    con = sqlite3.connect(db)
    q = """select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, count(*),
                  avg(k.end-k.start), min(k.end-k.start)
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
           where s.value like ? group by 1,2,3,4,5,6 order by count(*) desc limit ?"""
    out = ["%16s %9s %8s %9s %9s  %s" % ("grid", "block", "n", "avg us", "min us", "kernel")]
    for nm, gx, gy, gz, bx, by, n, us, mn in con.execute(q, (pattern, limit)):
        out.append("%16s %9s %8d %9.1f %9.1f  %s" % (
            "%dx%dx%d" % (gx, gy, gz), "%dx%d" % (bx, by), n, us / 1000, mn / 1000, nm[:56]))
    return "\n".join(out)


def report_text(r):
    out = []
    out.append(f"tokens {r['tokens']}   kernels/token {r['kernels_per_token']:.1f}")
    out.append(f"span         {r['span_ms']:8.3f} ms/token   (inflated by the profiler)")
    out.append(f"kernel busy  {r['kernel_busy_ms']:8.3f} ms/token   <- the GPU actually working")
    out.append(f"small gaps   {r['small_gaps_ms']:8.3f} ms/token   ({r['small_gaps_n']:.0f} of them: launch overhead)")
    out.append(f"big gaps     {r['big_gaps_ms']:8.3f} ms/token   ({r['big_gaps_n']:.0f} of them: the CPU splits)")
    out.append("")
    out.append("kernel duration histogram, per token:")
    for lbl, h in r["histogram"].items():
        out.append(f"  {lbl:>8}  n={h['n']:7.1f}  {h['ms']:7.3f} ms")
    out.append("")
    out.append(f"{'kernel':<66}{'n/tok':>7}{'ms/tok':>9}{'us avg':>8}")
    for k in r["by_kernel"]:
        out.append(f"{k['name']:<66}{k['n']:>7.1f}{k['ms']:>9.3f}{k['us_avg']:>8.1f}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default="gemma")
    p.add_argument("--llama-bin", default=DEFAULT_BIN)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ncmoe", type=int, default=None)
    p.add_argument("-t", "--threads", type=int, default=6)
    p.add_argument("-C", "--cpu-mask", default="555")
    p.add_argument("--ctk", default=None)
    p.add_argument("--ctv", default=None)
    p.add_argument("-n", "--n-gen", type=int, default=64)
    p.add_argument("-r", "--reps", type=int, default=2)
    p.add_argument("--marker", default="softcap_f32", help="kernel that runs once per decode step")
    p.add_argument("--stride", type=int, default=1, help="marker instances per decode step")
    p.add_argument("--gap-us", type=float, default=20.0, help="small/big gap boundary")
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--report", default=None, help="re-analyse an existing .nsys-rep")
    p.add_argument("--grids", default=None, metavar="LIKE",
                   help="also group kernels matching this SQL LIKE by launch geometry, "
                        "e.g. %%mul_mat_vec%% -- gridX is the matvec's output rows")
    p.add_argument("--out", default=None, help="write the numbers here as json")
    p.add_argument("rest", nargs="*", help="extra args passed to llama bench")
    a = p.parse_args()
    a.model = MODELS.get(a.model, a.model)

    if a.report:
        rep = a.report
    else:
        tag = os.path.splitext(os.path.basename(a.model))[0][:24]
        base = os.path.join(os.environ.get("TEMP", "."), f"gpuprof-{tag}")
        rep = profile(a, base)

    db = export_sqlite(rep)
    r = analyse(db, a.marker, a.stride, a.gap_us, a.top)
    r["report"] = rep
    print()
    print(report_text(r))
    if a.grids:
        r["grids"] = grid_table(db, a.grids)
        print()
        print(r["grids"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"\n[+] {a.out}")


if __name__ == "__main__":
    sys.exit(main())
