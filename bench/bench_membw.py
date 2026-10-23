#!/usr/bin/env python3
"""
bench_membw.py - STREAM-like memory bandwidth, swept over thread count.

Step 0 of IGPU-PLAN.md, and the cheapest way to kill that plan. Generation
currently pulls ~19.8 GB/s from DDR4 (587 MiB of expert weights per token at
33.8 t/s). The iGPU idea rests on six cores being unable to saturate the memory
controller -- an integrated GPU adds no bandwidth, since it is the same DRAM, but
it can hold far more requests in flight than a handful of cores can.

So the question is only: what does this machine actually reach, and does it keep
climbing past 6 threads? If the practical ceiling is near 22 GB/s we are already
at ~90% and there is nothing for an iGPU to pick up.

Method is the classic STREAM kernels on numpy arrays far larger than the 16 MB
L3, with the array split into one contiguous slice per thread. numpy releases
the GIL inside these ufuncs, so Python threads do give real parallelism here.
Every kernel writes into a preallocated `out=` buffer: without that numpy
allocates a temporary per call and the measurement turns into an allocator
benchmark.

Bytes moved is counted as the traffic the kernel must generate -- for copy, 8
read plus 8 written per element. Write-allocate means a written line is usually
read first too, so on x86 the true DRAM traffic for copy and scale is higher
than counted and these figures are, if anything, conservative.

    python bench_membw.py
    python bench_membw.py --mib 512 --threads 1 2 4 6 8 12 --reps 7
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def make_slices(n, k):
    """k contiguous ranges covering [0, n), so threads never share a cache line."""
    step = -(-n // k)
    return [(i * step, min((i + 1) * step, n)) for i in range(k)]


def run_kernel(pool, slices, fn):
    list(pool.map(lambda s: fn(*s), slices))


def bench(n, threads, reps, scalar=3.0):
    a = np.ones(n, dtype=np.float64)
    b = np.full(n, 2.0, dtype=np.float64)
    c = np.full(n, 0.5, dtype=np.float64)
    elem = a.itemsize

    # Kernel -> (function over a slice, bytes counted per element)
    #
    # No STREAM triad here. `np.add(b, scalar * c, out=a)` looks right and is
    # not: the inner product allocates a temporary array per call, so it times
    # the allocator rather than the memory controller. numpy has no fused
    # a = b + s*c, and splitting it into two passes is no longer triad.
    #
    # `read` replaces it, and is the kernel this machine actually cares about:
    # loading expert weights is a pure streaming read with no writes at all.
    kernels = {
        "read":  (lambda i, j: np.add.reduce(b[i:j]),                      1 * elem),
        "copy":  (lambda i, j: np.copyto(a[i:j], b[i:j]),                  2 * elem),
        "scale": (lambda i, j: np.multiply(b[i:j], scalar, out=a[i:j]),    2 * elem),
        "add":   (lambda i, j: np.add(b[i:j], c[i:j], out=a[i:j]),         3 * elem),
    }

    out = {}
    for k in threads:
        slices = make_slices(n, k)
        with ThreadPoolExecutor(max_workers=k) as pool:
            row = {}
            for name, (fn, per_elem) in kernels.items():
                run_kernel(pool, slices, fn)          # warm: fault pages, settle clocks
                best = 0.0
                for _ in range(reps):
                    t0 = time.perf_counter()
                    run_kernel(pool, slices, fn)
                    dt = time.perf_counter() - t0
                    if dt > 0:
                        best = max(best, n * per_elem / dt / 1e9)
                row[name] = best
            out[k] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=384, help="size of each of the 3 arrays")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()

    n = a.mib * 1024 * 1024 // 8
    print(f"3 arrays x {a.mib} MiB float64 ({n:,} elements each), "
          f"best of {a.reps}, GB/s\n")
    res = bench(n, a.threads, a.reps)

    names = ["read", "copy", "scale", "add"]
    print(f"{'threads':>8} " + " ".join(f"{x:>8}" for x in names) + f" {'best':>8}")
    peak = 0.0
    for k, row in res.items():
        best = max(row.values())
        peak = max(peak, best)
        print(f"{k:>8} " + " ".join(f"{row[x]:8.1f}" for x in names) + f" {best:8.1f}")

    print(f"\npeak measured            {peak:6.1f} GB/s")
    print(f"DDR4-3200 dual channel     51.2 GB/s theoretical "
          f"({100 * peak / 51.2:.0f}% reached)")
    print(f"generation pulls           19.8 GB/s "
          f"({100 * 19.8 / peak:.0f}% of the measured peak)")


if __name__ == "__main__":
    main()
