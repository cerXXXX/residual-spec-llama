#!/usr/bin/env python3
"""
bench_moe_gather.py - does the MoE access pattern cost bandwidth, or not?

G7 measured the shortfall and refuted its own explanation. Generation pulls
~19.8 GB/s out of DDR4 while `bench_membw.py` reaches 38.0 GB/s on memcpy, and
the replacement story -- "picking 8 experts of 128 is a scattered gather, not a
stream" -- was written as reasoning and never measured.

Arithmetic already makes that story look weak. Gemma has 128 experts per layer
and a layer weighs 408.4 MiB, so one expert is 3.19 MiB, and inside it the three
tensors (gate, up, down) are ~1.06 MiB each and are read end to end by the
matvec. So a layer is 24 contiguous runs of about a megabyte, scattered in
address space but not in any sense that troubles a prefetcher. Millions of
random 64-byte touches would; two dozen megabyte-long runs should not.

This measures it directly. One arena far larger than the 16 MB L3, read twice:
once contiguously, once as randomly ordered blocks of a chosen size, with the
same kernel and the same thread count so the reduction's own overhead cancels
out and only the ratio is read.

    read_seq / read_gather at block size B  ->  what the pattern costs at B

If ~1 MiB blocks come out level with sequential, the access-pattern explanation
is dead and the shortfall belongs to Q4_K dequantisation, the matvec arithmetic
and the per-layer thread synchronisation instead -- which is a different repair
entirely, in the kernel rather than in the memory system.

    python bench_moe_gather.py
    python bench_moe_gather.py --mib 4096 --threads 6 --reps 5
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Gemma 4 26B-A4B, from the GGUF header: 408.4 MiB per expert layer over 128
# experts is 3.19 MiB per expert, in three tensors of ~1.06 MiB.
EXPERT_MIB = 3.19
TENSOR_MIB = 1.06


def read_blocks(arr, starts, blk):
    """Sum each block. np.add.reduce releases the GIL, so threads are real."""
    total = 0.0
    for s in starts:
        total += np.add.reduce(arr[s:s + blk])
    return total


def bench(arr, blk_elems, threads, reps, rng):
    """Best-of GB/s reading the whole arena as blocks of blk_elems, in random order."""
    n = arr.size
    nblocks = n // blk_elems
    order = rng.permutation(nblocks) * blk_elems

    # Contiguous ranges of the *shuffled* list, so each thread still reads
    # 1/threads of the arena and no two threads share a block.
    per = -(-nblocks // threads)
    parts = [order[i * per:(i + 1) * per] for i in range(threads)]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        run = lambda: list(pool.map(lambda st: read_blocks(arr, st, blk_elems), parts))
        run()                                   # warm: fault pages, settle clocks
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            run()
            dt = time.perf_counter() - t0
            if dt > 0:
                best = max(best, nblocks * blk_elems * arr.itemsize / dt / 1e9)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=3072, help="arena size, must dwarf L3")
    ap.add_argument("--threads", type=int, default=6, help="production setting is 6")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    elem = 8                                    # float64, as in bench_membw.py
    n = a.mib * 1024 * 1024 // elem
    print(f"arena {a.mib} MiB float64, {a.threads} threads, best of {a.reps}\n")

    arr = np.full(n, 1.5, dtype=np.float64)
    rng = np.random.default_rng(a.seed)

    # KiB per block. The two marked sizes are the ones the model actually uses.
    sizes = [4, 16, 64, 256, 1024, 1088, 3268, 16384, 65536]
    labels = {1088: "  <- one expert tensor (1.06 MiB)",
              3268: "  <- one whole expert (3.19 MiB)"}

    seq = bench(arr, n // a.threads, a.threads, a.reps, rng)
    print(f"{'block':>12} {'GB/s':>8} {'vs seq':>8}")
    print(f"{'contiguous':>12} {seq:8.1f} {'1.00':>8}")
    for kib in sizes:
        blk = kib * 1024 // elem
        if blk > n // a.threads:
            continue
        gb = bench(arr, blk, a.threads, a.reps, rng)
        name = f"{kib} KiB" if kib < 1024 else f"{kib / 1024:.2f} MiB"
        print(f"{name:>12} {gb:8.1f} {gb / seq:8.2f}{labels.get(kib, '')}")

    print(f"\ngeneration pulls 19.8 GB/s; contiguous here is {seq:.1f} GB/s")
    print("If the model's own block sizes sit near 1.00, the access pattern is "
          "not what costs\nthe 1.8x -- look at dequant, matvec and per-layer sync.")


if __name__ == "__main__":
    main()
