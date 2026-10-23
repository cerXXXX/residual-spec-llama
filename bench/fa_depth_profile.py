#!/usr/bin/env python3
"""Decode-step kernels at depth, from an nsys sqlite export (hypothesis No. 17).

`llama bench -d <depth> -n <N>` first fills the KV cache, then decodes N
tokens, so the decode tokens are the *last* windows of the trace.  A window is
delimited by a kernel that runs a fixed number of times per token (Tiel: 30
GatedDeltaNet layers -> --marker gated_delta_net --stride 30).

Per decode token it prints kernel time by kernel name, and for the flash
attention kernels the effective KV read rate:  depth * KV bytes per token per
layer / kernel time.  The KV bytes are what the kernel has to read at least
once; if the kernel re-reads K/V per Q head, the rate comes out far below the
card's own matvec ceiling (~217-240 MiB/ms, S0-04).

    python bench/fa_depth_profile.py bench-logs/no17-fa-q8/nsys-q8_0-d65536.sqlite --depth 65536 --kv q8_0
"""
import argparse
import sqlite3
from collections import defaultdict

# Tiel: 10 full-attention layers, 2 KV heads, head dim 256.
N_ATTN, N_KV_HEADS, D = 10, 2, 256
ROW_BYTES = {"f16": D * 2, "q8_0": D // 32 * 34}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--kv", choices=ROW_BYTES, required=True)
    p.add_argument("--marker", default="gated_delta_net")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--tokens", type=int, default=24, help="last N decode windows")
    p.add_argument("--top", type=int, default=14)
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    rows = list(con.execute(
        "select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName order by k.start"))
    marks = [r[0] for r in rows if a.marker in r[2]][::a.stride]
    wins = list(zip(marks[:-1], marks[1:]))[-a.tokens:]

    per_t, per_n, grids = defaultdict(int), defaultdict(int), defaultdict(set)
    busy = span = 0
    i = 0
    for w0, w1 in wins:
        while rows[i][0] < w0:
            i += 1
        j = i
        while j < len(rows) and rows[j][0] < w1:
            st, en, nm, gx, gy, gz = rows[j]
            key = nm.split("(")[0][:90]
            per_t[key] += en - st
            per_n[key] += 1
            grids[key].add((gx, gy, gz))
            busy += en - st
            j += 1
        span += w1 - w0
    n = len(wins)

    kv_mib = a.depth * N_ATTN * N_KV_HEADS * 2 * ROW_BYTES[a.kv] / 2**20
    print(f"{a.db}\ndepth {a.depth}, KV {a.kv}: {kv_mib:.1f} MiB of K+V per decode token (all 10 layers)")
    print(f"tokens {n}, kernel busy {busy / n / 1e6:.3f} ms/token, span {span / n / 1e6:.3f} (inflated)\n")
    print(f"{'kernel':<92}{'n/tok':>6}{'ms/tok':>9}{'us avg':>9}  grid")
    fa_ms = 0.0
    for k, t in sorted(per_t.items(), key=lambda x: -x[1])[:a.top]:
        ms = t / n / 1e6
        if "flash_attn" in k:
            fa_ms += ms
        g = ",".join("%dx%dx%d" % x for x in sorted(grids[k])[:2])
        print(f"{k:<92}{per_n[k] / n:>6.1f}{ms:>9.3f}{t / per_n[k] / 1e3:>9.1f}  {g}")
    if fa_ms:
        print(f"\nflash_attn* total {fa_ms:.3f} ms/token -> {kv_mib / fa_ms:.1f} MiB/ms effective KV read")


if __name__ == "__main__":
    main()
