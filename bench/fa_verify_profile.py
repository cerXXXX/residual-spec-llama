#!/usr/bin/env python3
"""Flash attention cost of small batches at depth, from an nsys sqlite export (FA2).

`llama bench -d <depth> -p 2,4 -n 1` fills the KV cache before every test and then
runs one forward of 2, 4 or 1 token.  A forward is delimited by the 30 GatedDeltaNet
launches of Tiel (--marker gated_delta_net --stride 30); prefill forwards of 2048
tokens are long, the test forwards short, so only forwards with kernel busy below
--max-busy-ms are broken down.  For each one: kernel time by name, with the
q8_0 -> f16 K/V conversion (convert_unary / dequantize_block) and the FA kernels
summed separately.

    python bench/fa_verify_profile.py bench-logs/fa2-verify/nsys-gqa8-q8_0-d65536-p2p4n1.sqlite
"""
import argparse
import sqlite3
from collections import defaultdict

CONVERT = ("convert_unary", "dequantize_block", "dequantize_q8_0", "convert_")


def kind(name):
    if "flash_attn" in name:
        return "fa"
    if any(c in name for c in CONVERT):
        return "convert"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("--marker", default="gated_delta_net")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--max-busy-ms", type=float, default=300.0)
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    rows = list(con.execute(
        "select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName order by k.start"))
    gdn = [i for i, r in enumerate(rows) if a.marker in r[2]]
    firsts, lasts = gdn[::a.stride], gdn[a.stride - 1::a.stride]
    print(f"{a.db}: {len(rows)} kernels, {len(gdn)} {a.marker} launches, {len(firsts)} forwards\n")
    # Tiel's last layer (39) is attention, so a forward ends after its last GDN launch;
    # cut between forwards at the largest idle gap on the GPU.
    bounds = [0]
    for f in range(len(firsts) - 1):
        lo, hi = lasts[f] + 1, firsts[f + 1]
        while lo < hi and "flash_attn" not in rows[lo][2]:
            lo += 1
        while lo < hi and "flash_attn" in rows[lo][2]:  # layer 39 FA (+ its combine kernel)
            lo += 1
        lo = min(lo + 1, hi)
        cut = max(range(lo, hi + 1), key=lambda j: rows[j][0] - rows[j - 1][1])
        bounds.append(cut)
    bounds.append(len(rows))
    for f in range(len(bounds) - 1):
        seg = rows[bounds[f]:bounds[f + 1]]
        busy = sum(r[1] - r[0] for r in seg) / 1e6
        if busy > a.max_busy_ms:
            continue
        per_t, per_n, grids, by_kind = defaultdict(int), defaultdict(int), defaultdict(set), defaultdict(int)
        for st, en, nm, gx, gy, gz in seg:
            key = nm.split("(")[0][:100]
            per_t[key] += en - st
            per_n[key] += 1
            grids[key].add((gx, gy, gz))
            by_kind[kind(nm)] += en - st
        print(f"forward {f}: kernels {len(seg)}, busy {busy:.3f} ms, span {(seg[-1][1] - seg[0][0]) / 1e6:.3f} ms; "
              f"FA {by_kind['fa'] / 1e6:.3f}, K/V->f16 {by_kind['convert'] / 1e6:.3f}, other {by_kind['other'] / 1e6:.3f} ms")
        for k, t in sorted(per_t.items(), key=lambda x: -x[1])[:a.top]:
            g = ",".join("%dx%dx%d" % x for x in sorted(grids[k])[:2])
            print(f"   {k:<100}{per_n[k]:>5}{t / 1e6:>9.3f} ms{t / per_n[k] / 1e3:>9.1f} us  {g}")
        print()


if __name__ == "__main__":
    main()
