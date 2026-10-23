#!/usr/bin/env python3
"""Where the PCIe bus sits idle during a bulk prefill, from prefill_xfer_server.py output.

With expert prefetch the upload is the longer of the two streams, so every
millisecond the bus is idle is prefill time lost. This lists the idle gaps in
the HtoD stream (copies of 1 MiB and more), says whether the GPU was computing
during each one, and groups the uploads into graphs (ubatches) by the long
pauses between them.

    python bench/prefetch_timeline.py bench-logs/gemma-g18-nsys/pf2-L6.json
"""
import json
import sys


def main(path, gap_ms=2.0):
    d = json.load(open(path, encoding="utf-8"))
    tl = d["prefill_timeline"]
    cps = sorted(tl["copies"])
    ker = tl["kernels"]
    if not cps:
        raise SystemExit("no copies in the prefill window")

    def busy(a, b):
        return sum(max(0, min(e, b) - max(s, a)) for s, e in ker) / 1e6

    t_end = max(e for _, e in ker) if ker else cps[-1][1]
    print(f"prefill window: copies {cps[0][0] / 1e6:.1f} .. {cps[-1][1] / 1e6:.1f} ms, "
          f"last kernel {t_end / 1e6:.1f} ms, {len(cps)} copies >= 1 MiB, "
          f"{sum(c[2] for c in cps) / 2**30:.2f} GiB")
    sizes = {}
    for s, e, b, sid in cps:
        k = (round(b / 2**20), sid)
        n, t = sizes.get(k, (0, 0))
        sizes[k] = (n + 1, t + (e - s))
    print("\nsize MiB  stream  count  mean ms  GB/s")
    for (mib, sid), (n, t) in sorted(sizes.items()):
        print(f"{mib:8d}  {sid:6d}  {n:5d}  {t / n / 1e6:7.2f}  {mib * 2**20 * n / t:5.2f}")

    gaps = []
    end = cps[0][1]
    for s, e, b, sid in cps[1:]:
        if s - end > gap_ms * 1e6:
            gaps.append((end, s))
        end = max(end, e)
    tot_gap = sum(b - a for a, b in gaps) / 1e6
    span = (cps[-1][1] - cps[0][0]) / 1e6
    print(f"\nbus span {span:.1f} ms, idle gaps > {gap_ms} ms: {len(gaps)}, total {tot_gap:.1f} ms "
          f"({100 * tot_gap / span:.1f} %); tail after last copy {(t_end - cps[-1][1]) / 1e6:.1f} ms")
    print("\n  start ms   gap ms   GPU busy in gap ms")
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:25]:
        print(f"  {a / 1e6:8.1f} {(b - a) / 1e6:8.1f} {busy(a, b):12.1f}")
    small = [g for g in gaps if (g[1] - g[0]) < 50e6]
    print(f"\ngaps under 50 ms: {len(small)}, {sum(b - a for a, b in small) / 1e6:.1f} ms total, "
          f"GPU busy in them {sum(busy(a, b) for a, b in small):.1f} ms")


if __name__ == "__main__":
    main(*sys.argv[1:2], *(float(x) for x in sys.argv[2:3]))
