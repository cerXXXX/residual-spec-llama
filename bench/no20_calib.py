#!/usr/bin/env python3
"""No. 20, step 2 in place: fit the hybrid partition's cost model to the stats lines.

Reads bench-logs/<dir>/results.json written by `offload_crossover.py --set no20-cal`
(GGML_SCHED_HYBRID_STATS=1), where the CPU took every expert with at most c tokens.
A stats line gives, per graph (40 layers): experts on the GPU and the CPU per layer,
the CPU part's wall time, the upload's wall time, its copies and bytes, and the
predicted times of the model the run used (a = 55 us, b = 38 us per token), from
which the CPU part's token slots are recovered: sum(c) = (pred - a*n) / b.

    python bench/no20_calib.py bench-logs/no20-cal
"""
import json
import re
import sys

import numpy as np

RE = re.compile(r"hybrid moe: (\d+) slots, (\d+) layers, experts gpu ([\d.]+) cpu ([\d.]+) per layer, "
                r"cpu part ([\d.]+) ms \(pred ([\d.]+)\), upload ([\d.]+) ms \(pred ([\d.]+), (\d+) copies, "
                r"([\d.]+) MiB\), wait ([\d.]+) ms")
A0, B0 = 55.0, 38.0  # us, the model of the calibration run


def rows(path):
    res = json.load(open(path, encoding="utf-8"))
    for cfg in res["configs"]:
        for r in cfg["rows"]:
            for h in r.get("hybrid", []):
                m = RE.search(h)
                if not m:
                    continue
                slots, layers, gpu, cpu, t_cpu, p_cpu, t_up, p_up, copies, mib, wait = map(float, m.groups())
                n_cpu = cpu*layers
                sum_c = (p_cpu*1e3 - A0*n_cpu)/B0
                yield dict(cfg=cfg["name"], size=r["size"], tokens=slots/8, layers=layers, n_cpu=n_cpu,
                           n_gpu=gpu*layers, sum_c=sum_c, t_cpu=t_cpu, t_up=t_up, copies=copies, mib=mib,
                           wait=wait, pp_ms=r.get("pp_ms"))


def main():
    d = list(rows(sys.argv[1] + "/results.json"))
    print(f"{len(d)} graphs")

    # bus: upload time against bytes, CPU idle (all on the GPU) and CPU busy (the rest)
    for name, sel in (("CPU idle", [r for r in d if r["cfg"] == "maxc-1"]),
                      ("CPU busy", [r for r in d if r["cfg"] not in ("maxc-1", "maxc100000")])):
        mib = np.array([r["mib"] for r in sel]); t = np.array([r["t_up"] for r in sel])
        cp = np.array([r["copies"] for r in sel])
        X = np.c_[mib, cp]
        coef = np.linalg.lstsq(X, t, rcond=None)[0]
        print(f"bus, {name}: {1/coef[0]:.2f} MiB/ms ({1.048576/coef[0]:.2f} GB/s) + {coef[1]*1e3:.1f} us per copy"
              f"  [simple {mib.sum()/t.sum():.2f} MiB/ms]")

    # CPU alone (no bus): t = o*layers + a*n + b*sum_c
    sel = [r for r in d if r["cfg"] == "maxc100000"]
    X = np.array([[r["layers"], r["n_cpu"], r["sum_c"]] for r in sel]); t = np.array([r["t_cpu"] for r in sel])
    o, a, b = np.linalg.lstsq(X, t, rcond=None)[0]
    print(f"CPU alone: {o*1e3:.0f} us per layer + {a*1e3:.1f} us per expert + {b*1e3:.1f} us per token slot")
    for r in sel:
        print(f"   {r['tokens']:5.0f} tok: {r['t_cpu']:7.1f} ms, model {o*r['layers'] + a*r['n_cpu'] + b*r['sum_c']:7.1f}")

    # CPU under the upload: the part of its time the bus overlaps runs slower by g;
    # with t_plain from the fit above, t = t_plain + (g-1)*min(t, t_up) -> g = 1 + (t - t_plain)/min(t, t_up)
    print("CPU under the upload (g = 1 + extra / overlapped time):")
    gs = []
    for r in d:
        if r["cfg"] in ("maxc-1", "maxc100000"):
            continue
        plain = o*r["layers"] + a*r["n_cpu"] + b*r["sum_c"]
        ov = min(r["t_cpu"], r["t_up"])
        g = 1 + (r["t_cpu"] - plain)/ov
        gs.append(g)
        print(f"   {r['cfg']:8s} {r['tokens']:5.0f} tok: cpu {r['t_cpu']:6.1f} ms, alone {plain:6.1f}, "
              f"upload {r['t_up']:6.1f}, wait {r['wait']:6.1f}, g {g:.2f}, pp {r['pp_ms']}")
    print(f"g median {np.median(gs):.2f}, mean {np.mean(gs):.2f}")

    # joint fit with a per-token term (quantising the input rows, the combine): the CPU part costs
    #   o*L + a*n + b*sum_c + p*T*L    alone,   times g while the upload runs (it outlasts the CPU part
    # in all but one hybrid graph here); grid over g, least squares for the rest
    best = None
    for g in np.arange(1.0, 2.01, 0.01):
        X, t = [], []
        for r in d:
            if r["cfg"] == "maxc-1":
                continue
            k = 1.0 if r["cfg"] == "maxc100000" else g
            X.append([k*r["layers"], k*r["n_cpu"], k*r["sum_c"], k*r["tokens"]*r["layers"]])
            t.append(r["t_cpu"])
        X, t = np.array(X), np.array(t)
        coef, *_ = np.linalg.lstsq(X, t, rcond=None)
        err = np.sqrt(np.mean((X @ coef - t)**2))
        if best is None or err < best[0]:
            best = (err, g, coef)
    err, g, (o, a, b, p) = best
    print(f"joint: g {g:.2f}, {o*1e3:.0f} us per layer + {a*1e3:.1f} us per expert + {b*1e3:.1f} us per token slot"
          f" + {p*1e3:.2f} us per token per layer, rms {err:.1f} ms")
    for r in d:
        if r["cfg"] == "maxc-1":
            continue
        k = 1.0 if r["cfg"] == "maxc100000" else g
        m = k*(o*r["layers"] + a*r["n_cpu"] + b*r["sum_c"] + p*r["tokens"]*r["layers"])
        print(f"   {r['cfg']:10s} {r['tokens']:5.0f} tok: cpu {r['t_cpu']:7.1f} ms, model {m:7.1f}")


if __name__ == "__main__":
    main()
