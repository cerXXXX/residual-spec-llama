"""Hypothesis No. 8: would a static set of "hot" expert slices kept in VRAM
take a measurable share of Tiel's DDR4 traffic? Reads traces from
bench/expert_trace.exe ([40][n_tokens][8] uint8) and reports

  * skew: per layer, the share of selections that the most frequent 5 / 10 /
    25 % of experts take, and the effective number of experts (exp(entropy));
  * hit rate of a hot set picked greedily (count per MiB, across all layers)
    on one trace and scored on another, for several VRAM budgets. Hit rate is
    byte-weighted: the share of the 586.25 MiB/token the CPU would not read;
  * what that would buy per token, from the step-0 profile (CLAUDE.md, S0-03/04).

Usage: python bench/expert_hist.py .cache/expert-trace
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama-tiel" / "gguf-py"))
import gguf  # noqa: E402

MODEL = ROOT / "models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
N_LAYER, N_EXPERT, K = 40, 256, 8
# MiB. 84 / 446 / 656: planner headroom of the launcher default (f16, ub 2048,
# K=1), of q8_0 + ub 4096 with K=2, and without the second slot; 929: the old
# "at most". 1500 / 2100: VRAM idle during decode only (compute buffer of a
# 4096 ubatch + prefetch slots + headroom), refilled after each long prefill.
BUDGETS = (84, 256, 446, 656, 929, 1500, 2100)
# step-0 profile of a Tiel token (CLAUDE.md, "Профиль токена Tiel"), ms
TOKEN_MS, CPU_EXPERTS_MS, GPU_MIB_PER_MS = 33.6, 17.5, 216.7


def slice_mib():
    r = gguf.GGUFReader(str(MODEL))
    out = np.zeros(N_LAYER)
    for t in r.tensors:
        parts = t.name.split(".")
        if len(parts) > 2 and parts[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            il = int(parts[1])
            if il < N_LAYER:
                out[il] += int(t.n_bytes) / N_EXPERT / 2**20
    return out


def load(path):
    a = np.fromfile(path, dtype=np.uint8)
    return a.reshape(N_LAYER, -1, K)


def counts(ids):
    c = np.zeros((N_LAYER, N_EXPERT))
    for il in range(N_LAYER):
        c[il] = np.bincount(ids[il].ravel(), minlength=N_EXPERT)
    return c


def hot_set(c, mib, budget):
    """Greedy by selections per MiB over all (layer, expert) slices."""
    score = (c / mib[:, None]).ravel()
    order = np.argsort(-score, kind="stable")
    size = np.repeat(mib, N_EXPERT)[order]
    take = order[np.cumsum(size) <= budget]
    m = np.zeros(N_LAYER * N_EXPERT, bool)
    m[take] = True
    return m.reshape(N_LAYER, N_EXPERT)


def hit(c, mib, mask):
    """Byte-weighted share of selections that land in the mask."""
    w = c * mib[:, None]
    return (w * mask).sum() / w.sum()


def skew(c):
    rows = []
    for il in range(N_LAYER):
        s = np.sort(c[il])[::-1] / c[il].sum()
        p = s[s > 0]
        rows.append((s[:13].sum(), s[:26].sum(), s[:64].sum(), np.exp(-(p * np.log(p)).sum())))
    return np.array(rows)


def main():
    d = Path(sys.argv[1])
    mib = slice_mib()
    total = mib.sum() * N_EXPERT
    print(f"expert slices: {mib.min():.3f}..{mib.max():.3f} MiB, all experts {total:.1f} MiB, "
          f"per token {mib.sum() * K:.2f} MiB")

    tr = {}
    for name in ("real", "wiki"):
        p = d / f"{name}.prefill.u8"
        if p.exists():
            tr[name] = load(p)
    gens = [load(p) for p in sorted(d.glob("gen-*.gen.u8"))]
    if gens:
        tr["gen"] = np.concatenate(gens, axis=1)
    for p in sorted(d.glob("gen-*.gen.u8")):
        tr[p.name.split(".")[0]] = load(p)
    if "real" in tr:
        n = tr["real"].shape[1]
        tr["real-1"], tr["real-2"] = tr["real"][:, : n // 2], tr["real"][:, n // 2:]
    for k, v in tr.items():
        print(f"  {k:12s} {v.shape[1]:6d} tokens")
    c = {k: counts(v) for k, v in tr.items()}

    print("\n== skew: share of a layer's selections taken by its top 5 / 10 / 25 % experts; "
          "effective experts exp(H) of 256 (uniform: 5.1 / 10.2 / 25.0 %, 256)")
    for k in ("real", "wiki", "gen"):
        if k not in c:
            continue
        s = skew(c[k])
        print(f"  {k:5s} mean {100*s[:,0].mean():5.1f} / {100*s[:,1].mean():5.1f} / {100*s[:,2].mean():5.1f} %"
              f"   exp(H) mean {s[:,3].mean():6.1f}, min {s[:,3].min():6.1f} (layer {s[:,3].argmin()})"
              f"   top-5 % max {100*s[:,0].max():.1f} (layer {s[:,0].argmax()})")

    pairs = [("real", "real"), ("real-1", "real-2"), ("real", "wiki"), ("wiki", "real"),
             ("real", "gen"), ("gen", "gen"), ("wiki", "gen"),
             ("gen-cpp", "gen-py"), ("gen-py", "gen-review"), ("gen-review", "gen-cpp")]
    pairs = [p for p in pairs if p[0] in c and p[1] in c]
    print("\n== byte-weighted hit rate of a static hot set: picked on A, scored on B")
    print("   budget MiB        " + "  ".join(f"{b:>6d}" for b in BUDGETS))
    print("   uniform           " + "  ".join(f"{100*b/total:5.1f}%" for b in BUDGETS))
    res = {}
    for a, b in pairs:
        row = [hit(c[b], mib, hot_set(c[a], mib, bud)) for bud in BUDGETS]
        res[(a, b)] = row
        print(f"   {a:>10s}->{b:<10s}" + "  ".join(f"{100*h:5.1f}%" for h in row))

    gnames = sorted(k for k in tr if k.startswith("gen-"))
    if len(gnames) >= 2 and "real" in c:
        print("\n== leave-one-out over generations: hot set from real + wiki + the other generations, "
              "each corpus normalised to equal weight")
        print("   budget MiB        " + "  ".join(f"{b:>6d}" for b in BUDGETS))
        loo = []
        for g in gnames:
            train = sum(c[k] / c[k].sum() for k in ["real", "wiki"] + [x for x in gnames if x != g] if k in c)
            row = [hit(c[g], mib, hot_set(train, mib, bud)) for bud in BUDGETS]
            loo.append(row)
            print(f"   ->{g:<17s}" + "  ".join(f"{100*h:5.1f}%" for h in row))
        res[("loo", "gen")] = list(np.mean(loo, axis=0))
        print(f"   {'mean':<19s}" + "  ".join(f"{100*h:5.1f}%" for h in res[("loo", "gen")]))

        print("\n== adaptive: hot set from the previous W generated tokens, scored on the next W "
              "(per generation, then averaged); refill cost = new slices over PCIe at 6.56 GB/s")
        for w in (64, 256):
            hs, refill = [], []
            for g in gnames:
                ids = tr[g]
                prev = None
                for s in range(w, ids.shape[1] - w + 1, w):
                    cw = counts(ids[:, s - w: s])
                    row, rf = [], []
                    for bud in BUDGETS:
                        m = hot_set(cw, mib, bud)
                        row.append(hit(counts(ids[:, s: s + w]), mib, m))
                        rf.append(m)
                    if prev is not None:
                        refill.append([((m & ~p) * mib[:, None]).sum() for m, p in zip(rf, prev)])
                    prev = rf
                    hs.append(row)
            hs, refill = np.mean(hs, axis=0), np.mean(refill, axis=0)
            print(f"   W={w:<4d} hit      " + "  ".join(f"{100*h:5.1f}%" for h in hs))
            print(f"          refill   " + "  ".join(f"{r / 6.56e3 * 2**20 / 1e3 / w:5.2f}" for r in refill)
                  + "   ms of PCIe per generated token")
            res[("adapt", f"W{w}")] = list(hs)

    print("\n== what it buys per token (step-0 profile, 33.6 ms/token = 29.8 t/s)")
    print("   saved = hit * 17.5 ms of CPU experts - hit * 586.25 MiB / 216.7 MiB/ms of extra GPU matvec")
    for key in [("real", "gen"), ("loo", "gen"), ("adapt", "W256"), ("gen", "gen")]:
        if key not in res:
            continue
        for bud, h in zip(BUDGETS, res[key]):
            gpu = h * mib.sum() * K / GPU_MIB_PER_MS
            saved = h * CPU_EXPERTS_MS - gpu
            print(f"   {key[0]}->{key[1]} {bud:4d} MiB: hit {100*h:4.1f} %, CPU -{h*CPU_EXPERTS_MS:.2f} ms, "
                  f"GPU +{gpu:.2f} ms, net -{saved:.2f} ms -> {1000/(TOKEN_MS-saved):.2f} t/s "
                  f"({100*(TOKEN_MS/(TOKEN_MS-saved)-1):+.1f} %)")


if __name__ == "__main__":
    main()
