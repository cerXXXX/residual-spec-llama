import argparse
import json
import os
import random

import torch

POLICIES = ("base", "mtp", "tree", "mtp1_base")
GAMMAS = (1, 2, 3, 4, 6, 8)


def run_base(hits, p, g):
    k = 0
    while k < g and p + k < len(hits) and hits[p + k]:
        k += 1
    return k


def simulate(policy, first_hit, run_len, hits, g):
    tokens = rounds = 0
    i, n = 0, len(hits)
    while i < n:
        if policy == "base":
            k = run_base(hits, i, g)
        elif policy == "mtp":
            k = min(run_len[i], g)
        elif policy == "tree":
            k = max(min(run_len[i], g), run_base(hits, i, g))
        else:
            k = (1 + run_base(hits, i + 1, g - 1)) if (first_hit[i] or hits[i]) else 0
        step = min(k + 1, n - i)
        tokens += step
        rounds += 1
        i += step
    return tokens, rounds


def bootstrap_ci(per, n_boot=2000, seed=1):
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sel = [rng.randrange(len(per)) for _ in per]
        stats.append(sum(per[s][0] for s in sel) / max(1, sum(per[s][1] for s in sel)))
    stats.sort()
    cut = n_boot // 40
    return [stats[cut], stats[n_boot - cut - 1]]


def tokens_per_round(policy, first_hit, run_len, hits, idx, g):
    trials = [simulate(policy, first_hit[i], run_len[i], hits[i], g) for i in idx]
    return sum(x[0] for x in trials) / sum(x[1] for x in trials)


def distinct4(t):
    grams = [tuple(t[j:j + 4]) for j in range(len(t) - 3)]
    return len(set(grams)) / max(1, len(grams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="results/e06c_gemma_eval.pt")
    ap.add_argument("--hf", default="results/e06_gemma.json")
    ap.add_argument("--out", default="results/e06d_draft_policies.json")
    a = ap.parse_args()
    S = torch.load(a.arrays, weights_only=False)
    with open(a.hf) as f:
        hf = json.load(f)["hf_drafter"]
    first_hit, run_len, base_hit = S["drafter_first_hit"], S["drafter_run_len"], S["base_hit"]
    sources, texts, prompt_lens = S["sources"], S["texts"], S["prompt_lens"]
    d4 = [distinct4(t[pl:].tolist()) for t, pl in zip(texts, prompt_lens)]
    res = {"hf_measured": {g: hf[g]["tokens_per_round"] for g in hf}, "distinct_4gram_median": sorted(d4)[len(d4) // 2],
           "looping_texts": sum(x < 0.8 for x in d4), "groups": {}}
    for grp in ["all", "no_loops(4g>=0.8)"] + sorted(set(sources)):
        if grp == "all":
            idx = list(range(len(texts)))
        elif grp.startswith("no_loops"):
            idx = [i for i in range(len(texts)) if d4[i] >= 0.8]
        else:
            idx = [i for i in range(len(texts)) if sources[i] == grp]
        if not idx:
            continue
        r = {"texts": len(idx), "positions": sum(len(base_hit[i]) for i in idx),
             "agreement_depth1_drafter": sum(sum(first_hit[i]) for i in idx) / sum(len(first_hit[i]) for i in idx),
             "agreement_base": sum(sum(base_hit[i]) for i in idx) / sum(len(base_hit[i]) for i in idx)}
        for pol in POLICIES:
            for g in GAMMAS:
                r[f"{pol}_g{g}"] = tokens_per_round(pol, first_hit, run_len, base_hit, idx, g)
            for g in (4, 8):
                per = [simulate(pol, first_hit[i], run_len[i], base_hit[i], g) for i in idx]
                r[f"{pol}_g{g}_CI95"] = bootstrap_ci(per)
        res["groups"][grp] = r
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("HF measured:", res["hf_measured"], "| looping texts (4g<0.8):", res["looping_texts"])
    for grp, r in res["groups"].items():
        print(f"\n[{grp}] texts {r['texts']}, positions {r['positions']}; depth-1 agreement: "
              f"drafter {r['agreement_depth1_drafter']:.3f}, base {r['agreement_base']:.3f}")
        for pol in POLICIES:
            row = "  ".join(f"g{g}: {r[f'{pol}_g{g}']:.2f}" for g in GAMMAS)
            print(f"  {pol:10s} {row}   g8 CI {[round(x, 2) for x in r[f'{pol}_g8_CI95']]}")


if __name__ == "__main__":
    main()
