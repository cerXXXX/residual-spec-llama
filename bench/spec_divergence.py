#!/usr/bin/env python3
"""
spec_divergence.py - does speculative decoding change Tiel's greedy output, and by how much.

The speculative path of llama-server does not fill per-token probabilities
(server-context.cpp: "TODO: set result.probs"), so tiel_reference.py cannot be
run with drafts on. Instead: a reference run without speculation stores token
ids and top-5 log-probabilities; a speculative run stores token ids only
(`return_tokens`). Comparison is the same as tiel_reference.py: first diverging
position and the reference's top1 - top2 margin there. Verifying k+1 tokens in
one batch takes other kernels than decoding one (GPU matmuls, CPU expert path),
so near-tie flips are expected; a flip at a margin of whole nats is not.

Prompts: the four excerpt prompts of sweep_bench.py (copy and free) and the five
short chat cases of tiel_reference.py.

    python bench/spec_divergence.py run --label ref --llama-bin <bin> --probs --extra ...
    python bench/spec_divergence.py run --label x16 --llama-bin <bin> --extra ...
    python bench/spec_divergence.py compare ref x16
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_bench  # noqa: E402
import tiel_reference as tr  # noqa: E402

OUTDIR = os.path.join(tr.WS, "bench-logs", "no16-divergence")


def cases():
    p = sweep_bench.load_prompts()
    out = [(f"copy-{i}", m[0]["content"], 1000) for i, m in enumerate(p["copy"])]
    out += [(f"free-{i}", m[0]["content"], 400) for i, m in enumerate(p["free"])]
    out += [(n, t, tr.MAX_SHORT) for n, t in tr.SHORT]
    return out


def generate(text, max_tokens, probs):
    prompt = tr.post("/apply-template", {"messages": [{"role": "user", "content": text}]})["prompt"]
    r = tr.post("/completion", {
        "prompt": prompt, "n_predict": max_tokens, "n_probs": 5 if probs else 0,
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 1234,
        "cache_prompt": False, "stream": False, "return_tokens": True,
    })
    tm = r.get("timings", {})
    out = {"ids": r["tokens"], "n_gen": tm.get("predicted_n"), "gen_ms": tm.get("predicted_ms"),
           "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted")}
    if probs:
        out["top"] = [[[t["id"], t["logprob"]] for t in p["top_logprobs"]]
                      for p in r["completion_probabilities"]]
    return out


def cmd_run(a):
    os.makedirs(OUTDIR, exist_ok=True)
    extra = list(a.extra or []) + ["--llama-bin", a.llama_bin]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tr.WS,
                            capture_output=True, text=True).stdout.strip()
    out = {"label": a.label, "commit": commit, "extra": extra, "probs": a.probs, "cases": {}}
    with tr.Server(extra, os.path.join(OUTDIR, f"{a.label}.log")) as srv:
        out["header"] = srv.header()
        for name, text, mx in cases():
            r = generate(text, mx, a.probs)
            out["cases"][name] = r
            tps = r["n_gen"] / (r["gen_ms"] / 1000) if r["gen_ms"] else 0
            print(f"  {name:12s} gen {r['n_gen']:4d} tok {tps:6.2f} t/s"
                  f"  drafts {r['draft_n']}/{r['draft_n_accepted']}", flush=True)
    path = os.path.join(OUTDIR, f"{a.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}")


def cmd_compare(a):
    load = lambda l: json.load(open(os.path.join(OUTDIR, f"{l}.json"), encoding="utf-8"))
    ref, oth = load(a.ref), load(a.other)
    print(f"{'case':12s} {'tokens':>11s} {'diverge@':>9s} {'ref margin':>11s}")
    for name, r in ref["cases"].items():
        o = oth["cases"].get(name)
        if o is None:
            continue
        ta, tb = r["ids"], o["ids"]
        n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        m = "-"
        if div is not None and "top" in r and len(r["top"][div]) > 1:
            top = r["top"][div]
            m = f"{top[0][1] - top[1][1]:.4f}"
        print(f"{name:12s} {len(ta):5d}/{len(tb):<5d} {'-' if div is None else div:>9} {m:>11s}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--llama-bin", required=True)
    r.add_argument("--probs", action="store_true", help="store top-5 log-probabilities (reference run)")
    r.add_argument("--extra", nargs=argparse.REMAINDER, help="further start_server.py arguments, must come last")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
