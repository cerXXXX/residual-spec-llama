"""Hypothesis No. 4, step 2: how well can layer L's experts be guessed before
layer L's attention has run? Scores early-router predictors offline against
the real top-8, on dumps from bench/router_probe.exe.

Predictors for layer L (all available before the GPU part of layer L ends):
  prev_in   router_L(attn_post_norm_{L-1})   -- the CPU already holds it: it
            is the input of layer L-1's experts
  resid     router_L(rmsnorm(l_out_{L-1}) * post_norm_w_L)  -- layer L's input
            residual; lives on the GPU, shown as the best-informed case
  prev_tok  the real top-8 of the previous token at layer L  -- free
Metric hits@m: of the m best-ranked predicted experts, how many are in the
real top-8 (averaged, per token and layer). Only ~4-5 experts' bytes fit in
the GPU window (step 3), so hits@4..5 is what a prefetch could use.

Usage: python bench/router_probe.py .cache/router-probe/real [more dirs]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama-tiel" / "gguf-py"))
import gguf  # noqa: E402

MODEL = ROOT / "models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
N_LAYER, N_EMBD, N_EXPERT, K = 40, 2048, 256, 8
EPS = 1e-6
MS = (1, 2, 4, 5, 6, 8)


def load_weights():
    r = gguf.GGUFReader(str(MODEL))
    t = {x.name: x for x in r.tensors}
    router = [np.asarray(t[f"blk.{i}.ffn_gate_inp.weight"].data, dtype=np.float32).reshape(N_EXPERT, N_EMBD)
              for i in range(N_LAYER)]
    post = [np.asarray(t[f"blk.{i}.post_attention_norm.weight"].data, dtype=np.float32).reshape(N_EMBD)
            for i in range(N_LAYER)]
    return router, post


def dump(d, name, il, width):
    a = np.fromfile(d / f"{name}-{il}.f32", dtype=np.float32)
    return a.reshape(-1, width)


def topk(logits, m):
    return np.argpartition(-logits, m - 1, axis=1)[:, :m] if m < logits.shape[1] else np.argsort(-logits, axis=1)


def ranked(logits, m):
    idx = np.argsort(-logits, axis=1)
    return idx[:, :m]


def hits(pred_ranked, real_top8, m):
    p = pred_ranked[:, :m]
    return np.mean([len(set(a) & set(b)) for a, b in zip(p, real_top8)])


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + EPS) * w


def main():
    router, post = load_weights()
    for d in map(Path, sys.argv[1:]):
        print(f"== {d}")
        sums = {k: np.zeros(len(MS)) for k in ("prev_in", "resid", "prev_tok")}
        check = []
        rows = []
        for il in range(N_LAYER):
            x = dump(d, "attn_post_norm", il, N_EMBD)
            lg = dump(d, "ffn_moe_logits", il, N_EXPERT)
            mine = x @ router[il].T
            check.append(np.max(np.abs(mine - lg)) / (np.max(np.abs(lg)) + 1e-9))
            real = ranked(lg, K)
            if il == 0:
                continue
            xp = dump(d, "attn_post_norm", il - 1, N_EMBD)
            lo = dump(d, "l_out", il - 1, N_EMBD)
            preds = {
                "prev_in":  ranked(xp @ router[il].T, 8),
                "resid":    ranked(rmsnorm(lo, post[il]) @ router[il].T, 8),
                "prev_tok": np.vstack([real[:1], real[:-1]]),
            }
            row = [il]
            for k, p in preds.items():
                h = np.array([hits(p[1:], real[1:], m) for m in MS])
                sums[k] += h
                row.append(h)
            rows.append(row)
        print(f"router check: max rel err of my logits vs dumped {max(check):.2e}")
        n = N_LAYER - 1
        print("predictor   " + "  ".join(f"hits@{m}" for m in MS) + "   (out of min(m, 8))")
        for k, s in sums.items():
            print(f"{k:10s}  " + "  ".join(f"{v:6.2f}" for v in s / n))
        print("per layer, hits@5: layer prev_in resid prev_tok")
        for r in rows:
            print(f"  {r[0]:2d}  {r[1][3]:.2f}  {r[2][3]:.2f}  {r[3][3]:.2f}")
        save_ids(d, router)


def save_ids(d, router):
    """ids.i32 for bench/l3_prefetch_synth.exe: [n_tokens][40][real 8, prev_in ranked 8].
    Layer 0 has no prev_in; its prediction slot repeats the real ids (never used)."""
    real = [ranked(dump(d, "ffn_moe_logits", il, N_EXPERT), K) for il in range(N_LAYER)]
    pred = [real[0]] + [ranked(dump(d, "attn_post_norm", il - 1, N_EMBD) @ router[il].T, K)
                        for il in range(1, N_LAYER)]
    out = np.stack([np.stack([real[il], pred[il]], axis=1) for il in range(N_LAYER)], axis=1)
    out.astype(np.int32).tofile(d / "ids.i32")
    print(f"wrote {d / 'ids.i32'}: {out.shape}")


if __name__ == "__main__":
    main()
