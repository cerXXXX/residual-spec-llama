import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.hadamard import layer_rot
from twobit.quantizers import lloyd_max, rht_lm2

SEQ = 2048
NUM_CHUNKS = 4
KS = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--out", default="results/e01_error_ledger.json")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
t0 = time.time()
levels = None
residuals = {}
col_cv = {}
for n, m in linears.items():
    W = m.weight.detach().float()
    rot = layer_rot(n, W.shape[0], W.shape[1], W.device)
    if levels is None:
        Wt = rot.fwd(W)
        G = Wt.reshape(Wt.shape[0], -1, 128)
        v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
        levels = lloyd_max(v[torch.randperm(v.numel(), device=v.device)[:2_000_000]])
    R = W - rht_lm2(W, 128, levels, rot)
    col_norms = R.norm(dim=0)
    col_cv[n] = float(col_norms.std() / col_norms.mean())
    residuals[n] = R.to("cpu", torch.bfloat16).pin_memory()
print(f"residuals computed: {len(residuals)} layers, {time.time() - t0:.0f} s", flush=True)
acc = {n: {"WX": 0.0, "RX": 0.0, "X": 0.0, "dot": 0.0, "RmX": [0.0] * len(KS), "aX": [0.0] * len(KS)} for n in linears}


@torch.no_grad()
def hook(mod, inp, out, n):
    X = inp[0].float().reshape(-1, inp[0].shape[-1])
    W = mod.weight.float()
    R = residuals[n].to(X.device, non_blocking=True).float()
    WX = X @ W.T
    RX = X @ R.T
    A = acc[n]
    A["WX"] += float((WX * WX).sum())
    A["RX"] += float((RX * RX).sum())
    A["X"] += float((X * X).sum())
    A["dot"] += float(((WX - RX) * WX).sum())
    ax = X.abs()
    d = X.shape[1]
    for i, kf in enumerate(KS):
        k = max(1, int(round(kf * d)))
        idx = ax.topk(k, dim=-1).indices
        Xm = X.clone()
        Xm.scatter_(-1, idx, 0.0)
        Y = Xm @ R.T
        A["RmX"][i] += float((Y * Y).sum())
        A["aX"][i] += float((X * X).sum() - (Xm * Xm).sum())


hooks = [m.register_forward_hook(lambda mod, i, o, n=n: hook(mod, i, o, n)) for n, m in linears.items()]
ids = torch.load(args.eval_data)
t1 = time.time()
with torch.no_grad():
    for c in range(NUM_CHUNKS):
        model(input_ids=ids[c * SEQ:(c + 1) * SEQ].unsqueeze(0).cuda())
for h in hooks:
    h.remove()
print(f"ledger: {NUM_CHUNKS} x {SEQ} validation tokens, {time.time() - t1:.0f} s", flush=True)
res = {}
for n, A in acc.items():
    res[n] = {
        "e0": A["RX"] / A["WX"],
        "shr": A["dot"] / A["WX"],
        "cv": col_cv[n],
        "WX": A["WX"],
        "RX": A["RX"],
        "c": [1 - A["RmX"][i] / A["RX"] for i in range(len(KS))],
        "a": [A["aX"][i] / A["X"] for i in range(len(KS))],
    }
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump({"k_fracs": KS, "layers": res}, fh, indent=1)


def layer_type(n):
    return n.split(".", 3)[-1]


def median(values):
    return float(torch.tensor(values).median())


header = f"\n{'layer type':28s} {'n':>3s} {'e0 med':>7s} {'shr':>6s} {'CV|R_i|':>8s} | "
print(header + " | ".join(f"k={k * 100:g}%: c/a" for k in KS))
for t in sorted({layer_type(n) for n in res}):
    names = [n for n in res if layer_type(n) == t]
    row = (
        f"{t:28s} {len(names):3d} {median([res[n]['e0'] for n in names]):7.4f} "
        f"{median([res[n]['shr'] for n in names]):6.3f} {median([res[n]['cv'] for n in names]):8.3f} | "
    )
    cols = " | ".join(
        f"{median([res[n]['c'][i] for n in names]):.2f}/{median([res[n]['a'][i] for n in names]):.2f}" for i in range(len(KS))
    )
    print(row + cols)
total_RX = sum(A["RX"] for A in acc.values())
captured = [1 - sum(A["RmX"][i] for A in acc.values()) / total_RX for i in range(len(KS))]
print("\nTOTAL (weighted by error energy): " + " | ".join(f"k={k * 100:g}%: c={captured[i]:.3f}" for i, k in enumerate(KS)))
