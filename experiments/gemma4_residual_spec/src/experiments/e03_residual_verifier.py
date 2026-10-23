import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.hadamard import layer_rot
from twobit.quantizers import CorrLinear, int8_row, lloyd_max, rht_lm2

SEQ = 2048

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--out", default="results/e03_residual_verifier.json")
parser.add_argument("--limit", type=int, default=20)
parser.add_argument("--draft_writers", action="store_true")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}
levels = None
W_int8 = {}
W_hat = {}
R_int8 = {}
W_draft = {}
t0 = time.time()
for n, m in linears.items():
    W = orig[n].to("cuda", torch.float32)
    rot = layer_rot(n, W.shape[0], W.shape[1], W.device)
    if levels is None:
        Wt = rot.fwd(W)
        G = Wt.reshape(Wt.shape[0], -1, 128)
        v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
        levels = lloyd_max(v[torch.randperm(v.numel(), device=v.device)[:2_000_000]])
    Wq = rht_lm2(W, 128, levels, rot)
    W_int8[n] = int8_row(W).to("cpu", torch.bfloat16).pin_memory()
    W_hat[n] = Wq.to("cpu", torch.bfloat16).pin_memory()
    R_int8[n] = int8_row(W - Wq).to("cpu", torch.bfloat16).pin_memory()
    if args.draft_writers and (n.endswith("linear_attn.in_proj_a") or n.endswith("linear_attn.in_proj_b")):
        W_draft[n] = orig[n].pin_memory()
    elif args.draft_writers and (n.endswith("self_attn.k_proj") or n.endswith("self_attn.v_proj")):
        W_draft[n] = W_int8[n]
print(f"weights prepared ({time.time() - t0:.0f} s)", flush=True)
for n, R in R_int8.items():
    parent_name, child = n.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    setattr(parent, child, CorrLinear(getattr(parent, child), R, None, None))
ids = torch.load(args.eval_data)
num_chunks = min(ids.numel() // SEQ, args.limit)
hidden_states = {}


def load(weights):
    for n, m in linears.items():
        m.weight.data.copy_(weights[n].to("cuda", non_blocking=True))


@torch.no_grad()
def collect_hidden(tag, k_frac):
    CorrLinear.cfg.update({"k_frac": k_frac, "basis": "orig", "static": False, "first_tokens": 0})
    hidden_states[tag] = []
    for c in range(num_chunks):
        x = ids[c * SEQ:(c + 1) * SEQ].unsqueeze(0).cuda()
        hidden_states[tag].append(model.model(input_ids=x).last_hidden_state[0].to("cpu"))


t1 = time.time()
load(orig)
collect_hidden("bf16", 0.0)
load(W_int8)
collect_hidden("int8", 0.0)
load(W_hat)
collect_hidden("W_v", 1.0)
collect_hidden("draft", 0.0)
if args.draft_writers:
    load({n: W_draft.get(n, W_hat[n]) for n in linears})
    collect_hidden("draft_writers", 0.0)
print(f"hidden states computed ({time.time() - t1:.0f} s)", flush=True)
tags = list(hidden_states)
stats = {t: {"nll": 0.0, "kl": 0.0, "flip": 0, "n": 0} for t in tags}
agree = {t: 0 for t in tags}
lm = model.lm_head
with torch.no_grad():
    for c in range(num_chunks):
        y = ids[c * SEQ + 1:(c + 1) * SEQ + 1].cuda()
        for s0 in range(0, SEQ - 1, 256):
            s1 = min(s0 + 256, SEQ - 1)
            lp = {t: torch.log_softmax(lm(hidden_states[t][c][s0:s1].cuda()).float(), -1) for t in tags}
            pb = lp["bf16"].exp()
            ab = lp["bf16"].argmax(-1)
            av = lp["W_v"].argmax(-1)
            for t in tags:
                stats[t]["nll"] -= float(lp[t].gather(-1, y[s0:s1, None]).sum())
                stats[t]["kl"] += float((pb * (lp["bf16"] - lp[t])).sum())
                stats[t]["flip"] += int((lp[t].argmax(-1) != ab).sum())
                stats[t]["n"] += s1 - s0
                agree[t] += int((lp[t].argmax(-1) == av).sum())
res = {}
for t in tags:
    n = stats[t]["n"]
    p = agree[t] / n
    res[t] = {
        "ppl": math.exp(stats[t]["nll"] / n),
        "kl_bf16": stats[t]["kl"] / n,
        "flips_vs_bf16": stats[t]["flip"] / n,
        "agreement_with_W_v": p,
        "E_L_gamma4": (1 - p ** 5) / (1 - p) if p < 1 else 5.0,
        "E_L_gamma8": (1 - p ** 9) / (1 - p) if p < 1 else 9.0,
    }
    print(
        f"{t:14s} PPL {res[t]['ppl']:8.4f}  KL(bf16||.) {res[t]['kl_bf16']:.5f}  flips {res[t]['flips_vs_bf16'] * 100:6.2f} %  "
        f"agreement with W_v {p * 100:6.2f} %  E[L] gamma4 {res[t]['E_L_gamma4']:.2f} gamma8 {res[t]['E_L_gamma8']:.2f}",
        flush=True,
    )
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump({"chunks": num_chunks, "tokens": stats["bf16"]["n"], "variants": res}, fh, indent=1)
