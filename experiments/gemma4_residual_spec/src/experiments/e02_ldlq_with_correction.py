import argparse
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from twobit.base import GROUP
from twobit.hadamard import SEED, layer_rot, rotate_hessian
from twobit.quantizers import CorrLinear, lloyd_max

SEQ = 2048

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--calib_data", default="data/qwen_c4_calib.pt")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--out", default="results/ppl_validation.jsonl")
parser.add_argument("--k_design", type=float, default=0.02)
parser.add_argument("--calib_chunks", type=int, default=32)
parser.add_argument("--damp", type=float, default=0.01)
parser.add_argument("--k", default="0,0.01,0.02,0.05")
parser.add_argument("--limit", type=int, default=40)
parser.add_argument("--variants", default="H,Ht")
parser.add_argument("--save_dir", default="weights/ldlq_H")
parser.add_argument("--no_eval", action="store_true")
args = parser.parse_args()
model_name = args.model.rstrip("/").split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}


def layer_of(n):
    return int(n.split(".layers.")[1].split(".")[0])


num_layers = max(layer_of(n) for n in linears) + 1
c4 = torch.load(args.calib_data)
first = next(iter(linears))
W = orig[first].to("cuda", torch.float32)
rot = layer_rot(first, W.shape[0], W.shape[1], W.device)
Wt = rot.fwd(W)
G = Wt.reshape(Wt.shape[0], -1, 128)
v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
levels = lloyd_max(v[torch.randperm(v.numel(), device=v.device)[:2_000_000]])
thresholds = (levels[1:] + levels[:-1]) / 2
print("levels:", [round(x, 4) for x in levels.tolist()], flush=True)


@torch.no_grad()
def ldlq(Wt, Hr):
    W = Wt.clone()
    d_out, d_in = W.shape
    H = Hr.clone()
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += args.damp * torch.mean(torch.diag(H)) * torch.eye(d_in, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, d_in, GROUP):
        i2 = min(i1 + GROUP, d_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        s = W1.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        for i in range(i2 - i1):
            w = W1[:, i:i + 1]
            q = levels[torch.bucketize(w / s, thresholds)] * s
            Q1[:, i:i + 1] = q
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err @ Hb[i:i + 1, i:]
            E1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        if i2 < d_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


variants = args.variants.split(",")
W_hat = {v: {} for v in variants}
t0 = time.time()
for L in range(num_layers):
    mods = [n for n in linears if layer_of(n) == L]
    acc = {n: {"H": None, "Ht": None, "count": 0} for n in mods}

    def make_hook(n):
        def hook(mod, inp, out):
            X = inp[0].float().reshape(-1, inp[0].shape[-1])
            A = acc[n]
            if "H" in variants:
                A["H"] = X.T @ X if A["H"] is None else A["H"] + X.T @ X
            if "Ht" in variants:
                k = max(1, int(round(args.k_design * X.shape[1])))
                Z = X.clone()
                Z.scatter_(-1, X.abs().topk(k, dim=-1).indices, 0.0)
                A["Ht"] = Z.T @ Z if A["Ht"] is None else A["Ht"] + Z.T @ Z
            A["count"] += X.shape[0]
        return hook

    hooks = [linears[n].register_forward_hook(make_hook(n)) for n in mods]
    with torch.no_grad():
        for c in range(args.calib_chunks):
            model(input_ids=c4[c * SEQ:(c + 1) * SEQ].unsqueeze(0).cuda())
    for h in hooks:
        h.remove()
    for n in mods:
        W = orig[n].to("cuda", torch.float32)
        rot = layer_rot(n, W.shape[0], W.shape[1], W.device)
        Wt = rot.fwd(W)
        for v in variants:
            Hm = acc[n][v] / acc[n]["count"]
            W_hat[v][n] = rot.inv(ldlq(Wt, rotate_hessian(Hm, rot))).to("cpu", torch.bfloat16)
        acc[n] = None
    torch.cuda.empty_cache()
    print(f"layer {L + 1}/{num_layers} done ({time.time() - t0:.0f} s)", flush=True)

if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)
    for v in variants:
        save_file({n: t.contiguous() for n, t in W_hat[v].items()}, os.path.join(args.save_dir, f"W_hat_{v}.safetensors"))
        meta = {
            "variant": v,
            "k_design": args.k_design,
            "calib_chunks": args.calib_chunks,
            "damp": args.damp,
            "seed": SEED,
            "levels": levels.tolist(),
        }
        with open(os.path.join(args.save_dir, f"W_hat_{v}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
    print("saved W_hat to", args.save_dir, flush=True)
if args.no_eval:
    sys.exit(0)
ids = torch.load(args.eval_data)
num_chunks = min(ids.numel() // SEQ, args.limit)


@torch.no_grad()
def perplexity():
    nll, count = 0.0, 0
    for i in range(num_chunks):
        x = ids[i * SEQ:(i + 1) * SEQ].unsqueeze(0).cuda()
        nll += model(input_ids=x, labels=x).loss.float().item() * (SEQ - 1)
        count += SEQ - 1
    return math.exp(nll / count)


os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
out_file = open(args.out, "a")
for v in variants:
    corr = {}
    for n, m in linears.items():
        m.weight.data.copy_(W_hat[v][n].to("cuda"))
        corr[n] = (orig[n].float() - W_hat[v][n].float()).to(torch.bfloat16).pin_memory()
    for n, R in corr.items():
        parent_name, child = n.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        cur = getattr(parent, child)
        if isinstance(cur, CorrLinear):
            cur.R_cpu = R
        else:
            setattr(parent, child, CorrLinear(cur, R, None, None))
    for k in [float(x) for x in args.k.split(",")]:
        CorrLinear.cfg.update({"k_frac": k, "basis": "orig", "static": False, "first_tokens": 0})
        p = perplexity()
        rec = {
            "model": model_name,
            "variant": f"ldlq_{v}",
            "k": k,
            "k_design": args.k_design if v == "Ht" else None,
            "calib_chunks": args.calib_chunks,
            "damp": args.damp,
            "residual": "bf16" if k > 0 else None,
            "tag": f"e02_ldlq_{v}",
            "split": "validation",
            "chunks": num_chunks,
            "ppl": round(p, 4),
        }
        out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_file.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)
