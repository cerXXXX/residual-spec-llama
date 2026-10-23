import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.hadamard import SEED, layer_rot
from twobit.quantizers import CorrLinear, int8_row, lloyd_max, quant_res, rht_lm2, rtn

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--variants", default="bf16")
parser.add_argument("--split", default="test")
parser.add_argument("--eval_data", default=None)
parser.add_argument("--calib_data", default="data/qwen_c4_calib.pt")
parser.add_argument("--out", default=None)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--k", default="0")
parser.add_argument("--res", default="bf16")
parser.add_argument("--basis", default="orig")
parser.add_argument("--seq", type=int, default=2048)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--writers", default="none")
parser.add_argument("--kappa", action="store_true")
parser.add_argument("--static", action="store_true")
parser.add_argument("--first_tokens", type=int, default=0)
parser.add_argument("--tag", default="")
parser.add_argument("--calib_chunks", type=int, default=8)
args = parser.parse_args()
eval_path = args.eval_data or f"data/qwen_wikitext2_{args.split}.pt"
out_path = args.out or f"results/ppl_{args.split}.jsonl"
model_name = args.model.rstrip("/").split("/")[-1]

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"model {type(model).__name__} loaded in {time.time() - t0:.0f} s; VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
num_params = sum(m.weight.numel() for m in linears.values())
total_params = sum(p.numel() for p in model.parameters())
suffixes = sorted({n.split(".", 3)[-1] for n in linears})
print(f"linear layers to quantize: {len(linears)}, parameters {num_params / 1e9:.2f} B of {total_params / 1e9:.2f} B; types: {suffixes}", flush=True)
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}
ids = torch.load(eval_path)
num_chunks = ids.numel() // args.seq
if args.limit:
    num_chunks = min(num_chunks, args.limit)


def writer_mode(n):
    if args.writers != "ab_kv":
        return None
    if n.endswith("linear_attn.in_proj_a") or n.endswith("linear_attn.in_proj_b"):
        return "bf16"
    if n.endswith("self_attn.k_proj") or n.endswith("self_attn.v_proj"):
        return "int8"
    return None


@torch.no_grad()
def perplexity():
    nll, count, ts = 0.0, 0, time.time()
    for i in range(num_chunks):
        x = ids[i * args.seq:(i + 1) * args.seq].unsqueeze(0).cuda()
        output = model(input_ids=x, labels=x)
        nll += output.loss.float().item() * (args.seq - 1)
        count += args.seq - 1
    return math.exp(nll / count), time.time() - ts


levels = None
corr_mods = {}


def apply(variant):
    global levels
    corr_mods.clear()
    need_corr = variant == "rht2" and any(float(k) > 0 for k in args.k.split(","))
    writer_params = {"bf16": 0, "int8": 0}
    for n, m in linears.items():
        W = orig[n].to("cuda", torch.float32)
        mode = writer_mode(n)
        if mode == "bf16":
            Wq = W
            writer_params["bf16"] += W.numel()
        elif mode == "int8" and variant != "bf16":
            Wq = int8_row(W)
            writer_params["int8"] += W.numel()
        elif variant == "bf16":
            Wq = W
        elif variant == "int8":
            Wq = int8_row(W)
        elif variant.startswith("rtn"):
            bits = int(variant[3])
            Wq = rtn(W, bits, 64 if bits == 2 else 128)
        elif variant == "rht2":
            rot = layer_rot(n, W.shape[0], W.shape[1], W.device, seed=args.seed)
            if levels is None:
                Wt = rot.fwd(W)
                G = Wt.reshape(Wt.shape[0], -1, 128)
                v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
                levels = lloyd_max(v[torch.randperm(v.numel(), device=v.device)[:2_000_000]])
                print("Lloyd-Max levels:", [round(x, 4) for x in levels.tolist()], flush=True)
            Wq = rht_lm2(W, 128, levels, rot, kappa=args.kappa)
            if need_corr:
                R = quant_res(W - Wq, args.res).to("cpu", torch.bfloat16).pin_memory()
                R_rot = None
                if args.basis == "rot":
                    R_rot = quant_res(rot.fwd_in(W - Wq), args.res).to("cpu", torch.bfloat16).pin_memory()
                corr_mods[n] = (R, R_rot, rot)
        else:
            raise ValueError(variant)
        m.weight.data.copy_(Wq.to(torch.bfloat16))
        del W
    if args.writers != "none":
        extra_bits = (writer_params["bf16"] * 14 + writer_params["int8"] * 6) / num_params
        print(
            f"memory writers: bf16 {writer_params['bf16'] / 1e6:.1f} M params, int8 {writer_params['int8'] / 1e6:.1f} M params "
            f"(+{extra_bits:.3f} bits/weight vs 2-bit)",
            flush=True,
        )


@torch.no_grad()
def calib_static():
    acc, hooks = {}, []
    for n in corr_mods:
        def hook(mod, inp, out, n=n):
            x = inp[0].float().reshape(-1, inp[0].shape[-1])
            acc[n] = acc.get(n, 0) + (x * x).sum(0)
        hooks.append(linears[n].register_forward_hook(hook))
    c4 = torch.load(args.calib_data)
    for i in range(args.calib_chunks):
        model(input_ids=c4[i * args.seq:(i + 1) * args.seq].unsqueeze(0).cuda())
    for h in hooks:
        h.remove()
    return {n: torch.argsort(v, descending=True) for n, v in acc.items()}


def wrap(static_orders):
    for n, (R, R_rot, rot) in corr_mods.items():
        parent_name, child = n.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        cur = getattr(parent, child)
        order = static_orders.get(n) if static_orders else None
        if isinstance(cur, CorrLinear):
            cur.R_cpu, cur.R_rot_cpu, cur.rot = R, R_rot, rot
            cur.static_order = order
        else:
            setattr(parent, child, CorrLinear(cur, R, R_rot, rot, order))


os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
out_file = open(out_path, "a")
for variant in args.variants.split(","):
    ta = time.time()
    apply(variant)
    tq = time.time() - ta
    ks = [float(k) for k in args.k.split(",")] if variant == "rht2" else [0.0]
    orders = None
    if variant == "rht2" and corr_mods:
        if args.static:
            CorrLinear.cfg.update({"k_frac": 0.0})
            orders = calib_static()
            print(f"S0 calibrated on {args.calib_chunks} x {args.seq} C4 tokens", flush=True)
        wrap(orders)
    for k in ks:
        CorrLinear.cfg.update({"k_frac": k, "basis": args.basis, "static": args.static, "first_tokens": args.first_tokens})
        p, tp = perplexity()
        rec = {
            "model": model_name,
            "variant": variant,
            "k": k,
            "residual": args.res if k > 0 else None,
            "selection_basis": args.basis if k > 0 else None,
            "writers": args.writers,
            "kappa": args.kappa,
            "static": args.static if k > 0 else False,
            "first_tokens": args.first_tokens,
            "tag": args.tag,
            "split": args.split,
            "chunks": num_chunks,
            "seq": args.seq,
            "ppl": round(p, 4),
            "time_s": round(tp, 1),
            "quant_s": round(tq, 1),
            "vram_max_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_file.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)
