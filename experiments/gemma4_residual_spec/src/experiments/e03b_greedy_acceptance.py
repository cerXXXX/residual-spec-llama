import argparse
import json
import math
import os
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from twobit.hadamard import layer_rot
from twobit.quantizers import int8_row, lloyd_max, rht_lm2

SEQ = 2048

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--base_b", default="weights/ldlq_H/W_hat_H.safetensors")
parser.add_argument("--out", default="results/e03b_greedy_acceptance.json")
parser.add_argument("--limit", type=int, default=20)
parser.add_argument("--prompts", type=int, default=16)
parser.add_argument("--prompt_len", type=int, default=256)
parser.add_argument("--gen", type=int, default=128)
parser.add_argument("--gammas", default="2,4,8")
parser.add_argument("--bases", default="A,B")
parser.add_argument("--res8", default="row,g128")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}


def int8_group(R, g=128):
    d_out, d_in = R.shape
    G = R.reshape(d_out, d_in // g, g)
    s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    return ((G / s).round().clamp(-127, 127) * s).reshape(d_out, d_in)


levels = None
W_base_a = {}
for n in linears:
    W = orig[n].to("cuda", torch.float32)
    rot = layer_rot(n, W.shape[0], W.shape[1], W.device)
    if levels is None:
        Wt = rot.fwd(W)
        G = Wt.reshape(Wt.shape[0], -1, 128)
        v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
        levels = lloyd_max(v[torch.randperm(v.numel(), device=v.device)[:2_000_000]])
    W_base_a[n] = rht_lm2(W, 128, levels, rot).to("cpu", torch.bfloat16)
BASES = {"A": W_base_a}
if "B" in args.bases.split(","):
    BASES["B"] = load_file(args.base_b)


def load(kind, base=None, res8=None):
    for n, m in linears.items():
        W = orig[n].to("cuda", torch.float32)
        if kind == "bf16":
            Wn = W
        elif kind == "int8":
            Wn = int8_row(W)
        else:
            Wh = BASES[base][n].to("cuda", torch.float32)
            if kind == "draft":
                Wn = Wh
            else:
                Wn = Wh + (int8_row(W - Wh) if res8 == "row" else int8_group(W - Wh))
        m.weight.data.copy_(Wn.to(torch.bfloat16))


ids = torch.load(args.eval_data)
num_chunks = min(ids.numel() // SEQ, args.limit)


@torch.no_grad()
def hidden():
    return [model.model(input_ids=ids[c * SEQ:(c + 1) * SEQ].unsqueeze(0).cuda()).last_hidden_state[0].to("cpu") for c in range(num_chunks)]


@torch.no_grad()
def precision(Hb, Hv):
    nll = kl = 0.0
    flips = n = 0
    for c in range(num_chunks):
        y = ids[c * SEQ + 1:(c + 1) * SEQ + 1].cuda()
        for s0 in range(0, SEQ - 1, 256):
            s1 = min(s0 + 256, SEQ - 1)
            lb = torch.log_softmax(model.lm_head(Hb[c][s0:s1].cuda()).float(), -1)
            lv = torch.log_softmax(model.lm_head(Hv[c][s0:s1].cuda()).float(), -1)
            nll -= float(lv.gather(-1, y[s0:s1, None]).sum())
            kl += float((lb.exp() * (lb - lv)).sum())
            flips += int((lv.argmax(-1) != lb.argmax(-1)).sum())
            n += s1 - s0
    return {"ppl": math.exp(nll / n), "kl_bf16": kl / n, "flips": flips / n}


prompts = [ids[(num_chunks + 2 + p) * SEQ:(num_chunks + 2 + p) * SEQ + args.prompt_len] for p in range(args.prompts)]


@torch.no_grad()
def verifier_texts():
    outs = []
    for p in prompts:
        o = model.generate(p.unsqueeze(0).cuda(), max_new_tokens=args.gen, do_sample=False, pad_token_id=0)
        outs.append(o[0].to("cpu"))
    return outs


@torch.no_grad()
def draft_agreement(texts):
    agreement = []
    for t in texts:
        logits = model(input_ids=t.unsqueeze(0).cuda()).logits[0]
        pred = logits[:-1].argmax(-1).to("cpu")
        L = args.prompt_len
        agreement.append((pred[L - 1:] == t[L:]).to(torch.int8))
    return agreement


def simulate(agreement, gamma):
    rounds = toks = 0
    for v in agreement:
        i, n = 0, len(v)
        while i < n:
            k = 0
            while k < gamma and i + k < n and v[i + k] == 1:
                k += 1
            step = min(k + 1, n - i)
            toks += step
            rounds += 1
            i += step
    return toks / rounds


t0 = time.time()
res = {"precision": {}, "speculation": {}}
load("bf16")
Hb = hidden()
load("int8")
res["precision"]["int8"] = precision(Hb, hidden())
print(f"int8: {res['precision']['int8']}", flush=True)
for b in args.bases.split(","):
    for r8 in args.res8.split(","):
        load("verifier", b, r8)
        key = f"W_v[{b},{r8}]"
        res["precision"][key] = precision(Hb, hidden())
        print(f"{key}: {res['precision'][key]}", flush=True)
    load("verifier", b, "g128")
    texts = verifier_texts()
    load("draft", b)
    res["precision"][f"draft[{b}]"] = precision(Hb, hidden())
    agreement = draft_agreement(texts)
    p = float(torch.cat(agreement).float().mean())
    res["speculation"][b] = {"agreement": p, **{f"E_L_gamma{g}": simulate(agreement, int(g)) for g in args.gammas.split(",")}}
    print(f"draft[{b}]: {res['precision'][f'draft[{b}]']} | speculation: {res['speculation'][b]} ({time.time() - t0:.0f} s)", flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump({"chunks": num_chunks, "prompts": args.prompts, "prompt_len": args.prompt_len, "gen": args.gen, **res}, fh, indent=1)
