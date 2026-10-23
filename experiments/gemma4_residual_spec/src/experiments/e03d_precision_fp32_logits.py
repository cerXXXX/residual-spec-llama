import argparse
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from twobit.hadamard import SEED

SEQ = 2048
NUM_CHUNKS = 40

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--base_b", default="weights/ldlq_H/W_hat_H.safetensors")
parser.add_argument("--out", default="results/e03d_precision_fp32_logits.json")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}
base_b = load_file(args.base_b)
lm_weight = model.lm_head.weight.detach().float()


def qsym(W, bits, g=None):
    qm = 2 ** (bits - 1) - 1
    if g is None:
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qm
        return (W / s).round().clamp(-qm, qm) * s
    d_out, d_in = W.shape
    G = W.reshape(d_out, d_in // g, g)
    s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qm
    return ((G / s).round().clamp(-qm, qm) * s).reshape(d_out, d_in)


def ulp_perturb(Wb, frac, gen):
    I = Wb.view(torch.int16)
    r = torch.rand(Wb.shape, generator=gen, device=Wb.device)
    sgn = torch.where(torch.rand(Wb.shape, generator=gen, device=Wb.device) < 0.5, -1, 1).to(torch.int16)
    return (I + torch.where((r < frac) & (Wb != 0), sgn, torch.zeros_like(sgn))).view(torch.bfloat16)


@torch.no_grad()
def load(kind, bits=None, g=None, frac=None):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)
    changed = total = 0
    for n, m in linears.items():
        W = orig[n].to("cuda", torch.float32)
        if kind == "bf16":
            Wn = W
        elif kind == "int8":
            Wn = qsym(W, 8, g)
        elif kind == "ulp":
            m.weight.data.copy_(ulp_perturb(orig[n].to("cuda"), frac, gen))
            continue
        else:
            Wh = base_b[n].to("cuda", torch.float32)
            Wn = Wh + qsym(W - Wh, bits, 128)
        Wb = Wn.to(torch.bfloat16)
        changed += int((Wb != orig[n].to("cuda")).sum())
        total += Wb.numel()
        m.weight.data.copy_(Wb)
    return changed / max(1, total)


ids = torch.load(args.eval_data)


@torch.no_grad()
def hidden():
    return [model.model(input_ids=ids[c * SEQ:(c + 1) * SEQ].unsqueeze(0).cuda()).last_hidden_state[0].to("cpu") for c in range(NUM_CHUNKS)]


@torch.no_grad()
def per_chunk(Hb, Hv):
    out = []
    for c in range(NUM_CHUNKS):
        y = ids[c * SEQ + 1:(c + 1) * SEQ + 1].cuda()
        nll = kl = 0.0
        flips = n = 0
        for s0 in range(0, SEQ - 1, 256):
            s1 = min(s0 + 256, SEQ - 1)
            lb = torch.log_softmax(Hb[c][s0:s1].cuda().float() @ lm_weight.T, -1)
            lv = torch.log_softmax(Hv[c][s0:s1].cuda().float() @ lm_weight.T, -1)
            nll -= float(lv.gather(-1, y[s0:s1, None]).sum())
            kl += float((lb.exp() * (lb - lv)).sum())
            flips += int((lv.argmax(-1) != lb.argmax(-1)).sum())
            n += s1 - s0
        out.append((nll, kl, flips, n))
    return out


def aggregate(pc):
    n = sum(x[3] for x in pc)
    return {"ppl": math.exp(sum(x[0] for x in pc) / n), "kl_bf16": sum(x[1] for x in pc) / n, "flips": sum(x[2] for x in pc) / n}


t0 = time.time()
load("bf16")
Hb = hidden()
PC = {}
res = {"precision": {}, "changed_weight_fraction": {}, "bits_per_weight": {}}
PC["bf16_vs_bf16"] = per_chunk(Hb, Hb)
plan = [
    ("int8_row", dict(kind="int8"), 8.0),
    ("int8_g128", dict(kind="int8", g=128), 8.125),
    ("int8_g64", dict(kind="int8", g=64), 8.25),
    ("int8_g32", dict(kind="int8", g=32), 8.5),
    ("W_v[B,q6]", dict(kind="verifier", bits=6), 8.25),
    ("W_v[B,q8]", dict(kind="verifier", bits=8), 10.25),
]
for key, kw, bpw in plan:
    f = load(**kw)
    PC[key] = per_chunk(Hb, hidden())
    res["precision"][key] = aggregate(PC[key])
    res["changed_weight_fraction"][key] = f
    res["bits_per_weight"][key] = bpw
    print(f"{key:12s} {bpw:6.3f} bits/weight  changed {f * 100:5.1f} %  {res['precision'][key]}  ({time.time() - t0:.0f} s)", flush=True)
f_q8 = res["changed_weight_fraction"]["W_v[B,q8]"]
for key, frac in ((f"floor_ulp_f{f_q8:.3f}", f_q8), ("floor_ulp_all", 1.0)):
    load("ulp", frac=frac)
    PC[key] = per_chunk(Hb, hidden())
    res["precision"][key] = aggregate(PC[key])
    print(f"{key:24s} {res['precision'][key]}  ({time.time() - t0:.0f} s)", flush=True)
res["precision"]["bf16_vs_bf16"] = aggregate(PC["bf16_vs_bf16"])


def paired(key_a, key_b, metric):
    rng = random.Random(7)
    diffs = []

    def mean(pc, sel):
        return sum(pc[i][metric] for i in sel) / sum(pc[i][3] for i in sel)

    point = mean(PC[key_a], range(NUM_CHUNKS)) - mean(PC[key_b], range(NUM_CHUNKS))
    for _ in range(4000):
        sel = [rng.randrange(NUM_CHUNKS) for _ in range(NUM_CHUNKS)]
        diffs.append(mean(PC[key_a], sel) - mean(PC[key_b], sel))
    diffs.sort()
    return {"difference": point, "CI95": [diffs[100], diffs[3899]]}


floor_key = f"floor_ulp_f{f_q8:.3f}"
pairs = [
    ("W_v[B,q8]", "int8_g32"),
    ("W_v[B,q8]", "int8_g128"),
    ("W_v[B,q8]", floor_key),
    ("W_v[B,q6]", "int8_row"),
    ("W_v[B,q6]", "int8_g128"),
    ("int8_g32", floor_key),
]
res["paired"] = {f"{x} - {y}": {"KL": paired(x, y, 1), "flips": paired(x, y, 2)} for x, y in pairs}
for k, v in res["paired"].items():
    kl, fl = v["KL"], v["flips"]
    print(
        f"{k:40s} KL {kl['difference']:+.6f} [{kl['CI95'][0]:+.6f}, {kl['CI95'][1]:+.6f}]   "
        f"flips {fl['difference'] * 100:+.3f} pp [{fl['CI95'][0] * 100:+.3f}, {fl['CI95'][1] * 100:+.3f}]",
        flush=True,
    )
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
print(f"DONE ({time.time() - t0:.0f} s)", flush=True)
