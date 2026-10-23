import argparse
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from twobit.hadamard import SEED, layer_rot
from twobit.quantizers import lloyd_max, rht_lm2

SEQ = 2048
GAMMAS = (2, 4, 8)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--calib_data", default="data/qwen_c4_calib.pt")
parser.add_argument("--base_b", default="weights/ldlq_H/W_hat_H.safetensors")
parser.add_argument("--out", default="results/e03c_controls.json")
parser.add_argument("--texts_out", default="results/e03c_texts.pt")
parser.add_argument("--limit", type=int, default=20)
parser.add_argument("--prompts", type=int, default=16)
parser.add_argument("--prompt_len", type=int, default=256)
parser.add_argument("--gen", type=int, default=128)
parser.add_argument("--skip_precision", action="store_true")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}


def qsym(W, bits, g=None):
    qm = 2 ** (bits - 1) - 1
    if g is None:
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qm
        return (W / s).round().clamp(-qm, qm) * s
    d_out, d_in = W.shape
    G = W.reshape(d_out, d_in // g, g)
    s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qm
    return ((G / s).round().clamp(-qm, qm) * s).reshape(d_out, d_in)


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
BASES = {"A": W_base_a, "B": load_file(args.base_b)}


def ulp_perturb(Wb, frac, gen):
    I = Wb.view(torch.int16)
    r = torch.rand(Wb.shape, generator=gen, device=Wb.device)
    sgn = torch.where(torch.rand(Wb.shape, generator=gen, device=Wb.device) < 0.5, -1, 1).to(torch.int16)
    d = torch.where((r < frac) & (Wb != 0), sgn, torch.zeros_like(sgn))
    return (I + d).view(torch.bfloat16)


def weights(kind, base=None, bits=None, frac=None):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)
    for n in linears:
        W = orig[n].to("cuda", torch.float32)
        if kind == "bf16":
            Wn = W
        elif kind == "int8_row":
            Wn = qsym(W, 8)
        elif kind == "int8_g128":
            Wn = qsym(W, 8, 128)
        elif kind == "int4_g128":
            Wn = qsym(W, 4, 128)
        elif kind == "ulp":
            yield n, ulp_perturb(orig[n].to("cuda"), frac, gen)
            continue
        else:
            Wh = BASES[base][n].to("cuda", torch.float32)
            Wn = Wh if kind == "draft" else Wh + qsym(W - Wh, bits, 128)
        yield n, Wn.to(torch.bfloat16)


def load(*spec, **options):
    for n, Wn in weights(*spec, **options):
        linears[n].weight.data.copy_(Wn)


ids = torch.load(args.eval_data)
num_chunks = min(ids.numel() // SEQ, args.limit)
c4 = torch.load(args.calib_data)


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


t0 = time.time()
res = {"precision": {}, "bits_per_weight": {}, "acceptance": {}}
BPW = {"int8_row": 8.0, "int8_g128": 8.125, "int4_g128": 4.125}
if not args.skip_precision:
    with torch.no_grad():
        changed = total = 0
        for n, Wn in weights("verifier", "B", 8):
            changed += int((Wn != orig[n].to("cuda")).sum())
            total += Wn.numel()
    f_q8 = changed / total
    res["changed_weight_fraction_W_v_B_q8"] = f_q8
    print(f"W_v[B,q8]: changed bf16 weights = {f_q8 * 100:.2f} %", flush=True)
    load("bf16")
    Hb = hidden()
    plan = [("int8_row", dict(kind="int8_row")), ("int8_g128", dict(kind="int8_g128")), ("int4_g128", dict(kind="int4_g128"))]
    plan += [(f"W_v[B,q{b}]", dict(kind="verifier", base="B", bits=b)) for b in (4, 5, 6, 8)]
    plan += [(f"W_v[A,q{b}]", dict(kind="verifier", base="A", bits=b)) for b in (4, 6)]
    plan += [(f"floor_ulp_f{f_q8:.3f}", dict(kind="ulp", frac=f_q8)), ("floor_ulp_all", dict(kind="ulp", frac=1.0))]
    for key, kw in plan:
        load(**kw)
        res["precision"][key] = precision(Hb, hidden())
        if kw["kind"] == "verifier":
            BPW[key] = 2.125 + kw["bits"] + 0.125
        if key in BPW:
            res["bits_per_weight"][key] = BPW[key]
        print(f"{key:28s} {res['precision'][key]}  ({time.time() - t0:.0f} s)", flush=True)
    del Hb

PL, GEN = args.prompt_len, args.gen
prompts = [("wiki", ids[(num_chunks + 2 + p) * SEQ:(num_chunks + 2 + p) * SEQ + PL]) for p in range(args.prompts)]
prompts += [("c4", c4[(64 + p) * SEQ:(64 + p) * SEQ + PL]) for p in range(args.prompts)]


@torch.no_grad()
def generate(mode, T):
    outs = []
    for i, (_, p) in enumerate(prompts):
        torch.manual_seed(SEED + i)
        kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=T, top_k=0, top_p=1.0)
        o = model.generate(p.unsqueeze(0).cuda(), max_new_tokens=GEN, pad_token_id=0, **kw)
        outs.append(o[0].to("cpu"))
    return outs


@torch.no_grad()
def gen_logits(t):
    logits = model(input_ids=t.unsqueeze(0).cuda()).logits[0]
    return logits[PL - 1:t.numel() - 1].float()


def simulate_greedy(v, gamma):
    rounds = toks = 0
    i, n = 0, len(v)
    while i < n:
        k = 0
        while k < gamma and i + k < n and v[i + k]:
            k += 1
        step = min(k + 1, n - i)
        toks += step
        rounds += 1
        i += step
    return toks, rounds


def simulate_sampled(alpha, gamma, reps=200, seed=0):
    rng = random.Random(seed)
    toks = rounds = 0
    n = len(alpha)
    for _ in range(reps):
        i = 0
        while i < n:
            k = 0
            while k < gamma and i + k < n and rng.random() < alpha[i + k]:
                k += 1
            step = min(k + 1, n - i)
            toks += step
            rounds += 1
            i += step
    return toks, rounds


def distinct4(t):
    grams = [tuple(t[j:j + 4]) for j in range(len(t) - 3)]
    return len(set(grams)) / max(1, len(grams))


def per_text_rounds(per_text, kind):
    return {
        g: [simulate_greedy(v, g) if kind == "greedy" else simulate_sampled(v, g, seed=i) for i, v in enumerate(per_text)]
        for g in GAMMAS
    }


def summarize(per_text, rounds, idx):
    def expected_length(g, sel):
        return sum(rounds[g][i][0] for i in sel) / sum(rounds[g][i][1] for i in sel)

    s = {"mean_agreement_or_alpha": float(sum(sum(per_text[i]) for i in idx) / sum(len(per_text[i]) for i in idx))}
    s.update({f"E_L_g{g}": expected_length(g, idx) for g in GAMMAS})
    rng = random.Random(1)
    boots = sorted(expected_length(8, [rng.choice(idx) for _ in idx]) for _ in range(2000))
    s["E_L_g8_CI95"] = [boots[50], boots[1949]]
    return s


store = {"prompts": [(src, p) for src, p in prompts]}
for mode, T in (("greedy", None), ("T0.7", 0.7), ("T1.0", 1.0)):
    load("verifier", "B", 8)
    texts = generate(mode, T)
    P = [torch.softmax(gen_logits(t) / T, -1).to(torch.float16).cpu() for t in texts] if mode != "greedy" else None
    load("draft", "B")
    per = []
    for j, t in enumerate(texts):
        logits = gen_logits(t)
        if mode == "greedy":
            per.append([bool(x) for x in (logits.argmax(-1).cpu() == t[PL:]).tolist()])
        else:
            q = torch.softmax(logits / T, -1)
            p = P[j].cuda().float()
            per.append(torch.minimum(p, q).sum(-1).clamp(0, 1).cpu().tolist())
    idx_all = list(range(len(texts)))
    idx_wiki = [i for i in idx_all if prompts[i][0] == "wiki"]
    idx_c4 = [i for i in idx_all if prompts[i][0] == "c4"]
    d4 = [distinct4(t[PL:].tolist()) for t in texts]
    rounds = per_text_rounds(per, "greedy" if mode == "greedy" else "sampled")
    r = {
        "all": summarize(per, rounds, idx_all),
        "wiki": summarize(per, rounds, idx_wiki),
        "c4": summarize(per, rounds, idx_c4),
        "distinct_4grams_median": sorted(d4)[len(d4) // 2],
        "looping_texts_(4grams<0.5)": sum(x < 0.5 for x in d4),
    }
    non_looping = [i for i in idx_all if d4[i] >= 0.5]
    if non_looping and len(non_looping) < len(idx_all):
        r["without_looping"] = summarize(per, rounds, non_looping)
    r["first_32"] = float(sum(sum(v[:32]) for v in per) / sum(len(v[:32]) for v in per))
    r["after_first_32"] = float(sum(sum(v[32:]) for v in per) / max(1, sum(len(v[32:]) for v in per)))
    res["acceptance"][mode] = r
    store[mode] = {"texts": texts, "per_position": per, "distinct4": d4}
    print(f"[{mode}] {json.dumps(r, ensure_ascii=False)}  ({time.time() - t0:.0f} s)", flush=True)
    del P
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
os.makedirs(os.path.dirname(args.texts_out) or ".", exist_ok=True)
torch.save(store, args.texts_out)
print("DONE", flush=True)
