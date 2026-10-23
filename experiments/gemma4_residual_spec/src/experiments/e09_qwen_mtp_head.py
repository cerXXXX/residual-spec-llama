import argparse
import json
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from twobit.mtp_qwen import QwenMTPHead, model_dir, text_model

SEQ = 2048
PL = 256
GAMMAS = (1, 2, 3, 4, 8)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--eval_data", default="data/qwen_wikitext2_validation.pt")
parser.add_argument("--base_b", default="weights/ldlq_H/W_hat_H.safetensors")
parser.add_argument("--texts", default="results/e03c_texts.pt")
parser.add_argument("--chains_out", default="results/e09_chains.pt")
parser.add_argument("--out", default="results/e09_qwen_mtp_head.json")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--wiring_chunks", type=int, default=4)
args = parser.parse_args()

path = model_dir(args.model)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
tm = text_model(model)
lm_head = model.lm_head
head = QwenMTPHead(model, path)
print(f"MTP head: {head.num_tensors} tensors; full-attention layer, head_dim {head.head_dim}, KV groups {head.kv_groups}", flush=True)
captured = {}
tm.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("pre", i[0]))
hidden_source = "post"
chain_source = "post"

t0 = time.time()
res = {"wiring": {}}
ids = torch.load(args.eval_data)
with torch.no_grad():
    x = ids[:SEQ].cuda()
    h_post = model.model(input_ids=x[None]).last_hidden_state[0]
    X = head.inputs(h_post[:400], x[1:401], "eh")[None]
    ref = head.layer(X, position_embeddings=head.rotary(X, head.positions(0, 400)), attention_mask=None)
    mine, _, k_full, v_full = head.step(X, 0)
    res["wiring"]["max_diff_custom_vs_module"] = float((ref.float() - mine.float()).abs().max())
    prefix, _, _, _ = head.step(X[:, :300], 0)
    steps = []
    k_cache, v_cache = k_full[:, :, :300], v_full[:, :, :300]
    for j in range(300, 310):
        hj, _, kj, vj = head.step(X[:, j:j + 1], j, k_cache, v_cache)
        k_cache = torch.cat([k_cache, kj], 2)
        v_cache = torch.cat([v_cache, vj], 2)
        steps.append(float((hj[0, 0].float() - mine[0, j].float()).abs().max()))
    res["wiring"]["max_diff_stepwise_vs_full"] = max(steps)
    res["wiring"]["max_diff_causality"] = float((prefix.float() - mine[:, :300].float()).abs().max())
print(f"path tests: {res['wiring']}", flush=True)
acc = {f"{o}_{h}": [0, 0] for o in ("eh", "he") for h in ("post", "pre")}
main_acc = [0, 0]
with torch.no_grad():
    for c in range(args.wiring_chunks):
        x = ids[c * SEQ:(c + 1) * SEQ].cuda()
        h_post = model.model(input_ids=x[None]).last_hidden_state[0]
        h_pre = captured["pre"][0]
        main_acc[0] += int((head.argmax_tokens(h_post[:-1]) == x[1:]).sum())
        main_acc[1] += SEQ - 1
        for o in ("eh", "he"):
            for hn, H in (("post", h_post), ("pre", h_pre)):
                _, out, _, _ = head.step(head.inputs(H[:-1], x[1:], o)[None], 0)
                acc[f"{o}_{hn}"][0] += int((head.argmax_tokens(out[0, :-1]) == x[2:]).sum())
                acc[f"{o}_{hn}"][1] += SEQ - 2
res["wiring"]["model_next_token_accuracy"] = main_acc[0] / main_acc[1]
res["wiring"]["mtp_depth1_accuracy"] = {k: v[0] / v[1] for k, v in acc.items()}
best = max(acc, key=lambda k: acc[k][0] / acc[k][1])
head.order, hidden_source = best.split("_")
depth1_rounded = {k: round(v, 4) for k, v in res["wiring"]["mtp_depth1_accuracy"].items()}
print(
    f"model accuracy (next token) {res['wiring']['model_next_token_accuracy']:.4f}; MTP depth 1: {depth1_rounded} -> choice {best}",
    flush=True,
)
depth2 = {"post": [0, 0], "pre": [0, 0]}
with torch.no_grad():
    x = ids[:SEQ].cuda()
    h_post = model.model(input_ids=x[None]).last_hidden_state[0]
    H = h_post if hidden_source == "post" else captured["pre"][0]
    hs_all, out_all, K, V = head.step(head.inputs(H[:-1], x[1:])[None], 0)
    for i in range(200, 1800, 4):
        for chain, src in (("post", out_all), ("pre", hs_all)):
            _, o2, _, _ = head.step(head.inputs(src[0, i:i + 1], x[i + 2:i + 3])[None], i + 1, K[:, :, :i + 1], V[:, :, :i + 1])
            depth2[chain][0] += int(lm_head(o2[0, -1]).argmax() == x[i + 3])
            depth2[chain][1] += 1
res["wiring"]["mtp_depth2_accuracy"] = {k: v[0] / v[1] for k, v in depth2.items()}
chain_source = max(depth2, key=lambda k: depth2[k][0] / depth2[k][1])
print(f"MTP depth 2: {res['wiring']['mtp_depth2_accuracy']} -> chain {chain_source} ({time.time() - t0:.0f} s)", flush=True)

linears = {
    n: m for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n and "vision" not in n
}
base_b = load_file(args.base_b)


def qsym(W, bits, g):
    qm = 2 ** (bits - 1) - 1
    d_out, d_in = W.shape
    G = W.reshape(d_out, d_in // g, g)
    s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qm
    return ((G / s).round().clamp(-qm, qm) * s).reshape(d_out, d_in)


with torch.no_grad():
    for n, m in linears.items():
        W = m.weight.data.float()
        Wh = base_b[n].to("cuda", torch.float32)
        m.weight.data.copy_((Wh + qsym(W - Wh, 8, 128)).to(torch.bfloat16))
store = torch.load(args.texts, weights_only=False)


@torch.no_grad()
def verifier_hidden(t):
    hp = model.model(input_ids=t[None].cuda()).last_hidden_state[0]
    return hp if hidden_source == "post" else captured["pre"][0]


@torch.no_grad()
def mtp_chains(t, depth):
    n = t.numel()
    tc = t.cuda()
    H = verifier_hidden(t)
    hs_all, out_all, K, V = head.step(head.inputs(H[:-1], tc[1:])[None], 0)
    depth1_hits, chain_lengths = [], []
    for p in range(PL, n):
        s = p - 2
        src = out_all if chain_source == "post" else hs_all
        pred = int(lm_head(out_all[0, s]).argmax())
        depth1_hits.append(pred == int(t[p]))
        k = 0
        k_cache, v_cache = K[:, :, :s + 1], V[:, :, :s + 1]
        prev = src[0, s:s + 1]
        while k < depth and p + k < n and pred == int(t[p + k]):
            k += 1
            if k == depth or p + k >= n:
                break
            h2, o2, k2, v2 = head.step(head.inputs(prev, tc[p + k - 1:p + k])[None], s + k, k_cache, v_cache)
            k_cache = torch.cat([k_cache, k2], 2)
            v_cache = torch.cat([v_cache, v2], 2)
            prev = (o2 if chain_source == "post" else h2)[0]
            pred = int(lm_head(o2[0, -1]).argmax())
        chain_lengths.append(k)
    return depth1_hits, chain_lengths


def run_base(b, p, g):
    k = 0
    while k < g and p + k < len(b) and b[p + k]:
        k += 1
    return k


def simulate(policy, m1, kk, b, g):
    toks = rounds = 0
    i, n = 0, len(b)
    while i < n:
        if policy == "base":
            k = run_base(b, i, g)
        elif policy == "mtp":
            k = min(kk[i], g)
        elif policy == "tree":
            k = max(min(kk[i], g), run_base(b, i, g))
        else:
            k = (1 + run_base(b, i + 1, g - 1)) if (m1[i] or b[i]) and g >= 1 else 0
        step = min(k + 1, n - i)
        toks += step
        rounds += 1
        i += step
    return toks, rounds


texts = store["greedy"]["texts"]
base_agreement = store["greedy"]["per_position"]
M1, KK = [], []
for t in texts:
    m1, kk = mtp_chains(t, args.depth)
    M1.append(m1)
    KK.append(kk)
print(f"MTP chains computed ({time.time() - t0:.0f} s)", flush=True)
os.makedirs(os.path.dirname(args.chains_out) or ".", exist_ok=True)
torch.save({"depth1_hits": M1, "chain_lengths": KK}, args.chains_out)
sources = [s for s, _ in store["prompts"]]
d4 = store["greedy"]["distinct4"]
res["greedy"] = {}
groups = (
    ("all", range(len(texts))),
    ("wiki", [i for i in range(len(texts)) if sources[i] == "wiki"]),
    ("c4", [i for i in range(len(texts)) if sources[i] == "c4"]),
    ("without_looping", [i for i in range(len(texts)) if d4[i] >= 0.5]),
)
for group, idx in groups:
    idx = list(idx)
    r = {
        "mtp_depth1_agreement": sum(sum(M1[i]) for i in idx) / sum(len(M1[i]) for i in idx),
        "base_agreement": sum(sum(base_agreement[i]) for i in idx) / sum(len(base_agreement[i]) for i in idx),
    }
    for policy in ("base", "mtp", "tree", "mtp1_base"):
        for g in GAMMAS:
            tr = [simulate(policy, M1[i], KK[i], base_agreement[i], g) for i in idx]
            r[f"{policy}_g{g}"] = sum(x[0] for x in tr) / sum(x[1] for x in tr)
        rng = random.Random(1)
        per_text = [simulate(policy, M1[i], KK[i], base_agreement[i], 8) for i in idx]
        boots = []
        for _ in range(2000):
            sel = [rng.randrange(len(idx)) for _ in idx]
            boots.append(sum(per_text[s][0] for s in sel) / sum(per_text[s][1] for s in sel))
        boots.sort()
        r[f"{policy}_g8_CI95"] = [boots[50], boots[1949]]
    res["greedy"][group] = r
    rounded = {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}
    print(f"[{group}] " + json.dumps(rounded, ensure_ascii=False), flush=True)

res["sampling_depth1"] = {}
for mode, T in (("T0.7", 0.7), ("T1.0", 1.0)):
    alphas = []
    with torch.no_grad():
        for t in store[mode]["texts"]:
            n = t.numel()
            tc = t.cuda()
            logits = model(input_ids=t[None].cuda()).logits[0]
            H = model.model(input_ids=t[None].cuda()).last_hidden_state[0]
            if hidden_source == "pre":
                H = captured["pre"][0]
            _, out, _, _ = head.step(head.inputs(H[:-1], tc[1:])[None], 0)
            p = torch.softmax(logits[PL - 1:n - 1].float() / T, -1)
            q = torch.softmax(lm_head(out[0, PL - 2:n - 2]).float() / T, -1)
            alphas += torch.minimum(p, q).sum(-1).clamp(0, 1).tolist()
    base_alphas = [x for v in store[mode]["per_position"] for x in v]
    res["sampling_depth1"][mode] = {"alpha_mtp": sum(alphas) / len(alphas), "alpha_base": sum(base_alphas) / len(base_alphas)}
    print(f"[{mode}] {res['sampling_depth1'][mode]}", flush=True)
res["config"] = {"order": head.order, "hidden": hidden_source, "chain": chain_source, "depth": args.depth}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
print(f"DONE ({time.time() - t0:.0f} s)", flush=True)
