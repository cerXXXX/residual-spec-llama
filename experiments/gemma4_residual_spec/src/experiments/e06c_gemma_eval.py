import argparse
import gc
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from experiments.e06_gemma_drafter import drafter_chain, fp32_logits, paired_bootstrap, ulp_perturb
from experiments.e10_quantize_layerwise import make_logger
from twobit.base import GROUP
from twobit.hadamard import SEED
from twobit.residual import symmetric_quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--drafter", default="google/gemma-4-12B-it-assistant")
    ap.add_argument("--base-weights", default="weights/ldlq_c4_b2/base.safetensors")
    ap.add_argument("--prompts", default="data/prompts.pt")
    ap.add_argument("--out", default="results/e06c_gemma_eval")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    dev = "cpu" if a.dry else "cuda"
    if a.dry:
        a.n_prompts, a.gen, a.depth, a.bs = 2, 8, 4, 2
    out_prefix = a.out + ("_dry" if a.dry else "")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    log = make_logger(time.time())
    S = {}
    RES = {}

    def save():
        torch.save(S, out_prefix + ".pt")
        with open(out_prefix + ".json", "w") as f:
            json.dump(RES, f, indent=1, ensure_ascii=False)

    def clean(tag=""):
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
            log(f"{tag} VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev).eval()
    drafter = AutoModelForCausalLM.from_pretrained(a.drafter, dtype=torch.bfloat16, device_map=dev).eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    emb = model.get_input_embeddings()
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model.layers." in n}
    orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in lins.items()}
    base = None if a.dry else load_file(a.base_weights)
    prompts = torch.load(a.prompts)[: a.n_prompts]
    prompt_lens = [p.numel() for _, p in prompts]
    S["sources"] = [s for s, _ in prompts]
    clean("start:")

    @torch.no_grad()
    def set_weights(kind, bits=None, g=None, frac=None):
        gen = torch.Generator(device=dev)
        gen.manual_seed(SEED)
        changed = total = 0
        for n, m in lins.items():
            if kind == "ulp":
                m.weight.data.copy_(ulp_perturb(orig[n].to(dev), frac, gen))
                continue
            W = orig[n].to(dev, torch.float32)
            if kind == "bf16":
                Wn = W
            elif kind == "int8":
                Wn = symmetric_quant(W, 8, g)
            else:
                Wh = base[n].to(dev, torch.float32)
                Wn = Wh if kind == "draft" else Wh + symmetric_quant(W - Wh, bits, GROUP)
            Wb = Wn.to(torch.bfloat16)
            changed += int((Wb != orig[n].to(dev)).sum())
            total += Wb.numel()
            m.weight.data.copy_(Wb)
        return changed / max(1, total)

    @torch.no_grad()
    def gen_one(p, **kw):
        return model.generate(input_ids=p[None].to(dev), attention_mask=torch.ones(1, p.numel(), dtype=torch.long, device=dev),
                              max_new_tokens=a.gen, **kw)[0].cpu()

    @torch.no_grad()
    def fwd_text(t):
        o = model(input_ids=t[None].to(dev), output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
        H, KV = o.hidden_states[-1], o.shared_kv_states
        del o
        return H, KV

    def chain(H, KV, t, s, depth, stop=True):
        return drafter_chain(drafter, emb, H, KV, t, s, depth, 0, dev, stop)

    def z_fp32(x):
        return fp32_logits(x, model.lm_head.weight, softcap)

    @torch.no_grad()
    def hidden_batch(texts, idx):
        n = max(texts[i].numel() for i in idx)
        ids = torch.zeros(len(idx), n, dtype=torch.long)
        am = torch.zeros(len(idx), n, dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :texts[i].numel()] = texts[i]
            am[k, :texts[i].numel()] = 1
        return model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False).last_hidden_state

    RES["changed_frac_W_v_q8"] = None if a.dry else set_weights("verif", bits=8)
    if a.dry:
        set_weights("bf16")
    tg = time.time()
    texts = [gen_one(p, do_sample=False, top_k=None, top_p=None) for _, p in prompts]
    tg = time.time() - tg
    nt = sum(t.numel() - pl for t, pl in zip(texts, prompt_lens))
    RES["generation"] = {"tokens": nt, "s": tg, "tok_s": nt / tg}
    S["texts"] = texts
    S["prompt_lens"] = prompt_lens
    save()
    log("1) texts saved:", RES["generation"])
    clean("after 1:")

    first_hit, run_len = [], []
    for j, t in enumerate(texts):
        H, KV = fwd_text(t)
        m1, kk = [], []
        for p in range(prompt_lens[j], t.numel()):
            em, _ = chain(H, KV, t, p - 1, a.depth)
            k = 0
            while k < len(em) and p + k < t.numel() and em[k] == int(t[p + k]):
                k += 1
            kk.append(k)
            m1.append(bool(em[0] == int(t[p])))
        first_hit.append(m1)
        run_len.append(kk)
        del H, KV
    S["drafter_first_hit"], S["drafter_run_len"] = first_hit, run_len
    save()
    log("2) drafter chains:", sum(map(sum, first_hit)), "/", sum(map(len, first_hit)), "depth-1 hits")
    clean("after 2:")

    B = len(texts)
    batches = [list(range(i, min(i + a.bs, B))) for i in range(0, B, a.bs)]
    set_weights("bf16")
    LB = [None] * B
    for idx in batches:
        H = hidden_batch(texts, idx)
        for k, i in enumerate(idx):
            LB[i] = torch.log_softmax(z_fp32(H[k, prompt_lens[i] - 1:texts[i].numel() - 1]), -1).cpu()
        del H
    clean("after bf16 reference:")
    plan = [("int8_row", dict(kind="int8")), ("int8_g128", dict(kind="int8", g=128)), ("int8_g32", dict(kind="int8", g=32))]
    if not a.dry:
        plan += [("W_v[B,q6]", dict(kind="verif", bits=6)), ("W_v[B,q8]", dict(kind="verif", bits=8)), ("draft[B]", dict(kind="draft")),
                 ("ulp_floor", dict(kind="ulp", frac=RES["changed_frac_W_v_q8"]))]
    else:
        plan += [("ulp_floor", dict(kind="ulp", frac=0.5))]
    RES["prec"] = {}
    S["per_text"] = {}
    for key, kw in plan:
        set_weights(**kw)
        pc = [None] * B
        hits = [None] * B
        for idx in batches:
            H = hidden_batch(texts, idx)
            for k, i in enumerate(idx):
                lv = torch.log_softmax(z_fp32(H[k, prompt_lens[i] - 1:texts[i].numel() - 1]), -1)
                lb = LB[i].to(dev)
                y = texts[i][prompt_lens[i]:].to(dev)
                pc[i] = (float(-lv.gather(-1, y[:, None]).sum()), float((lb.exp() * (lb - lv)).sum()),
                         int((lv.argmax(-1) != lb.argmax(-1)).sum()), int(y.numel()))
                hits[i] = (lv.argmax(-1) == y).cpu().tolist()
            del H
        n = sum(x[3] for x in pc)
        RES["prec"][key] = {"ppl": math.exp(sum(x[0] for x in pc) / n), "kl_bf16": sum(x[1] for x in pc) / n,
                            "flips": sum(x[2] for x in pc) / n, "tokens": n}
        S["per_text"][key] = pc
        if key == "draft[B]":
            S["base_hit"] = hits
        save()
        log("3)", key, RES["prec"][key])
    rng = random.Random(7)
    pairs = [("int8_g32", "ulp_floor")]
    if not a.dry:
        pairs += [("W_v[B,q8]", "int8_g32"), ("W_v[B,q8]", "int8_g128"), ("W_v[B,q8]", "ulp_floor"), ("W_v[B,q6]", "int8_g128")]
    per_text = S["per_text"]
    RES["paired"] = {f"{x} - {y}": {"KL": paired_bootstrap(per_text[x], per_text[y], 1, rng), "flips": paired_bootstrap(per_text[x], per_text[y], 2, rng)}
                     for x, y in pairs}
    save()
    log("3) paired:", json.dumps(RES["paired"]))
    del LB
    clean("after 3:")

    T = a.temperature
    if not a.dry:
        set_weights("verif", bits=8)
    else:
        set_weights("bf16")
    stexts = []
    for j, (_, p) in enumerate(prompts):
        torch.manual_seed(SEED + j)
        stexts.append(gen_one(p, do_sample=True, temperature=T, top_k=0, top_p=1.0))
    S["sampled_texts"] = stexts
    save()
    P, alpha_drafter, pmax_drafter = [], [], []
    for j, t in enumerate(stexts):
        H, KV = fwd_text(t)
        zp = z_fp32(H[0, prompt_lens[j] - 1:t.numel() - 1])
        p = torch.softmax(zp / T, -1)
        P.append(p.half().cpu())
        for q_i, pp in enumerate(range(prompt_lens[j], t.numel())):
            _, l1 = chain(H, KV, t, pp - 1, 1, stop=False)
            q = torch.softmax(l1 / T, -1)
            alpha_drafter.append(float(torch.minimum(p[q_i], q).sum()))
            pmax_drafter.append(float(p[q_i, q.argmax()]))
        del H, KV, zp, p
    alpha_base, pmax_base = [], []
    if not a.dry:
        set_weights("draft")
        for j, t in enumerate(stexts):
            H = hidden_batch(stexts, [j])
            q = torch.softmax(z_fp32(H[0, prompt_lens[j] - 1:t.numel() - 1]) / T, -1)
            p = P[j].to(dev).float()
            alpha_base += torch.minimum(p, q).sum(-1).tolist()
            pmax_base += p.gather(-1, q.argmax(-1, keepdim=True))[:, 0].tolist()
            del H
    RES["sampling"] = {"T": T, "alpha_drafter_depth1": sum(alpha_drafter) / max(1, len(alpha_drafter)),
                       "p_argmax_drafter": sum(pmax_drafter) / max(1, len(pmax_drafter)),
                       "alpha_base": sum(alpha_base) / max(1, len(alpha_base)) if alpha_base else None,
                       "p_argmax_base": sum(pmax_base) / max(1, len(pmax_base)) if pmax_base else None, "positions": len(alpha_drafter)}
    S["alpha"] = {"drafter": alpha_drafter, "base": alpha_base}
    save()
    log("4) sampling:", RES["sampling"])
    log("DONE")


if __name__ == "__main__":
    main()
