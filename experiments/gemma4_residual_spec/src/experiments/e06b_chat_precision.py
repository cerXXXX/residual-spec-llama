import argparse
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from experiments.e06_gemma_drafter import fp32_logits, paired_bootstrap, ulp_perturb
from experiments.e10_quantize_layerwise import make_logger
from twobit.hadamard import SEED
from twobit.residual import symmetric_quant


def to_gpu(module, args, kwargs):
    module.to("cuda")


def to_cpu(module, args, output):
    module.to("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--base-weights", default="weights/ldlq_c4_b2/base.safetensors")
    ap.add_argument("--arrays", default="results/e06_gemma_arrays.pt")
    ap.add_argument("--prompts", default="data/prompts.pt")
    ap.add_argument("--ulp-frac", type=float, default=0.576)
    ap.add_argument("--out", default="results/e06b_chat_precision.json")
    a = ap.parse_args()
    log = make_logger(time.time())
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    lm = model.model.language_model
    lm.embed_tokens.to("cuda")
    lm.norm.to("cuda")
    lm.rotary_emb.to("cuda")
    for layer in lm.layers:
        layer.register_forward_pre_hook(to_gpu, with_kwargs=True)
        layer.register_forward_hook(to_cpu)
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model.layers." in n}
    orig = {n: m.weight.detach().clone() for n, m in lins.items()}
    base = load_file(a.base_weights)
    arrays = torch.load(a.arrays, weights_only=False)
    prompts = torch.load(a.prompts)
    texts = arrays["texts"]
    prompt_lens = [p.numel() for _, p in prompts[:len(texts)]]
    N = max(t.numel() for t in texts)
    B = len(texts)
    ids = torch.zeros(B, N, dtype=torch.long)
    am = torch.zeros(B, N, dtype=torch.long)
    for i, t in enumerate(texts):
        ids[i, :t.numel()] = t
        am[i, :t.numel()] = 1
    log(f"{B} texts, max {N} tokens, response positions: {sum(t.numel() - pl for t, pl in zip(texts, prompt_lens))}")

    @torch.no_grad()
    def set_weights(kind, bits=None, g=None, frac=None):
        gen = torch.Generator()
        gen.manual_seed(SEED)
        for n, m in lins.items():
            if kind == "ulp":
                m.weight.data.copy_(ulp_perturb(orig[n], frac, gen))
                continue
            W = orig[n].to("cuda", torch.float32)
            if kind == "bf16":
                Wn = W
            elif kind == "int8":
                Wn = symmetric_quant(W, 8, g)
            else:
                Wh = base[n].to("cuda", torch.float32)
                Wn = Wh if kind == "draft" else Wh + symmetric_quant(W - Wh, bits, 128)
            m.weight.data.copy_(Wn.to("cpu", torch.bfloat16))

    @torch.no_grad()
    def hidden():
        return model.model(input_ids=ids.cuda(), attention_mask=am.cuda(), use_cache=False).last_hidden_state

    @torch.no_grad()
    def logp(h, i):
        return torch.log_softmax(fp32_logits(h[i, prompt_lens[i] - 1:texts[i].numel() - 1], model.lm_head.weight, softcap), -1)

    set_weights("bf16")
    Hb = hidden()
    LB = [logp(Hb, i).cpu() for i in range(B)]
    del Hb
    log("bf16 done")
    PC, res = {}, {"prec": {}}
    plan = [("bf16_check", dict(kind="bf16")), ("int8_row", dict(kind="int8")), ("int8_g128", dict(kind="int8", g=128)),
            ("int8_g32", dict(kind="int8", g=32)), ("W_v[B,q6]", dict(kind="verif", bits=6)), ("W_v[B,q8]", dict(kind="verif", bits=8)),
            ("draft[B]", dict(kind="draft")), ("ulp_floor", dict(kind="ulp", frac=a.ulp_frac))]
    for key, kw in plan:
        set_weights(**kw)
        H = hidden()
        pc = []
        for i in range(B):
            lb = LB[i].cuda()
            lv = logp(H, i)
            y = texts[i][prompt_lens[i]:].cuda()
            pc.append((float(-lv.gather(-1, y[:, None]).sum()), float((lb.exp() * (lb - lv)).sum()),
                       int((lv.argmax(-1) != lb.argmax(-1)).sum()), y.numel()))
        del H
        PC[key] = pc
        n = sum(x[3] for x in pc)
        res["prec"][key] = {"ppl": math.exp(sum(x[0] for x in pc) / n), "kl_bf16": sum(x[1] for x in pc) / n,
                            "flips": sum(x[2] for x in pc) / n, "tokens": n}
        log(key, res["prec"][key])
    rng = random.Random(7)
    pairs = (("W_v[B,q8]", "int8_g32"), ("W_v[B,q8]", "int8_g128"), ("W_v[B,q8]", "ulp_floor"), ("W_v[B,q6]", "int8_g128"), ("int8_g32", "ulp_floor"))
    res["paired"] = {f"{x} - {y}": {"KL": paired_bootstrap(PC[x], PC[y], 1, rng), "flips": paired_bootstrap(PC[x], PC[y], 2, rng)} for x, y in pairs}
    for k, v in res["paired"].items():
        log(f"{k:32s} KL {v['KL']['diff']:+.6f} {[round(x, 6) for x in v['KL']['CI95']]}  "
            f"flips {v['flips']['diff'] * 100:+.3f} pp {[round(x * 100, 3) for x in v['flips']['CI95']]}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log("DONE")


if __name__ == "__main__":
    main()
