import argparse
import gc
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM

from experiments.e06_gemma_drafter import fp32_logits
from experiments.e06d_draft_policies import GAMMAS, POLICIES, bootstrap_ci, simulate, tokens_per_round
from experiments.e10_quantize_layerwise import make_logger
from twobit.base import reconstruct
from twobit.residual import symmetric_quant


def to_gpu(module, args, kwargs):
    module.to("cuda")


def to_cpu(module, args, output):
    module.to("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="weights/ldlq_chat_b2/codes.pt")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--residual-bits", default="8")
    ap.add_argument("--texts", default="results/e06c_gemma_eval.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = os.path.basename(os.path.dirname(os.path.abspath(a.codes)))
    out_path = a.out or f"results/e10_eval_{name}.json"
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
    codes = torch.load(a.codes, weights_only=False)
    levels = torch.tensor(codes["meta"]["levels"])
    S = torch.load(a.texts, weights_only=False)
    texts, prompt_lens = S["texts"], S["prompt_lens"]
    first_hit, run_len, sources = S["drafter_first_hit"], S["drafter_run_len"], S["sources"]
    B = len(texts)
    log(f"base {a.codes}: {codes['meta']['bits']} bits, calibration {codes['meta']['calib']}; {B} texts")

    @torch.no_grad()
    def set_weights(kind, residual_bits=8):
        for n, m in lins.items():
            if kind == "bf16":
                m.weight.data.copy_(orig[n])
                continue
            Wh = reconstruct(codes["layers"][n], levels, n)
            Wn = Wh if kind == "draft" else Wh + symmetric_quant(orig[n].to("cuda").float() - Wh, residual_bits, 128)
            m.weight.data.copy_(Wn.to("cpu", torch.bfloat16))

    @torch.no_grad()
    def logps(i_list):
        n = max(texts[i].numel() for i in i_list)
        ids = torch.zeros(len(i_list), n, dtype=torch.long)
        am = torch.zeros(len(i_list), n, dtype=torch.long)
        for k, i in enumerate(i_list):
            ids[k, :texts[i].numel()] = texts[i]
            am[k, :texts[i].numel()] = 1
        H = model.model(input_ids=ids.cuda(), attention_mask=am.cuda(), use_cache=False).last_hidden_state
        return {i: torch.log_softmax(fp32_logits(H[k, prompt_lens[i] - 1:texts[i].numel() - 1], model.lm_head.weight, softcap), -1)
                for k, i in enumerate(i_list)}

    batches = [list(range(i, min(i + a.bs, B))) for i in range(0, B, a.bs)]
    set_weights("bf16")
    LB = {}
    for b in batches:
        LB.update({i: v.cpu() for i, v in logps(b).items()})
        gc.collect()
        torch.cuda.empty_cache()
    log("bf16 reference done")
    res = {"prec": {}}
    base_hit = None
    for key in ["draft"] + [f"verif_q{r}" for r in a.residual_bits.split(",")]:
        if key == "draft":
            set_weights("draft", 8)
        else:
            set_weights("verif", int(key.split("_q")[1]))
        pc, hits = {}, {}
        for b in batches:
            for i, lv in logps(b).items():
                lb = LB[i].cuda()
                y = texts[i][prompt_lens[i]:].cuda()
                pc[i] = (float(-lv.gather(-1, y[:, None]).sum()), float((lb.exp() * (lb - lv)).sum()),
                         int((lv.argmax(-1) != lb.argmax(-1)).sum()), int(y.numel()))
                hits[i] = (lv.argmax(-1) == y).cpu().tolist()
            gc.collect()
            torch.cuda.empty_cache()
        n = sum(x[3] for x in pc.values())
        res["prec"][key] = {"ppl": math.exp(sum(x[0] for x in pc.values()) / n), "kl_bf16": sum(x[1] for x in pc.values()) / n,
                            "flips": sum(x[2] for x in pc.values()) / n}
        if key == "draft":
            base_hit = [hits[i] for i in range(B)]
        log(key, res["prec"][key])

    res["policies"] = {}
    for grp in ["all"] + sorted(set(sources)):
        idx = [i for i in range(B) if grp == "all" or sources[i] == grp]
        r = {"agreement_depth1_drafter": sum(sum(first_hit[i]) for i in idx) / sum(len(first_hit[i]) for i in idx),
             "agreement_base": sum(sum(base_hit[i]) for i in idx) / sum(len(base_hit[i]) for i in idx)}
        for pol in POLICIES:
            for g in GAMMAS:
                r[f"{pol}_g{g}"] = tokens_per_round(pol, first_hit, run_len, base_hit, idx, g)
            per = [simulate(pol, first_hit[i], run_len[i], base_hit[i], 8) for i in idx]
            r["CI95_g8_" + pol] = bootstrap_ci(per)
        res["policies"][grp] = r
        log(f"[{grp}] depth-1 agreement: drafter {r['agreement_depth1_drafter']:.3f}, base {r['agreement_base']:.3f} | g8: "
            f"base {r['base_g8']:.2f}, drafter {r['mtp_g8']:.2f}, tree {r['tree_g8']:.2f}, hybrid {r['mtp1_base_g8']:.2f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log("DONE")


if __name__ == "__main__":
    main()
