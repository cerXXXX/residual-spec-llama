import argparse
import gc
import json
import math
import os
import time
from collections import UserDict

import torch
from transformers import AutoModelForCausalLM

from experiments.e06_gemma_drafter import fp32_logits
from experiments.e10_quantize_layerwise import capture_layer_inputs, capture_layer_kwargs, gram_hook, layer_linears, make_logger, to_device
from twobit.hadamard import layer_rot, rotate_hessian
from twobit.residual import quantize_residual, residual_levels, rms_normalized, rotated_base

REFERENCE_KEYS = ("int8_row", "int8_g128", "int8_g32", "W_v[B,q8]", "ulp_floor")


def to_gpu(module, args, kwargs):
    module.to("cuda")


def to_cpu(module, args, output):
    module.to("cpu")


def parse_variant(name):
    grid = "lm" if name.startswith("lm") else "uni"
    bits = int(name[len(grid)])
    use_gptq = name.endswith("g")
    return grid, bits, use_gptq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="lm4g,lm5g,lm6g,uni6g,uni8r")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--codes", default="weights/ldlq_chat_b2/codes.pt")
    ap.add_argument("--calib", default="data/calib_chat.pt")
    ap.add_argument("--texts", default="results/e06c_gemma_eval.pt")
    ap.add_argument("--refs", default="results/e06c_gemma_eval.json")
    ap.add_argument("--out", default="results/e16_residual_gptq.json")
    a = ap.parse_args()
    log = make_logger(time.time())
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    lm = model.model.language_model
    layers = lm.layers
    layer_types = lm.config.layer_types
    for _, child in model.model.named_children():
        if child is lm:
            for sub_name, sub in lm.named_children():
                if sub_name != "layers":
                    sub.to("cuda")
        else:
            child.to("cuda")
    model.lm_head.to("cuda")
    linears = layer_linears(layers)
    lins = {n: m for d in linears.values() for n, m in d.items()}
    orig = {n: m.weight.detach().clone() for n, m in lins.items()}
    codes = torch.load(a.codes, weights_only=False)
    base_levels = torch.tensor(codes["meta"]["levels"]).cuda()
    C = torch.load(a.calib)
    n = C.shape[0]
    S = torch.load(a.texts, weights_only=False)
    texts, prompt_lens = S["texts"], S["prompt_lens"]
    B = len(texts)
    with open(a.refs) as f:
        refs = json.load(f)["prec"]

    stream = [L.register_forward_pre_hook(to_gpu, with_kwargs=True) for L in layers] + [L.register_forward_hook(to_cpu) for L in layers]
    kwargs_cap = capture_layer_kwargs(lm, layers, layer_types, C[0:1].cuda())
    for h in stream:
        h.remove()
    layers[layer_types.index("full_attention")].to("cpu")
    X0 = capture_layer_inputs(lm, layers[0], C, "cuda")
    kwargs_dev = {t: to_device(v, "cuda") for t, v in kwargs_cap.items()}
    log(f"captured kwargs {list(kwargs_cap)}; layer 0 inputs: {len(X0)}")
    if len(kwargs_cap) < 2:
        raise SystemExit("missing kwargs for both layer types")
    level_cache = {}

    @torch.no_grad()
    def build_variant(grid, bits, use_gptq):
        X = [x for x in X0]
        weights = {}
        for L in range(len(layers)):
            layer = layers[L].to("cuda")
            kw = kwargs_dev[layer_types[L]]
            mods = linears[L]
            acc = {nm: [None, 0] for nm in mods}
            hooks = [m.register_forward_hook(gram_hook(acc, nm)) for nm, m in mods.items()] if use_gptq else []
            Xn = [layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw).to(torch.bfloat16).cpu() for s in range(n)]
            for h in hooks:
                h.remove()
            for nm, m in mods.items():
                W = orig[nm].to("cuda", torch.float32)
                rot = layer_rot(nm, W.shape[0], W.shape[1], W.device)
                Qb = rotated_base(codes["layers"][nm], base_levels)
                Rt = (rot.fwd(W) - Qb).contiguous()
                levels = residual_levels(grid, bits, rms_normalized(Rt), level_cache)
                Hr = rotate_hessian(acc[nm][0] / acc[nm][1], rot) if use_gptq else None
                acc[nm][0] = None
                Qr = quantize_residual(Rt, Hr, grid, levels, use_gptq, a.damp)
                weights[nm] = rot.inv(Qb + Qr).to("cpu", torch.bfloat16)
                del W, Qb, Rt, Qr, Hr
            torch.cuda.empty_cache()
            layer.to("cpu")
            X = Xn
        return weights

    @torch.no_grad()
    def evaluate(weights):
        hooks = []
        for L in layers:
            hooks += [L.register_forward_pre_hook(to_gpu, with_kwargs=True), L.register_forward_hook(to_cpu)]
        for nm, m in lins.items():
            m.weight.data.copy_(orig[nm] if weights is None else weights[nm])
        W = model.lm_head.weight
        out = {}
        for s0 in range(0, B, a.bs):
            idx = list(range(s0, min(s0 + a.bs, B)))
            nmax = max(texts[i].numel() for i in idx)
            ids = torch.zeros(len(idx), nmax, dtype=torch.long)
            am = torch.zeros(len(idx), nmax, dtype=torch.long)
            for k, i in enumerate(idx):
                ids[k, :texts[i].numel()] = texts[i]
                am[k, :texts[i].numel()] = 1
            H = model.model(input_ids=ids.cuda(), attention_mask=am.cuda(), use_cache=False).last_hidden_state
            for k, i in enumerate(idx):
                out[i] = torch.log_softmax(fp32_logits(H[k, prompt_lens[i] - 1:texts[i].numel() - 1], W, softcap), -1).cpu()
            del H
            gc.collect()
            torch.cuda.empty_cache()
        for h in hooks:
            h.remove()
        for nm, m in lins.items():
            m.weight.data.copy_(orig[nm])
        return out

    LB = evaluate(None)
    log("bf16 reference done")
    if os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)
    else:
        res = {"variants": {}}
    res["references"] = {k: refs[k] for k in REFERENCE_KEYS if k in refs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for vname in a.variants.split(","):
        grid, bits, use_gptq = parse_variant(vname)
        tq = time.time()
        weights = build_variant(grid, bits, use_gptq)
        tq = time.time() - tq
        LV = evaluate(weights)
        del weights
        gc.collect()
        nll = kl = fl = cnt = 0.0
        for i in range(B):
            lb, lv = LB[i], LV[i]
            y = texts[i][prompt_lens[i]:]
            nll -= float(lv.gather(-1, y[:, None]).sum())
            kl += float((lb.exp() * (lb - lv)).sum())
            fl += int((lv.argmax(-1) != lb.argmax(-1)).sum())
            cnt += y.numel()
        del LV
        r = {"grid": grid, "bits": bits, "gptq": use_gptq, "residual_bits_per_weight": bits + 0.125, "ppl": math.exp(nll / cnt),
             "kl_bf16": kl / cnt, "flips": fl / cnt, "quant_time_s": tq}
        res["variants"][vname] = r
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        log(f"{vname:6s} residual {bits + 0.125:.3f} b/w: KL {r['kl_bf16']:.5f}  flips {r['flips'] * 100:.2f} %  PPL {r['ppl']:.4f}  "
            f"(quantization {tq:.0f} s)")
    log("references (same texts): " + "; ".join(f"{k} KL {v['kl_bf16']:.5f} flips {v['flips'] * 100:.2f} %" for k, v in res["references"].items()))
    log("DONE")


if __name__ == "__main__":
    main()
