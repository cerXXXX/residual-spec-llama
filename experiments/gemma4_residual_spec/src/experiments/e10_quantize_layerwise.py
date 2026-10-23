import argparse
import gc
import json
import os
import time
from collections import UserDict

import torch
from transformers import AutoModelForCausalLM

from twobit.base import GROUP, gptq_codes, pack
from twobit.hadamard import SEED, layer_rot, rotate_hessian
from twobit.quantizers import lloyd_max

LEVEL_SAMPLE = 2_000_000


class Stop(Exception):
    pass


def make_logger(t0):
    def log(*x):
        print(f"[{time.time() - t0:6.0f} s]", *x, flush=True)
    return log


def layer_linears(layers):
    return {L: {f"model.language_model.layers.{L}.{k}": m for k, m in layers[L].named_modules() if isinstance(m, torch.nn.Linear)}
            for L in range(len(layers))}


def capture_layer_kwargs(lm, layers, layer_types, ids):
    captured = {}
    first_full = layer_types.index("full_attention")

    def make_hook(kind, stop):
        def hook(module, args, kwargs):
            captured.setdefault(kind, {k: v for k, v in kwargs.items() if k != "shared_kv_states"})
            if stop:
                raise Stop()
        return hook

    handles = [layers[0].register_forward_pre_hook(make_hook(layer_types[0], False), with_kwargs=True),
               layers[first_full].register_forward_pre_hook(make_hook("full_attention", True), with_kwargs=True)]
    with torch.no_grad():
        try:
            lm(input_ids=ids, use_cache=False)
        except Stop:
            pass
    for h in handles:
        h.remove()
    return captured


def capture_layer_inputs(lm, first_layer, ids, device):
    inputs = []

    def hook(module, args, kwargs):
        inputs.append((args[0] if args else kwargs["hidden_states"]).detach().to(torch.bfloat16).cpu())
        raise Stop()

    handle = first_layer.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        for s in range(ids.shape[0]):
            try:
                lm(input_ids=ids[s:s + 1].to(device), use_cache=False)
            except Stop:
                pass
    handle.remove()
    gc.collect()
    return inputs


def to_device(o, device):
    if torch.is_tensor(o):
        return o.to(device)
    if isinstance(o, tuple):
        return tuple(to_device(x, device) for x in o)
    if isinstance(o, list):
        return [to_device(x, device) for x in o]
    if isinstance(o, dict):
        return {k: to_device(v, device) for k, v in o.items()}
    return o


def gram_hook(acc, name):
    def hook(module, inp, out):
        x = inp[0].float().reshape(-1, inp[0].shape[-1])
        entry = acc[name]
        entry[0] = x.T @ x if entry[0] is None else entry[0].addmm_(x.T, x)
        entry[1] += x.shape[0]
    return hook


def pop_rotated_hessian(entry, rot):
    H = entry[0]
    entry[0] = None
    H.div_(entry[1])
    return rotate_hessian(H, rot)


def fit_base_levels(name, W, bits, seed=SEED):
    rot = layer_rot(name, W.shape[0], W.shape[1], W.device)
    Wt = rot.fwd(W)
    G = Wt.reshape(Wt.shape[0], -1, GROUP)
    v = (G / G.pow(2).mean(-1, keepdim=True).sqrt()).flatten()
    g = torch.Generator(device=W.device)
    g.manual_seed(seed)
    return lloyd_max(v[torch.randperm(v.numel(), device=v.device, generator=g)[:LEVEL_SAMPLE]], L=2 ** bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--calib", default="data/calib_chat.pt")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--out", default="weights/ldlq_chat_b2")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--non-sequential", action="store_true")
    ap.add_argument("--n-samples", type=int, default=0)
    a = ap.parse_args()
    log = make_logger(time.time())
    os.makedirs(a.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval()
    lm = model.model.language_model
    layers = lm.layers
    layer_types = lm.config.layer_types
    n_layers = len(layers)
    C = torch.load(a.calib)
    C = C[: a.n_samples] if a.n_samples else C
    n = C.shape[0]
    linears = layer_linears(layers)
    mode = "non-sequential" if a.non_sequential else "sequential"
    log(f"model {a.model.split('/')[-1]}: {n_layers} layers, {sum(len(v) for v in linears.values())} linear; "
        f"calib {n} x {C.shape[1]}; bits {a.bits}; {mode}")

    kwargs_cpu = capture_layer_kwargs(lm, layers, layer_types, C[0:1])
    X = capture_layer_inputs(lm, layers[0], C, "cpu")
    kwargs_dev = {t: to_device(v, "cuda") for t, v in kwargs_cpu.items()}
    log(f"captured: kwargs {list(kwargs_cpu)} (keys {sorted(kwargs_cpu[layer_types[0]])}); layer 0 inputs: {len(X)} x {tuple(X[0].shape)}")

    n0 = next(iter(linears[0]))
    W0 = linears[0][n0].weight.detach().to("cuda", torch.float32)
    levels = fit_base_levels(n0, W0, a.bits)
    del W0
    log("Lloyd-Max levels:", [round(x, 4) for x in levels.tolist()])

    out = {}
    for L in range(n_layers):
        layer = layers[L].to("cuda")
        kw = kwargs_dev[layer_types[L]]
        mods = linears[L]
        acc = {nm: [None, 0] for nm in mods}
        hooks = [m.register_forward_hook(gram_hook(acc, nm)) for nm, m in mods.items()]
        with torch.no_grad():
            for s in range(n):
                layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw)
        for h in hooks:
            h.remove()
        orig_w = {nm: m.weight.data.clone() for nm, m in mods.items()}
        for nm, m in mods.items():
            W = orig_w[nm].float()
            rot = layer_rot(nm, W.shape[0], W.shape[1], W.device)
            Q, codes, scales = gptq_codes(rot.fwd(W), pop_rotated_hessian(acc[nm], rot), levels, a.damp, GROUP)
            out[nm] = {"P": pack(codes, a.bits).cpu(), "s": scales.cpu(), "shape": tuple(W.shape)}
            m.weight.data.copy_(rot.inv(Q).to(torch.bfloat16))
            del W, Q, codes, scales
            torch.cuda.empty_cache()
        if a.non_sequential:
            for nm, m in mods.items():
                m.weight.data.copy_(orig_w[nm])
        with torch.no_grad():
            X = [layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw).to(torch.bfloat16).cpu() for s in range(n)]
        for nm, m in mods.items():
            m.weight.data.copy_(orig_w[nm])
        layer.to("cpu")
        del orig_w, acc
        gc.collect()
        torch.cuda.empty_cache()
        log(f"layer {L + 1}/{n_layers} ({layer_types[L]}) done; VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    meta = {"model": a.model, "calib": a.calib, "bits": a.bits, "levels": levels.tolist(), "seed": SEED, "group": GROUP,
            "damp": a.damp, "sequential": not a.non_sequential, "samples": n}
    torch.save({"meta": meta, "layers": out}, os.path.join(a.out, "codes.pt"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    size_gb = sum(v["P"].numel() + 2 * v["s"].numel() for v in out.values()) / 1e9
    log(f"saved {os.path.join(a.out, 'codes.pt')} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
