import os
import sys
import gc
import time
from collections import UserDict
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from twobit.base import GROUP
from twobit.hadamard import SEED, layer_rot, rotate_hessian
from twobit.residual import (
    quantize_residual,
    residual_levels,
    rms_normalized,
    rotated_base,
)
from experiments.e10_quantize_layerwise import (
    capture_layer_inputs,
    capture_layer_kwargs,
    fit_base_levels,
    gram_hook,
    layer_linears,
    make_logger,
    to_device,
)
from run_quantization import gptq_codes_with_hinv, gptq_codes

def test_l10():
    t0 = time.time()
    def log(*x):
        print(f"[{time.time() - t0:6.1f} s]", *x, flush=True)

    model_dir = r"D:\localllamacpp\models\gemma4-12b\bf16"
    calib_file = r"D:\localllamacpp\experiments\gemma4_residual_spec\data\calib_chat.pt"
    ckpt_dir = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints"

    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu").eval()
    lm = model.model.language_model
    layers = lm.layers
    layer_types = lm.config.layer_types
    linears = layer_linears(layers)

    C = torch.load(calib_file, weights_only=False)
    n = C.shape[0]

    log("Capturing kwargs & inputs...")
    kwargs_cpu = capture_layer_kwargs(lm, layers, layer_types, C[0:1])
    X = capture_layer_inputs(lm, layers[0], C, "cpu")
    kwargs_dev = {t: to_device(v, "cuda") for t, v in kwargs_cpu.items()}

    n0 = next(iter(linears[0]))
    W0 = linears[0][n0].weight.detach().to("cuda", torch.float32)
    levels = fit_base_levels(n0, W0, 2)
    del W0

    log("Replaying cached layers 0..9...")
    for L in range(10):
        ckpt_path = os.path.join(ckpt_dir, f"layer_{L}.pt")
        saved = torch.load(ckpt_path, weights_only=False)
        layer_base = saved["base"]
        layer_res_q8 = saved["res_q8"]
        del saved
        gc.collect()

        layer = layers[L].to("cuda")
        kw = kwargs_dev[layer_types[L]]
        mods = linears[L]
        for nm, m in mods.items():
            d_out, d_in = layer_base[nm]["shape"]
            rot = layer_rot(nm, d_out, d_in, "cuda")
            Qb = rotated_base(layer_base[nm], levels, device="cuda")
            Qr = layer_res_q8[nm].to("cuda", torch.float32)
            m.weight.data.copy_(rot.inv(Qb + Qr).to(torch.bfloat16))
            del Qb, Qr, rot
        with torch.no_grad():
            X = [layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw).to(torch.bfloat16).cpu() for s in range(n)]
        layer.to("cpu")
        del layer_base, layer_res_q8
        gc.collect()
        torch.cuda.empty_cache()
        log(f"Replayed layer {L} -> activations forwarded.")

    log(f"Starting Layer 10 (Layer 11/48, type: {layer_types[10]})...")
    L = 10
    layer = layers[L].to("cuda")
    kw = kwargs_dev[layer_types[L]]
    mods = linears[L]
    acc = {nm: [None, 0] for nm in mods}
    hooks = [m.register_forward_hook(gram_hook(acc, nm)) for nm, m in mods.items()]
    log("Accumulating Gram matrices...")
    with torch.no_grad():
        for s in range(n):
            layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw)
    for h in hooks:
        h.remove()
    log("Gram accumulation done!")

    for nm in mods:
        if acc[nm][0] is not None:
            acc[nm][0] = acc[nm][0].cpu()

    layer.to("cpu")
    torch.cuda.empty_cache()
    log("Layer offloaded to CPU. Quantizing modules...")

    orig_w = {nm: m.weight.data.clone() for nm, m in mods.items()}
    damp = 0.01
    level_cache = {}

    for nm, m in mods.items():
        d_out, d_in = orig_w[nm].shape
        log(f"Module {nm.split('.')[-1]} (shape {d_out}x{d_in})...")
        W = orig_w[nm].float().cuda()
        rot = layer_rot(nm, d_out, d_in, "cuda")
        W_rot = rot.fwd(W)
        del W
        torch.cuda.empty_cache()

        H_cpu = acc[nm][0] / acc[nm][1]
        acc[nm][0] = None

        if d_in > 4096:
            log("  Running CPU Cholesky for large matrix...")
            rot_cpu = layer_rot(nm, d_out, d_in, "cpu")
            H_rot = rotate_hessian(H_cpu, rot_cpu)
            del H_cpu, rot_cpu
            dead = torch.diag(H_rot) <= 0
            H_rot[dead, dead] = 1.0
            H_rot += damp * torch.mean(torch.diag(H_rot)) * torch.eye(d_in)
            Lc = torch.linalg.cholesky(H_rot)
            del H_rot
            Hi = torch.cholesky_inverse(Lc)
            del Lc
            Hinv = torch.linalg.cholesky(Hi, upper=True).cuda()
            del Hi
            log("  Running GPTQ on GPU with precomputed Hinv...")
            Q_rot, codes, scales = gptq_codes_with_hinv(W_rot, Hinv, levels, GROUP)
            del Hinv
            torch.cuda.empty_cache()
        else:
            log("  Running fast GPU GPTQ...")
            H_rot = rotate_hessian(H_cpu.cuda(), rot)
            del H_cpu
            Q_rot, codes, scales = gptq_codes(W_rot, H_rot, levels, damp, GROUP)
            del H_rot
            torch.cuda.empty_cache()

        log("  Module 2-bit GPTQ complete!")

    log("TEST LAYER 10 SUCCESS!")

if __name__ == "__main__":
    test_l10()
