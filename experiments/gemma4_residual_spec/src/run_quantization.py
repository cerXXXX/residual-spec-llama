import os
import sys

# Prevent CUDA memory fragmentation on 6GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import time
from collections import UserDict

import torch
from transformers import AutoModelForCausalLM

# Add src to sys.path
sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from twobit.base import GROUP, gptq_codes, pack
from twobit.hadamard import SEED, layer_rot, rotate_hessian
from twobit.quantizers import lloyd_max
from twobit.residual import (
    quantize_residual,
    residual_levels,
    rms_normalized,
    rotated_base,
    group_scale,
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

@torch.no_grad()
def gptq_codes_with_hinv(Wt, Hinv, levels, group=GROUP):
    thr = (levels[1:] + levels[:-1]) / 2
    W = Wt.clone()
    d_out, d_in = W.shape
    codes = torch.zeros(d_out, d_in, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(d_out, d_in // group, dtype=torch.float16, device=W.device)
    for i1 in range(0, d_in, group):
        i2 = i1 + group
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        sc = W1.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8).half()
        scales[:, i1 // group] = sc[:, 0]
        sc = sc.float()
        for i in range(group):
            w = W1[:, i:i + 1]
            idx = torch.bucketize(w / sc, thr)
            q = levels[idx] * sc
            Q1[:, i:i + 1] = q
            codes[:, i1 + i] = idx[:, 0]
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err @ Hb[i:i + 1, i:]
            E1[:, i:i + 1] = err
        if i2 < d_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return None, codes, scales

def dequantize_res_entry(entry, levels, device="cuda"):
    if isinstance(entry, torch.Tensor):
        return entry.to(device, torch.float32)
    codes = entry["codes"].to(device).long()
    sc = entry["scales"].to(device).float().repeat_interleave(GROUP, dim=1)
    return levels.to(device)[codes] * sc

def run_quantization():
    t0 = time.time()
    log = make_logger(t0)

    model_dir = r"D:\localllamacpp\models\gemma4-12b\bf16"
    calib_file = r"D:\localllamacpp\experiments\gemma4_residual_spec\data\calib_chat.pt"
    ckpt_dir = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints"
    out_b2 = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\base_b2"
    out_q6 = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\residual_q6"
    out_q8 = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\residual_q8"
    damp = 0.01

    for d in [ckpt_dir, out_b2, out_q6, out_q8]:
        os.makedirs(d, exist_ok=True)

    log(f"Loading model from {model_dir} (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    lm = model.model.language_model
    layers = lm.layers
    layer_types = lm.config.layer_types
    n_layers = len(layers)

    C = torch.load(calib_file, weights_only=False)
    n = C.shape[0]
    linears = layer_linears(layers)
    log(f"Model: {n_layers} layers, {sum(len(v) for v in linears.values())} linear projections; Calib {n} x {C.shape[1]}")

    kwargs_cpu = capture_layer_kwargs(lm, layers, layer_types, C[0:1])
    X = capture_layer_inputs(lm, layers[0], C, "cpu")
    kwargs_dev = {t: to_device(v, "cuda") for t, v in kwargs_cpu.items()}
    log(f"Captured kwargs and initial activations: {len(X)} samples")

    n0 = next(iter(linears[0]))
    W0 = linears[0][n0].weight.detach().to("cuda", torch.float32)
    levels = fit_base_levels(n0, W0, 2)
    del W0
    log("Fitted 2-bit Lloyd-Max base levels:", [round(x, 4) for x in levels.tolist()])

    levels_q8 = torch.arange(-127, 128, dtype=torch.float32)
    levels_q8_dev = levels_q8.to("cuda")

    base_out = {}
    level_cache = {}

    for L in range(n_layers):
        ckpt_path = os.path.join(ckpt_dir, f"layer_{L}.pt")
        mods = linears[L]

        # Check if layer already quantized in checkpoint
        if os.path.exists(ckpt_path):
            log(f"Loading cached layer {L + 1}/{n_layers} from checkpoint...")
            saved = torch.load(ckpt_path, weights_only=False)
            layer_base = saved["base"]
            layer_res_q8 = saved["res_q8"]
            del saved
            gc.collect()

            for k, v in layer_base.items():
                base_out[k] = v

            # Propagate activations using reconstructed weights
            layer = layers[L].to("cuda")
            kw = kwargs_dev[layer_types[L]]
            for nm, m in mods.items():
                d_out, d_in = layer_base[nm]["shape"]
                rot = layer_rot(nm, d_out, d_in, "cuda")
                Qb = rotated_base(layer_base[nm], levels, device="cuda")
                Qr = dequantize_res_entry(layer_res_q8[nm], levels_q8_dev, "cuda")
                m.weight.data.copy_(rot.inv(Qb + Qr).to(torch.bfloat16))
                del Qb, Qr, rot
            with torch.no_grad():
                X = [layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw).to(torch.bfloat16).cpu() for s in range(n)]
            layer.to("cpu")
            del layer_base, layer_res_q8
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # 1. Accumulate Gram matrix (Hessian H = X^T X) on GPU
        layer = layers[L].to("cuda")
        kw = kwargs_dev[layer_types[L]]
        acc = {nm: [None, 0] for nm in mods}
        hooks = [m.register_forward_hook(gram_hook(acc, nm)) for nm, m in mods.items()]
        with torch.no_grad():
            for s in range(n):
                layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw)
        for h in hooks:
            h.remove()

        # Offload accumulated Gram matrices to CPU immediately to free ~1.3 GB VRAM
        for nm in mods:
            if acc[nm][0] is not None:
                acc[nm][0] = acc[nm][0].cpu()

        # Offload layer weights to CPU to free ~0.7 GB VRAM during GPTQ
        layer.to("cpu")
        torch.cuda.empty_cache()

        orig_w = {nm: m.weight.data.clone() for nm, m in mods.items()}
        layer_base = {}
        layer_res_q6 = {}
        layer_res_q8 = {}

        # 2. Quantize each linear module in layer (isolated on GPU one by one)
        for nm, m in mods.items():
            d_out, d_in = orig_w[nm].shape
            W = orig_w[nm].float().cuda()
            rot = layer_rot(nm, d_out, d_in, "cuda")
            W_rot = rot.fwd(W)
            del W
            torch.cuda.empty_cache()

            # A. 2-bit GPTQ Base
            H_cpu = acc[nm][0] / acc[nm][1]
            acc[nm][0] = None  # Free CPU memory

            if d_in > 4096:
                # Ultra-safe CPU rotation and Cholesky for giant matrices (down_proj) - 0 MB VRAM!
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
                Q_rot, codes, scales = gptq_codes_with_hinv(W_rot, Hinv, levels, GROUP)
                del Hinv
                torch.cuda.empty_cache()
            else:
                # Fast GPU path for small matrices (d_in <= 4096)
                H_rot = rotate_hessian(H_cpu.cuda(), rot)
                del H_cpu
                Q_rot, codes, scales = gptq_codes(W_rot, H_rot, levels, damp, GROUP)
                del H_rot
                torch.cuda.empty_cache()

            entry_base = {
                "P": pack(codes, 2).cpu(),
                "s": scales.cpu(),
                "shape": tuple(orig_w[nm].shape),
            }
            base_out[nm] = entry_base
            layer_base[nm] = entry_base

            # B. Residual R_t = W_rot - Q_rot_base
            Qb = rotated_base(entry_base, levels, device="cuda")
            Rt = (W_rot - Qb).contiguous()
            del W_rot, Q_rot, codes, scales

            # C. 6-bit Residual (Lloyd-Max 64 levels)
            levels_q6 = residual_levels("lm", 6, rms_normalized(Rt), level_cache)
            thr_q6 = (levels_q6[1:] + levels_q6[:-1]) / 2
            qmax_q6 = (levels_q6.numel() - 1) / 2
            G_q6 = Rt.view(d_out, d_in // GROUP, GROUP)
            sc_q6 = group_scale(G_q6.reshape(-1, GROUP), "lm", qmax_q6).view(d_out, d_in // GROUP, 1)
            idx_q6 = torch.bucketize(G_q6 / sc_q6, thr_q6)
            layer_res_q6[nm] = {
                "codes": idx_q6.view(d_out, d_in).to(torch.uint8).cpu(),
                "scales": sc_q6.squeeze(-1).to(torch.float16).cpu(),
                "shape": (d_out, d_in),
            }
            del G_q6, sc_q6, idx_q6

            # D. 8-bit Residual (Uniform 256 levels)
            thr_q8 = (levels_q8_dev[1:] + levels_q8_dev[:-1]) / 2
            qmax_q8 = (levels_q8_dev.numel() - 1) / 2
            G_q8 = Rt.view(d_out, d_in // GROUP, GROUP)
            sc_q8 = group_scale(G_q8.reshape(-1, GROUP), "uni", qmax_q8).view(d_out, d_in // GROUP, 1)
            idx_q8 = torch.bucketize(G_q8 / sc_q8, thr_q8)
            layer_res_q8[nm] = {
                "codes": idx_q8.view(d_out, d_in).to(torch.uint8).cpu(),
                "scales": sc_q8.squeeze(-1).to(torch.float16).cpu(),
                "shape": (d_out, d_in),
            }

            # Reconstruct Q8 for subsequent layer activation propagation
            sc_unp = layer_res_q8[nm]["scales"].to("cuda").float().repeat_interleave(GROUP, dim=1)
            Qr_8 = levels_q8_dev[layer_res_q8[nm]["codes"].to("cuda").long()] * sc_unp
            m.weight.data.copy_(rot.inv(Qb + Qr_8).to(torch.bfloat16))

            del Qb, Rt, Qr_8, sc_unp, G_q8, sc_q8, idx_q8, rot
            torch.cuda.empty_cache()

        # Save per-layer checkpoint (compact representation: ~400 MB instead of ~956 MB)
        torch.save({
            "base": layer_base,
            "res_q6": layer_res_q6,
            "res_q8": layer_res_q8,
        }, ckpt_path)

        # Update layer inputs for next layer
        layer.to("cuda")
        with torch.no_grad():
            X = [layer(X[s].to("cuda"), shared_kv_states=UserDict(), **kw).to(torch.bfloat16).cpu() for s in range(n)]

        # Restore original weight in CPU model
        for nm, m in mods.items():
            m.weight.data.copy_(orig_w[nm])

        layer.to("cpu")
        del orig_w, acc, layer_base, layer_res_q6, layer_res_q8
        gc.collect()
        torch.cuda.empty_cache()
        log(f"Layer {L + 1}/{n_layers} ({layer_types[L]}) quantized & checkpointed. Peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    meta = {
        "model": model_dir,
        "calib": calib_file,
        "bits": 2,
        "levels": levels.tolist(),
        "seed": SEED,
        "group": GROUP,
        "damp": damp,
        "samples": n,
    }
    log("Consolidating base codes...")
    torch.save({"meta": meta, "layers": base_out}, os.path.join(out_b2, "codes.pt"))
    with open(os.path.join(out_b2, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # Save residual meta info
    meta_q6 = {"grid": "lm", "bits": 6, "levels": level_cache.get(6, torch.empty(0)).tolist()}
    with open(os.path.join(out_q6, "meta.json"), "w") as f:
        json.dump(meta_q6, f, indent=1)
    meta_q8 = {"grid": "uni", "bits": 8, "levels": levels_q8.tolist()}
    with open(os.path.join(out_q8, "meta.json"), "w") as f:
        json.dump(meta_q8, f, indent=1)

    log(f"SUCCESS: Saved base_b2 ({os.path.getsize(os.path.join(out_b2, 'codes.pt')) / 1e9:.2f} GB)")
    log("Quantization pipeline completed successfully across all 48 layers!")

if __name__ == "__main__":
    run_quantization()
