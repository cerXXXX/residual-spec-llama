import os
import sys
import gc
import json
import torch
from transformers import AutoModelForCausalLM

# Add src to sys.path
sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from twobit.base import GROUP
from twobit.hadamard import layer_rot
from twobit.residual import rotated_base

def dequantize_res_entry(entry, levels, device="cpu"):
    if isinstance(entry, torch.Tensor):
        return entry.to(device, torch.float32)
    codes = entry["codes"].to(device).long()
    sc = entry["scales"].to(device).float().repeat_interleave(GROUP, dim=1)
    return levels.to(device)[codes] * sc

def load_reconstructed_model(
    model_path: str,
    base_codes_path: str = None,
    residual_path: str = None,
    residual_variant: str = None,  # "base_b2", "residual_q6", "residual_q8"
    ckpt_dir: str = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Loads Gemma 4 12B and replaces linear weights with reconstructed weights:
        W_rec = Rot^{-1}(W_base + W_res)
    Supports layer-by-layer streaming from checkpoints to keep peak RAM under 25 GB.
    """
    print(f"[Loader] Loading base model config & non-quantized weights from {model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
    ).eval()

    lm = model.model.language_model
    layers = lm.layers
    n_layers = len(layers)

    # Determine base levels
    base_levels = torch.tensor([-1.5043, -0.4523, 0.4523, 1.5043], dtype=torch.float32)
    meta_path = os.path.join(os.path.dirname(ckpt_dir), "base_b2", "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            base_levels = torch.tensor(json.load(f)["levels"], dtype=torch.float32)

    # Determine residual levels if requested
    levels_res = None
    res_key = None
    if residual_variant == "residual_q8":
        res_key = "res_q8"
        levels_res = torch.arange(-127, 128, dtype=torch.float32)
    elif residual_variant == "residual_q6":
        res_key = "res_q6"
        q6_meta = os.path.join(os.path.dirname(ckpt_dir), "residual_q6", "meta.json")
        if os.path.exists(q6_meta):
            with open(q6_meta) as f:
                levels_res = torch.tensor(json.load(f)["levels"], dtype=torch.float32)

    print(f"[Loader] Reconstructing weights across {n_layers} layers (variant: {residual_variant or 'base_b2'})...", flush=True)

    # Streaming layer-by-layer load from checkointed files
    for L in range(n_layers):
        ckpt_path = os.path.join(ckpt_dir, f"layer_{L}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for layer {L}: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        layer_base = ckpt["base"]
        layer_res = ckpt.get(res_key) if res_key else None

        layer = layers[L]
        for name, module in layer.named_modules():
            full_name = f"model.language_model.layers.{L}.{name}"
            if isinstance(module, torch.nn.Linear) and full_name in layer_base:
                entry = layer_base[full_name]
                d_out, d_in = entry["shape"]

                # 1. Reconstruct rotated base
                Qb = rotated_base(entry, base_levels, device="cpu")

                # 2. Add residual if present
                if layer_res and full_name in layer_res:
                    Qr = dequantize_res_entry(layer_res[full_name], levels_res, device="cpu")
                    W_rot = Qb + Qr
                    del Qr
                else:
                    W_rot = Qb

                # 3. Inverse Hadamard rotation back to original weight space
                rot = layer_rot(full_name, d_out, d_in, device="cpu")
                W_rec = rot.inv(W_rot).to(dtype)

                module.weight.data.copy_(W_rec)
                del Qb, W_rot, W_rec, rot

        del ckpt, layer_base, layer_res
        gc.collect()

        if (L + 1) % 10 == 0 or L == n_layers - 1:
            print(f"  Layer {L + 1}/{n_layers} reconstructed.", flush=True)

    gc.collect()
    print("[Loader] Model reconstruction complete!", flush=True)
    return model

if __name__ == "__main__":
    print("reconstruct_model module loaded successfully.")
