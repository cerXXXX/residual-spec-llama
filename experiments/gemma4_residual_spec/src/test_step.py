import os
import sys
import torch
import safetensors.torch

print("Step 1: torch loaded", flush=True)
ckpt_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints\layer_0.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("Step 2: ckpt loaded", flush=True)

st_path = r"D:\localllamacpp\models\gemma4-12b\bf16\model.safetensors"
print("Step 3: opening safetensors", flush=True)
with safetensors.torch.safe_open(st_path, framework="pt") as f:
    print("Step 4: opened safetensors successfully", flush=True)
    t = f.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
    print("Step 5: got tensor:", t.shape, t.dtype, flush=True)
print("Done!")
