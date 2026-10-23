import os
import sys
import json
import traceback
import torch
import safetensors.torch

# Ensure proper paths
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from twobit.base import GROUP
from twobit.hadamard import layer_rot
from twobit.residual import rotated_base
from reconstruct_model import dequantize_res_entry

def run():
    try:
        print("[Audit] Starting Layer 0 audit...", flush=True)
        ckpt_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints\layer_0.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Audit] Checkpoint not found: {ckpt_path}", flush=True)
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_meta_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\base_b2\meta.json"
        with open(base_meta_path) as f:
            base_levels = torch.tensor(json.load(f)["levels"], dtype=torch.float32)
        levels_res = torch.arange(-127, 128, dtype=torch.float32)

        st_path = r"D:\localllamacpp\models\gemma4-12b\bf16\model.safetensors"
        f_st = safetensors.torch.safe_open(st_path, framework="pt")

        projs = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.gate_proj",
        ]

        for proj_name in projs:
            weight_key = proj_name + ".weight"
            W_orig = f_st.get_tensor(weight_key).float()
            entry_base = ckpt["base"][proj_name]
            d_out, d_in = entry_base["shape"]
            print(f"\n[Audit] Target: {proj_name}", flush=True)
            print(f"  Shape: {d_out} x {d_in}", flush=True)

            Qb = rotated_base(entry_base, base_levels, device="cpu")
            Qr = dequantize_res_entry(ckpt["res_q8"][proj_name], levels_res, device="cpu")
            W_rot = Qb + Qr

            rot = layer_rot(proj_name, d_out, d_in, device="cpu")
            W_rec = rot.inv(W_rot)

            dot = torch.sum(W_orig * W_rec).item()
            norm_orig = torch.norm(W_orig).item()
            norm_rec = torch.norm(W_rec).item()
            cos_sim = dot / (norm_orig * norm_rec)

            diff_frob = torch.norm(W_orig - W_rec).item()
            rel_frob = diff_frob / norm_orig

            print(f"  ||W_orig||_F: {norm_orig:.4f}", flush=True)
            print(f"  ||W_rec||_F:  {norm_rec:.4f}", flush=True)
            print(f"  Cosine Similarity:        {cos_sim:.6f}", flush=True)
            print(f"  Relative Frobenius Error: {rel_frob * 100:.4f}%", flush=True)

        print("\n[Audit] Layer 0 check completed successfully!", flush=True)

    except Exception as e:
        print(f"[Audit ERROR]: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    run()
