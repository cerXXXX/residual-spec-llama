import os
import sys
import json
import struct
import torch

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from twobit.base import GROUP
from twobit.hadamard import layer_rot
from twobit.residual import rotated_base
from reconstruct_model import dequantize_res_entry

def read_safetensor(st_path, key):
    with open(st_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        info = header[key]
        start, end = info["data_offsets"]
        shape = info["shape"]
        f.seek(8 + header_len + start)
        raw = f.read(end - start)
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).float()

def audit():
    ckpt_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\checkpoints\layer_0.pt"
    st_path = r"D:\localllamacpp\models\gemma4-12b\bf16\model.safetensors"
    base_meta_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\base_b2\meta.json"

    with open(base_meta_path) as f:
        base_levels = torch.tensor(json.load(f)["levels"], dtype=torch.float32)
    levels_res = torch.arange(-127, 128, dtype=torch.float32)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    projs = [
        ("self_attn.q_proj", "model.language_model.layers.0.self_attn.q_proj"),
        ("mlp.gate_proj", "model.language_model.layers.0.mlp.gate_proj"),
    ]

    for short_name, full_name in projs:
        print(f"\n==========================================")
        print(f"Auditing {short_name} ({full_name})")
        print(f"==========================================")

        W_orig = read_safetensor(st_path, full_name + ".weight")
        entry_base = ckpt["base"][full_name]
        d_out, d_in = entry_base["shape"]

        print(f"Shape: [{d_out}, {d_in}]")

        # 1. Base 2-bit
        Qb = rotated_base(entry_base, base_levels, device="cpu")

        # 2. Residual Q8
        res_entry = ckpt["res_q8"][full_name]
        Qr = dequantize_res_entry(res_entry, levels_res, device="cpu")

        W_rot = Qb + Qr

        # 3. Inverse Hadamard rotation
        rot = layer_rot(full_name, d_out, d_in, device="cpu")
        W_rec = rot.inv(W_rot)

        # Metrics
        dot = torch.sum(W_orig * W_rec).item()
        norm_orig = torch.norm(W_orig).item()
        norm_rec = torch.norm(W_rec).item()
        cos_sim = dot / (norm_orig * norm_rec)

        diff = W_orig - W_rec
        norm_diff = torch.norm(diff).item()
        rel_frob = norm_diff / norm_orig

        print(f"  ||W_orig||_F:             {norm_orig:.6f}")
        print(f"  ||W_rec||_F:              {norm_rec:.6f}")
        print(f"  Cosine Similarity:        {cos_sim:.8f}")
        print(f"  Relative Frobenius Error: {rel_frob * 100:.4f}%")

        # Also check Base 2-bit alone
        W_rec_base = rot.inv(Qb)
        dot_b = torch.sum(W_orig * W_rec_base).item()
        norm_rec_b = torch.norm(W_rec_base).item()
        cos_sim_b = dot_b / (norm_orig * norm_rec_b)
        rel_frob_b = torch.norm(W_orig - W_rec_base).item() / norm_orig
        print(f"  [Base 2-bit only] Cosine Sim: {cos_sim_b:.6f}, Rel Frob: {rel_frob_b * 100:.2f}%")

if __name__ == "__main__":
    audit()
