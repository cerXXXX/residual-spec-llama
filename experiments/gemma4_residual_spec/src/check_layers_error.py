import os
import sys
import json
import torch
import struct

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
        return torch.frombuffer(bytearray(f.read(end - start)), dtype=torch.bfloat16).reshape(shape).float()

st_path = r"D:\localllamacpp\models\gemma4-12b\bf16\model.safetensors"
base_meta_path = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights\base_b2\meta.json"
with open(base_meta_path) as f:
    base_levels = torch.tensor(json.load(f)["levels"], dtype=torch.float32)
levels_res = torch.arange(-127, 128, dtype=torch.float32)

print("Checking layers across model:")
for L in [0, 5, 9, 10, 11, 20, 35, 47]:
    ckpt_path = f"D:\\localllamacpp\\experiments\\gemma4_residual_spec\\weights\\checkpoints\\layer_{L}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj_name = f"model.language_model.layers.{L}.self_attn.q_proj"
    W_orig = read_safetensor(st_path, proj_name + ".weight")
    entry_base = ckpt["base"][proj_name]
    d_out, d_in = entry_base["shape"]
    Qb = rotated_base(entry_base, base_levels, device="cpu")
    Qr = dequantize_res_entry(ckpt["res_q8"][proj_name], levels_res, device="cpu")
    rot = layer_rot(proj_name, d_out, d_in, device="cpu")
    W_rec = rot.inv(Qb + Qr)
    cos_sim = (torch.sum(W_orig * W_rec) / (torch.norm(W_orig) * torch.norm(W_rec))).item()
    rel_frob = (torch.norm(W_orig - W_rec) / torch.norm(W_orig)).item()
    t_name = type(ckpt["res_q8"][proj_name]).__name__
    print(f"Layer {L:2d}: Cos Sim = {cos_sim:.6f}, Rel Frob = {rel_frob * 100:.3f}% (res type: {t_name})")
