import os
import sys
import torch

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from twobit.hadamard import layer_rot

def test_hadamard_roundtrip():
    print("=== Testing Hadamard Roundtrip on all Gemma 4 shapes ===")
    shapes = [
        ("q_proj", 4096, 3840),
        ("k_proj", 2048, 3840),
        ("v_proj", 2048, 3840),
        ("o_proj", 3840, 4096),
        ("gate_proj", 15360, 3840),
        ("up_proj", 15360, 3840),
        ("down_proj", 3840, 15360),
    ]

    for name, d_out, d_in in shapes:
        rot = layer_rot(f"test.{name}", d_out, d_in, device="cpu")
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        W_rot = rot.fwd(W)
        W_rec = rot.inv(W_rot)
        max_err = torch.max(torch.abs(W - W_rec)).item()
        rel_frob = (torch.norm(W - W_rec) / torch.norm(W)).item()
        print(f"  {name:10s} ({d_out}x{d_in}): bo={rot.bo:4d}, bi={rot.bi:4d} | max_err={max_err:.2e}, rel_frob={rel_frob:.2e}")

if __name__ == "__main__":
    test_hadamard_roundtrip()
