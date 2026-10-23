import sys
import traceback
import safetensors.torch

st_path = r"D:\localllamacpp\models\gemma4-12b\bf16\model.safetensors"
try:
    with safetensors.torch.safe_open(st_path, framework="pt", device="cpu") as f:
        print("Keys count:", len(f.keys()))
        k = "model.language_model.layers.0.self_attn.q_proj.weight"
        sl = f.get_slice(k)
        print("Slice shape:", sl.get_shape())
        t = sl[:]
        print("Slice data loaded:", t.shape, t.dtype)
except BaseException as e:
    print("Caught exception:", type(e), e)
    traceback.print_exc()
