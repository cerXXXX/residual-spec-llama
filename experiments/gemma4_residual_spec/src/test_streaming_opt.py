import torch
import time
import os

os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

from transformers import AutoModelForCausalLM

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"

print("[Test] Loading model config...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
).eval()

lm = model.model.language_model
layers = lm.layers

print("[Test] Pinning CPU parameters and buffers...", flush=True)
cpu_params = {}
cpu_buffers = {}
for i, layer in enumerate(layers):
    cpu_params[i] = {name: p.data for name, p in layer.named_parameters()}
    cpu_buffers[i] = {name: b.data for name, b in layer.named_buffers()}

# Place non-layer modules on GPU
for _, child in model.model.named_children():
    if child is lm:
        for name, sub in lm.named_children():
            if name != "layers":
                sub.to("cuda")
    else:
        child.to("cuda")
model.lm_head.to("cuda")

# Register zero-copy hooks
for i, layer in enumerate(layers):
    def make_pre(idx):
        def pre_hook(module, args, kwargs):
            for name, p in module.named_parameters():
                p.data = cpu_params[idx][name].to("cuda", non_blocking=True)
            for name, b in module.named_buffers():
                b.data = cpu_buffers[idx][name].to("cuda", non_blocking=True)
        return pre_hook
    def make_post(idx):
        def post_hook(module, args, output):
            for name, p in module.named_parameters():
                p.data = cpu_params[idx][name]
            for name, b in module.named_buffers():
                b.data = cpu_buffers[idx][name]
        return post_hook

    layer.register_forward_pre_hook(make_pre(i), with_kwargs=True)
    layer.register_forward_hook(make_post(i))

print(f"[Test] Setup complete! CUDA allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

# Run 3 test forwards
dummy_input = torch.randint(0, 1000, (1, 64), device="cuda")
for step in range(3):
    t0 = time.time()
    with torch.no_grad():
        out = model(dummy_input)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  Step {step+1}: {dt:.3f}s | CUDA alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

print("[SUCCESS] All steps executed successfully without leaks!")
