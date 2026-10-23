import os
import sys
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"

def test_layer_streaming_bit_exact():
    print("=== Test 1: Single Layer Bit-for-Bit on CUDA ===", flush=True)
    # Load model config and 1 layer on CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    lm = model.model.language_model
    layer0 = lm.layers[0]

    # Dummy input
    torch.manual_seed(42)
    x = torch.randn(1, 16, lm.config.hidden_size, dtype=torch.bfloat16, device="cuda")

    # Native CUDA execution:
    layer0_cuda = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu"
    ).eval().model.language_model.layers[0].to("cuda")

    with torch.no_grad():
        out_native = layer0_cuda(x.clone())[0]

    # Hooked streaming execution:
    # layer0 starts on CPU
    cpu_params = {name: p.data.clone() for name, p in layer0.named_parameters()}
    cpu_buffers = {name: b.data.clone() for name, b in layer0.named_buffers()}

    def pre_hook(module, args, kwargs):
        for name, p in module.named_parameters():
            p.data = cpu_params[name].to("cuda", non_blocking=True)
        for name, b in module.named_buffers():
            b.data = cpu_buffers[name].to("cuda", non_blocking=True)

    def post_hook(module, args, output):
        for name, p in module.named_parameters():
            p.data = cpu_params[name]
        for name, b in module.named_buffers():
            b.data = cpu_buffers[name]

    h1 = layer0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer0.register_forward_hook(post_hook)

    with torch.no_grad():
        out_streamed = layer0(x.clone())[0]

    h1.remove()
    h2.remove()

    is_equal = torch.equal(out_native, out_streamed)
    diff = torch.max(torch.abs(out_native.float() - out_streamed.float())).item()
    print(f"  torch.equal(out_streamed, out_native): {is_equal}")
    print(f"  Max absolute difference: {diff}")

if __name__ == "__main__":
    test_layer_streaming_bit_exact()
