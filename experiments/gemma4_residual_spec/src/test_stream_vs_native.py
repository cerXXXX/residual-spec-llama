import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"

def test_streaming_vs_native():
    print("Testing streaming vs native on a 2-layer test model...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg.text_config.num_hidden_layers = 2
    cfg.text_config.layer_types = cfg.text_config.layer_types[:2]

    # 1. Native CUDA model
    torch.manual_seed(1234)
    model_native = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to("cuda").eval()

    # 2. Streaming model with identical weights
    torch.manual_seed(1234)
    model_stream = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to("cpu").eval()
    model_stream.load_state_dict(model_native.state_dict())

    # Setup streaming hooks
    lm = model_stream.model.language_model
    layers = lm.layers

    for _, child in model_stream.model.named_children():
        if child is lm:
            for sub_name, sub in lm.named_children():
                if sub_name != "layers":
                    sub.to("cuda")
        else:
            child.to("cuda")
    model_stream.lm_head.to("cuda")

    cpu_params = {}
    cpu_buffers = {}
    for i, layer in enumerate(layers):
        cpu_params[i] = {name: p.data for name, p in layer.named_parameters()}
        cpu_buffers[i] = {name: b.data for name, b in layer.named_buffers()}

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

    input_ids = torch.tensor([[100, 200, 300, 400]], device="cuda")

    with torch.no_grad():
        out_native = model_native(input_ids).logits
        out_stream = model_stream(input_ids).logits

    is_equal = torch.equal(out_native, out_stream)
    max_diff = torch.max(torch.abs(out_native.float() - out_stream.float())).item()
    print(f"torch.equal(logits_hooked, logits_native): {is_equal}", flush=True)
    print(f"Max difference: {max_diff}", flush=True)

if __name__ == "__main__":
    test_streaming_vs_native()
