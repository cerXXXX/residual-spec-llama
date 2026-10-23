import os
import sys
import time
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"

def test_template():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    tok = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu").eval()
    hooks = setup_gpu_streaming(model)

    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to("cuda")

    print("[BF16 Official Template] Generating 20 tokens...", flush=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.1,
            eos_token_id=[1, 106],
        )
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print("Generated raw:\n", repr(gen))
    print("Decoded (clean):\n", tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    test_template()
