import os
import sys
import time
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"

def test_gen():
    print("[Test] Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    tok = processor.tokenizer

    print("[Test] Loading model on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu").eval()

    print("[Test] Setting up GPU streaming hooks...", flush=True)
    hooks = setup_gpu_streaming(model)

    prompt = "<start_of_turn>user\nWhat is the capital of France? Answer in one short sentence.<end_of_turn>\n<start_of_turn>model\n"
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    print(f"[Test] Generating 15 tokens... Prompt tokens: {inputs.input_ids.shape[1]}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tok.eos_token_id or 1,
            eos_token_id=[1, 106],
        )
    dt = time.time() - t0
    gen_text = tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[Test] Time: {dt:.2f}s ({out_ids.shape[1] - inputs.input_ids.shape[1]} tokens, {dt / (out_ids.shape[1] - inputs.input_ids.shape[1]):.2f} s/tok)", flush=True)
    print(f"[Test] Output: {gen_text!r}", flush=True)

if __name__ == "__main__":
    test_gen()
