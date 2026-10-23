import os
import sys
import time
import torch
from transformers import AutoProcessor

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from reconstruct_model import load_reconstructed_model
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"

def test_residual_gen():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    tok = processor.tokenizer

    base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
    print("[Residual_Q8] Loading reconstructed model...", flush=True)
    model = load_reconstructed_model(
        model_path=MODEL_DIR,
        base_codes_path=base_codes_path,
        residual_variant="residual_q8",
        device="cpu",
    )
    print("[Residual_Q8] Setting up GPU streaming hooks...", flush=True)
    hooks = setup_gpu_streaming(model)

    prompts = [
        ("Level 1.1: Capital of France", "What is the capital of France? Answer in one short sentence."),
        ("Level 1.2: Arithmetic", "Calculate: 17 * 4 = "),
        ("Level 1.3: Greeting", "Introduce yourself in two sentences."),
    ]

    for label, p_text in prompts:
        print(f"\n--- {label} ---", flush=True)

        # Test with Canonical Template
        msgs = [{"role": "user", "content": p_text}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to("cuda")

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.1,
                eos_token_id=[1, 106],
            )
        dt = time.time() - t0
        gen_raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        gen_clean = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[Canonical Template] ({dt:.2f}s): {gen_clean!r}")
        print(f"  Raw: {gen_raw!r}")

        # Test with User's Manual Template
        manual_p = f"<start_of_turn>user\n{p_text}<end_of_turn>\n<start_of_turn>model\n"
        inputs_man = tok(manual_p, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out_man = model.generate(
                **inputs_man,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.1,
                eos_token_id=[1, 106],
            )
        dt_man = time.time() - t0
        gen_man_clean = tok.decode(out_man[0][inputs_man.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[Manual Template]    ({dt_man:.2f}s): {gen_man_clean!r}")

if __name__ == "__main__":
    test_residual_gen()
