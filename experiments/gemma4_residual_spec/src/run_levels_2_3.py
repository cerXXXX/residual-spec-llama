import os
import sys
import time
import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from reconstruct_model import load_reconstructed_model
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"
OUT_JSON = r"D:\localllamacpp\experiments\gemma4_residual_spec\results\audit_generation_results.json"

PROMPTS_L2 = [
    ("L2.1_palindrome", "Write a Python function 'is_palindrome(s: str) -> bool' that checks if a string is a palindrome, ignoring spaces, punctuation, and capitalization. Provide 2 assert tests.", 120),
    ("L2.2_sally", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have in total? Explain your answer briefly.", 90),
    ("L2.3_sky_blue", "Explain in three clear bullet points why the sky appears blue during the day (Rayleigh scattering).", 120),
    ("L2.4_russian_idiom", "Переведи на английский язык и объясни смысл поговорки: 'Дело в шляпе'.", 120),
]

PROMPTS_L3 = [
    ("L3.1_rate_limiter", "Write a Python class 'SlidingWindowRateLimiter' that tracks requests per user within a rolling time window (e.g. 100 requests per 60 seconds). Implement thread-safety with threading.Lock, add type hints, docstrings, and handle edge cases (expired timestamps cleanup).", 220),
    ("L3.2_river_crossing", "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item at a time. If left unattended, the wolf will eat the goat, and the goat will eat the cabbage. Provide a step-by-step solution explaining each crossing and the state of both riverbanks.", 220),
    ("L3.3_json_extraction", "Extract structured entities from this sentence into valid JSON without markdown fences: 'John Doe bought 3 units of RTX 4090 on March 15 for $1599 each from Newegg.' Schema: {buyer: str, items: [{name: str, quantity: int, price_usd: float}], merchant: str, date: str}.", 140),
    ("L3.4_flash_attention", "Подробно объясни архитектурный принцип механизма FlashAttention: чем он отличается от стандартного Scaled Dot-Product Attention, как используется тайлинг (tiling) в SRAM GPU и почему он снижает требования к объему памяти HBM с O(N^2) до O(N).", 250),
]

def run_suite():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    tok = processor.tokenizer

    results = {"level1": {
        "capital_france": {
            "canonical": "The capital of France is Paris.",
            "manual": "model\\nmodel\\nmodel\\nmodel\\n..."
        },
        "arithmetic": {
            "canonical": "17 * 4 = 68",
            "manual": "............................"
        },
        "greeting": {
            "canonical": "I am Gemma 4, a large language model with open weights developed by Google DeepMind. I can process text and images to assist you with a",
            "manual": "<start_start__________________________"
        }
    }, "level2": {}, "level3": {}}

    print("=== Starting Residual_Q8 Evaluation on Level 2 & Level 3 ===", flush=True)
    base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
    model = load_reconstructed_model(
        model_path=MODEL_DIR,
        base_codes_path=base_codes_path,
        residual_variant="residual_q8",
        device="cpu",
    )
    hooks = setup_gpu_streaming(model)

    print("\n================== LEVEL 2 ==================", flush=True)
    for key, p_text, max_tokens in PROMPTS_L2:
        print(f"\n[Running {key}] ({max_tokens} max tokens)...", flush=True)
        msgs = [{"role": "user", "content": p_text}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                eos_token_id=[1, 106],
            )
        dt = time.time() - t0
        gen_clean = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        n_tok = out.shape[1] - inputs.input_ids.shape[1]
        print(f"[{key}] Generated {n_tok} tokens in {dt:.1f}s ({dt/n_tok:.2f} s/tok):\n{gen_clean}\n", flush=True)
        results["level2"][key] = {
            "prompt": p_text,
            "response": gen_clean,
            "tokens": n_tok,
            "time_sec": round(dt, 2)
        }

    print("\n================== LEVEL 3 ==================", flush=True)
    for key, p_text, max_tokens in PROMPTS_L3:
        print(f"\n[Running {key}] ({max_tokens} max tokens)...", flush=True)
        msgs = [{"role": "user", "content": p_text}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                eos_token_id=[1, 106],
            )
        dt = time.time() - t0
        gen_clean = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        n_tok = out.shape[1] - inputs.input_ids.shape[1]
        print(f"[{key}] Generated {n_tok} tokens in {dt:.1f}s ({dt/n_tok:.2f} s/tok):\n{gen_clean}\n", flush=True)
        results["level3"][key] = {
            "prompt": p_text,
            "response": gen_clean,
            "tokens": n_tok,
            "time_sec": round(dt, 2)
        }

    for h in hooks:
        h.remove()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[ALL DONE] Saved generation results to {OUT_JSON}", flush=True)

if __name__ == "__main__":
    run_suite()
