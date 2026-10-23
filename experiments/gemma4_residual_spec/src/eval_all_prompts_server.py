import os
import sys
import json
import time
import urllib.request

URL = "http://127.0.0.1:9095/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-localllama-rtx3060-securekey"
}

ALL_PROMPTS = [
    # Level 1
    ("L1.1_capital", 1, "What is the capital of France? Answer in one short sentence.", 100),
    ("L1.2_arithmetic", 1, "Calculate: 17 * 4 = ", 100),
    ("L1.3_greeting", 1, "Introduce yourself in two sentences.", 150),

    # Level 2
    ("L2.1_palindrome", 2, "Write a Python function 'is_palindrome(s: str) -> bool' that checks if a string is a palindrome, ignoring spaces, punctuation, and capitalization. Provide 2 assert tests.", 350),
    ("L2.2_sally", 2, "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have in total? Explain your answer briefly.", 200),
    ("L2.3_sky_blue", 2, "Explain in three clear bullet points why the sky appears blue during the day (Rayleigh scattering).", 300),
    ("L2.4_russian_idiom", 2, "Переведи на английский язык и объясни смысл поговорки: 'Дело в шляпе'.", 300),

    # Level 3
    ("L3.1_rate_limiter", 3, "Write a Python class 'SlidingWindowRateLimiter' that tracks requests per user within a rolling time window (e.g. 100 requests per 60 seconds). Implement thread-safety with threading.Lock, add type hints, docstrings, and handle edge cases (expired timestamps cleanup).", 550),
    ("L3.2_river_crossing", 3, "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item at a time. If left unattended, the wolf will eat the goat, and the goat will eat the cabbage. Provide a step-by-step solution explaining each crossing and the state of both riverbanks.", 550),
    ("L3.3_json_extraction", 3, "Extract structured entities from this sentence into valid JSON without markdown fences: 'John Doe bought 3 units of RTX 4090 on March 15 for $1599 each from Newegg.' Schema: {buyer: str, items: [{name: str, quantity: int, price_usd: float}], merchant: str, date: str}.", 250),
    ("L3.4_flash_attention", 3, "Подробно объясни архитектурный принцип механизма FlashAttention: чем он отличается от стандартного Scaled Dot-Product Attention, как используется тайлинг (tiling) в SRAM GPU и почему он снижает требования к объему памяти HBM с O(N^2) до O(N).", 600),
]

def run_suite(model_name, output_path):
    results = {}
    print(f"\n=======================================================", flush=True)
    print(f"  Running 3-Level Evaluation on {model_name}", flush=True)
    print(f"=======================================================\n", flush=True)

    for key, level, prompt, max_tokens in ALL_PROMPTS:
        print(f"[{key}] (Level {level}) Sending request...", flush=True)
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        req = urllib.request.Request(URL, data=json.dumps(data).encode("utf-8"), headers=HEADERS)

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0

        msg = res["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = res.get("usage", {})
        timings = res.get("timings", {})

        n_tok = usage.get("completion_tokens", 0)
        speed = timings.get("predicted_per_second", n_tok / dt if dt > 0 else 0)

        json_valid = None
        if "json" in key:
            try:
                parsed = json.loads(content.strip().strip("`").strip())
                json_valid = True
            except Exception as e:
                json_valid = False

        print(f"  -> Generated {n_tok} tokens in {dt:.2f}s ({speed:.1f} tok/s)", flush=True)
        print(f"  -> Preview: {repr(content[:120])}...", flush=True)
        if json_valid is not None:
            print(f"  -> JSON Valid: {json_valid}", flush=True)

        results[key] = {
            "level": level,
            "prompt": prompt,
            "content": content,
            "reasoning": reasoning,
            "tokens": n_tok,
            "time_sec": round(dt, 2),
            "tok_per_sec": round(speed, 2),
            "json_valid": json_valid
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {output_path}!\n", flush=True)

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else r"D:\localllamacpp\experiments\gemma4_residual_spec\results\eval_server_iq2m.json"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Gemma-4-12B-it-UD-IQ2_M"
    run_suite(model_name, out_file)
