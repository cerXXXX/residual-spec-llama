#!/usr/bin/env python3
"""
Benchmark utility for llama.cpp server.
Measures Time-To-First-Token (TTFT), token generation speed (t/s), and total throughput.
"""

import time
import json
import urllib.request
import urllib.error
import argparse
import sys

def test_completion(api_url, api_key, prompt, max_tokens=256, model="gemma-4-E4B-it-Q4_K_M"):
    req_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    endpoint = f"{api_url.rstrip('/')}/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(req_data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            usage = data.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            tps = comp_tokens / elapsed if elapsed > 0 else 0
            content = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "tps": tps,
                "content_preview": content[:120].replace("\n", " ")
            }
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_loaded_model(api_url, api_key):
    try:
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0].get("id", "Tiel-Coder-35B-A3B")
    except Exception:
        pass
    return "Tiel-Coder-35B-A3B"

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp API Server")
    parser.add_argument("--url", default="http://127.0.0.1:9095", help="Base URL of server (default: http://127.0.0.1:9095)")
    parser.add_argument("--api-key", default="sk-localllama-rtx3060-securekey", help="API Key")
    parser.add_argument("--model", "-m", default="", help="Model alias to test (default: auto-detected from /v1/models or Tiel-Coder-35B-A3B)")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens to generate per test (default: 300)")
    args = parser.parse_args()

    model_name = args.model if args.model else get_loaded_model(args.url, args.api_key)

    prompts = [
        ("Code Python", "Write a python implementation of merge sort algorithm with comments and docstring."),
        ("HTML/CSS UI", "Create a responsive HTML and CSS card component with header, image, description, and button."),
        ("SQL Query", "Write a complex SQL query joining users, orders, and order_items with aggregation and window functions."),
        ("Structured JSON", "Generate a JSON schema and sample data for an e-commerce product catalog with 5 items.")
    ]
    
    print(f"Connecting to: {args.url}")
    print(f"Model:         {model_name}")
    print(f"{'Task':<18} | {'Prompt':<6} | {'Gen':<5} | {'Time (s)':<8} | {'Speed (t/s)':<12}")
    print("-" * 57)
    
    speeds = []
    for name, p in prompts:
        res = test_completion(args.url, args.api_key, p, max_tokens=args.max_tokens, model=model_name)
        if res["success"]:
            print(f"{name:<18} | {res['prompt_tokens']:<6} | {res['completion_tokens']:<5} | {res['elapsed']:<8.2f} | {res['tps']:<12.2f}")
            speeds.append(res["tps"])
        else:
            print(f"{name:<18} | ERROR: {res['error']}")
            
    if speeds:
        print("-" * 57)
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        print(f"Average generation speed: {avg_speed:.2f} t/s (Peak: {max_speed:.2f} t/s)")

if __name__ == "__main__":
    main()
