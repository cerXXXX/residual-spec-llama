#!/usr/bin/env python3
"""
sweep_moe_bench.py - Automated MoE Offload Sweep Benchmarker for llama.cpp

Systematically sweeps through different values of --n-cpu-moe (how many layers
of expert weights stay in CPU RAM vs GPU VRAM) using `llama.exe bench` to find
the exact peak performance configuration on your specific hardware (e.g. RTX 3060 Laptop 6GB).
"""

import os
import sys
import json
import time
import argparse
import subprocess

def get_vram_usage():
    """Returns (used_mib, total_mib) from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5
        ).strip()
        used, total = out.split(",")
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0.0, 0.0

def find_default_model(workspace_dir):
    """Finds the preferred Tiel-Coder-35B model in models directory."""
    models_dir = os.path.join(workspace_dir, "models")
    if not os.path.exists(models_dir):
        return None
    candidates = []
    for root, _, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f.lower():
                path = os.path.join(root, f)
                name = f.lower()
                is_tiel = "tiel" in name or "35b" in name
                is_q4 = "q4_k_xl" in name
                candidates.append((is_tiel, is_q4, os.path.getmtime(path), path))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][3]
    return None

def run_bench_step(llama_bin, model_path, n_cpu_moe, threads=10, n_prompt=128, n_gen=16):
    """Executes llama.exe bench for a specific n-cpu-moe value and returns stats."""
    cmd = [
        llama_bin, "bench",
        "-m", model_path,
        "-ngl", "99",
        "-ncmoe", str(n_cpu_moe),
        "-fa", "1",
        "-t", str(threads),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "1",
        "-o", "json"
    ]

    print(f"\n[*] Testing --n-cpu-moe {n_cpu_moe} (GPU experts: {max(0, 40 - n_cpu_moe)} layers)...")
    vram_before, vram_total = get_vram_usage()

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        elapsed = time.time() - t0
        output = proc.stdout
    except Exception as e:
        return {
            "n_cpu_moe": n_cpu_moe,
            "error": str(e),
            "pp_ts": 0.0,
            "tg_ts": 0.0,
            "vram_peak": 0.0,
            "vram_total": vram_total
        }

    vram_after, _ = get_vram_usage()
    vram_peak = max(vram_before, vram_after)

    # Parse JSON output from bench
    pp_ts = 0.0
    tg_ts = 0.0
    try:
        # Find JSON block in output
        json_start = output.find("[")
        json_end = output.rfind("]")
        if json_start != -1 and json_end != -1:
            raw_json = output[json_start:json_end + 1]
            data = json.loads(raw_json)
            for item in data:
                if item.get("n_prompt", 0) > 0 and item.get("n_gen", 0) == 0:
                    pp_ts = float(item.get("avg_ts", 0.0))
                elif item.get("n_gen", 0) > 0:
                    tg_ts = float(item.get("avg_ts", 0.0))
    except Exception as e:
        print(f"[!] Warning: failed to parse JSON ({e}). Raw snippet:\n{output[-500:]}")

    return {
        "n_cpu_moe": n_cpu_moe,
        "gpu_layers": max(0, 40 - n_cpu_moe),
        "pp_ts": pp_ts,
        "tg_ts": tg_ts,
        "vram_peak": vram_peak,
        "vram_total": vram_total,
        "elapsed": elapsed
    }

def main():
    parser = argparse.ArgumentParser(description="Sweep llama.cpp MoE offload settings to find optimal --n-cpu-moe")
    parser.add_argument("--model", "-m", default="", help="Model file path (default: auto-detect Tiel Q4_K_XL)")
    parser.add_argument("--layers", "-l", default="40,38,36,34,32",
                        help="Comma-separated list of --n-cpu-moe values to sweep (default: 40,38,36,34,32)")
    parser.add_argument("--threads", "-t", type=int, default=10, help="CPU threads (default: 10)")
    parser.add_argument("--prompt", "-p", type=int, default=128, help="Prompt tokens for bench (default: 128)")
    parser.add_argument("--gen", "-n", type=int, default=16, help="Generation tokens for bench (default: 16)")
    parser.add_argument("--llama-bin", default="llama.exe", help="Path to llama binary")
    args = parser.parse_args()

    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model if args.model else find_default_model(workspace_dir)

    if not model_path or not os.path.exists(model_path):
        print(f"[!] Error: Model not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        layers_to_test = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    except ValueError:
        print("[!] Error: --layers must be comma-separated integers, e.g. 40,38,36", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("   llama.cpp MoE Expert Offload Sweep Benchmarker")
    print("=" * 70)
    print(f"[*] Target Model:   {os.path.basename(model_path)}")
    print(f"[*] Test Sweeps:    --n-cpu-moe in {layers_to_test}")
    print(f"[*] CPU Threads:    {args.threads}")
    print(f"[*] Benchmark Size: Prompt {args.prompt} tokens, Gen {args.gen} tokens")
    _, vram_total = get_vram_usage()
    print(f"[*] Detected VRAM:  {vram_total:.0f} MiB")

    results = []
    for n_cpu_moe in layers_to_test:
        res = run_bench_step(
            args.llama_bin,
            model_path,
            n_cpu_moe,
            threads=args.threads,
            n_prompt=args.prompt,
            n_gen=args.gen
        )
        results.append(res)
        print(f"    -> Gen Speed: {res['tg_ts']:.2f} t/s | Prompt Speed: {res['pp_ts']:.2f} t/s")

    print("\n" + "=" * 70)
    print(f"{'--n-cpu-moe':<13} | {'GPU Experts':<12} | {'Prompt (t/s)':<14} | {'Gen (t/s)':<12} | {'Status':<15}")
    print("-" * 70)

    best_res = None
    max_tg = 0.0
    for r in results:
        nc = r["n_cpu_moe"]
        gpu_exp = r.get("gpu_layers", 0)
        pp = r["pp_ts"]
        tg = r["tg_ts"]
        
        status = "OK"
        if tg > max_tg:
            max_tg = tg
            best_res = r

        print(f"{nc:<13} | {gpu_exp:<12} | {pp:<14.2f} | {tg:<12.2f} | {status:<15}")

    print("-" * 70)
    if best_res:
        print(f"\n[+] Recommended setting for your hardware:")
        print(f"    --n-cpu-moe {best_res['n_cpu_moe']} (Peak Gen Speed: {best_res['tg_ts']:.2f} t/s)")
        if best_res['n_cpu_moe'] == 40:
            print("    (All expert weights on CPU, max VRAM headroom for 200k context)")
        else:
            print(f"    ({best_res.get('gpu_layers', 0)} expert layers on GPU, remaining {best_res['n_cpu_moe']} on CPU)")
    print("=" * 70)

if __name__ == "__main__":
    main()
