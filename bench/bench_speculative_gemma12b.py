import os
import sys
import subprocess
import time
import re
import json
import threading
import psutil

LLAMA_CLI = r"D:\localllamacpp\bin\b11099\llama-cli.exe"
TARGET_MODEL = r"D:\localllamacpp\models\gemma4-12b\gemma-4-12b-it-Q4_K_M.gguf"
MTP_MODEL = r"D:\localllamacpp\models\drafts\MTP\mtp-gemma-4-12b-it-Q8_0.gguf"
DFLASH_MODEL = r"D:\localllamacpp\models\drafts\gemma-4-12B-it-DFlash-Q4_K_M.gguf"
DSPARK_MODEL = r"D:\localllamacpp\models\drafts\gemma4_12b-DSpark-Q4_K_M.gguf"

PROMPT = "Write a Python function to check if a number is prime, with explanation."
N_GEN = 64
CTX = 2048
BATCH = 512
UBATCH = 512

def get_vram_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

def monitor_memory(stop_event, stats):
    peak_vram = get_vram_used()
    peak_ram = psutil.virtual_memory().used // (1024 * 1024)
    while not stop_event.is_set():
        v = get_vram_used()
        r = psutil.virtual_memory().used // (1024 * 1024)
        if v > peak_vram:
            peak_vram = v
        if r > peak_ram:
            peak_ram = r
        time.sleep(0.05)
    stats["peak_vram"] = peak_vram
    stats["peak_ram"] = peak_ram

def run_test(name, extra_args):
    cmd = [
        LLAMA_CLI,
        "-m", TARGET_MODEL,
        "-c", str(CTX),
        "-b", str(BATCH),
        "-ub", str(UBATCH),
        "-p", PROMPT,
        "-n", str(N_GEN),
        "-st",
        "-fit", "off"
    ] + extra_args

    print(f"\n=======================================================")
    print(f"Running test: {name}")
    print(f"Cmd: {' '.join(cmd)}")
    
    stop_event = threading.Event()
    mem_stats = {"peak_vram": 0, "peak_ram": 0}
    t_mon = threading.Thread(target=monitor_memory, args=(stop_event, mem_stats))
    t_mon.start()

    start_t = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=120)
        output = proc.stdout
        code = proc.returncode
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
        code = -1
    except Exception as e:
        output = str(e)
        code = -2
    finally:
        stop_event.set()
        t_mon.join()
        wall_t = time.time() - start_t

    # parse output
    # e.g.: [ Prompt: 28.1 t/s | Generation: 20.5 t/s ]
    # or prompt eval time = ... / eval time = ...
    pp_ts = None
    tg_ts = None
    m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
    if m:
        pp_ts = float(m.group(1))
        tg_ts = float(m.group(2))

    error_snippet = None
    status = "SUCCESS"
    if code != 0 or tg_ts is None:
        status = "FAILED"
        # find relevant error lines
        lines = [line.strip() for line in output.split("\n") if "error" in line.lower() or "failed" in line.lower() or "exception" in line.lower() or "invalid" in line.lower()]
        error_snippet = " | ".join(lines[-3:]) if lines else f"Exit code {code}"

    res = {
        "name": name,
        "extra_args": extra_args,
        "status": status,
        "pp_ts": pp_ts,
        "tg_ts": tg_ts,
        "wall_time_s": round(wall_t, 2),
        "peak_vram_mib": mem_stats["peak_vram"],
        "peak_ram_mib": mem_stats["peak_ram"],
        "error": error_snippet
    }
    print(f"Result: {status} | TG: {tg_ts} t/s | PP: {pp_ts} t/s | VRAM: {mem_stats['peak_vram']} MiB | RAM: {mem_stats['peak_ram']} MiB | Error: {error_snippet}")
    return res

def main():
    configs = [
        # 1. Baselines (No draft)
        ("Baseline (Sweet spot, -ngl 37)", ["-ngl", "37"]),
        ("Baseline (-ngl 32)", ["-ngl", "32"]),
        ("Baseline (Conservative, -ngl 30)", ["-ngl", "30"]),

        # 2. Google MTP
        ("Google MTP (-ngl 32, -ngld 99, n-max 1)", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "1"]),
        ("Google MTP (-ngl 32, -ngld 99, n-max 2)", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "2"]),
        ("Google MTP (-ngl 32, -ngld 99, n-max 4)", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "4"]),
        ("Google MTP (-ngl 30, -ngld 99, n-max 2)", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "30", "-ngld", "99", "--spec-draft-n-max", "2"]),
        ("Google MTP (-ngl 37, -ngld 0 [CPU], n-max 2)", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "37", "-ngld", "0", "--spec-draft-n-max", "2"]),
        ("Google MTP (-ngl 37, -ngld 99 [VRAM overflow attempt])", ["-md", MTP_MODEL, "--spec-type", "draft-mtp", "-ngl", "37", "-ngld", "99", "--spec-draft-n-max", "2"]),

        # 3. z-lab DFlash
        ("z-lab DFlash (-ngl 32, -ngld 99, n-max 2)", ["-md", DFLASH_MODEL, "--spec-type", "draft-dflash", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "2"]),
        ("z-lab DFlash (-ngl 32, -ngld 99, n-max 4)", ["-md", DFLASH_MODEL, "--spec-type", "draft-dflash", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "4"]),
        ("z-lab DFlash (-ngl 32, -ngld 99, n-max 8)", ["-md", DFLASH_MODEL, "--spec-type", "draft-dflash", "-ngl", "32", "-ngld", "99", "--spec-draft-n-max", "8"]),
        ("z-lab DFlash (-ngl 30, -ngld 99, n-max 2)", ["-md", DFLASH_MODEL, "--spec-type", "draft-dflash", "-ngl", "30", "-ngld", "99", "--spec-draft-n-max", "2"]),
        ("z-lab DFlash (-ngl 37, -ngld 0 [CPU], n-max 2)", ["-md", DFLASH_MODEL, "--spec-type", "draft-dflash", "-ngl", "37", "-ngld", "0", "--spec-draft-n-max", "2"]),

        # 4. DSpark
        ("DSpark (-ngl 30, -ngld 99, n-max 2)", ["-md", DSPARK_MODEL, "--spec-type", "draft-dspark", "-ngl", "30", "-ngld", "99", "--spec-draft-n-max", "2"])
    ]

    results = []
    for name, args in configs:
        res = run_test(name, args)
        results.append(res)
        time.sleep(1.0) # cool down

    out_json = r"D:\localllamacpp\experiments\gemma4_residual_spec\results\SPECULATIVE_BENCHMARK_2026_10_23.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll tests completed! Saved to {out_json}")

if __name__ == "__main__":
    main()
