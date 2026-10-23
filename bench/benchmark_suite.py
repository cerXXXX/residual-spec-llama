import os
import sys
import time
import subprocess
import json
import re
import urllib.request

MODEL_PATH = r"D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
PORT = 9095
API_KEY = "test-benchmark-key"

def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        used, total = out.split(",")
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0, 0

def run_llama_bench(args_list):
    """Runs llama.exe bench with specified arguments and returns markdown/parsed stats."""
    cmd = ["llama.exe", "bench", "-m", MODEL_PATH, "-o", "json"] + args_list
    print(f"\n[+] Running bench: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
        return res.stdout
    except Exception as e:
        return str(e)

def test_server_speed(extra_flags, prompt="Write a fast Dijkstra shortest path algorithm in Python with detailed comments.", n_predict=128):
    """Starts llama.exe serve in background, sends request, measures t/s and draft acceptance."""
    used_start, total_vram = get_vram_mb()
    cmd = [
        "llama.exe", "serve",
        "-m", MODEL_PATH,
        "-c", "200000",
        "-ngl", "99",
        "-ot", ".ffn_.*_exps.=CPU",
        "-fa", "on",
        "-b", "2048",
        "-ub", "2048",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-t", "10",
        "-tb", "10",
        "--prio", "2",
        "--poll", "50",
        "--mlock",
        "--jinja",
        "--alias", "Tiel-Coder-35B-A3B",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--api-key", API_KEY
    ] + extra_flags

    print(f"\n[+] Launching test server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Wait for server ready
    url = f"http://127.0.0.1:{PORT}/v1/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    req = urllib.request.Request(url, headers=headers)
    
    ready = False
    start_wait = time.time()
    while time.time() - start_wait < 90:
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            print("[!] Server crashed on startup:\n", stdout[-2000:])
            return None
        try:
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(1)

    if not ready:
        print("[!] Server did not become ready in 90s.")
        proc.terminate()
        return None

    vram_peak, _ = get_vram_mb()
    print(f"[*] Server ready. VRAM usage: {vram_peak:.0f} / {total_vram:.0f} MiB")

    # Send completion request
    chat_url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "model": "Tiel-Coder-35B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.2
    }
    data = json.dumps(payload).encode("utf-8")
    chat_req = urllib.request.Request(chat_url, data=data, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})

    t0 = time.time()
    with urllib.request.urlopen(chat_req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()

    total_time = t1 - t0
    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    gen_speed = completion_tokens / total_time if total_time > 0 else 0

    print(f"[+] Response received: {completion_tokens} tokens in {total_time:.2f}s ({gen_speed:.2f} t/s)")

    # Terminate server and read logs for draft acceptance
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    stdout, _ = proc.communicate()
    
    # Parse draft acceptance rate if speculative decoding was used
    draft_match = re.search(r"draft acceptance rate:\s+([\d\.]+)\s*%", stdout)
    draft_rate = float(draft_match.group(1)) if draft_match else None
    
    # Parse exact timings if logged
    # e.g. eval time = ... ms / ... runs ( ... ms per token, ... tokens per second)
    eval_match = re.search(r"eval time =\s*([\d\.]+) ms \/\s*(\d+) runs\s*\(\s*([\d\.]+) ms per token,\s*([\d\.]+) tokens per second\)", stdout)
    exact_speed = float(eval_match.group(4)) if eval_match else gen_speed

    return {
        "flags": " ".join(extra_flags),
        "tokens": completion_tokens,
        "time": total_time,
        "speed": exact_speed,
        "vram_used": vram_peak,
        "draft_acceptance": draft_rate
    }

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at {MODEL_PATH}")
        sys.exit(1)
    print(f"[*] Benchmark suite ready for {MODEL_PATH}")
