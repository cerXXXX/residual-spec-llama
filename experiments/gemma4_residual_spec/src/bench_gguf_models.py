import subprocess
import time
import re
import json
import psutil

LLAMA_CLI = r"D:\localllamacpp\bin\b11099\llama-cli.exe"
MODELS = {
    "llama_cpp_Q4_K_M": {
        "path": r"D:\localllamacpp\models\gemma4-12b\gemma-4-12b-it-Q4_K_M.gguf",
        "ngl": 30
    },
    "llama_cpp_IQ2_M": {
        "path": r"D:\localllamacpp\models\gemma4-12b\gemma-4-12b-it-UD-IQ2_M.gguf",
        "ngl": 99
    }
}

PROMPTS = [
    ("Capital of France", "What is the capital of France? Answer in one short sentence."),
    ("Arithmetic", "Calculate: 17 * 4 = "),
    ("Greeting", "Introduce yourself in two sentences.")
]

def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(out.strip())
    except Exception:
        return 0.0

def run_bench():
    results = {}
    for m_name, m_info in MODELS.items():
        print(f"\n=======================================================", flush=True)
        print(f"Testing {m_name} (ngl={m_info['ngl']})...", flush=True)
        print(f"=======================================================", flush=True)
        results[m_name] = []

        for p_label, p_text in PROMPTS:
            formatted_prompt = f"<bos><|turn>user\n{p_text}<turn|>\n<|turn>model\n"
            cmd = [
                LLAMA_CLI,
                "-m", m_info["path"],
                "-ngl", str(m_info["ngl"]),
                "-c", "2048",
                "-b", "2048",
                "-ub", "512",
                "-n", "30",
                "--reasoning", "off",
                "-st",
                "-p", formatted_prompt
            ]
            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Poll memory during execution
            peak_vram = 0.0
            peak_ram_mb = 0.0
            p_obj = psutil.Process(proc.pid)
            
            while proc.poll() is None:
                try:
                    peak_vram = max(peak_vram, get_vram())
                    peak_ram_mb = max(peak_ram_mb, p_obj.memory_info().rss / (1024 * 1024))
                except Exception:
                    pass
                time.sleep(0.05)
                
            stdout, _ = proc.communicate()
            dt = time.time() - t0
            
            # Parse output
            m_gen = re.search(r"Generation:\s*([0-9\.]+)\s*t/s", stdout)
            m_prompt = re.search(r"Prompt:\s*([0-9\.]+)\s*t/s", stdout)
            gen_speed = float(m_gen.group(1)) if m_gen else 0.0
            prompt_speed = float(m_prompt.group(1)) if m_prompt else 0.0
            
            # Extract generated response text
            ans_match = re.search(r"<\|turn>model\s*\n(.*?)(?=\n\[ Prompt|\Z)", stdout, re.DOTALL)
            ans_text = ans_match.group(1).strip() if ans_match else stdout[-300:].strip()
            
            item = {
                "prompt": p_label,
                "response": ans_text,
                "gen_tok_per_sec": gen_speed,
                "prompt_tok_per_sec": prompt_speed,
                "wall_time_sec": round(dt, 2),
                "peak_vram_mb": peak_vram,
                "peak_ram_mb": round(peak_ram_mb, 1)
            }
            print(f"[{p_label}] Wall: {dt:.2f}s | Gen: {gen_speed} t/s | Peak VRAM: {peak_vram} MB | Peak RAM: {peak_ram_mb:.1f} MB", flush=True)
            print(f"  Response: {ans_text!r}", flush=True)
            results[m_name].append(item)

    with open(r"D:\localllamacpp\experiments\gemma4_residual_spec\results\live_bench_gguf.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nGGUF Benchmark completed successfully!", flush=True)

if __name__ == "__main__":
    run_bench()
