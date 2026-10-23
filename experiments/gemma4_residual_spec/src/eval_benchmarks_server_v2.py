import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datasets import load_dataset

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

URL = "http://127.0.0.1:9095/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-localllama-rtx3060-securekey"
}
RESULTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\results"

def extract_answer_letter(text, valid_choices):
    text = text.strip()
    if not text:
        return ""
    first_char = text[0].upper()
    if first_char in valid_choices:
        return first_char
    m = re.search(r'\b([A-J])\b', text)
    if m and m.group(1) in valid_choices:
        return m.group(1)
    return first_char

def query_llama_server(prompt, max_tokens=4):
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    req = urllib.request.Request(URL, data=json.dumps(data).encode("utf-8"), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    msg = res["choices"][0]["message"]
    return (msg.get("content") or "").strip()

def wait_for_server(max_wait=30):
    print("Waiting for llama-server to be ready on port 9095...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            req = urllib.request.Request("http://127.0.0.1:9095/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("Server is ready!\n", flush=True)
                    return True
        except Exception:
            time.sleep(1)
    print("Warning: server health check timed out, proceeding anyway...", flush=True)
    return False

def run_evaluation(variant_name, n_samples=100):
    wait_for_server()
    print(f"=================================================================", flush=True)
    print(f"  Evaluating GGUF on Server: {variant_name} ({n_samples} samples)", flush=True)
    print(f"=================================================================\n", flush=True)

    # 1. ARC-Challenge
    print(f"[1/2] Evaluating ARC-Challenge ({n_samples} questions, greedy temp=0.0)...", flush=True)
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    choices_letters = ["A", "B", "C", "D"]
    arc_correct = 0
    t0 = time.time()

    for idx, item in enumerate(arc_ds):
        if idx >= n_samples:
            break
        q = item["question"]
        choices = item["choices"]
        lbls = choices["label"]
        txts = choices["text"]
        opts_formatted = "\n".join([f"{l}. {t}" for l, t in zip(lbls, txts)])
        prompt_text = (
            f"Choose the correct answer for the following question. Respond with ONLY the single letter of the correct option (e.g. A, B, C, D).\n\n"
            f"Question: {q}\n\nChoices:\n{opts_formatted}\n\nAnswer:"
        )
        try:
            resp = query_llama_server(prompt_text, max_tokens=4)
            pred = extract_answer_letter(resp, choices_letters)
            expected = item["answerKey"]
            if pred == expected:
                arc_correct += 1
        except Exception as e:
            print(f"  Error on Q{idx}: {e}", flush=True)

        if (idx + 1) % 25 == 0 or (idx + 1) == n_samples:
            dt = time.time() - t0
            print(f"  ARC progress: {idx+1}/{n_samples} | Acc: {arc_correct / (idx+1) * 100:.2f}% [{dt:.1f}s, {(idx+1)/dt:.1f} q/s]", flush=True)

    arc_acc = (arc_correct / n_samples) * 100.0
    print(f"  ARC-Challenge Final: {arc_acc:.2f}%\n", flush=True)

    # 2. MMLU-Pro
    print(f"[2/2] Evaluating MMLU-Pro ({n_samples} questions, greedy temp=0.0)...", flush=True)
    mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    choices_pro = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    mmlu_correct = 0
    t0 = time.time()

    for idx, item in enumerate(mmlu_ds):
        if idx >= n_samples:
            break
        q = item["question"]
        opts = item["options"]
        ans = item["answer"]
        opts_formatted = "\n".join([f"{choices_pro[i]}. {opt}" for i, opt in enumerate(opts)])
        prompt_text = (
            f"The following is a multiple-choice question. Please answer with ONLY the single letter of the correct option (e.g. A, B, C, D...).\n\n"
            f"Question: {q}\n\nOptions:\n{opts_formatted}\n\nAnswer:"
        )
        try:
            resp = query_llama_server(prompt_text, max_tokens=4)
            pred = extract_answer_letter(resp, choices_pro)
            expected = ans
            if pred == expected:
                mmlu_correct += 1
        except Exception as e:
            print(f"  Error on Q{idx}: {e}", flush=True)

        if (idx + 1) % 25 == 0 or (idx + 1) == n_samples:
            dt = time.time() - t0
            print(f"  MMLU-Pro progress: {idx+1}/{n_samples} | Acc: {mmlu_correct / (idx+1) * 100:.2f}% [{dt:.1f}s, {(idx+1)/dt:.1f} q/s]", flush=True)

    mmlu_acc = (mmlu_correct / n_samples) * 100.0
    print(f"  MMLU-Pro Final: {mmlu_acc:.2f}%\n", flush=True)

    out_file = os.path.join(RESULTS_DIR, f"v2_eval_{variant_name}.json")
    res = {
        "variant": variant_name,
        "arc_challenge_acc": arc_acc,
        "mmlu_pro_acc": mmlu_acc,
        "samples": n_samples,
        "temperature": 0.0,
        "engine": "llama.cpp",
        "timestamp": time.time(),
    }
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[DONE] Saved server results to {out_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", default="llama_cpp_Q4_K_M")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    run_evaluation(args.variant, args.samples)
