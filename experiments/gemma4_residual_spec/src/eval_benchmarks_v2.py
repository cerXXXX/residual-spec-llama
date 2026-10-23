import argparse
import gc
import json
import math
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

# Add src to sys.path
sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from reconstruct_model import load_reconstructed_model
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
DATA_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\data"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"
RESULTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\results"

def eval_wikitext_ppl(model, test_ids, seqlen=2048, n_chunks=8, device="cuda"):
    """
    Evaluates WikiText-2 Perplexity with explicit <bos> token (ID 2) prepended
    to each chunk as an attention sink, preventing positional drift on instruction models.
    """
    print(f"\n[Eval] Evaluating WikiText-2 Perplexity ({n_chunks} chunks x {seqlen} tokens with <bos>)...", flush=True)
    n_tokens = test_ids.numel()
    max_chunks = n_tokens // seqlen
    nll_sum = 0.0
    total_eval_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(min(max_chunks, n_chunks)):
            t0 = time.time()
            raw_chunk = test_ids[i * seqlen : (i + 1) * seqlen].unsqueeze(0).to(device)
            # Prepend <bos> (ID 2)
            bos = torch.tensor([[2]], device=device)
            inp_chunk = torch.cat([bos, raw_chunk], dim=1)
            target = raw_chunk # Target is the raw chunk tokens

            outputs = model(inp_chunk)
            logits = outputs.logits[:, :-1, :] # Predict target from <bos> onwards
            loss = torch.nn.functional.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                target.contiguous().view(-1),
            )
            val_loss = loss.item()
            nll_sum += val_loss * seqlen
            total_eval_tokens += seqlen
            dt = time.time() - t0

            del outputs, logits, inp_chunk, raw_chunk, target, loss
            torch.cuda.empty_cache()

            print(f"  Chunk {i+1}/{n_chunks} loss: {val_loss:.4f} (running PPL: {math.exp(nll_sum / total_eval_tokens):.4f}) [{dt:.2f}s]", flush=True)

    ppl = math.exp(nll_sum / total_eval_tokens)
    print(f"  WikiText-2 Final PPL: {ppl:.4f} (over {total_eval_tokens} tokens)", flush=True)
    return ppl

def extract_answer_letter(text, valid_choices):
    text = text.strip()
    if not text:
        return ""
    # Check if first character is in valid choices
    first_char = text[0].upper()
    if first_char in valid_choices:
        return first_char
    # Search for pattern like (A), **A**, Answer: A
    m = re.search(r'\b([A-J])\b', text)
    if m and m.group(1) in valid_choices:
        return m.group(1)
    return first_char

def eval_arc_challenge(model, processor, dataset, n_samples=100, batch_size=8, device="cuda"):
    """
    Evaluates ARC-Challenge using canonical Gemma 4 chat template and greedy decoding (temperature=0.0).
    """
    print(f"\n[Eval] Evaluating ARC-Challenge ({n_samples} questions, batch_size={batch_size}, greedy temp=0.0)...", flush=True)
    choices_letters = ["A", "B", "C", "D"]
    prompts = []
    answers = []

    for idx, item in enumerate(dataset):
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
        msgs = [{"role": "user", "content": prompt_text}]
        formatted = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(formatted)
        answers.append(item["answerKey"])

    total = len(prompts)
    correct = 0
    t0 = time.time()

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id or 0

    model.eval()
    with torch.no_grad():
        for b_start in range(0, total, batch_size):
            b_end = min(b_start + batch_size, total)
            b_prompts = prompts[b_start:b_end]

            inp = processor.tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **inp,
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=[1, 106]
            )

            prompt_len = inp.input_ids.shape[1]
            for b_idx in range(len(b_prompts)):
                q_idx = b_start + b_idx
                gen_tokens = out[b_idx, prompt_len:]
                gen_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                pred = extract_answer_letter(gen_text, choices_letters)
                expected = answers[q_idx]
                is_correct = (pred == expected)
                if is_correct:
                    correct += 1

            del inp, out
            torch.cuda.empty_cache()

            dt = time.time() - t0
            print(f"  ARC-Challenge progress: {b_end}/{total} | Current Acc: {correct / b_end * 100:.2f}% [{dt:.1f}s, {b_end/dt:.2f} q/s]", flush=True)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"  ARC-Challenge Final Accuracy: {acc:.2f}% ({correct}/{total})", flush=True)
    return acc

def eval_mmlu_pro(model, processor, dataset, n_samples=100, batch_size=4, device="cuda"):
    """
    Evaluates MMLU-Pro using canonical Gemma 4 chat template and greedy decoding (temperature=0.0).
    """
    print(f"\n[Eval] Evaluating MMLU-Pro ({n_samples} questions, batch_size={batch_size}, greedy temp=0.0)...", flush=True)
    choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    prompts = []
    answers = []

    for idx, item in enumerate(dataset):
        if idx >= n_samples:
            break
        q = item["question"]
        opts = item["options"]
        ans = item["answer"]
        opts_formatted = "\n".join([f"{choices[i]}. {opt}" for i, opt in enumerate(opts)])
        prompt_text = (
            f"The following is a multiple-choice question. Please answer with ONLY the single letter of the correct option (e.g. A, B, C, D...).\n\n"
            f"Question: {q}\n\nOptions:\n{opts_formatted}\n\nAnswer:"
        )
        msgs = [{"role": "user", "content": prompt_text}]
        formatted = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(formatted)
        answers.append(ans)

    total = len(prompts)
    correct = 0
    t0 = time.time()

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id or 0

    model.eval()
    with torch.no_grad():
        for b_start in range(0, total, batch_size):
            b_end = min(b_start + batch_size, total)
            b_prompts = prompts[b_start:b_end]

            inp = processor.tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **inp,
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=[1, 106]
            )

            prompt_len = inp.input_ids.shape[1]
            for b_idx in range(len(b_prompts)):
                q_idx = b_start + b_idx
                gen_tokens = out[b_idx, prompt_len:]
                gen_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                pred = extract_answer_letter(gen_text, choices)
                expected = answers[q_idx]
                is_correct = (pred == expected)
                if is_correct:
                    correct += 1

            del inp, out
            torch.cuda.empty_cache()

            dt = time.time() - t0
            print(f"  MMLU-Pro progress: {b_end}/{total} | Current Acc: {correct / b_end * 100:.2f}% [{dt:.1f}s, {b_end/dt:.2f} q/s]", flush=True)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"  MMLU-Pro Final Accuracy: {acc:.2f}% ({correct}/{total})", flush=True)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["bf16", "residual_q8", "residual_q6", "base_b2"], default="residual_q8")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=================================================================", flush=True)
    print(f"  Running V2 Canonical Validation Benchmark: {args.variant}", flush=True)
    print(f"=================================================================\n", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    if args.variant == "bf16":
        print(f"[Loader] Loading original ground-truth BF16 model from {MODEL_DIR}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()
    else:
        base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
        model = load_reconstructed_model(
            model_path=MODEL_DIR,
            base_codes_path=base_codes_path,
            residual_variant=args.variant,
            device="cpu",
        )

    print("[Streamer] Setting up zero-copy GPU layer streaming...", flush=True)
    hooks = setup_gpu_streaming(model)
    print(f"[Streamer] Active. GPU VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n", flush=True)

    # 1. WikiText-2 PPL
    test_ids = torch.load(os.path.join(DATA_DIR, "gemma_wikitext2_test.pt"))
    ppl = eval_wikitext_ppl(model, test_ids, n_chunks=args.ppl_chunks, device="cuda")

    # 2. ARC-Challenge
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_acc = eval_arc_challenge(model, processor, arc_ds, n_samples=args.samples, batch_size=args.batch_size, device="cuda")

    # 3. MMLU-Pro
    mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu_acc = eval_mmlu_pro(model, processor, mmlu_ds, n_samples=args.samples, batch_size=args.batch_size, device="cuda")

    for h in hooks:
        h.remove()

    res = {
        "variant": args.variant,
        "wikitext_ppl": ppl,
        "arc_challenge_acc": arc_acc,
        "mmlu_pro_acc": mmlu_acc,
        "samples": args.samples,
        "ppl_chunks": args.ppl_chunks,
        "batch_size": args.batch_size,
        "temperature": 0.0,
        "chat_template": "canonical_gemma4",
        "timestamp": time.time(),
    }

    out_file = os.path.join(RESULTS_DIR, f"v2_eval_{args.variant}.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[DONE] Saved validated results to {out_file}", flush=True)
    print(f"  WikiText-2 PPL:   {ppl:.4f}")
    print(f"  ARC-Challenge:    {arc_acc:.2f}%")
    print(f"  MMLU-Pro Acc:     {mmlu_acc:.2f}%")

if __name__ == "__main__":
    main()
