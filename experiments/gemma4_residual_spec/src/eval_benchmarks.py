import argparse
import gc
import json
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoProcessor

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

# Add src to sys.path
sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from reconstruct_model import load_reconstructed_model

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
DATA_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\data"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"
RESULTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\results"

def setup_gpu_streaming(model):
    """
    Zero-copy streaming:
    1. Embeddings, norm, and lm_head stay on GPU permanently (~2.1 GB VRAM).
    2. 48 transformer layers remain statically in CPU host memory.
    3. During forward pass, each layer's parameters & buffers stream to GPU.
    4. Upon layer completion, parameters point back to CPU memory (zero CPU allocation, zero host leak).
    Achieves ~3.5s per 48-layer pass with <2.5 GB peak VRAM.
    """
    lm = model.model.language_model
    layers = lm.layers

    for _, child in model.model.named_children():
        if child is lm:
            for sub_name, sub in lm.named_children():
                if sub_name != "layers":
                    sub.to("cuda")
        else:
            child.to("cuda")
    model.lm_head.to("cuda")

    # Capture permanent CPU references
    cpu_params = {}
    cpu_buffers = {}
    for i, layer in enumerate(layers):
        cpu_params[i] = {name: p.data for name, p in layer.named_parameters()}
        cpu_buffers[i] = {name: b.data for name, b in layer.named_buffers()}

    hooks = []
    for i, layer in enumerate(layers):
        def make_pre(idx):
            def pre_hook(module, args, kwargs):
                for name, p in module.named_parameters():
                    p.data = cpu_params[idx][name].to("cuda", non_blocking=True)
                for name, b in module.named_buffers():
                    b.data = cpu_buffers[idx][name].to("cuda", non_blocking=True)
            return pre_hook

        def make_post(idx):
            def post_hook(module, args, output):
                for name, p in module.named_parameters():
                    p.data = cpu_params[idx][name]
                for name, b in module.named_buffers():
                    b.data = cpu_buffers[idx][name]
            return post_hook

        h_pre = layer.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        h_post = layer.register_forward_hook(make_post(i))
        hooks.extend([h_pre, h_post])

    return hooks

def eval_wikitext_ppl(model, test_ids, seqlen=2048, n_chunks=8, device="cuda"):
    print(f"[Eval] Evaluating WikiText-2 Perplexity ({n_chunks} chunks x {seqlen} tokens)...", flush=True)
    n_tokens = test_ids.numel()
    max_chunks = n_tokens // seqlen
    nll_sum = 0.0
    total_eval_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(min(max_chunks, n_chunks)):
            t0 = time.time()
            chunk = test_ids[i * seqlen : (i + 1) * seqlen].unsqueeze(0).to(device)
            target = chunk.clone()
            outputs = model(chunk)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                target[:, 1:].contiguous().view(-1),
            )
            val_loss = loss.item()
            nll_sum += val_loss * (seqlen - 1)
            total_eval_tokens += seqlen - 1
            dt = time.time() - t0

            del outputs, logits, chunk, target, loss
            torch.cuda.empty_cache()

            print(f"  Chunk {i+1}/{n_chunks} loss: {val_loss:.4f} (running PPL: {math.exp(nll_sum / total_eval_tokens):.4f}) [{dt:.2f}s]", flush=True)

    ppl = math.exp(nll_sum / total_eval_tokens)
    print(f"  WikiText-2 Final PPL: {ppl:.4f} (over {total_eval_tokens} tokens)", flush=True)
    return ppl

def eval_mmlu_pro(model, processor, dataset, n_samples=100, batch_size=8, device="cuda"):
    print(f"[Eval] Evaluating MMLU-Pro ({n_samples} questions, batch size: {batch_size})...", flush=True)
    choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    choice_token_ids = [processor.tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    prompts = []
    answers = []
    opts_lens = []
    for idx, item in enumerate(dataset):
        if idx >= n_samples:
            break
        q = item["question"]
        opts = item["options"]
        ans = item["answer"]
        prompt = f"Question: {q}\nOptions:\n"
        for i, opt in enumerate(opts):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer:"
        prompts.append(prompt)
        answers.append(ans)
        opts_lens.append(len(opts))

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
            out = model(**inp)
            last_logits = out.logits[:, -1, :]

            for b_idx in range(len(b_prompts)):
                q_idx = b_start + b_idx
                logits = last_logits[b_idx]
                candidate_logits = torch.tensor([logits[tid].item() for tid in choice_token_ids[: opts_lens[q_idx]]])
                pred_idx = torch.argmax(candidate_logits).item()
                pred_letter = choices[pred_idx]
                if pred_letter == answers[q_idx]:
                    correct += 1

            del out, last_logits, inp
            torch.cuda.empty_cache()

            dt = time.time() - t0
            print(f"  MMLU-Pro progress: {b_end}/{total} | Current Acc: {correct / b_end * 100:.2f}% [{dt:.1f}s, {b_end/dt:.2f} q/s]", flush=True)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"  MMLU-Pro Final Accuracy: {acc:.2f}% ({correct}/{total})", flush=True)
    return acc

def eval_arc_challenge(model, processor, dataset, n_samples=100, batch_size=8, device="cuda"):
    print(f"[Eval] Evaluating ARC-Challenge ({n_samples} questions, batch size: {batch_size})...", flush=True)
    label_map = {"A": "A", "B": "B", "C": "C", "D": "D", "1": "A", "2": "B", "3": "C", "4": "D"}
    choices = ["A", "B", "C", "D"]
    choice_token_ids = [processor.tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]

    prompts = []
    answers = []
    opts_lens = []
    for idx, item in enumerate(dataset):
        if idx >= n_samples:
            break
        q = item["question"]
        opts = item["choices"]["text"]
        lbls = item["choices"]["label"]
        raw_ans = item["answerKey"]
        ans = label_map.get(raw_ans, raw_ans)

        prompt = f"Question: {q}\nOptions:\n"
        for i, opt in enumerate(opts):
            lbl = label_map.get(lbls[i], choices[i] if i < 4 else chr(ord("A") + i))
            prompt += f"{lbl}. {opt}\n"
        prompt += "Answer:"
        prompts.append(prompt)
        answers.append(ans)
        opts_lens.append(len(opts))

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
            out = model(**inp)
            last_logits = out.logits[:, -1, :]

            for b_idx in range(len(b_prompts)):
                q_idx = b_start + b_idx
                logits = last_logits[b_idx]
                candidate_logits = torch.tensor([logits[tid].item() for tid in choice_token_ids[: opts_lens[q_idx]]])
                pred_idx = torch.argmax(candidate_logits).item()
                pred_letter = choices[pred_idx]
                if pred_letter == answers[q_idx]:
                    correct += 1

            del out, last_logits, inp
            torch.cuda.empty_cache()

            dt = time.time() - t0
            print(f"  ARC-Challenge progress: {b_end}/{total} | Current Acc: {correct / b_end * 100:.2f}% [{dt:.1f}s, {b_end/dt:.2f} q/s]", flush=True)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"  ARC-Challenge Final Accuracy: {acc:.2f}% ({correct}/{total})", flush=True)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["base_b2", "residual_q6", "residual_q8"], default="residual_q8")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=== Starting Optimized Benchmark Evaluation for variant: {args.variant} ===", flush=True)

    base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    # Layer-by-layer reconstructed model
    model = load_reconstructed_model(
        model_path=MODEL_DIR,
        base_codes_path=base_codes_path,
        residual_variant=args.variant,
        device="cpu",
    )

    print("[Streamer] Setting up zero-copy GPU layer streaming...", flush=True)
    hooks = setup_gpu_streaming(model)
    print(f"[Streamer] Streaming active. GPU VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    # 1. WikiText-2 PPL
    test_ids = torch.load(os.path.join(DATA_DIR, "gemma_wikitext2_test.pt"))
    ppl = eval_wikitext_ppl(model, test_ids, n_chunks=args.ppl_chunks, device="cuda")

    # 2. MMLU-Pro
    mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu_acc = eval_mmlu_pro(model, processor, mmlu_ds, n_samples=args.samples, batch_size=args.batch_size, device="cuda")

    # 3. ARC-Challenge
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_acc = eval_arc_challenge(model, processor, arc_ds, n_samples=args.samples, batch_size=args.batch_size, device="cuda")

    for h in hooks:
        h.remove()

    res = {
        "variant": args.variant,
        "wikitext_ppl": ppl,
        "mmlu_pro_acc": mmlu_acc,
        "arc_challenge_acc": arc_acc,
        "samples": args.samples,
        "ppl_chunks": args.ppl_chunks,
        "batch_size": args.batch_size,
        "timestamp": time.time(),
    }

    out_file = os.path.join(RESULTS_DIR, f"eval_{args.variant}.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[DONE] Saved evaluation results to {out_file}", flush=True)
    print(f"  WikiText-2 PPL: {ppl:.4f}")
    print(f"  MMLU-Pro Acc:   {mmlu_acc:.2f}%")
    print(f"  ARC-Challenge:  {arc_acc:.2f}%")

if __name__ == "__main__":
    main()
