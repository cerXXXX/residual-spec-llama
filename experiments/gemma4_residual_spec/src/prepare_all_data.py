import os
import sys
import time
import json
import torch

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

# Add src to sys.path
sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")

from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer
from twobit.calibration import build_chat_calibration, OTHER_LANGS

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
DATA_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\data"

C4_CALIB_DOCS = 3000
C4_CALIB_TOKENS = 256 * 2047
C4_PROMPT_DOCS = (3000, 3016)
DOC_CHARS = 1200

MANUAL_PROMPTS = [
    ("en", "Explain how a refrigerator works to a 12-year-old."),
    ("en", "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring."),
    ("en", "A train leaves at 9:40 and arrives at 13:05. How long is the journey? Show your reasoning."),
    ("en", "List five practical tips for reducing household electricity usage."),
    ("en", "Write a short, polite email asking a colleague to review a document by Friday."),
    ("en", "What are the main differences between TCP and UDP?"),
    ("en", "Write a four-line poem about autumn rain."),
    ("en", "Summarize the causes of the French Revolution in one paragraph."),
    ("en", "Give me a recipe for a simple tomato soup."),
    ("en", "Explain the difference between a list and a tuple in Python, with examples."),
    ("en", "What should I check before buying a used car?"),
    ("en", "Describe the water cycle step by step."),
    ("pl", "Wyjaśnij krótko, czym jest inflacja i jak wpływa na oszczędności."),
    ("pl", "Napisz krótkie ogłoszenie o sprzedaży roweru miejskiego."),
    ("pl", "Jakie są zalety i wady pracy zdalnej? Wypisz po trzy."),
    ("pl", "Opisz w кількох zdaniach historię starożytnego Rzymu."),
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    print("[1/5] Loading local tokenizer & processor...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    print("[2/5] Downloading and preparing WikiText-2...", flush=True)
    ds_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    ids_val = tok("\n\n".join(ds_val["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    torch.save(ids_val, os.path.join(DATA_DIR, "gemma_wikitext2_validation.pt"))
    print(f"  WikiText-2 validation saved: {ids_val.numel()} tokens", flush=True)

    ds_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids_test = tok("\n\n".join(ds_test["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    torch.save(ids_test, os.path.join(DATA_DIR, "gemma_wikitext2_test.pt"))
    print(f"  WikiText-2 test saved: {ids_test.numel()} tokens", flush=True)

    print("[3/5] Downloading and preparing C4 calibration slice...", flush=True)
    c4 = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    c4_texts = [x for x in c4["text"][:C4_CALIB_DOCS]]
    ids_c4 = tok("\n\n".join(c4_texts), return_tensors="pt", add_special_tokens=False).input_ids[0][:C4_CALIB_TOKENS]
    torch.save(ids_c4, os.path.join(DATA_DIR, "gemma_c4_calib.pt"))
    print(f"  C4 calibration saved: {ids_c4.numel()} tokens", flush=True)

    print("  Preparing prompt suite...", flush=True)
    prompts = []
    for doc in c4["text"][C4_PROMPT_DOCS[0]:C4_PROMPT_DOCS[1]]:
        prompts.append(("c4_summary", "Summarize the following text in three sentences:\n\n" + doc[:DOC_CHARS]))
    for lang, u in MANUAL_PROMPTS:
        prompts.append((f"manual_{lang}", u))
    out, info = [], []
    for src, u in prompts:
        msgs = [{"role": "user", "content": u}]
        x = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
        out.append((src, x["input_ids"][0].clone()))
        info.append({"source": src, "prompt": u, "tokens": int(x["input_ids"].shape[1])})
    torch.save(out, os.path.join(DATA_DIR, "prompts.pt"))
    with open(os.path.join(DATA_DIR, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    print(f"  Prompts saved: {len(out)} items", flush=True)

    print("[4/5] Downloading UltraChat 200k & OASST2 for multilingual chat calibration...", flush=True)
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    oasst = load_dataset("OpenAssistant/oasst2", split="train")

    print("[5/5] Building chat calibration chunks (calib_chat.pt)...", flush=True)
    chunks, stats, primary = build_chat_calibration(
        processor, ultrachat, oasst, seq=2048, n_en=14, lang="pl", n_lang=12,
        other_langs=OTHER_LANGS, other_per_lang=400, n_other=6
    )
    torch.save(torch.tensor(chunks, dtype=torch.long), os.path.join(DATA_DIR, "calib_chat.pt"))
    with open(os.path.join(DATA_DIR, "calib_chat.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(f"  Chat calibration saved: {stats}", flush=True)
    print(f"All calibration & evaluation data prepared successfully in {time.time() - t0:.1f}s!")

if __name__ == "__main__":
    main()
