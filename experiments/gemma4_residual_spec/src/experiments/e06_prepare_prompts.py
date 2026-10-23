import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

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
    ("pl", "Opisz w kilku zdaniach historię starożytnego Rzymu."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    torch.save(ids, os.path.join(a.data_dir, "gemma_wikitext2_validation.pt"))
    print(f"wikitext2 validation: {ids.numel()} tokens", flush=True)
    c4 = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    ids = tok("\n\n".join(c4["text"][:C4_CALIB_DOCS]), return_tensors="pt", add_special_tokens=False).input_ids[0][:C4_CALIB_TOKENS]
    torch.save(ids, os.path.join(a.data_dir, "gemma_c4_calib.pt"))
    print(f"c4 calibration: {ids.numel()} tokens", flush=True)
    processor = AutoProcessor.from_pretrained(a.model)
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
    torch.save(out, os.path.join(a.data_dir, "prompts.pt"))
    with open(os.path.join(a.data_dir, "prompts.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    print(f"prompts: {len(out)}; token lengths: min {min(t['tokens'] for t in info)}, max {max(t['tokens'] for t in info)}", flush=True)
    print("first manual prompt (decoded):", repr(tok.decode(out[16][1])[:300]), flush=True)


if __name__ == "__main__":
    main()
