import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

C4_DOCS = 3000
C4_TOKENS = 256 * 2048

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
parser.add_argument("--out_dir", default="data")
args = parser.parse_args()

t0 = time.time()
os.makedirs(args.out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.model)
for split in ("test", "validation"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    torch.save(ids, os.path.join(args.out_dir, f"qwen_wikitext2_{split}.pt"))
    print(f"wikitext2 {split}: {ids.numel()} tokens", flush=True)
c4 = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
texts = [x for x in c4["text"][:C4_DOCS]]
ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0][:C4_TOKENS]
torch.save(ids, os.path.join(args.out_dir, "qwen_c4_calib.pt"))
print(f"c4 calibration: {ids.numel()} tokens", flush=True)
print("data download done", round(time.time() - t0), "s")
