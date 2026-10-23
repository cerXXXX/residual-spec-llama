import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor

from twobit.calibration import OTHER_LANGS, build_chat_calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--out", default="data/calib_chat.pt")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--n-en", type=int, default=14)
    ap.add_argument("--lang", default="pl")
    ap.add_argument("--n-lang", type=int, default=12)
    ap.add_argument("--other-langs", default=",".join(OTHER_LANGS))
    ap.add_argument("--other-per-lang", type=int, default=400)
    ap.add_argument("--n-other", type=int, default=6)
    a = ap.parse_args()
    processor = AutoProcessor.from_pretrained(a.model)
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    oasst = load_dataset("OpenAssistant/oasst2", split="train")
    chunks, stats, primary = build_chat_calibration(processor, ultrachat, oasst, seq=a.seq, n_en=a.n_en, lang=a.lang, n_lang=a.n_lang,
                                                    other_langs=tuple(a.other_langs.split(",")), other_per_lang=a.other_per_lang,
                                                    n_other=a.n_other)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(torch.tensor(chunks, dtype=torch.long), a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(stats, f, indent=1)
    print(stats)
    print(f"{a.lang} sample:", processor.tokenizer.decode(primary[0][:120]) if primary else "-")


if __name__ == "__main__":
    main()
