import json
import os
from collections import defaultdict
from datasets import load_dataset

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

DATA_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\data"
OUT_FILE = os.path.join(DATA_DIR, "balanced_benchmark_suite.json")

print("[1/2] Loading ARC-Challenge (100% of test split)...", flush=True)
arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc_list = []
for idx, item in enumerate(arc_ds):
    arc_list.append({
        "id": item.get("id", f"arc_{idx}"),
        "idx": idx,
        "question": item["question"],
        "choices": {
            "label": item["choices"]["label"],
            "text": item["choices"]["text"],
        },
        "answerKey": item["answerKey"]
    })
print(f"  Loaded all {len(arc_list)} ARC-Challenge questions.", flush=True)

print("\n[2/2] Loading MMLU-Pro (Stratified sample: up to 350 per category)...", flush=True)
mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

by_category = defaultdict(list)
for idx, item in enumerate(mmlu_ds):
    cat = item["category"]
    by_category[cat].append({
        "question_id": item.get("question_id", idx),
        "idx": idx,
        "category": cat,
        "question": item["question"],
        "options": item["options"],
        "answer": item["answer"],
        "src": item.get("src", "")
    })

TARGET_PER_CAT = 350
balanced_mmlu = []
cat_stats = {}
for cat, items in sorted(by_category.items()):
    selected = items[:TARGET_PER_CAT]
    balanced_mmlu.extend(selected)
    cat_stats[cat] = len(selected)

print(f"  MMLU-Pro category breakdown (target up to {TARGET_PER_CAT} per cat):")
for cat, count in cat_stats.items():
    print(f"    - {cat:18s}: {count}/{len(by_category[cat])}")
print(f"  Total balanced MMLU-Pro questions: {len(balanced_mmlu)} across {len(cat_stats)} categories.", flush=True)

suite = {
    "arc_challenge": arc_list,
    "mmlu_pro": balanced_mmlu,
    "metadata": {
        "arc_total": len(arc_list),
        "mmlu_total": len(balanced_mmlu),
        "target_per_cat": TARGET_PER_CAT,
        "categories": cat_stats
    }
}

os.makedirs(DATA_DIR, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(suite, f, indent=1, ensure_ascii=False)

print(f"\n[DONE] Saved complete benchmark suite to {OUT_FILE}!")
print(f"Total workload: {len(arc_list)} ARC + {len(balanced_mmlu)} MMLU-Pro = {len(arc_list) + len(balanced_mmlu)} questions.")
