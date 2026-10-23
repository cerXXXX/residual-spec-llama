import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoProcessor

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from reconstruct_model import load_reconstructed_model
from eval_benchmarks import setup_gpu_streaming
from eval_benchmarks_v2 import extract_answer_letter

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

print("Loading base_b2 model...", flush=True)
base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
model = load_reconstructed_model(
    model_path=MODEL_DIR,
    base_codes_path=base_codes_path,
    residual_variant="base_b2",
    device="cpu",
)
hooks = setup_gpu_streaming(model)

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
preds = []
targets = []

print("\nEvaluating first 20 questions of MMLU-Pro on base_b2:", flush=True)
for idx in range(20):
    item = mmlu[idx]
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
    inp = processor.tokenizer([formatted], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=2,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=[1, 106]
        )
    gen = processor.tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    letter = extract_answer_letter(gen, choices)
    preds.append(letter)
    targets.append(ans)
    print(f"Q{idx+1}: Raw Gen='{gen}' -> Pred='{letter}', Target='{ans}', Correct={letter == ans}", flush=True)

from collections import Counter
print("\nPredictions distribution:", Counter(preds))
print("Targets distribution:", Counter(targets))
print(f"Accuracy: {sum(p == t for p, t in zip(preds, targets)) / len(preds) * 100:.1f}%")

for h in hooks:
    h.remove()
