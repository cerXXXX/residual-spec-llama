import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

sys.path.insert(0, r"D:\localllamacpp\experiments\gemma4_residual_spec\src")
from reconstruct_model import load_reconstructed_model
from eval_benchmarks import setup_gpu_streaming

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
WEIGHTS_DIR = r"D:\localllamacpp\experiments\gemma4_residual_spec\weights"

print("Loading processor & dataset...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

print("Loading residual_q8 model...", flush=True)
base_codes_path = os.path.join(WEIGHTS_DIR, "base_b2", "codes.pt")
model = load_reconstructed_model(
    model_path=MODEL_DIR,
    base_codes_path=base_codes_path,
    residual_variant="residual_q8",
    device="cpu",
)
hooks = setup_gpu_streaming(model)

print("\nTesting 3 ARC-Challenge questions with greedy generation (temp=0.0):", flush=True)
for i in range(3):
    item = arc[i]
    q = item['question']
    choices = item['choices']
    opts = '\n'.join([f"{lbl}. {txt}" for lbl, txt in zip(choices['label'], choices['text'])])
    prompt = f"Choose the correct answer for the following question. Respond with ONLY the single letter of the correct option (e.g. A, B, C, D).\n\nQuestion: {q}\n\nChoices:\n{opts}\n\nAnswer:"
    msgs = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to("cuda")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=[1, 106]
        )
    gen_text = processor.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"Q{i+1}: Expected '{item['answerKey']}', Model generated: '{gen_text}'", flush=True)

for h in hooks:
    h.remove()
