import os
import sys
from datasets import load_dataset
from transformers import AutoProcessor

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_DATASETS_CACHE"] = r"D:\localllamacpp\.cache\huggingface\datasets"

MODEL_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
processor = AutoProcessor.from_pretrained(MODEL_DIR)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

for i in range(3):
    item = arc[i]
    q = item['question']
    choices = item['choices']
    opts = '\n'.join([f"{lbl}. {txt}" for lbl, txt in zip(choices['label'], choices['text'])])
    prompt = f"Choose the correct answer for the following question. Respond with ONLY the single letter of the correct answer.\n\nQuestion: {q}\n\nChoices:\n{opts}\n\nAnswer:"
    msgs = [{"role": "user", "content": prompt}]
    formatted = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"=== Q{i+1} (Correct: {item['answerKey']}) ===")
    print(formatted)
    print("-" * 50)
