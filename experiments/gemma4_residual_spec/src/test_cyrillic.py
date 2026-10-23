import os
import sys
from transformers import AutoProcessor

p = AutoProcessor.from_pretrained(r"D:\localllamacpp\models\gemma4-12b\bf16")
tok = p.tokenizer

phrase = "Дело в шляпе"
tokens = tok.encode(phrase, add_special_tokens=False)
print("Phrase:", phrase)
print("Tokens:", tokens)
for t in tokens:
    s = tok.decode([t])
    print(f"  ID {t:6d}: {s!r} (bytes: {list(s.encode('utf-8'))})")

# Check what happens with single bytes vs multi-byte UTF-8 in SentencePiece/BPE
test_single_byte = 237  # byte fallback token
print(f"\nByte fallback test: ID {test_single_byte}: {tok.decode([test_single_byte])!r}")
