import os
import sys
import time

# Strictly ensure HF caches and temp files stay on D: drive, never C:
os.environ["HF_HOME"] = r"D:\localllamacpp\.cache\huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

def download():
    target_dir = r"D:\localllamacpp\models\gemma4-12b\bf16"
    print(f"Starting download of google/gemma-4-12B-it into {target_dir}...")
    t0 = time.time()
    snapshot_download(
        repo_id="google/gemma-4-12B-it",
        local_dir=target_dir,
        ignore_patterns=[".git*", "*.msgpack"],
        max_workers=4,
    )
    elapsed = time.time() - t0
    print(f"Download complete in {elapsed:.1f}s.")

if __name__ == "__main__":
    download()
