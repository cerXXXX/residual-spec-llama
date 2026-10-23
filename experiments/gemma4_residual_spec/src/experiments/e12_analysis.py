import argparse
import json
import os
import time

import torch

from twobit.trees import compare_sources

START = time.time()


def log(*parts):
    print(f"[{time.time() - START:6.0f} s]", *parts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-kv-ranks", default="results/e12_exact_kv.pt")
    parser.add_argument("--pool-ranks", default="results/e11_candidate_pool.pt")
    parser.add_argument("--out", default="results/e12_exact_kv.json")
    args = parser.parse_args()

    exact = torch.load(args.exact_kv_ranks, weights_only=False)
    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    pool = torch.load(args.pool_ranks, weights_only=False)
    results["comparison"] = compare_sources(pool["base_ranks"], exact["band_ranks"], pool["drafter_ranks"], log)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
