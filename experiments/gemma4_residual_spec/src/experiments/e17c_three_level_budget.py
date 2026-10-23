import argparse
import json

from twobit.staged import MODELS, add_rank_arguments, base_led_policy, hybrid_paths, load_rank_set, run_policy

CONFIGS = [
    (2, 2, 8, 8, 8),
    (2, 2, 4, 2, 8),
    (2, 2, 4, 3, 8),
    (2, 3, 2, 2, 8),
    (2, 2, 6, 1, 8),
    (3, 2, 4, 2, 8),
    (3, 2, 2, 2, 8),
    (2, 2, 3, 2, 7),
    (2, 1, 0, 4, 8),
    (3, 1, 0, 3, 8),
]


def evaluate(base, mtp, lengths):
    results = {}
    for max_passes, width, alt_depth, branch_len, chain_len in CONFIGS:
        policy = base_led_policy(max_passes, width, alt_depth, branch_len, chain_len)
        expected, passes, _ = run_policy(base, mtp, lengths, policy)
        nodes = chain_len + (width - 1) * alt_depth + branch_len * max_passes
        key = f"H2_P{max_passes}_k{width}_A{alt_depth}_C{branch_len}_D{chain_len}"
        results[key] = {"EL": expected, "base_passes": passes, "max_nodes": nodes}
    return results


def main():
    parser = argparse.ArgumentParser()
    add_rank_arguments(parser)
    parser.add_argument("--out", default="results/e17c_three_level_budget.json")
    args = parser.parse_args()
    results = {}
    for model in MODELS:
        band_ranks, mtp_ranks, lengths = load_rank_set(model, args.pool_ranks, args.exact_kv_ranks, args.streaming_state)
        base, mtp = hybrid_paths(band_ranks, mtp_ranks, lengths)
        results[model] = evaluate(base, mtp, lengths)
        print(f"== {model}")
        for key, value in results[model].items():
            print(f"  {key:22s} nodes <= {value['max_nodes']:2d}   E[L] {value['EL']:.2f}   base passes {value['base_passes']:.2f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
