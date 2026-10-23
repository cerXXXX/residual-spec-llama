import argparse
import json

from twobit.staged import DEPTH, MODELS, add_rank_arguments, hybrid_paths, load_rank_set
from twobit.trees import accepted_chain, accepted_tree, cross_validated_length, expected_length


def first_level_union(base, mtp, lengths):
    def make_accept(tree):
        def accept(i, s):
            rb, rg = base[i][s], mtp[i][s]
            if not rb:
                return 0
            if (rb[0],) in tree:
                return max(accepted_tree(rb, tree), accepted_chain(rg, 1))
            if rg and rg[0] == 0:
                rest = base[i][s + 1] if s + 1 < lengths[i] else []
                return 1 + accepted_tree(rest, tree)
            return 0
        return accept
    return make_accept


def evaluate(base, mtp, lengths):
    all_idx = list(range(len(lengths)))
    results = {}
    for gamma in (3, 4):
        results[f"MTP_g{gamma}"] = expected_length(lengths, all_idx, lambda i, s, gamma=gamma: accepted_chain(mtp[i][s], gamma))
    for budget in (8, 16, 32):
        results[f"base_N{budget}"] = cross_validated_length(base, lengths, budget, DEPTH, lambda tree: (lambda i, s: accepted_tree(base[i][s], tree)))
        for gamma in (3, 4):
            def union(tree, gamma=gamma):
                return lambda i, s: max(accepted_tree(base[i][s], tree), accepted_chain(mtp[i][s], gamma))
            results[f"base_N{budget}+MTP_g{gamma}"] = cross_validated_length(base, lengths, budget, DEPTH, union)
        results[f"depth1_base|MTP_N{budget}"] = cross_validated_length(base, lengths, budget, DEPTH, first_level_union(base, mtp, lengths))
    return results


def main():
    parser = argparse.ArgumentParser()
    add_rank_arguments(parser)
    parser.add_argument("--out", default="results/e17_mtp_branch.json")
    args = parser.parse_args()
    results = {}
    for model in MODELS:
        band_ranks, mtp_ranks, lengths = load_rank_set(model, args.pool_ranks, args.exact_kv_ranks, args.streaming_state)
        base, mtp = hybrid_paths(band_ranks, mtp_ranks, lengths)
        results[model] = evaluate(base, mtp, lengths)
        print(f"[{model}] " + "  ".join(f"{k} {v:.2f}" for k, v in results[model].items()), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
