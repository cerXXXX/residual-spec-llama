import argparse
import json

from twobit.staged import (
    MODELS,
    add_rank_arguments,
    base_chain_policy,
    base_led_policy,
    hybrid_paths,
    load_rank_set,
    mtp_chain_policy,
    mtp_led_policy,
    run_policy,
)


def summarize(base, mtp, lengths, policy):
    expected, passes, steps = run_policy(base, mtp, lengths, policy)
    return {"EL": expected, "base_passes": passes, "MTP_steps": steps}


def evaluate(base, mtp, lengths):
    results = {
        "MTP_g3": summarize(base, mtp, lengths, mtp_chain_policy(3)),
        "MTP_g4": summarize(base, mtp, lengths, mtp_chain_policy(4)),
        "MTP_g8": summarize(base, mtp, lengths, mtp_chain_policy(8)),
        "base_chain_D8": summarize(base, mtp, lengths, base_chain_policy),
    }
    for gamma in (3, 4, 6, 8):
        for width in (1, 2, 4):
            results[f"H1_g{gamma}_k{width}"] = summarize(base, mtp, lengths, mtp_led_policy(gamma, width))
    for max_passes in (1, 2, 3, 8):
        for width in (1, 2, 4):
            results[f"H2_P{max_passes}_k{width}"] = summarize(base, mtp, lengths, base_led_policy(max_passes, width))
    return results


def main():
    parser = argparse.ArgumentParser()
    add_rank_arguments(parser)
    parser.add_argument("--out", default="results/e17b_three_level.json")
    args = parser.parse_args()
    results = {}
    for model in MODELS:
        band_ranks, mtp_ranks, lengths = load_rank_set(model, args.pool_ranks, args.exact_kv_ranks, args.streaming_state)
        base, mtp = hybrid_paths(band_ranks, mtp_ranks, lengths)
        results[model] = evaluate(base, mtp, lengths)
        print(f"== {model}")
        for key, value in results[model].items():
            print(f"  {key:16s} E[L] {value['EL']:.2f}   base passes {value['base_passes']:.2f}   MTP steps {value['MTP_steps']:.1f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
