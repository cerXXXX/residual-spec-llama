import math

import torch

from twobit.trees import accepted_chain, band_paths, drafter_paths

DEPTH = 8
MODELS = ("12B", "31B")


def load_rank_set(model, pool_path, exact_kv_path, streaming_path):
    if model == "31B":
        state = torch.load(streaming_path, weights_only=False)
        return state["band_ranks"], state["drafter_ranks"], [len(v) for v in state["base_ranks"]]
    exact = torch.load(exact_kv_path, weights_only=False)
    pool = torch.load(pool_path, weights_only=False)
    return exact["band_ranks"], pool["drafter_ranks"], [len(v) for v in pool["base_ranks"]]


def hybrid_paths(band_ranks, drafter_rank_lists, lengths, depth=DEPTH):
    return band_paths(band_ranks, lengths, depth), drafter_paths(drafter_rank_lists, lengths)


def add_rank_arguments(parser):
    parser.add_argument("--pool-ranks", default="results/e11_candidate_pool.pt")
    parser.add_argument("--exact-kv-ranks", default="results/e12_exact_kv.pt")
    parser.add_argument("--streaming-state", default="results/e13_gemma31_streaming.pt")


def run_policy(base, mtp, lengths, policy):
    tokens = 0
    rounds = 0
    passes = 0
    steps = 0
    for i in range(len(lengths)):
        s = 0
        while s < lengths[i]:
            k, used_passes, used_steps = policy(base[i][s], mtp[i][s])
            n = min(k + 1, lengths[i] - s)
            tokens += n
            rounds += 1
            passes += used_passes
            steps += used_steps
            s += n
    return tokens / rounds, passes / rounds, steps / rounds


def mtp_chain_policy(limit):
    def policy(base, mtp):
        return accepted_chain(mtp, limit), 0, limit
    return policy


def base_chain_policy(base, mtp):
    return accepted_chain(base), DEPTH, 0


def mtp_led_policy(limit, base_width):
    def policy(base, mtp):
        n = min(len(base), len(mtp))
        j = accepted_chain(mtp, min(limit, n))
        if j < n and base[j] < base_width:
            j += 1
        return j, 1, limit
    return policy


def base_led_policy(max_passes, base_width, alt_depth=math.inf, branch_len=math.inf, chain_len=math.inf):
    def policy(base, mtp):
        n = min(len(base), len(mtp), chain_len)
        j = 0
        passes = 1
        steps = n
        while j < n:
            if base[j] == 0 and mtp[j] == 0:
                j += 1
                continue
            if base[j] == 0:
                if passes < max_passes:
                    passes += 1
                    steps += n - j - 1
                    j += 1
                    continue
                j += 1
                steps += n - j
                while j < n and mtp[j] == 0:
                    j += 1
                return j, passes, steps
            if mtp[j] == 0:
                end = min(n, j + branch_len)
                j += 1
                while j < end and mtp[j] == 0:
                    j += 1
                if j < n and j < end + 1 and base[j] < base_width and j < alt_depth:
                    j += 1
                return j, passes, steps
            return j + (1 if (base[j] < base_width and j < alt_depth) else 0), passes, steps
        return j, passes, steps
    return policy
