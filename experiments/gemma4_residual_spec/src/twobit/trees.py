import heapq

import torch

BUDGETS = (4, 8, 16, 32, 64)


@torch.no_grad()
def target_ranks(hidden, lm_head_weight, softcap, targets, chunk=65536):
    x = hidden.float()
    logits = torch.cat([x @ lm_head_weight[j:j + chunk].float().T for j in range(0, lm_head_weight.shape[0], chunk)], -1)
    logits = torch.tanh(logits / softcap) * softcap
    y = targets.to(logits.device)
    return (logits > logits.gather(-1, y[:, None])).sum(-1).cpu().tolist()


@torch.no_grad()
def answer_ranks(hidden_states, idx, texts, prompt_lens, lm_head_weight, softcap):
    ranks = {}
    for row, i in enumerate(idx):
        n = texts[i].numel()
        ranks[i] = target_ranks(hidden_states[row, prompt_lens[i] - 1:n - 1], lm_head_weight, softcap, texts[i][prompt_lens[i]:])
    return ranks


@torch.no_grad()
def drafter_ranks(drafter, embed, text, prompt_len, hidden, shared_kv, row, depth):
    device = hidden.device
    tokens = text.to(device)
    n = tokens.numel()
    result = []
    for s in range(prompt_len - 1, n - 1):
        kv = {kind: (v[0][row:row + 1, :, :s], v[1][row:row + 1, :, :s]) for kind, v in shared_kv.items()}
        h = hidden[row:row + 1, s - 1:s]
        position = torch.tensor([[s]], device=device)
        ranks = []
        for j in range(depth):
            if s + 1 + j >= n:
                break
            inputs = torch.cat([embed(tokens[s + j:s + j + 1].view(1, 1)), h], -1)
            out = drafter(inputs_embeds=inputs, position_ids=position, shared_kv_states=kv, attention_mask=None, use_cache=False)
            logits = out.logits[0, -1].float()
            ranks.append(int((logits > logits[tokens[s + 1 + j]]).sum()))
            h = out.last_hidden_state
        result.append(ranks)
    return result


def prefix_paths(ranks, depth):
    return [[r[s:s + depth] for s in range(len(r))] for r in ranks]


def band_paths(band_ranks, lengths, depth):
    bands = sorted(band_ranks)
    paths = []
    for i, n in enumerate(lengths):
        text_paths = []
        for s in range(n):
            text_paths.append([band_ranks[min(b for b in bands if b >= j + 1)][i][s + j] for j in range(min(depth, n - s))])
        paths.append(text_paths)
    return paths


def drafter_paths(ranks, lengths):
    return [[ranks[i][s] for s in range(n)] for i, n in enumerate(lengths)]


def accepted_chain(ranks, limit=None):
    n = len(ranks) if limit is None else min(limit, len(ranks))
    k = 0
    while k < n and ranks[k] == 0:
        k += 1
    return k


def accepted_full_tree(ranks, width, depth):
    k = 0
    while k < min(depth, len(ranks)) and ranks[k] < width:
        k += 1
    return k


def accepted_top_then_chain(ranks, width, depth):
    if not ranks or ranks[0] >= width:
        return 0
    k = 1
    while k < min(depth, len(ranks)) and ranks[k] == 0:
        k += 1
    return k


def accepted_tree(ranks, tree):
    k = 0
    while k < len(ranks) and tuple(ranks[:k + 1]) in tree:
        k += 1
    return k


def on_paths(paths, rule, *params):
    return lambda i, s: rule(paths[i][s], *params)


def expected_length(lengths, idx, accept):
    tokens = 0
    rounds = 0
    for i in idx:
        s = 0
        while s < lengths[i]:
            step = min(accept(i, s) + 1, lengths[i] - s)
            tokens += step
            rounds += 1
            s += step
    return tokens / rounds


def rank_probabilities(paths, idx, depth, max_rank=16):
    counts = [[0] * max_rank for _ in range(depth)]
    totals = [0] * depth
    for i in idx:
        for ranks in paths[i]:
            for d in range(len(ranks)):
                totals[d] += 1
                if ranks[d] < max_rank:
                    counts[d][ranks[d]] += 1
    return [[counts[d][r] / max(1, totals[d]) for r in range(max_rank)] for d in range(depth)]


def best_first_tree(paths, idx, budget, depth, max_rank=16):
    probs = rank_probabilities(paths, idx, depth, max_rank)
    tree = set()
    heap = [(-probs[0][r], (r,)) for r in range(max_rank)]
    heapq.heapify(heap)
    while heap and len(tree) < budget:
        neg_prob, path = heapq.heappop(heap)
        tree.add(path)
        if len(path) < depth:
            for r in range(max_rank):
                heapq.heappush(heap, (neg_prob * probs[len(path)][r], path + (r,)))
    return tree


def cross_validated_length(tree_paths, lengths, budget, depth, make_accept, max_rank=16):
    idx = list(range(len(lengths)))
    even, odd = idx[0::2], idx[1::2]
    tree_even = best_first_tree(tree_paths, even, budget, depth, max_rank)
    tree_odd = best_first_tree(tree_paths, odd, budget, depth, max_rank)
    total_odd = sum(lengths[i] for i in odd)
    total_even = sum(lengths[i] for i in even)
    total = sum(lengths[i] for i in idx)
    return (expected_length(lengths, odd, make_accept(tree_even)) * total_odd + expected_length(lengths, even, make_accept(tree_odd)) * total_even) / total


def tree_length(paths, lengths, budget, depth):
    return cross_validated_length(paths, lengths, budget, depth, lambda tree: on_paths(paths, accepted_tree, tree))


def recall_at(paths, idx, width, depth_index):
    hits = 0
    count = 0
    for i in idx:
        for ranks in paths[i]:
            if len(ranks) > depth_index:
                count += 1
                hits += ranks[depth_index] < width
    return hits / max(1, count)


def source_paths(base_ranks, band_ranks, drafter_rank_lists):
    depth = max(band_ranks)
    lengths = [len(v) for v in base_ranks]
    paths = {
        "base": prefix_paths(base_ranks, depth),
        "base_exact_kv": band_paths(band_ranks, lengths, depth),
        "drafter": drafter_paths(drafter_rank_lists, lengths),
    }
    return paths, lengths, depth


def source_summary(paths, lengths, depth, with_depth4=True, budgets=BUDGETS):
    idx = list(range(len(lengths)))
    summary = {"recall@1/2/4/8_depth1": [recall_at(paths, idx, width, 0) for width in (1, 2, 4, 8)]}
    if with_depth4:
        summary["recall@1_depth4"] = recall_at(paths, idx, 1, 3)
    summary["chain_g8"] = expected_length(lengths, idx, on_paths(paths, accepted_chain))
    for budget in budgets:
        summary[f"tree_N{budget}"] = tree_length(paths, lengths, budget, depth)
    return summary


def summary_line(name, summary, budgets=BUDGETS):
    line = f"[{name:17s}] recall depth1 @1/2/4/8: " + "/".join(f"{x:.3f}" for x in summary["recall@1/2/4/8_depth1"])
    if "recall@1_depth4" in summary:
        line += f" | @1 depth4 {summary['recall@1_depth4']:.3f}"
    line += f" | chain {summary['chain_g8']:.2f} | "
    return line + "  ".join(f"N{budget} {summary[f'tree_N{budget}']:.2f}" for budget in budgets)


def compare_sources(base_ranks, band_ranks, drafter_rank_lists, log, with_depth4=True):
    paths, lengths, depth = source_paths(base_ranks, band_ranks, drafter_rank_lists)
    comparison = {}
    for name, source in paths.items():
        comparison[name] = source_summary(source, lengths, depth, with_depth4)
        log(summary_line(name, comparison[name]))
    return comparison
