import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.exact_kv import apply_base_weights, decoder_linears, load_base, load_texts, pad_right, restore_weights, stream_layers
from twobit.trees import (
    BUDGETS,
    accepted_full_tree,
    accepted_top_then_chain,
    answer_ranks,
    drafter_paths,
    drafter_ranks,
    expected_length,
    on_paths,
    prefix_paths,
    recall_at,
    tree_length,
)

START = time.time()


def log(*parts):
    print(f"[{time.time() - START:6.0f} s]", *parts, flush=True)


def simulate_pool(base_ranks, mtp_ranks, depth):
    lengths = [len(v) for v in base_ranks]
    all_idx = list(range(len(base_ranks)))
    paths_by_source = {"base": prefix_paths(base_ranks, depth), "drafter": drafter_paths(mtp_ranks, lengths)}
    results = {"recall": {}, "full_tree": {}, "top_then_chain": {}, "best_first": {}}
    for source, paths in paths_by_source.items():
        recall = {width: [recall_at(paths, all_idx, width, d) for d in range(depth)] for width in (1, 2, 3, 4, 8, 16)}
        results["recall"][source] = recall
        results["full_tree"][source] = {
            f"P{width}_g{gamma}": expected_length(lengths, all_idx, on_paths(paths, accepted_full_tree, width, gamma))
            for width in (1, 2, 3, 4)
            for gamma in (2, 4, 8)
        }
        results["top_then_chain"][source] = {
            f"P{width}_g8": expected_length(lengths, all_idx, on_paths(paths, accepted_top_then_chain, width, 8))
            for width in (1, 2, 4, 8)
        }
        best_first = {}
        for budget in BUDGETS:
            best_first[f"N{budget}"] = tree_length(paths, lengths, budget, depth)
        results["best_first"][source] = best_first
        recall_text = "/".join(f"{recall[width][0]:.3f}" for width in (1, 2, 4, 8))
        tree_text = "  ".join(f"N{budget}: {value:.2f}" for budget, value in zip(BUDGETS, best_first.values()))
        log(f"[{source}] recall@1/2/4/8 (depth 1): {recall_text} | best-first tree E[L]: {tree_text}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--drafter", default="google/gemma-4-12B-it-assistant")
    parser.add_argument("--codes", default="weights/ldlq_chat_b2/codes.pt")
    parser.add_argument("--texts", default="results/e06c_gemma_eval.pt")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--ranks-out", default="results/e11_candidate_pool.pt")
    parser.add_argument("--out", default="results/e11_candidate_pool.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.ranks_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    depth = args.depth

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").eval()
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=torch.bfloat16, device_map="cuda").eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    lm = model.model.language_model
    embed = model.get_input_embeddings()
    lm.embed_tokens.to("cuda")
    lm.norm.to("cuda")
    lm.rotary_emb.to("cuda")
    stream_layers(lm.layers)
    linears = decoder_linears(model)
    originals = {name: m.weight.detach().clone() for name, m in linears.items()}
    checkpoint, levels = load_base(args.codes)
    texts, prompt_lens, sources = load_texts(args.texts)
    n_texts = len(texts)
    batches = [list(range(i, min(i + args.bs, n_texts))) for i in range(0, n_texts, args.bs)]

    apply_base_weights(linears, checkpoint, levels)
    base_ranks = [None] * n_texts
    with torch.no_grad():
        for idx in batches:
            ids, mask, _ = pad_right(texts, idx)
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for i, ranks in answer_ranks(hidden, idx, texts, prompt_lens, model.lm_head.weight, softcap).items():
                base_ranks[i] = ranks
            del hidden
            gc.collect()
            torch.cuda.empty_cache()
    total = sum(len(v) for v in base_ranks)
    recalls = tuple(sum(sum(r < width for r in v) for v in base_ranks) / total for width in (1, 4, 16))
    log("1) 2-bit base ranks: recall@1 %.3f, @4 %.3f, @16 %.3f" % recalls)

    restore_weights(linears, originals)
    mtp_ranks = [None] * n_texts
    with torch.no_grad():
        for idx in batches:
            ids, mask, _ = pad_right(texts, idx)
            out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_shared_kv_states=True)
            hidden, shared_kv = out.last_hidden_state, out.shared_kv_states
            del out
            for row, i in enumerate(idx):
                mtp_ranks[i] = drafter_ranks(drafter, embed, texts[i], prompt_lens[i], hidden, shared_kv, row, depth)
            del hidden, shared_kv
            gc.collect()
            torch.cuda.empty_cache()
            log(f"2) drafter: texts {idx[0]}-{idx[-1]} done")
    torch.save({"base_ranks": base_ranks, "drafter_ranks": mtp_ranks, "sources": sources}, args.ranks_out)

    results = simulate_pool(base_ranks, mtp_ranks, depth)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
