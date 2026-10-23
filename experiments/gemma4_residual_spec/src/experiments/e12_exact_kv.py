import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.exact_kv import (
    ExactKVAttention,
    apply_base_weights,
    decoder_linears,
    forward_texts,
    load_base,
    load_texts,
    restore_weights,
    stream_layers,
)
from twobit.trees import answer_ranks, compare_sources

START = time.time()


def log(*parts):
    print(f"[{time.time() - START:6.0f} s]", *parts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--codes", default="weights/ldlq_chat_b2/codes.pt")
    parser.add_argument("--texts", default="results/e06c_gemma_eval.pt")
    parser.add_argument("--pool-ranks", default="results/e11_candidate_pool.pt")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--bands", default="1,2,4,8")
    parser.add_argument("--ranks-out", default="results/e12_exact_kv.pt")
    parser.add_argument("--out", default="results/e12_exact_kv.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.ranks_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bands = [int(x) for x in args.bands.split(",")]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    lm = model.model.language_model
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
    attention = ExactKVAttention().install()

    def hidden(idx, mode, band=1, batch=0):
        return forward_texts(model, attention, texts, idx, mode, band, batch).last_hidden_state

    def ranks(h, idx):
        return answer_ranks(h, idx, texts, prompt_lens, model.lm_head.weight, softcap)

    restore_weights(linears, originals)
    h_original = hidden(batches[0][:2], "original")
    h_custom = hidden(batches[0][:2], "custom")
    first_len = texts[batches[0][0]].numel()
    max_diff = float((h_original[0, :first_len].float() - h_custom[0, :first_len].float()).abs().max())
    results = {"check_max_state_diff": max_diff}
    scale = float(h_original[0, :first_len].float().abs().max())
    log(f"custom attention vs sdpa check: max |diff| = {max_diff:.4f} (state scale ~{scale:.1f})")
    del h_original, h_custom
    for bi, idx in enumerate(batches):
        hidden(idx, "capture", batch=bi)
        gc.collect()
        torch.cuda.empty_cache()
    log(f"phase A: K/V captured ({len(attention.store)} entries)")
    apply_base_weights(linears, checkpoint, levels)
    band_ranks = {d: [None] * n_texts for d in bands}
    for bi, idx in enumerate(batches):
        for d in bands:
            for i, v in ranks(hidden(idx, "band", band=d, batch=bi), idx).items():
                band_ranks[d][i] = v
            gc.collect()
            torch.cuda.empty_cache()
        log(f"phase B: batch {bi + 1}/{len(batches)}")
    pool = torch.load(args.pool_ranks, weights_only=False)
    torch.save({"band_ranks": band_ranks, "sources": sources}, args.ranks_out)

    results["comparison"] = compare_sources(pool["base_ranks"], band_ranks, pool["drafter_ranks"], log)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
