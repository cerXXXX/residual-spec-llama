import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from twobit.exact_kv import ExactKVAttention, apply_base_weights, decoder_linears, forward_texts, load_base, stream_layers
from twobit.trees import answer_ranks, compare_sources, drafter_ranks

START = time.time()


def log(*parts):
    print(f"[{time.time() - START:6.0f} s]", *parts, flush=True)


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def place_non_layers_on_gpu(model):
    lm = model.model.language_model
    for _, child in model.model.named_children():
        if child is lm:
            for name, sub in lm.named_children():
                if name != "layers":
                    sub.to("cuda")
        else:
            child.to("cuda")
    model.lm_head.to("cuda")


@torch.no_grad()
def generate(model, prompts, idx, max_new_tokens, eos):
    prompt_ids = [prompts[i][1] for i in idx]
    n = max(p.numel() for p in prompt_ids)
    ids = torch.zeros(len(idx), n, dtype=torch.long)
    mask = torch.zeros(len(idx), n, dtype=torch.long)
    for row, p in enumerate(prompt_ids):
        ids[row, n - p.numel():] = p
        mask[row, n - p.numel():] = 1
    out = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=max_new_tokens, do_sample=False, top_k=None, top_p=None, pad_token_id=0).cpu()
    texts = []
    for row in range(len(idx)):
        generated = out[row, n:].tolist()
        cut = next((j + 1 for j, token in enumerate(generated) if token in eos), len(generated))
        texts.append(torch.cat([prompt_ids[row], torch.tensor(generated[:cut], dtype=torch.long)]))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--drafter", default="google/gemma-4-31B-it-assistant")
    parser.add_argument("--codes", default="weights/gemma31_ldlq_chat_b2/codes.pt")
    parser.add_argument("--prompts", default="data/prompts.pt")
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--gbs", type=int, default=16)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--bands", default="1,2,4,8")
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument("--state-out", default="results/e13_gemma31_streaming.pt")
    parser.add_argument("--out", default="results/e13_gemma31_streaming.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.state_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bands = [int(x) for x in args.bands.split(",")]
    depth = max(bands)
    state = {}

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu").eval()
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=torch.bfloat16, device_map="cuda").eval()
    softcap = model.config.get_text_config().final_logit_softcapping
    lm = model.model.language_model
    embed = model.get_input_embeddings()
    place_non_layers_on_gpu(model)
    eos = eos_ids(model)
    stream_layers(lm.layers)
    linears = decoder_linears(model)
    prompts = torch.load(args.prompts)[: args.n_prompts]
    prompt_lens = [p.numel() for _, p in prompts]
    sources = [s for s, _ in prompts]
    n_texts = len(prompts)
    log(f"31B: {len(lm.layers)} layers, {len(linears)} linear; EOS {sorted(eos)}; {n_texts} prompts")
    attention = ExactKVAttention().install()

    texts = []
    for start in range(0, n_texts, args.gbs):
        texts += generate(model, prompts, list(range(start, min(start + args.gbs, n_texts))), args.gen, eos)
        gc.collect()
        torch.cuda.empty_cache()
        log(f"0) generated {len(texts)}/{n_texts}")
    alone = generate(model, prompts, [0], args.gen, eos)[0]
    k16 = min(16, alone.numel() - prompt_lens[0], texts[0].numel() - prompt_lens[0])
    same = bool(torch.equal(alone[prompt_lens[0]:prompt_lens[0] + k16], texts[0][prompt_lens[0]:prompt_lens[0] + k16]))
    state.update(texts=texts, prompt_lens=prompt_lens, sources=sources, alone_matches_batch=same)
    torch.save(state, args.state_out)
    answer_positions = sum(t.numel() - p for t, p in zip(texts, prompt_lens))
    log(f"0) texts saved; answer positions {answer_positions}; prompt 0 alone = in batch (16 tokens): {same}")
    batches = [list(range(i, min(i + args.bs, n_texts))) for i in range(0, n_texts, args.bs)]

    def ranks(h, idx):
        return answer_ranks(h, idx, texts, prompt_lens, model.lm_head.weight, softcap)

    mtp_ranks = [None] * n_texts
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            out = forward_texts(model, attention, texts, idx, "capture", batch=bi, return_shared_kv_states=True)
            hidden, shared_kv = out.last_hidden_state, out.shared_kv_states
            del out
            for row, i in enumerate(idx):
                mtp_ranks[i] = drafter_ranks(drafter, embed, texts[i], prompt_lens[i], hidden, shared_kv, row, depth)
            del hidden, shared_kv
            gc.collect()
            torch.cuda.empty_cache()
            log(f"A) batch {bi + 1}/{len(batches)}")
    state["drafter_ranks"] = mtp_ranks
    torch.save(state, args.state_out)

    checkpoint, levels = load_base(args.codes)
    apply_base_weights(linears, checkpoint, levels)
    del checkpoint
    gc.collect()
    log("B) 2-bit base weights applied")
    base_ranks = [None] * n_texts
    band_ranks = {d: [None] * n_texts for d in bands}
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            for i, v in ranks(forward_texts(model, attention, texts, idx, "custom", batch=bi).last_hidden_state, idx).items():
                base_ranks[i] = v
            for d in bands:
                for i, v in ranks(forward_texts(model, attention, texts, idx, "band", band=d, batch=bi).last_hidden_state, idx).items():
                    band_ranks[d][i] = v
            gc.collect()
            torch.cuda.empty_cache()
            log(f"B) batch {bi + 1}/{len(batches)}")
    state.update(base_ranks=base_ranks, band_ranks=band_ranks)
    torch.save(state, args.state_out)

    results = {"alone_matches_batch": state["alone_matches_batch"], "positions": sum(len(v) for v in base_ranks)}
    results["comparison"] = compare_sources(base_ranks, band_ranks, mtp_ranks, log, with_depth4=False)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
