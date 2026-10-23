import random

from twobit.hadamard import SEED

OTHER_LANGS = ("de", "es", "fr", "ru", "it")


def chat_ids(processor, messages):
    turns = [{"role": "user" if m["role"] in ("user", "prompter") else "assistant", "content": m["content"]} for m in messages]
    x = processor.apply_chat_template(turns, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False, enable_thinking=False)
    return x["input_ids"][0].tolist()


def pack_chunks(processor, conversations, n_chunks, seq):
    buf, out = [], []
    for conv in conversations:
        buf += chat_ids(processor, conv)
        while len(buf) >= seq and len(out) < n_chunks:
            out.append(buf[:seq])
            buf = buf[seq:]
        if len(out) >= n_chunks:
            break
    return out


def oasst_pairs(rows):
    by_id = {r["message_id"]: r for r in rows}
    pairs = {}
    for r in rows:
        if r["role"] == "assistant" and r.get("rank") == 0 and r["parent_id"] in by_id and by_id[r["parent_id"]]["role"] == "prompter":
            pair = [{"role": "user", "content": by_id[r["parent_id"]]["text"]}, {"role": "assistant", "content": r["text"]}]
            pairs.setdefault(r["lang"], []).append(pair)
    return pairs


def build_chat_calibration(processor, ultrachat, oasst, seq=2048, n_en=14, lang="pl", n_lang=12,
                           other_langs=OTHER_LANGS, other_per_lang=400, n_other=6, seed=SEED):
    rng = random.Random(seed)
    order = list(range(len(ultrachat)))
    rng.shuffle(order)
    en = pack_chunks(processor, (ultrachat[i]["messages"] for i in order), n_en, seq)
    pairs = oasst_pairs(oasst)
    for v in pairs.values():
        rng.shuffle(v)
    primary = pack_chunks(processor, pairs.get(lang, []), n_lang, seq)
    other_src = [c for code in other_langs for c in pairs.get(code, [])[:other_per_lang]]
    rng.shuffle(other_src)
    other = pack_chunks(processor, other_src, n_other, seq)
    chunks = en + primary + other
    stats = {"en": len(en), lang: len(primary), "other": len(other), f"{lang}_pairs_available": len(pairs.get(lang, [])),
             "chunks": len(chunks), "seq": seq}
    if len(primary) < n_lang:
        stats["note"] = f"{lang} only {len(primary)} chunks"
    return chunks, stats, primary
