import argparse
import itertools
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.generation import candidate_generator

from experiments.e06d_draft_policies import POLICIES, bootstrap_ci, simulate, tokens_per_round
from experiments.e10_quantize_layerwise import Stop, fit_base_levels, gram_hook, make_logger, pop_rotated_hessian
from twobit.base import GROUP
from twobit.hadamard import SEED, layer_rot
from twobit.residual import symmetric_quant

SEQ = 2048
GAMMAS = (1, 2, 3, 4, 8)
LOGIT_CHUNK = 65536


@torch.no_grad()
def fp32_logits(h, weight, softcap):
    hf = h.float()
    z = torch.cat([hf @ weight[i:i + LOGIT_CHUNK].float().T for i in range(0, weight.shape[0], LOGIT_CHUNK)], -1)
    return torch.tanh(z / softcap) * softcap if softcap else z


def ulp_perturb(Wb, frac, gen):
    bits = Wb.view(torch.int16)
    r = torch.rand(Wb.shape, generator=gen, device=Wb.device)
    sgn = torch.where(torch.rand(Wb.shape, generator=gen, device=Wb.device) < 0.5, -1, 1).to(torch.int16)
    return (bits + torch.where((r < frac) & (Wb != 0), sgn, torch.zeros_like(sgn))).view(torch.bfloat16)


def paired_bootstrap(pc_x, pc_y, k, rng, n_boot=4000):
    n = len(pc_x)

    def mean(pc, sel):
        return sum(pc[i][k] for i in sel) / sum(pc[i][3] for i in sel)

    d = []
    for _ in range(n_boot):
        sel = [rng.randrange(n) for _ in range(n)]
        d.append(mean(pc_x, sel) - mean(pc_y, sel))
    d.sort()
    cut = n_boot // 40
    return {"diff": mean(pc_x, range(n)) - mean(pc_y, range(n)), "CI95": [d[cut], d[n_boot - cut - 1]]}


def aggregate(pc):
    n = sum(x[3] for x in pc)
    return {"ppl": math.exp(sum(x[0] for x in pc) / n), "kl_bf16": sum(x[1] for x in pc) / n, "flips": sum(x[2] for x in pc) / n}


@torch.no_grad()
def drafter_chain(drafter, embed, H, KV, t, s, depth, incl, device, stop=True):
    kv = {k: (v[0][:, :, :s + incl], v[1][:, :, :s + incl]) for k, v in KV.items()}
    h = H[:, s - 1:s]
    tok = t[s:s + 1].view(1, 1).to(device)
    pos = torch.tensor([[s]], device=device)
    out_tokens, first_logits = [], None
    for j in range(depth):
        o = drafter(inputs_embeds=torch.cat([embed(tok), h], -1), position_ids=pos, shared_kv_states=kv, attention_mask=None, use_cache=False)
        if j == 0:
            first_logits = o.logits[0, -1].float()
        tok = o.logits[:, -1:].argmax(-1)
        h = o.last_hidden_state
        out_tokens.append(int(tok))
        if stop and (s + 1 + j >= t.numel() or out_tokens[-1] != int(t[s + 1 + j])):
            break
    return out_tokens, first_logits


@torch.no_grad()
def gptq_quantize(Wt, H, levels, damp, group=GROUP):
    thr = (levels[1:] + levels[:-1]) / 2
    W = Wt.clone()
    d_in = W.shape[1]
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(d_in, device=H.device)
    L = torch.linalg.cholesky(H)
    del H
    Hi = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Hi, upper=True)
    del Hi
    Q = torch.zeros_like(W)
    for i1 in range(0, d_in, group):
        i2 = min(i1 + group, d_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        sc = W1.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        for i in range(i2 - i1):
            w = W1[:, i:i + 1]
            q = levels[torch.bucketize(w / sc, thr)] * sc
            Q1[:, i:i + 1] = q
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err @ Hb[i:i + 1, i:]
            E1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        if i2 < d_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def patch_assistant_init(generator_cls):
    original_init = generator_cls.__init__

    def init(self, input_ids=None, assistant_model=None, target_model_input_embeddings=None, generation_config=None, model_kwargs=None,
             *args, **kwargs):
        original_init(self, input_ids, assistant_model, target_model_input_embeddings, generation_config,
                      model_kwargs if model_kwargs is not None else {}, *args, **kwargs)

    generator_cls.__init__ = init


def patch_recorder(generator_cls, record):
    get_candidates = generator_cls.get_candidates
    update_strategy = generator_cls.update_candidate_strategy

    def recording_get(self, input_ids, *args, **kwargs):
        cand, logits = get_candidates(self, input_ids, *args, **kwargs)
        if cand.shape[1] > input_ids.shape[1]:
            record["proposals"].append((record["prompt"], int(input_ids.shape[1]), cand[0, input_ids.shape[1]:].tolist()))
        return cand, logits

    def recording_update(self, input_ids, scores, num_matches, *args, **kwargs):
        record["rounds"].append(int(num_matches))
        return update_strategy(self, input_ids, scores, num_matches, *args, **kwargs)

    generator_cls.get_candidates = recording_get
    generator_cls.update_candidate_strategy = recording_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--drafter", default="google/gemma-4-12B-it-assistant")
    ap.add_argument("--prompts", default="data/prompts.pt")
    ap.add_argument("--c4", default="data/gemma_c4_calib.pt")
    ap.add_argument("--wikitext", default="data/gemma_wikitext2_validation.pt")
    ap.add_argument("--base-weights", default="weights/ldlq_c4_b2/base.safetensors")
    ap.add_argument("--out", default="results/e06_gemma.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--calib-chunks", type=int, default=32)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--hf-gammas", default="4,8")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--skip-precision", action="store_true")
    ap.add_argument("--skip-acceptance", action="store_true")
    a = ap.parse_args()
    out_path = a.out.replace(".json", "_dry.json") if a.dry else a.out
    dev = "cpu" if a.dry else a.device
    if a.dry:
        a.n_prompts, a.gen, a.hf_gammas, a.depth = 1, 8, "4", 4
    log = make_logger(time.time())
    res = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev).eval()
    tcfg = model.config.get_text_config()
    softcap = tcfg.final_logit_softcapping
    bos = tcfg.bos_token_id
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model.layers." in n}

    def layer_of(n):
        return int(n.split(".layers.")[1].split(".")[0])

    n_layers = max(layer_of(n) for n in lins) + 1
    lm = model.model.language_model
    emb = model.get_input_embeddings()
    orig = {n: m.weight.detach().to("cpu", copy=True) for n, m in lins.items()}
    prompts = torch.load(a.prompts)[: a.n_prompts]
    log(f"model {type(model).__name__}: {len(lins)} linear layers in {n_layers} blocks; softcap {softcap}; bos {bos}; example {list(lins)[:2]}")
    res["model"] = {"class": type(model).__name__, "linear": len(lins), "blocks": n_layers, "softcap": softcap}

    def logits32(h):
        return fp32_logits(h, model.lm_head.weight, softcap)

    base = None

    @torch.no_grad()
    def set_weights(kind, bits=None, g=None, frac=None):
        gen = torch.Generator(device=dev)
        gen.manual_seed(SEED)
        changed = total = 0
        for n, m in lins.items():
            if kind == "ulp":
                m.weight.data.copy_(ulp_perturb(orig[n].to(dev), frac, gen))
                continue
            W = orig[n].to(dev, torch.float32)
            if kind == "bf16":
                Wn = W
            elif kind == "int8":
                Wn = symmetric_quant(W, 8, g)
            else:
                Wh = base[n].to(dev, torch.float32)
                Wn = Wh if kind == "draft" else Wh + symmetric_quant(W - Wh, bits, GROUP)
            Wb = Wn.to(torch.bfloat16)
            changed += int((Wb != orig[n].to(dev)).sum())
            total += Wb.numel()
            m.weight.data.copy_(Wb)
        return changed / max(1, total)

    with torch.no_grad():
        x = prompts[0][1][None].to(dev)
        out = model(input_ids=x, output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
        lhs = model.model(input_ids=x).last_hidden_state
        kv_shapes = {k: (tuple(v[0].shape), tuple(v[1].shape)) for k, v in out.shared_kv_states.items()}
        d_h = float((out.hidden_states[-1].float() - lhs.float()).abs().max())
        d_lg = float((out.logits[0, -1].float() - logits32(lhs[0, -1:])[0]).abs().max())
        s = x.shape[1] - 1
        kv = {k: (v[0][:, :, :s], v[1][:, :, :s]) for k, v in out.shared_kv_states.items()}
        drafter = AutoModelForCausalLM.from_pretrained(a.drafter, dtype=torch.bfloat16, device_map=dev).eval()
        o = drafter(inputs_embeds=torch.cat([emb(x[:, s:s + 1]), out.hidden_states[-1][:, s - 1:s]], -1), position_ids=torch.tensor([[s]], device=dev),
                    shared_kv_states=kv, attention_mask=None, use_cache=False)
    res["api"] = {"shared_kv": kv_shapes, "max_d_hidden_vs_last": d_h, "max_d_logits_model_vs_fp32": d_lg, "drafter_logits": list(o.logits.shape),
                  "drafter_hidden": list(o.last_hidden_state.shape), "drafter_class": type(drafter).__name__}
    log("API:", res["api"])
    del out, lhs, o
    save()

    if not a.dry:
        if os.path.exists(a.base_weights):
            base = load_file(a.base_weights)
            log("base B loaded from disk")
        else:
            drafter.to("cpu")
            torch.cuda.empty_cache()
            c4 = torch.load(a.c4)

            def chunk(src, c):
                return torch.cat([torch.tensor([bos]), src[c * (SEQ - 1):(c + 1) * (SEQ - 1)]])

            n0 = next(iter(lins))
            levels = fit_base_levels(n0, orig[n0].to(dev, torch.float32), 2)
            log("Lloyd-Max levels:", [round(x, 4) for x in levels.tolist()])

            def raise_stop(m, i, o):
                raise Stop()

            base = {}
            for L in range(n_layers):
                mods = [n for n in lins if layer_of(n) == L]
                acc = {n: [None, 0] for n in mods}
                hooks = [lins[n].register_forward_hook(gram_hook(acc, n)) for n in mods] + [lm.layers[L].register_forward_hook(raise_stop)]
                with torch.no_grad():
                    for c in range(a.calib_chunks):
                        try:
                            model.model(input_ids=chunk(c4, c)[None].to(dev), use_cache=False)
                        except Stop:
                            pass
                for h in hooks:
                    h.remove()
                for n in mods:
                    W = orig[n].to(dev, torch.float32)
                    rot = layer_rot(n, W.shape[0], W.shape[1], W.device)
                    base[n] = rot.inv(gptq_quantize(rot.fwd(W), pop_rotated_hessian(acc[n], rot), levels, a.damp)).to("cpu", torch.bfloat16)
                    torch.cuda.empty_cache()
                log(f"LDLQ: block {L + 1}/{n_layers}")
            os.makedirs(os.path.dirname(a.base_weights) or ".", exist_ok=True)
            save_file({n: t.contiguous() for n, t in base.items()}, a.base_weights)
            with open(a.base_weights.replace(".safetensors", ".json"), "w") as f:
                json.dump({"levels": levels.tolist(), "calib": a.calib_chunks, "damp": a.damp, "seed": SEED}, f)
            log("base B saved")
            drafter.to(dev)
        res["base_B"] = a.base_weights
        save()

    if not a.dry and not a.skip_precision:
        wt = torch.load(a.wikitext)
        n_chunks = min(wt.numel() // (SEQ - 1), a.limit)

        def wchunk(c):
            return torch.cat([torch.tensor([bos]), wt[c * (SEQ - 1):(c + 1) * (SEQ - 1)]])

        @torch.no_grad()
        def hidden():
            return [model.model(input_ids=wchunk(c)[None].to(dev), use_cache=False).last_hidden_state[0].to("cpu") for c in range(n_chunks)]

        @torch.no_grad()
        def per_chunk(Hb, Hv):
            stats = []
            for c in range(n_chunks):
                y = wchunk(c)[1:].to(dev)
                nll = kl = 0.0
                fl = n = 0
                for s0 in range(0, SEQ - 1, 256):
                    s1 = min(s0 + 256, SEQ - 1)
                    lb = torch.log_softmax(logits32(Hb[c][s0:s1].to(dev)), -1)
                    lv = torch.log_softmax(logits32(Hv[c][s0:s1].to(dev)), -1)
                    nll -= float(lv.gather(-1, y[s0:s1, None]).sum())
                    kl += float((lb.exp() * (lb - lv)).sum())
                    fl += int((lv.argmax(-1) != lb.argmax(-1)).sum())
                    n += s1 - s0
                stats.append((nll, kl, fl, n))
            return stats

        set_weights("bf16")
        Hb = hidden()
        PC = {"bf16": per_chunk(Hb, Hb)}
        res["prec"] = {"bf16": aggregate(PC["bf16"])}
        res["changed"] = {}
        plan = [("int8_row", dict(kind="int8")), ("int8_g128", dict(kind="int8", g=128)), ("int8_g32", dict(kind="int8", g=32)),
                ("W_v[B,q6]", dict(kind="verif", bits=6)), ("W_v[B,q8]", dict(kind="verif", bits=8)), ("draft[B]", dict(kind="draft"))]
        for key, kw in plan:
            res["changed"][key] = set_weights(**kw)
            PC[key] = per_chunk(Hb, hidden())
            res["prec"][key] = aggregate(PC[key])
            log(key, res["prec"][key])
            save()
        f8 = res["changed"]["W_v[B,q8]"]
        set_weights("ulp", frac=f8)
        PC["ulp_floor"] = per_chunk(Hb, hidden())
        res["prec"][f"ulp_floor_f{f8:.3f}"] = aggregate(PC["ulp_floor"])
        log("ulp floor", res["prec"][f"ulp_floor_f{f8:.3f}"])
        rng = random.Random(7)
        pairs = (("W_v[B,q8]", "int8_g32"), ("W_v[B,q8]", "int8_g128"), ("W_v[B,q8]", "ulp_floor"), ("W_v[B,q6]", "int8_g128"), ("int8_g32", "ulp_floor"))
        res["paired"] = {f"{x} - {y}": {"KL": paired_bootstrap(PC[x], PC[y], 1, rng), "flips": paired_bootstrap(PC[x], PC[y], 2, rng)} for x, y in pairs}
        log("paired:", json.dumps(res["paired"]))
        del Hb
        save()

    if a.skip_acceptance:
        log("DONE (acceptance stage skipped)")
        return
    record = {"rounds": [], "proposals": [], "prompt": -1}
    generator_cls = candidate_generator.SinglePositionMultiTokenCandidateGenerator
    patch_recorder(generator_cls, record)
    patch_assistant_init(generator_cls)
    if a.dry:
        set_weights("bf16")
    else:
        res["changed_verifier"] = set_weights("verif", bits=8)

    @torch.no_grad()
    def gen_plain(p, **kw):
        return model.generate(input_ids=p[None].to(dev), attention_mask=torch.ones(1, p.numel(), dtype=torch.long, device=dev),
                              max_new_tokens=a.gen, **kw)[0].cpu()

    texts = []
    t_plain = time.time()
    for _, p in prompts:
        texts.append(gen_plain(p, do_sample=False, top_k=None, top_p=None))
    t_plain = time.time() - t_plain
    n_tok = sum(t.numel() - p.numel() for t, (_, p) in zip(texts, prompts))
    res["plain_generation"] = {"tokens": n_tok, "s": t_plain, "tok_s": n_tok / t_plain}
    log("plain:", res["plain_generation"])
    res["hf_drafter"] = {}
    record_by_gamma = {}
    for g in [int(x) for x in a.hf_gammas.split(",")]:
        drafter.generation_config.num_assistant_tokens = g
        drafter.generation_config.num_assistant_tokens_schedule = "constant"
        record["rounds"], record["proposals"] = [], []
        outs = []
        t_assist = time.time()
        for jp, (_, p) in enumerate(prompts):
            record["prompt"] = jp
            outs.append(gen_plain(p, do_sample=False, top_k=None, top_p=None, assistant_model=drafter))
        t_assist = time.time() - t_assist
        nt = sum(t.numel() - p.numel() for t, (_, p) in zip(outs, prompts))
        same = sum(int(torch.equal(x, y)) for x, y in zip(outs, texts))
        n_rounds = len(record["rounds"])
        res["hf_drafter"][g] = {"tokens": nt, "verify_rounds": n_rounds, "tokens_per_round": nt / max(1, n_rounds),
                                "mean_matches": sum(record["rounds"]) / max(1, n_rounds), "s": t_assist, "tok_s": nt / t_assist,
                                "identical_to_plain": f"{same}/{len(outs)}"}
        record_by_gamma[g] = {"proposals": list(record["proposals"]), "outs": outs}
        log(f"HF drafter g={g}:", res["hf_drafter"][g])
        save()

    @torch.no_grad()
    def full_pass(t):
        o = model(input_ids=t[None].to(dev), output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
        return o.hidden_states[-1], o.shared_kv_states, o.logits[0].float()

    def chain(H, KV, t, s, depth, incl, stop=True):
        return drafter_chain(drafter, emb, H, KV, t, s, depth, incl, dev, stop)

    g0 = int(a.hf_gammas.split(",")[0])
    val = {0: [0, 0], 1: [0, 0]}
    proposals = record_by_gamma[g0]["proposals"]
    outs0 = record_by_gamma[g0]["outs"]
    for j, t in enumerate(outs0):
        H, KV, _ = full_pass(t)
        for (jp, pos, draft) in proposals:
            if jp != j or pos > t.numel() or pos - 1 < 1:
                continue
            for incl in (0, 1):
                em, _ = chain(H, KV, t, pos - 1, len(draft), incl, stop=False)
                val[incl][0] += sum(int(x == y) for x, y in zip(em, draft))
                val[incl][1] += len(draft)
    res["emulation_validation"] = {f"kv_to_s+{i}": val[i][0] / max(1, val[i][1]) for i in (0, 1)}
    incl_best = max((0, 1), key=lambda i: val[i][0] / max(1, val[i][1]))
    log("emulation validation (agreement of proposals with HF):", res["emulation_validation"], "-> KV to s +", incl_best)
    save()

    first_hit, run_len, base_hit, sources = [], [], [], []
    for j, t in enumerate(texts):
        pl = prompts[j][1].numel()
        H, KV, _ = full_pass(t)
        m1, kk = [], []
        for p in range(pl, t.numel()):
            em, _ = chain(H, KV, t, p - 1, a.depth, incl_best)
            kk.append(sum(1 for _ in itertools.takewhile(lambda z: z[0] == int(z[1]), zip(em, t[p:p + a.depth]))))
            m1.append(bool(em[0] == int(t[p])))
        first_hit.append(m1)
        run_len.append(kk)
        sources.append(prompts[j][0])
    log("drafter chains computed")
    if not a.dry:
        set_weights("draft")
        for j, t in enumerate(texts):
            pl = prompts[j][1].numel()
            lg = model(input_ids=t[None].to(dev), use_cache=False).logits[0]
            base_hit.append([bool(x) for x in (lg[pl - 1:t.numel() - 1].argmax(-1).cpu() == t[pl:]).tolist()])
    else:
        base_hit = [[False] * len(m) for m in first_hit]

    res["policies"] = {}
    for grp in ["all"] + sorted(set(sources)):
        idx = [i for i in range(len(texts)) if grp == "all" or sources[i] == grp]
        if not idx or sum(len(first_hit[i]) for i in idx) == 0:
            continue
        r = {"agreement_depth1_drafter": sum(sum(first_hit[i]) for i in idx) / sum(len(first_hit[i]) for i in idx),
             "agreement_base": sum(sum(base_hit[i]) for i in idx) / sum(len(base_hit[i]) for i in idx)}
        for pol in POLICIES:
            for g in GAMMAS:
                r[f"{pol}_g{g}"] = tokens_per_round(pol, first_hit, run_len, base_hit, idx, g)
            per = [simulate(pol, first_hit[i], run_len[i], base_hit[i], 8) for i in idx]
            r[f"{pol}_g8_CI95"] = bootstrap_ci(per)
        res["policies"][grp] = r
        log(f"[{grp}]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    save()
    if not a.dry:
        T = 0.7
        set_weights("verif", bits=8)
        stexts, P, QM = [], [], []
        for j, (_, p) in enumerate(prompts):
            torch.manual_seed(SEED + j)
            stexts.append(gen_plain(p, do_sample=True, temperature=T, top_k=0, top_p=1.0))
        for j, t in enumerate(stexts):
            pl = prompts[j][1].numel()
            H, KV, lg = full_pass(t)
            P.append(torch.softmax(lg[pl - 1:t.numel() - 1] / T, -1).half().cpu())
            qm = []
            for pp in range(pl, t.numel()):
                _, l1 = chain(H, KV, t, pp - 1, 1, incl_best, stop=False)
                qm.append(torch.softmax(l1 / T, -1))
            QM.append(torch.stack(qm).half().cpu() if qm else torch.zeros(0))
        am = [float(torch.minimum(P[j].float(), QM[j].float()).sum(-1).mean()) for j in range(len(stexts)) if QM[j].numel()]
        amg = [float(P[j].float().gather(-1, QM[j].float().argmax(-1, keepdim=True)).mean()) for j in range(len(stexts)) if QM[j].numel()]
        set_weights("draft")
        ab, abg = [], []
        for j, t in enumerate(stexts):
            pl = prompts[j][1].numel()
            lg = model(input_ids=t[None].to(dev), use_cache=False).logits[0].float()
            q = torch.softmax(lg[pl - 1:t.numel() - 1] / T, -1)
            pj = P[j].to(dev).float()
            ab.append(float(torch.minimum(pj, q).sum(-1).mean()))
            abg.append(float(pj.gather(-1, q.argmax(-1, keepdim=True)).mean()))
        res["sampling_T0.7"] = {"alpha_drafter_depth1_sum_min": sum(am) / len(am), "alpha_base_sum_min": sum(ab) / len(ab),
                                "p_argmax_drafter": sum(amg) / len(amg), "p_argmax_base": sum(abg) / len(abg)}
        log("sampling:", res["sampling_T0.7"])
    torch.save({"drafter_first_hit": first_hit, "drafter_run_len": run_len, "base_hit": base_hit, "sources": sources, "texts": texts},
               out_path.replace(".json", "_arrays.pt"))
    save()
    log("DONE")


if __name__ == "__main__":
    main()
