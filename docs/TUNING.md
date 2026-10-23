# Tuning log — Tiel-Coder-35B-A3B on RTX 3060 6 GB

Running record of what was measured, what it means, and what is still open.
Newest section at the bottom.

---

## Hardware / build

| | |
|---|---|
| GPU | RTX 3060 Laptop, 6144 MiB |
| CPU | Ryzen 5 5600H, 6C/12T |
| RAM | 28 GiB usable |
| llama.cpp | `0.4.0-dev` build **10826**, commit `73a43d1f6`, MSVC, 2026-09-09 |

## Model facts (read straight from the GGUF metadata)

`models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`

| key | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `general.name` | `Ornith-1.5-35B` (Tiel is a finetune of that base) |
| `block_count` | 41 → blk.0–39 compute, **blk.40 is the MTP head** |
| `expert_count` / `expert_used_count` | **256 / 8** |
| `expert_feed_forward_length` | 512 |
| `embedding_length` | 2048 |
| `attention.head_count` / `_kv` | 16 / 2 |
| `full_attention_interval` | **4** → 1 attention layer per 4; the other ~30 are SSM |
| `ssm.state_size` / `ssm.inner_size` | 128 / 4096 |
| `nextn_predict_layers` | 1 |

So this is a **hybrid SSM + attention MoE**, not a plain transformer. 210 of the
753 tensors are SSM. That is the reason speculative decoding is delicate here:
every rejected draft has to roll back recurrent state in ~30 layers.

Expert tensors per layer: `ffn_gate/up_exps` Q4_K, `ffn_down_exps` Q5_K,
256 experts of 2048×512. `--n-cpu-moe` moves only `*_exps`; shared experts,
attention and SSM stay on the GPU (~2038 MiB, matches the planner constant).

### Derived cost model

> **Both claims below were tested and both were wrong.** Kept here with the
> corrections attached, because the arithmetic is still useful and the way it
> misled is worth remembering. Measurements: `bench-logs/threads-t/`, `threads-tb/`.

- Per generated token the CPU reads **~608 MiB** of expert weights
  (8 of 256 experts × 40 layers). At 30 t/s that is ~18 GB/s, from which I
  inferred that **generation is bound by DDR4 bandwidth and thread count will
  not matter**. ❌ It matters a great deal: TG ranges 19.31–31.84 t/s across
  `-t 4/6/8/12` and masks, a 65 % spread. Bandwidth is evidently not the only
  binding constraint. The practical answer is unchanged though — the existing
  `-t 6` on a physical-core mask is the best of everything measured.
- A prefill ubatch of ≥ ~50 tokens routes to nearly **all 256** experts, so it
  touches ~19.5 GiB per pass, independent of batch size — hence the fixed floor
  on every request. From this I inferred **prefill is bound by Q4_K dequant on
  6 cores and should scale with cores**. ❌ It does not scale at all: 28 035
  tokens take 37.81 s at `-tb 6` and 38.06 s at `-tb 12`. Prefill is **GPU-bound**
  — an `nvidia-smi` log over a five-config sweep showed exactly five sustained
  ≥95 % stretches of 36 s each, matching the five 37.6–38.1 s prefill windows.
  ~745 t/s is the 3060's ceiling for this model, and no CPU knob moves it.

## Baseline run (2026-09-16 log, ~2 h 43 min)

Command: `python start_server.py -m tiel -ctk q8_0 -ctv q8_0`
→ `-c 131072 -b 4096 -ub 4096 -ngl 99 --n-cpu-moe 40 -t 6 -tb 6 -C 555 --cpu-strict 1 --load-mode none`

Generation vs context (paired `eval t/s` ↔ `n_tokens` at release):

| ctx | TG t/s |
|---|---|
| 6–8K | 29.9–30.4 |
| 10–12K | 29.8–30.1 |
| 15–18K | 29.2–29.5 |
| 20–24K | 28.2–28.9 |
| 27–30K | 27.2–27.6 |
| 33–38K | 26.8–27.2 |
| 41–46K | 25.7–26.4 |

≈ **−0.33 % per 1K of context**, −14 % from 6K to 46K. Normal for FA + q8_0 KV.

Prefill vs batch size (single-chunk requests):

| prompt | prefill | effective t/s |
|---|---|---|
| 62–64 tok | 1.90–2.04 s | 31–33 |
| 110–113 tok | 2.43–2.57 s | 41–45 |
| 265–295 tok | 3.09–3.19 s | 83–95 |
| 4096 tok chunk | 5.30 s | ~770 |

**Every request carries a ~2 s floor** before the first token, however small the
new prompt is. For an agentic client doing many short turns this is 30–50 % of
per-turn latency, and it is the single biggest latency item in the log.

Other observations: max context actually reached was **45 700** tokens (of
131 072 allocated); after a 75 min idle the first request still hit 30.18 t/s,
so expert pages stay resident and RAM is sufficient; 4 client-side cancels and
4 context rewrites (`f_sim` 0.25–0.51) cost ~30–35 s of re-prefill each.

## MTP status — resolved

The build **does** support it: `--spec-type` accepts `draft-mtp`, alongside
`--spec-draft-n-max`, `--spec-draft-p-min`, `--spec-draft-ngl` and friends.
`blk.40.*` was logged as "unused tensor" only because speculative decoding was
off, not because the build cannot use it.

The head is the **trained** one: upstream shipped an untrained random-init MTP
block at first and replaced it on 2026-08-23; the local file was written
2026-09-10 and carries the four `blk.40.nextn.*` tensors. Upstream's own number
for the fixed head is **1.10× at 96 % acceptance with
`--spec-draft-n-max 1 --spec-draft-p-min 0.8`**.

`start_server.py` currently refuses MTP on Tiel, citing a local measurement of
**4× slower (24.7 → 6.3 t/s) at 77 % acceptance**. That configuration did not
use `--spec-draft-p-min`, which is what stops the low-confidence steps from
paying the verify cost — and the verify cost is exactly what is expensive when
experts live on the CPU. Worth re-measuring on this build. See open items.

The control is **Ornith-1.5-35B-A3B in the XL quant** — the base Tiel was
finetuned from, so byte-for-byte the same architecture, the same expert layout
and the same verify cost. That makes it a test of *head quality*, not of
mechanism: if Ornith's original head accepts well where Tiel's does not, the
finetune desynced the inherited MTP block from the body it predicts for.

(`models/Ornith-1.5-9B-Q4_K_M` is present locally but is a different animal —
dense `qwen35`, 33 blocks, MTP head at blk.32. It would isolate "SSM rollback"
from "CPU-MoE verify" because it fits entirely in VRAM, but it is not the
comparison being asked for. Keep it as an optional extra, not a plan item.)

## VRAM planner — the GPU-expert lever does not exist here

`start_server.py` already auto-sizes ubatch and plans the split. Output of
`best_tiel_ubatch` / `plan_tiel_vram` across the range (6144 MiB, q8_0):

| ctx | ub | ncmoe | expert layers on GPU | KV MiB | peak MiB | free MiB |
|---|---|---|---|---|---|---|
| 16384 | 8192 | 40 | 0 | 180 | 5283 | 661 |
| 32768 | 8192 | 40 | 0 | 361 | 5464 | 480 |
| 49152 | 6144 | 40 | 0 | 541 | 5015 | 929 |
| 65536 | 6144 | 40 | 0 | 722 | 5196 | 748 |
| 98304 | 6144 | 40 | 0 | 1082 | 5556 | 388 |
| 131072 | 4096 | 40 | 0 | 1443 | 5288 | 656 |
| 163840 | 2048 | 40 | 0 | 1804 | 5020 | 924 |
| 200000 | 2048 | 40 | 0 | 2202 | 5419 | 525 |
| 262144 | 2048 | 40 | 0 | 2886 | 6103 | −159 |

**Zero expert layers reach the GPU at any context size**, because
`TIEL_MOE_LAYER_MIB = 498` plus `TIEL_PREFILL_HEADROOM_MIB = 700` needs 1198 MiB
free and the planner never gets there — it spends the slack on a larger ubatch
instead. Even forcing it: at `-c 131072`, dropping ub 4096 → 2048 frees ~628 MiB
and buys exactly **one** layer of 40 (≈2.5 % of expert traffic, ≈+2–3 % TG) at
the price of roughly half the bulk prefill rate. A bad trade.

**Conclusion: on 6 GB VRAM this model will never get a meaningful number of
expert layers onto the GPU, so shrinking context or ubatch to make room is not
a useful optimisation.** This supersedes the earlier suggestion to cut `-c` to
65536 for that purpose. The remaining real levers are MTP and prefill threading.

## Housekeeping

Three byte-identical copies of the same 21.66 GiB file exist (verified by size +
first-1 MiB MD5 `301D238C…`):

- `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` (repo root)
- `models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf`
- `models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`

Two of them are redundant → **43.3 GiB reclaimable**. The `-MTP` / non-`-MTP`
names are meaningless: all three carry the MTP head.

`--backend-sampling` is logged as disabled on every request because the client
always sends a grammar (JSON schema / tool constraints). It is a silent no-op,
not an error, and sampling is a negligible share of a 33 ms token here. Nothing
to fix; it still applies on the Gemma / Ornith paths.

## Open items

1. Re-measure MTP on Tiel. Two axes, swept separately so they do not confound:
   `--spec-draft-p-min` (0.0 / 0.6 / 0.8 / 0.9) at `--spec-draft-n-max 1`, then
   `--spec-draft-n-max` **1 → 5** at whichever p-min won. Note that
   `nextn_predict_layers = 1`, so drafting more than one token runs the head
   autoregressively and acceptance is expected to decay with depth — expected,
   not established; measure it. Needs a flag in `start_server.py` to override
   the current hard refusal on Tiel.
2. Same matrix on **Ornith-1.5-35B-A3B, XL quant** as the control for head
   quality. Not downloaded yet.
3. Prefill threading: `-tb 8 / 10 / 12` with a widened CPU mask, `-t 6` kept.
   Target is the ~2 s floor, which is dequant-bound and should scale with cores.
4. Cheap sanity check on the cost model: TG at `-t 4 / 6 / 8`. If generation is
   DDR4-bound as derived, thread count should barely move it.

Quant policy: **XL only.** IQ4_XS measured worse and noticeably weaker in
quality, and XL fits, so XS is out of scope entirely — do not benchmark it.

---

# Gemma 4 26B-A4B — second model, for conversation

Tiel stays the coding model. This one is picked for talking to: multilingual
(Russian specifically), multimodal, and MoE so it runs at all on 6 GB.

## Build support — checked before downloading anything

Gemma 4 shipped 2026-04-02, the local `llama.exe` was built 2026-09-09, so
support is five months old rather than absent. Confirmed by reading the binary
rather than assuming: it contains `llama_model_gemma4`,
`llama_model_gemma4_assistant`, `clip_graph_gemma4v`, the `gemma4a`/`gemma4ua`
audio preprocessors, and a `peg_gemma4` tool-call parser. **No new build needed.**

## Which file

`unsloth/gemma-4-26B-A4B-it-qat-GGUF` → `UD-Q4_K_XL`, 14.25 GB.

The QAT checkpoint is trained *for* 4-bit rather than cut down to it, and the
Unsloth dynamic quant of it is both **smaller and more accurate** than their own
non-QAT `UD-Q4_K_XL` (14.2 GB vs 17 GB). That inverts the usual "want quality,
take a bigger quant" reasoning, and it matters most for Russian: rarer tokens
carry thinner probability mass, so they are what quantisation error eats first.

Stock `-it`, no finetune. QAT exists only for the official instruct weights, so
every finetune trades a model trained for Q4 for one naively cut to Q4; on top of
that the available tunes (StyleTune, the abliterations) are English-target and
multilingual behaviour is the first thing that degrades. Reasoning, not
measurement — recorded as such.

## Architecture, read from the GGUF and config.json

| | Tiel 35B-A3B | Gemma 4 26B-A4B |
|---|---|---|
| arch | `qwen35moe`, hybrid SSM+attn | `gemma4`, plain transformer |
| blocks | 41 | 30 |
| experts | 256, top-8 | 128, top-8, **+1 shared** |
| hidden / expert FF | 2048 / 512 | 2816 / 704 |
| attention pattern | 1 full per 4 | **25 sliding (1024) + 5 full**, 5:1 |
| head_dim | 128 | 256 sliding / **512 global** |
| KV per token | ~5.6 KiB | **~80 KiB f16** (global blocks only) |
| expert bulk | 19.6 GiB | ~12.2 GB |
| draft head | blk.40, inside the MoE | separate 225 MiB **dense** model |

Two consequences that break the Tiel configuration outright:

**Context cannot be 131072.** The 25 sliding blocks cost a fixed amount whatever
`-c` is, but the 5 global blocks carry head_dim 512 over 8 KV heads: ~80 KiB per
token at f16. That is ~1.45 GiB at 16k, ~2.7 GiB at 32k, and past the whole card
at 64k. The launcher defaults to 16384.

**KV should not be q8_0.** Every other model here runs `-ctk q8_0 -ctv q8_0`
because it was measured to be nearly free. Published KL divergence against f16
for this exact model is 0.377 at q8_0 and 1.088 at q4_0, against 0.108 for the
dense Gemma 4 31B on the same test — the MoE variant degrades ~3.5× worse than
its own dense sibling. Someone else's number; G1 checks it locally.

**The draft head is a different animal from Tiel's.** `mtp-gemma-4-26B-A4B-it.gguf`
is its own model: arch `gemma4-assistant`, 4 blocks, embedding 1024, 225 MiB,
`nextn_predict_layers = 4`, and **zero expert tensors**. It fits in VRAM whole
and drafting costs no CPU expert traffic. Tiel's head was blk.40 *of the MoE*,
so every draft step paid the same DDR4 bill as a real token, on top of rolling
back SSM state in 30 layers. Neither penalty exists here. That is why MTP is
worth re-measuring rather than inheriting Tiel's verdict.

## Derived numbers — arithmetic, not measurement

Per-token routed-expert traffic works out at ~730 MiB against Tiel's measured
604 MiB, i.e. ~1.2× heavier. Scaling Tiel's 30 t/s by that ratio gives ~25 t/s
with every expert layer on the CPU, ~28 t/s if the planner gets 4 layers onto the
GPU. **The linear model this rests on was already refuted once** (H4: thread
count alone moved TG by 65 %), so treat the bracket as 22–30 t/s and nothing
narrower until G3 runs.

## Hypotheses — all closed except G4

| ID | hypothesis | verdict |
|----|------------|---------|
| G1 | q8_0 KV costs visible quality here, unlike on Tiel | **half confirmed** — it changes every answer, but no quality loss is demonstrable, and at the default context it buys nothing anyway |
| G2 | Expert layers reach the GPU for the first time on this box, worth +10–15 % TG | **confirmed, understated** — +24 % |
| G3 | TG 22–30 t/s, prefill above Tiel's 745 ceiling, floor below 1.7 s | **confirmed; my TG range was too pessimistic** — 33.8 t/s |
| G4 | MTP pays off here even though it lost on Tiel | **blocked** — the draft head does not load on either build |
| G5 | `--mlock` removes page-fault variance | **refuted** — no measurable difference |
| G6 | Context is a speed knob here, unlike on Tiel | **confirmed** — 16k → 131k costs 29 % of generation |

## VRAM model, calibrated

`calibrate_vram.py` varies one term at a time and samples peak VRAM after the
warmup prefill. Points in `bench-logs/gemma-vram/calib.json`:

| ctx | ub | GPU expert layers | VRAM MiB |
|---|---|---|---|
| 4096 | 1024 | 0 | 2325 |
| 16384 | 1024 | 0 | 2711 |
| 32768 | 1024 | 0 | 3063 |
| 65536 | 1024 | 0 | 3767 |
| 4096 | 2048 | 0 | 2643 |
| 4096 | 1024 | 2 | 3141 |
| 4096 | 1024 | 4 | 3959 |

```
peak ≈ 1322.6 (dense weights, from the header)
     +  497   (CUDA context and runtime)
     +  KV(ctx, type) × 1.10
     +  0.3105 × ubatch
     +  408.5 × gpu_expert_layers
```

Reproduces the three saturated context points to within 1 MiB.

**The expert-layer slope is 408.5 MiB against the 408.4 that `inspect_gguf.py`
reads out of the header.** That is the result worth keeping: the constant did not
need calibrating at all, and a requantisation of this model will not need a new
sweep either. The same extraction rule, checked backwards against Tiel,
reproduces its hand-tuned `TIEL_DENSE_VRAM_MIB = 2038` to 1 MiB.

Two things the first version of the model got wrong, both worth an expert layer:
`token_embd.weight` has to stay on the GPU when embeddings are tied (Gemma) and
off it when they are not (Tiel); and the compute coefficient does **not** scale
with hidden size — Tiel's 0.307 and Gemma's 0.3105 are the same number at a 27 %
difference in `embedding_length`.

## G2 / G3 — the expert-layer split

`bench-logs/gemma-ncmoe/`, ctx 16384, f16 KV, ub 2048, `-t 6 -C 555`:

| `-ncmoe` | GPU expert layers | TG t/s | prefill t/s | floor s |
|---|---|---|---|---|
| 30 | 0 | 26.88 | 685.09 | 2.18 |
| 28 | 2 | 28.53 | 724.52 | 1.19 |
| 26 | 4 | 30.64 | 759.17 | 1.11 |
| 24 | 6 | 33.23 | 799.89 | 1.05 |
| 22 | 8 | 36.46 | **196.77** | 1.08 |

Generation and prefill improve together up to the point where the allocation
stops fitting. `-ncmoe 22` is that point — 6295 MiB planned against 6144
available — and the penalty is confined to bulk prefill, which collapses as WDDM
pages the staging buffer over PCIe. The floor barely notices.

### The frontier

`bench-logs/gemma-frontier/`:

| layers | ub | TG t/s | prefill t/s | floor s | free VRAM |
|---|---|---|---|---|---|
| 6 | 2048 | 33.82 / 33.24 | 798.6 | 1.05 | 666 |
| **7** | **2048** | **33.75** | **819.0** | **1.00** | **257** |
| 7 | 1024 | 34.48 | 516.3 | 1.00 | 575 |
| 8 | 1024 | 36.59 | 531.3 | 0.94 | 167 |
| 8 | 512 | 36.46 | 320.0 | 0.96 | 326 |

**Tiel's 700 MiB prefill headroom does not transfer.** Prefill is flat at ~800
t/s from 666 MiB free down to 257; it fails only when the allocation exceeds the
card outright. So the headroom budget was pure waste here and is set to 0, with
the reserve raised 200 → 250 to cover the ~50 MiB the model under-predicts by.

L8/ub1024 is the generation-maximum corner: +8.4 % TG and a 0.94 s floor against
531 t/s prefill. For a 500-token prompt and a 300-token answer it is ~8 % faster
end to end; for a 6000-token paste L7 is ~17 % faster. L7 is the default because
it gives up nothing measurable on short turns.

#### Addendum: ubatch alone, 2026-09-21 (`bench-logs/gemma-ub/`)

Run because `bench-logs/` had no `gemma-ub/` directory and the knob looked
unswept. It was not: the frontier table above is the same question asked the way
that matters, with `--n-cpu-moe` moving alongside. Sweeping ubatch on its own
mostly re-derives rows already there. Recorded anyway for the three things it
does add.

| `-ub` | compute MiB | layers on GPU | TG t/s | prefill t/s |
|---|---|---|---|---|
| 2048 (default) | 636 | 7 | 34.66 | 804.92 |
| 1024 | 318 | 7 | 34.46 | 512.46 |
| 4096 | 1272 | 5 | 31.77 | 402.26 |
| 6144 | 1908 | 3 | 26.96 | 347.79 |

1. **The planner never reaches 8 layers on its own**, at any ubatch. L8 exists
   only under an explicit `--n-cpu-moe 22`. The reason is arithmetic the banner
   hides: it prints `weights + KV + compute + experts` but the budget also
   carries `GEMMA4_FIXED_VRAM_MIB = 497` and `GEMMA4_VRAM_RESERVE_MIB = 250`.
   At ub 1024 the spare is 6144 − 250 − (1322.6 + 497 + 572 + 318) = 3184.4 MiB
   against 3267.2 for eight layers — **short by 82.8 MiB**. The prediction that
   halving ubatch would buy a layer omitted both terms and was wrong by exactly
   that. At ub 512 it does fit, and the planner takes `-ncmoe 22` by itself.
2. **Upward is closed for good.** ub 4096 and 6144 were not in the frontier
   sweep; both lose, 5 and 3 layers respectively, down to 26.96 t/s.
3. ub 2048 reproduced at 34.66 against the frontier's 33.75 — inside the 7 %
   noise floor measured in G6.

Nothing here moves the default. L7/ub2048 stands, and L8/ub1024 remains the
generation corner for anyone whose turns are short.

## G6 — context

`bench-logs/gemma-ctx/`:

| ctx | GPU expert layers | TG t/s | prefill t/s | floor s |
|---|---|---|---|---|
| 8192 | 7 | 34.73 | 821.9 | 1.00 |
| 16384 | 7 | 33.78 / 31.59 | 822.2 | 1.00 |
| 32768 | 6 | 29.93 | 797.7 | 1.08 |
| 65536 | 4 | 28.36 | 756.6 | 1.16 |
| 131072 | 0 | 23.89 | 677.8 | 1.38 |

Context is a speed knob here purely through the expert layers it displaces —
8192 and 16384 both get 7 layers and are indistinguishable. The 16384 repeat
also fixes the noise floor: **TG varies ~7 % between identical runs**, so nothing
smaller than that is a result.

## G1 — does a quantised KV cache change the answer?

`kv_quality_test.py`, 12 Russian prompts, greedy decoding, fixed seed, one server
process per KV type. `bench-logs/gemma-kv-quality/`:

| comparison | answers changed | mean similarity |
|---|---|---|
| f16 vs q8_0 | **12 / 12** | 0.462 |
| f16 vs f16 (control) | **0 / 12** | 1.000 |

The control is what makes this readable: the same KV type in a fresh process
reproduces byte for byte, so the 12/12 is quantisation and not non-determinism.

**But different is not worse, and this test cannot show that it is.** On the two
prompts with a checkable answer both runs agree: the arithmetic comes out 20⅓ in
each, and the одеть/надеть rule is correct in each. What changes is phrasing and
structure. Twelve prompts cannot establish a quality delta either way.

The decision comes out elsewhere and is not close: **at ctx 16384 a q8_0 cache
buys zero extra expert layers**, so it changes every answer for no speed at all.
It buys one layer at 32768 and two at 65536, which is where it becomes a real
question.

| ctx | KV f16 layers | KV q8_0 layers | q8_0 gains |
|---|---|---|---|
| 16384 | 7 | 7 | nothing |
| 32768 | 6 | 7 | +1 (~+3 % TG) |
| 65536 | 4 | 6 | +2 (~+7 % TG) |

## Threads — same answer as Tiel

`bench-logs/gemma-threads/`:

| config | TG t/s | prefill t/s | floor s |
|---|---|---|---|
| **`-t 6`, mask 555** | **31.56** | 821.7 | 1.01 |
| `-t 4`, mask fff | 17.86 | 819.7 | 1.06 |
| `-t 6`, mask fff | 24.43 | 822.5 | 1.01 |
| `-t 8`, mask fff | 29.87 | 820.4 | 1.01 |
| `-t 12`, mask fff | — | — | reset the display driver |

Nothing to change. Pinning six threads to physical cores beats the same six
threads on the full mask by 29 %, and prefill is deaf to threads as always.

**`-t 12` with `--cpu-strict 1 --prio 2` caused a TDR reset.** Tiel has the same
row and it merely ran badly. The difference is structural: Tiel puts 0 expert
layers on the GPU so a token barely crosses the CPU/GPU boundary, Gemma puts 7
there and every token does, so the GPU scheduler needs CPU time constantly and
starving it costs more. `start_server.py` now drops strict placement above the
physical core count, and the row is gone from the sweep.

## G5 — `--mlock`

| config | TG t/s | prefill t/s |
|---|---|---|
| `--load-mode none` | 31.95 | 819.4 |
| `--mlock` | 31.73 | 820.2 |
| `--load-mode none` (repeat) | 31.57 | 819.5 |

Refuted. It is now *possible* — Tiel could not use it, 21.7 GiB pinned in a 28
GiB system killed the server, while Gemma needs ~12 GiB host-side — but there is
nothing to win. 1.2 % spread across all three.

## G4 — MTP, blocked

The draft head does not load:

```
common_speculative_init_result: loading draft model 'mtp-gemma-4-26B-A4B-it.gguf'
llama_model_load: error loading model: invalid vector subscript
```

Ruled out in order: the file is byte-correct (sha256 matches the repo), both the
smart-Q4_0 and Q8_0 drafters fail identically, the documented `--model-draft`
invocation is exactly what was used, a bare command line with no local flags
fails the same way, and the head predates the build by three months (published
June 2026, MTP merged into llama.cpp 2026-06-07, build is 2026-09-09). Build
10909 fails identically.

### It is a known upstream regression, not our setup (2026-09-21)

[Issue #24795](https://github.com/ggml-org/llama.cpp/issues/24795) is our bug
exactly: the same `gemma4-assistant` draft, the same `invalid vector subscript`,
firing the instant `load_tensors` begins, before any tensor data is read. It is
a **regression**, and the reporter pinned the window: the same models and the
same command load cleanly on **b9553** (`9e3b928`, ~2026-06-19) and crash on
b9702 (`968c438`) and b9717 (`8141e73`). Our 10826 is roughly 1270 builds past
the break.

The issue is still open, still labelled `bug-unconfirmed`, never bisected, with
no maintainer response and no workaround other than going back to b9553.

So the history is not "Gemma 4 MTP was never supported". It was added by
[PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) (merged
2026-06-07), extended to E2B/E4B by #24282 the next day, worked for about two
weeks, and has been broken since. The path is narrow enough that it went
unnoticed: the drafter is a separate model few people run, and `llama-speculative`
and `llama-bench` cannot load E2B/E4B at all, so only `llama-server` users hit it.

**`--fit off` is not the fix — measured, 2026-09-21.** Our logs carry the message
from the neighbouring [issue #24350](https://github.com/ggml-org/llama.cpp/issues/24350),
`Gemma4Assistant requires ctx_other to be set (this warning is normal during
memory fitting)`, which there was worked around with `--fit off` (that issue was
closed as not-planned via PR #24590). Re-running our exact MTP command with
`--fit off` removes the `ctx_other` line and the failed fitting pass, and the
draft still dies on `invalid vector subscript` 25 s later. The fitting noise was
a red herring; the bug is #24795.

What the PR actually claims, for sizing the prize: >2× on dense models, and on
its own bench 290.01 s → 120.65 s at `--spec-draft-n-max 4` (2.4×) with
acceptance 0.5881; a user on an RTX PRO 6000 reports 40.0 → 83.3 t/s (2.08×) at
acceptance 0.5765. All of it on cards that hold the whole model, which is not
our regime — ours has 23 of 30 expert layers on the CPU.

### It is not the loader and not CUDA. The head does not fit in VRAM (2026-09-21)

**Read this section before the two below it — they were written while the cause
was still misdiagnosed, and are kept only because the route to the answer is
worth seeing.**

`invalid vector subscript` is llama.cpp failing to allocate the draft model in
VRAM. Nothing more. The working Gemma configuration leaves **258 MiB free on the
card** (`weights 1323 + KV 572 + compute 636 + experts 2859` of 6144) and the
head needs its 225 MiB of weights plus its own KV and compute buffers. It does
not fit, and the error says something else entirely.

Evicting expert layers makes room. The threshold is exact:

| `--n-cpu-moe` | expert layers on GPU | head at `-ngld 99` |
|---|---|---|
| 23 (working config) … 26 | 7 … 4 | `invalid vector subscript` |
| **27** | **3** | **loads** |
| 28, 29, 30 | 2, 1, 0 | loads |

So the entry price is four expert layers. Upstream
[issue #24795](https://github.com/ggml-org/llama.cpp/issues/24795) is most
likely the same thing on someone else's VRAM budget, and calling it a build
regression was wrong: b9553 is not special, it simply laid the memory out
differently.

### G4 — measured against its own hypothesis, refuted (`bench-logs/gemma-mtp-vram/`)

Head fully in VRAM (`-ngld 99`), `--n-cpu-moe 27` on every row including both
baselines, so the four evicted layers are paid by baseline and MTP alike.

| config | TG t/s | prefill t/s | floor, s | acceptance | mean len |
|---|---|---|---|---|---|
| A: baseline | **29.53** | 726.95 | 1.12 | — | — |
| n-max 1 | 29.06 | 720.78 | 1.12 | 0.80543 | 1.81 |
| n-max 2 | 28.58 | 719.20 | 1.11 | 0.73148 | 2.46 |
| n-max 3 | 25.02 | 714.12 | 1.15 | 0.64216 | 2.93 |
| n-max 4 | 22.68 | 719.98 | 1.16 | 0.53649 | 3.14 |
| n-max 6 | 18.46 | 719.31 | 1.13 | 0.43636 | 3.59 |
| Z: baseline repeat | **29.77** | 725.05 | 1.13 | — | — |

**The prefill tax is gone.** 720.8 against a baseline of 726.0 at n-max 1, −0.8 %,
where the CPU-resident head cost 302 against 804. That is the one thing this
sweep settles cleanly: the prefill collapse seen everywhere else is about where
the draft model runs, not about MTP.

**And it still does not pay.** Every row is at or below baseline, from −2.0 % at
n-max 1 to −38 % at n-max 6. Acceptance is healthy and unchanged from the CPU
run (0.805 at n-max 1), so the drafter is doing its job; what it cannot do is
make the verify step cheap. Verification reads the expert weights for the whole
draft batch, and with 27 of 30 expert layers in system RAM that lands on DDR4 at
close to full price per step. Drafting is free in VRAM; verifying never was.

Worse, the entry price dominates the whole question. Against the real working
configuration — `--n-cpu-moe 23`, no MTP, **34.89 t/s and 804 t/s prefill** —
the best MTP row here is 29.06 t/s and 720.8. Four expert layers cost −15 %
before speculation begins, and speculation does not give it back.

G4 is refuted, this time against the hypothesis it was actually written for: a
225 MiB dense head drafting entirely in VRAM at no DDR4 cost. It drafts exactly
as promised and the result is still worse than not doing it. Upstream's 2.08–2.4×
comes from machines where the *target* model is in VRAM too; that is the part we
cannot reproduce, and no flag will change it.

### How it was misdiagnosed first (kept for the method)

Everything above reads as "a defect in the `gemma4-assistant` loader, wait for a
new build". That was wrong, and the way it came apart is worth recording.

b9553 was downloaded to test the regression window. It loaded the head on the
first try — and that first try was **wrong**: the Windows CUDA release ships the
CUDA runtime as a *separate* asset, `cudart-llama-bin-win-cuda-13.3-x64.zip`,
which had not been fetched. Without `cudart`/`cublas`, `ggml-cuda.dll` cannot
load and the build falls back to CPU silently — `device_info` listed only the
Ryzen. The whole first sweep ran with no GPU (TG 15.09 against 33.8, prefill
34.76 against 820) and is discarded, not journalled.

With the runtime installed, b9553 fails exactly like every other build. But the
accident pointed at the answer. Two runs on b9553, CUDA up, differing in one
flag:

| | result |
|---|---|
| `-ngl 99 --n-cpu-moe 23 -ngld 0` | loads, server listens |
| `-ngl 99` (head offloaded to CUDA) | `invalid vector subscript` |

And on the production build 10826, `-ngld 0` loads the head and serves. **So G4
was never blocked by the build.**

The conclusion drawn here — "what breaks is offloading the draft model to CUDA"
— was wrong, and the b9553 tests above do not support it: both of them held
`--n-cpu-moe` fixed while moving the head, so they changed VRAM pressure and the
CUDA path together. `-ngld 0` works because the head stops competing for VRAM,
not because it stops touching CUDA. The section above has the discriminating
test: free four expert layers and `-ngld 99` loads fine.

`start_server.py` still defaults to `-ngld 0` with a separate head, because at
the working `--n-cpu-moe 23` that is the only thing that loads. `--ngld N`
overrides it, and the head goes to VRAM from `--n-cpu-moe 27` up.

### G4 measured with the head on the CPU (`bench-logs/gemma-mtp-g4/`)

Superseded by the VRAM sweep above; this one measures `-ngld 0`, which is not
what G4 proposed. Kept because it is the cleanest available demonstration of
what a CPU-resident draft head costs.

Build 10826, `-ngld 0`, everything else the working Gemma configuration.

| config | TG t/s | prefill t/s | prefill wall | floor, s | load, s | acceptance | mean len |
|---|---|---|---|---|---|---|---|
| A: baseline | **34.70** | **804.84** | 9.2 s | 1.00 | 39.1 | — | — |
| n-max 1 | 25.89 | 302.13 | 24.4 s | 0.97 | 45.1 | 0.82192 | 1.82 |
| n-max 2 | 20.18 | 301.99 | 24.4 s | 0.97 | 46.0 | 0.61345 | 2.22 |
| n-max 3 | 18.87 | 301.93 | 24.4 s | 0.97 | 45.0 | 0.58391 | 2.75 |
| n-max 4 | 18.23 | 302.25 | 24.4 s | 0.96 | 47.1 | 0.55894 | 3.22 |
| n-max 6 | 16.21 | 301.99 | 24.4 s | 0.98 | 45.0 | 0.47267 | 3.80 |
| Z: baseline repeat | **35.08** | **803.23** | 9.2 s | 0.97 | 39.1 | — | — |

Every configuration loses, monotonically: −26 % at n-max 1 down to −54 % at
n-max 6 against the baseline mean of 34.89. Prefill sits at 302 t/s whatever the
draft depth — 2.66× below baseline, 24.4 s of wall against 9.2 s — and is
remarkably steady, unlike Tiel's, because the draft prefill is a fixed cost per
prompt rather than per drafted token.

**The head is not the problem.** Acceptance lands in the same band upstream
reports: 0.82 at n-max 1, and 0.559 at n-max 4 against their 0.5881 at the same
depth. The drafter predicts as well here as it does on their hardware. What we
cannot reproduce is the payoff, and the reason is the workaround: `-ngld 0` puts
the head on the CPU, which is precisely the property that made G4 worth chasing
— a 225 MiB dense drafter that costs no DDR4 traffic. On the CPU it reads those
225 MiB per drafted token and the verify step lands on a model whose 23 expert
layers are already CPU-resident. The tokens saved do not cover it.

This is not G4. What it measures is a head on the CPU, which was never the
hypothesis — see the VRAM sweep above for the real one, which also fails, and
for the reason: the verify step, not the drafting, is what costs.
