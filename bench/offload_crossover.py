#!/usr/bin/env python3
"""Hypothesis 2: where the CPU stops beating the PCIe upload, by turn size.

A batch of n new tokens either runs its experts on the CPU (~66 t/s) or uploads
every expert the batch routes to over Gen3 x8 and runs them on the GPU. The
upload shrinks with n only as fast as the union of routed experts does, so the
GPU wins for big batches and loses for small ones. This sends turns of several
sizes to one server per configuration and reports llama.cpp's own prompt-eval
time for each, so the GGML_OP_OFFLOAD_MIN_BATCH threshold can be put at the
crossover instead of guessed.

Each size is sent twice with different text (no cache hit), second one scored.

    python bench/offload_crossover.py --logdir bench-logs/h2-crossover
    python bench/offload_crossover.py --set p2 --logdir bench-logs/p2-prefetch-min
    python bench/offload_crossover.py --set g20 --logdir bench-logs/g20-crossover

`--set g20` is the h2 pair on Gemma (G20). `--set no20` is the h2 pair on the
launcher's fork binary (hypothesis No. 20). `--set no22` refines that crossover
and adds the CPU path with GGML_CPU_TILED_MM=0 and GGML_CPU_TILED_MM_FORCE=1
(hypothesis No. 22).

`--set p2` (hypothesis P2) runs the same turns on the Tiel fork with expert
prefetch on, and moves GGML_SCHED_PREFETCH_MIN: below it a batch takes the
stock path (sync for the routed ids, upload only those experts, then compute,
no overlap); at or above it every expert tensor is uploaded whole on a second
stream while the previous one computes. Configs alternate so drift shows.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_bench as sb  # noqa: E402

# (model, sizes, binary, configs). Every env names GGML_OP_OFFLOAD_MIN_BATCH itself:
# start_server.py sets 128 for Tiel unless the environment already has one.
_P2_ENV = {"GGML_OP_OFFLOAD_MIN_BATCH": "128", "GGML_SCHED_PREFETCH": "2"}
_H2_CONFIGS = [
    ("gpu-default", {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
    ("cpu-forced", {"GGML_OP_OFFLOAD_MIN_BATCH": "8192"}),  # everything below a ubatch on the CPU
]
_NO22_SIZES = [112, 160, 192, 224, 256, 288, 320, 384, 448]
_NO20_ENV = {"GGML_OP_OFFLOAD_MIN_BATCH": "184", "GGML_SCHED_PREFETCH": "2", "GGML_CUDA_FA_VEC_GQA": "8"}
_NO20_HYB = {"LLAMA_MOE_HYBRID": "1", "GGML_SCHED_HYBRID_STATS": "1"}
_NO20_SIZES = [64, 112, 160, 192, 256, 384, 512, 768, 960]
_NO22_CONFIGS = [
    (f"{tag}-{i}", {"GGML_OP_OFFLOAD_MIN_BATCH": thr, "GGML_SCHED_PREFETCH": "2",
                    "GGML_CUDA_FA_VEC_GQA": "8", **extra})
    for i in (1, 2) for tag, thr, extra in (
        ("gpu", "32", {}),
        ("cpu", "8192", {}),
        ("cpu-notiled", "8192", {"GGML_CPU_TILED_MM": "0"}),
        ("cpu-force", "8192", {"GGML_CPU_TILED_MM_FORCE": "1"}),
    )
]
SETS = {
    # Hypothesis 2, 2026-10-01 (h2-crossover): stock 10826, threshold 32 vs CPU.
    "h2": ("tiel", [48, 96, 160, 256, 384, 640, 1024], "llama.exe", _H2_CONFIGS),
    # G20: the same on Gemma (launcher defaults: -c 16384, f16 KV, -ub 2048, -ncmoe 23).
    "g20": ("gemma", [48, 96, 160, 256, 384, 640, 1024], "llama.exe", _H2_CONFIGS),
    # g20 put Gemma's crossover between 59 and 107 new tokens; the fine grid places it
    "g20-fine": ("gemma", [24, 36, 48, 60, 72, 84, 96], "llama.exe", _H2_CONFIGS),
    # No. 20 (HybriMoE filter): the launcher's fork binary and env, GPU path (stock
    # selective upload below GGML_SCHED_PREFETCH_MIN 1024) against the CPU path, by
    # turn size, to calibrate the CPU cost of an expert and the GPU path's fixed part.
    "no20": ("tiel", [64, 128, 192, 256, 384, 512, 768, 960], sb.TIEL_FORK_NO17_REL_BIN, [
        (f"{tag}-{i}", {"GGML_OP_OFFLOAD_MIN_BATCH": thr, "GGML_SCHED_PREFETCH": "2",
                        "GGML_CUDA_FA_VEC_GQA": "8"})
        for i in (1, 2) for tag, thr in (("gpu", "128"), ("cpu", "8192"))
    ]),
    # No. 22: no20 put the fork's crossover at ~200-220 tokens; a finer grid around it,
    # GPU path at threshold 32 (so the small sizes take it too), and the CPU path with
    # the upstream tiled K-quant MUL_MAT_ID kernel on (default), off, and forced below
    # its 8-rows-per-expert floor. Alternated twice so drift shows.
    "no22": ("tiel", _NO22_SIZES, sb.TIEL_FORK_NO17_REL_BIN, _NO22_CONFIGS),
    # the same with the prompt cache off: 18 distinct prompts per server grow it by
    # gigabytes, and at -c 131072 that pages the experts out (first no22 run, No. 45)
    "no22-nocache": ("tiel", _NO22_SIZES, sb.TIEL_FORK_NO17_REL_BIN, _NO22_CONFIGS),
    # no22-nocache put the crossover between 145 and 194 tokens; the fine grid places it
    "no22-fine": ("tiel", [168, 176, 184, 200, 208, 216, 232, 240], sb.TIEL_FORK_NO17_REL_BIN,
                  [c for c in _NO22_CONFIGS if c[0] in ("gpu-1", "cpu-1", "gpu-2", "cpu-2")]),
    # No. 22 through the launcher: no --llama-bin, so start_server.py picks the fork and
    # its own threshold (184), against the same launch with the old 128 forced.
    "no22-launch": ("tiel", [112, 160, 184, 200, 224, 256, 448], None, [
        (f"{tag}-{i}", env) for i in (1, 2)
        for tag, env in (("launcher", {}), ("thr128", {"GGML_OP_OFFLOAD_MIN_BATCH": "128"}))
    ]),
    # No. 20, the patch: the launcher's env (threshold 184, K=2, grouped FA) on the No. 20 binary,
    # stock against LLAMA_MOE_HYBRID=1 (CPU computes the lightly loaded experts while the bus uploads
    # the rest), alternated twice; GGML_SCHED_HYBRID_STATS logs the per-graph split and timings
    "no20-smoke": ("tiel", [96, 192, 384, 768], sb.TIEL_FORK_NO20_BIN, [
        ("hyb", {**_NO20_ENV, **_NO20_HYB}),
    ]),
    "no20-hyb": ("tiel", _NO20_SIZES, sb.TIEL_FORK_NO20_BIN, [
        (f"{tag}-{i}", env) for i in (1, 2)
        for tag, env in (("stock", _NO20_ENV), ("hyb", {**_NO20_ENV, **_NO20_HYB}))
    ]),
    # one prefetch slot (K=1, q8_0 / ub 4096): the ring holds half as many GPU experts per layer
    "no20-k1": ("tiel", [112, 256, 512, 960], sb.TIEL_FORK_NO20_BIN, [
        (f"{tag}-{i}", {**env, "GGML_SCHED_PREFETCH": "1"}) for i in (1, 2)
        for tag, env in (("stock", _NO20_ENV), ("hyb", {**_NO20_ENV, **_NO20_HYB}))
    ]),
    # the launcher's f16 config: f16 KV, the launcher picks -ub 2048, one prefetch slot
    "no20-f16": ("tiel", [112, 256, 512, 960], sb.TIEL_FORK_NO20_BIN, [
        (f"{tag}-{i}", {**env, "GGML_SCHED_PREFETCH": "1"}) for i in (1, 2)
        for tag, env in (("stock", _NO20_ENV), ("hyb", {**_NO20_ENV, **_NO20_HYB}))
    ]),
    # No. 20 through the launcher: no --llama-bin, so start_server.py picks the fork with hybrid MoE on,
    # against the same launch with LLAMA_MOE_HYBRID=0
    "no20-launch": ("tiel", [64, 160, 256, 512, 960], None, [
        (f"{tag}-{i}", env) for i in (1, 2)
        for tag, env in (("launcher", {}), ("hyb0", {"LLAMA_MOE_HYBRID": "0"}))
    ]),
    # step 2 in place: the CPU takes every expert with at most c tokens, whatever the model says;
    # the stats lines give the CPU part's time against its experts and token slots
    "no20-cal": ("tiel", [128, 256, 512, 960], sb.TIEL_FORK_NO20_BIN, [
        (f"maxc{c}", {**_NO20_ENV, **_NO20_HYB, "LLAMA_MOE_HYBRID_CPU_MAXC": str(c)})
        for c in (-1, 1, 2, 4, 8, 100000)
    ]),
    "p2": ("tiel", [160, 256, 384, 512, 640, 768, 896, 1024], sb.TIEL_FORK_BIN, [
        ("min1024-1", {**_P2_ENV, "GGML_SCHED_PREFETCH_MIN": "1024"}),
        ("min128-1", {**_P2_ENV, "GGML_SCHED_PREFETCH_MIN": "128"}),
        ("min1024-2", {**_P2_ENV, "GGML_SCHED_PREFETCH_MIN": "1024"}),
        ("min128-2", {**_P2_ENV, "GGML_SCHED_PREFETCH_MIN": "128"}),
    ]),
}
# extra start_server.py arguments per set
_NO_CACHE = ["--llama-arg=--cache-ram", "--llama-arg=0"]
EXTRA_LAUNCH = {"no22-nocache": _NO_CACHE, "no22-fine": _NO_CACHE, "no22-launch": _NO_CACHE,
                "no20-smoke": _NO_CACHE, "no20-hyb": _NO_CACHE, "no20-cal": _NO_CACHE,
                "no20-k1": _NO_CACHE, "no20-f16": _NO_CACHE + sb.F16KV, "no20-launch": _NO_CACHE}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--set", default="h2", choices=sorted(SETS))
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    model, SIZES, llama_bin, CONFIGS = SETS[a.set]
    profile = sb.MODELS[model]

    src = open(os.path.join(sb.WS, "prompts", "prompt_real.txt"), encoding="utf-8", errors="replace").read()
    # llama_bin None = let start_server.py choose (the fork, with its own env defaults)
    launch = (profile["launch"] + ["--no-spec"] + (["--llama-bin", llama_bin] if llama_bin else [])
              + EXTRA_LAUNCH.get(a.set, []))
    commit = sb.subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sb.WS,
                               capture_output=True, text=True).stdout.strip()
    res = {"set": a.set, "model": model, "commit": commit, "launch": launch, "sizes": SIZES, "configs": []}
    for name, env in CONFIGS:
        srv = sb.Server([], os.path.join(a.logdir, f"{name}.log"), launch, env)
        entry = {"name": name, "env": env, "rows": []}
        try:
            srv.start()
            entry["load_s"] = round(srv.wait_ready(), 1)
            entry["executing"] = srv.executing_line()
            off = 0
            for n in SIZES:
                block = None
                for rep in range(2):
                    chars = int(n * 3.3)  # source code runs ~3.3 chars/token on this tokenizer
                    text = src[off:off + chars]
                    off += chars
                    mark = len(srv.lines)
                    wall, _ = sb.chat([{"role": "user", "content": f"[{n}-{rep}]\n" + text + "\nOK?"}], 1,
                                      alias=profile["alias"])
                    time.sleep(0.3)
                    block = sb.parse_block(srv.lines[mark:])
                    block["wall_s"] = round(wall, 3)
                    hyb = [l.strip() for l in srv.lines[mark:] if "hybrid moe:" in l]
                    if hyb:
                        block["hybrid"] = hyb
                entry["rows"].append({"size": n, **block})
                print(f"  {name:12s} n~{n:5d}  pp {block.get('pp_tokens')} tok  "
                      f"{block.get('pp_ms')} ms  wall {block['wall_s']} s", flush=True)
                for h in block.get("hybrid", []):
                    print(f"      {h[h.index('hybrid moe:'):]}", flush=True)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print("  !!", entry["error"], flush=True)
        finally:
            srv.stop()
        res["configs"].append(entry)
        with open(os.path.join(a.logdir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
