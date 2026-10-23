#!/usr/bin/env python3
"""
sweep_bench.py - config sweep harness for start_server.py.

Launches the server once per configuration, runs a fixed set of requests against
it, and reads the timings back out of llama.cpp's own `print_timing` lines rather
than timing on the client side. Kills the process tree between configs so each
run starts from the same state.

Protocol (see CLAUDE.md): temperature 0, every measurement taken twice with the
second one scored, so the page cache is warm. Per-model launch settings live in
MODELS below, because Tiel and Gemma do not share a context size or KV type and
comparing across them was never the point -- each sweep compares configurations
of one model against each other.

Prompts come from prompt_real.txt (a real source dump). prompt23k.txt is
deliberately NOT used: it is "the quick brown fox" repeated, and repetitive text
routes to a handful of experts and is trivially predictable, which would flatter
both prefill throughput and MTP acceptance.

The generation prompt is language-matched to the model's job. Which experts a
MoE routes to depends on what it is writing, and Gemma is being tuned for
Russian conversation, so measuring its generation on an English code explanation
would be measuring the wrong experts. Prefill keeps the English source dump:
a full ubatch routes to essentially every expert regardless of content, and it
keeps the prefill figure comparable with the existing journal.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PORT = 9095
API_KEY = "sk-localllama-rtx3060-securekey"
BASE = f"http://127.0.0.1:{PORT}"

RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_EVAL = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_TOTAL = re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_DRAFT = re.compile(r"(draft|accept|spec)", re.I)


# ---------------------------------------------------------------- prompts

# A Russian generation prompt for the models that are meant to hold a
# conversation. It asks for a long free-form answer on a concrete topic, so the
# reply is ordinary connected prose rather than a list or a code block, and it
# names the topic precisely enough that temperature 0 gives the same answer every
# run -- which is what makes two configurations comparable.
RU_TG_PROMPT = (
    "Расскажи подробно, как устроена система городского водоснабжения: "
    "откуда берётся вода, какие этапы очистки она проходит, как её доставляют "
    "в квартиры и что происходит со сточными водами. Пиши связным текстом, "
    "без списков и заголовков, примерно на две страницы."
)


def load_prompts(lang="en", prefill_chars=None):
    """prefill_chars caps the prompt-processing prompt.

    prompt_real.txt is 31 174 tokens, which is fine against Tiel's 131072 window
    and a hard 400 against Gemma's 16384 -- the whole first ncmoe sweep came back
    empty for that reason. The cap has to be a constant across a sweep, not a
    function of -c, or the gemma-ctx rows would each be measuring a different
    prompt; it is therefore sized to the smallest window any sweep uses.
    """
    with open(os.path.join(WS, "prompts", "prompt_real.txt"), "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    if prefill_chars:
        src = src[:prefill_chars]
    # ~4 chars/token, so 6000 chars is roughly 1.5k tokens: big enough to be a
    # realistic routing load, small enough that prefill does not dominate the
    # generation measurement.
    mid = src[:6000]

    # Hypothesis No. 16 (draftless n-gram speculation). Two excerpts per test, one
    # per repeat: an n-gram table persists across requests, so asking the same
    # thing twice would let the second answer be drafted from the first. Excerpts
    # are cut on line boundaries from the full file, not from the prefill cap.
    with open(os.path.join(WS, "prompts", "prompt_real.txt"), "r", encoding="utf-8", errors="replace") as f:
        full = f.read()

    def excerpt(start, n=6000):
        a = full.index("\n", start) + 1
        b = full.index("\n", a + n)
        return full[a:b]

    copy = [[{"role": "user", "content":
              "Here is an excerpt of a Python file:\n\n```python\n" + excerpt(s) + "\n```\n\n"
              "Reproduce this excerpt exactly, character for character, in one python code "
              "block, with a single change: add the line `# reviewed` as the very first line. "
              "Output only the code block."}]
            for s in (6000, 30000)]
    free = [[{"role": "user", "content":
              "Here is an excerpt of a Python file:\n\n```python\n" + excerpt(s) + "\n```\n\n"
              "Explain in plain prose what this code does and why it is written this way. "
              "Do not quote, copy or rewrite any of the code."}]
            for s in (50000, 70000)]
    # NG1 at depth: the same copy task behind ~85K tokens of context, so the KV
    # cache is actually written to and WDDM can no longer leave its tail off the
    # card. Both requests share the filler, so the second prefills only its tail.
    with open(os.path.join(WS, "prompts", "prompt_ru_wiki.txt"), "r", encoding="utf-8", errors="replace") as f:
        wiki = f.read()[:200000]
    filler = ("Reference material:\n\n```python\n" + full + "\n```\n\n" + wiki + "\n\n")
    deep = [[{"role": "user", "content": filler + m[0]["content"]}] for m in copy]
    en_tg = [{"role": "user", "content":
              "Here is part of a Python file:\n\n```python\n" + mid + "\n```\n\n"
              "Explain what this code does, then write a refactored version of the "
              "argument parsing section with type hints and docstrings."}]
    return {
        # generation-dominated: short realistic context, long answer
        "tg": [{"role": "user", "content": RU_TG_PROMPT}] if lang == "ru" else en_tg,
        "copy": copy,
        "free": free,
        "deep": deep,
        # prefill-dominated: the whole file, answer of length 1
        "prefill": [{"role": "user", "content":
                     "Here is a Python file:\n\n```python\n" + src + "\n```\n\nReply with the single word OK."}],
    }


# ---------------------------------------------------------------- server

class Server:
    def __init__(self, extra_args, log_path, launch_args=None, env=None):
        self.extra = extra_args
        self.log_path = log_path
        self.launch = launch_args or []
        self.env = env or {}
        self.lines = []
        self.proc = None
        self._fh = None

    def start(self):
        cmd = [sys.executable, os.path.join(WS, "start_server.py")] + self.launch + self.extra
        self._fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        if self.env:
            self._fh.write("[+] env: " + " ".join(f"{k}={v}" for k, v in self.env.items()) + "\n")
        self.proc = subprocess.Popen(
            cmd, cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, **self.env},
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        threading.Thread(target=self._pump, daemon=True).start()
        return cmd

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            self._fh.write(line)
            self._fh.flush()

    def executing_line(self):
        for i, l in enumerate(self.lines):
            if "[+] Executing:" in l:
                return l.split("[+] Executing:", 1)[1].strip()
        return ""

    def wait_ready(self, timeout=420, warmup_grace=180):
        """
        Ready = the launcher's warmup prefill has finished, so the expert pages
        are already faulted in and the first scored request is not paying for it.

        Gated on the log rather than on /health, whose presence in this build is
        not established. If warmup never reports (it runs on a background thread
        and could fail on its own), fall back to "listening" plus a grace period
        rather than hanging the whole sweep.
        """
        t0 = time.time()
        listening_at = None
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early, rc={self.proc.returncode}")
            joined = self.lines
            if any("[*] Warmup:" in l for l in joined):
                time.sleep(1.0)
                return time.time() - t0
            if listening_at is None and any("listening on" in l for l in joined):
                listening_at = time.time()
            if listening_at and time.time() - listening_at > warmup_grace:
                self.warmup_missing = True
                return time.time() - t0
            time.sleep(1.0)
        raise TimeoutError("server did not become ready in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        for _ in range(30):
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        if self._fh:
            self._fh.close()
        # llama.exe releases VRAM a moment after the handle closes
        time.sleep(3)


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------- requests

def chat(messages, max_tokens, timeout=900, alias="Tiel-Coder-35B-A3B"):
    body = json.dumps({
        "model": alias,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    return time.time() - t0, payload


def parse_block(lines):
    """Pull the timing numbers out of the log lines produced by one request."""
    out = {}
    for l in lines:
        m = RE_PROMPT.search(l)
        if m:
            out["pp_ms"], out["pp_tokens"], out["pp_tps"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
            continue
        m = RE_EVAL.search(l)
        if m and "prompt" not in l:
            out["tg_ms"], out["tg_tokens"], out["tg_tps"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
            continue
        m = RE_TOTAL.search(l)
        if m:
            out["total_ms"] = float(m.group(1))
    spec = [l.strip() for l in lines if RE_DRAFT.search(l) and "print_timing" in l]
    if spec:
        out["spec_lines"] = spec
    return out


def measure(srv, prompts, repeats=2, alias="Tiel-Coder-35B-A3B", tests=("tg", "floor", "prefill")):
    """
    Order and repeat counts are deliberate, because the slot's KV cache makes a
    naive "run it twice" wrong:

    - tg     : same prompt twice, score the second. The prompt is then a full
               cache hit, so prefill drops out and the number is pure generation.
    - floor  : several short follow-ups, each with a *different* suffix so every
               one carries genuinely new tokens, score the last. Repeating an
               identical follow-up would hit the cache and measure nothing.
    - prefill: exactly ONCE, and last. A second identical run would reuse the KV
               (f_sim = 1.0) and report a prefill that never happened. It also
               goes last because its big unrelated prompt evicts the tg context.
    """
    res = {}

    # Each of the three is recorded independently. The first version let one
    # failure abort the whole configuration, so a prefill prompt that overran the
    # context threw away the tg and floor numbers that had already been measured
    # correctly -- six configurations of wall time for nothing.
    def scored(label, fn):
        try:
            res[label] = fn()
        except Exception as e:
            res[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  !! {label}: {res[label]['error']}", flush=True)

    def run_tg():
        block = None
        for _ in range(repeats):
            mark = len(srv.lines)
            wall, _ = chat(prompts["tg"], 400, alias=alias)
            block = parse_block(srv.lines[mark:])
            block["wall_s"] = round(wall, 2)
        return block

    def run_floor():
        block = None
        for i in range(repeats + 1):
            follow = prompts["tg"] + [
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content":
                 f"Now restate point number {i + 1} of your answer in one short sentence, "
                 f"and mention the identifier QX{i:02d}{i * 7:03d} verbatim."}]
            mark = len(srv.lines)
            wall, _ = chat(follow, 1, alias=alias)
            block = parse_block(srv.lines[mark:])
            block["wall_s"] = round(wall, 2)
        return block

    def run_prefill():
        mark = len(srv.lines)
        wall, _ = chat(prompts["prefill"], 1, alias=alias)
        block = parse_block(srv.lines[mark:])
        block["wall_s"] = round(wall, 2)
        return block

    # No. 16: a different excerpt per repeat (see load_prompts), the last one is
    # scored. The reply's hash tells whether speculation changed the output.
    def excerpts(key, max_tokens):
        def run():
            block = None
            for msgs in prompts[key]:
                mark = len(srv.lines)
                wall, payload = chat(msgs, max_tokens, alias=alias)
                block = parse_block(srv.lines[mark:])
                block["wall_s"] = round(wall, 2)
                # Tiel thinks first: within 1000 tokens the copy lands in reasoning_content
                msg = payload["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + "\n\x00\n" + (msg.get("content") or "")
                block["reply_md5"] = hashlib.md5(text.encode("utf-8")).hexdigest()
                block["reply_head"] = text[:120]
                block["n_prompt"] = (payload.get("usage") or {}).get("prompt_tokens")
                block["restores"] = sum("restore checkpoint" in l for l in srv.lines[mark:])
                block["rounds"] = sum("draft tokens" in l and "accepted" in l for l in srv.lines[mark:])
            return block
        return run

    for label, fn in (("tg", run_tg), ("floor", run_floor), ("free", excerpts("free", 400)),
                      ("copy", excerpts("copy", 1000)), ("deep", excerpts("deep", 600)),
                      ("prefill", run_prefill)):
        if label in tests:
            scored(label, fn)

    spec = [l for l in srv.lines if RE_DRAFT.search(l)]
    if spec:
        res["spec_log_tail"] = spec[-12:]
    return res


# ---------------------------------------------------------------- sweeps

MODELS = {
    "tiel": {
        "alias": "Tiel-Coder-35B-A3B",
        "lang": "en",
        # -c 131072 and q8_0 KV are what every existing journal row was taken at,
        # so they stay pinned here or the old numbers stop being comparable.
        "launch": ["-m", "tiel", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "131072"],
    },
    "gemma": {
        "alias": "gemma-4-26B-A4B-it",
        "lang": "ru",
        # Context and KV type come from the launcher's own Gemma defaults
        # (16384, f16) so a sweep measures what the model actually ships with.
        "launch": ["-m", "gemma"],
        # ~6k tokens of English source: three full ubatches at ub 2048, enough to
        # reach steady state, and it still fits the 8192 window that gemma-ctx
        # uses at its narrowest.
        "prefill_chars": 24000,
    },
}


# Own llama.cpp fork for Tiel: llama-tiel/ (branch tiel), b11338 dcd387a41 +
# expert prefetch patch 4f66802e1, snapshot copied out of the build tree.
TIEL_FORK_BIN = os.path.join(WS, "bin", "tiel-fork-4f66802e1", "llama.exe")
# hypothesis No. 15: adaptive hot experts in VRAM (fork branch probe/no15-hot-experts)
TIEL_FORK_NO15_BIN = os.path.join(WS, "bin", "tiel-fork-no15-f9fcb16ab", "llama.exe")
# The same fork + early-router L3 prefetch of CPU experts (hypothesis No. 4),
# 4143919d6 + 784c24565 + 7bad047d0 on top of 4f66802e1 (patches/tiel-moe-l3-prefetch.patch);
# LLAMA_MOE_PREFETCH unset = previous behaviour.
TIEL_FORK_L3_BIN = os.path.join(WS, "bin", "tiel-fork-7bad047d0", "llama.exe")
# 4f66802e1 + diagnostics for hypothesis No. 12 (fork branch probe/no12-layer-centric):
# GGML_SCHED_PREFETCH_PROBE uploads expert weights only in the first prefetch graph
# (the output is wrong), LLAMA_UBATCH_TIMING logs host time per ubatch phase.
TIEL_FORK_PROBE12_BIN = os.path.join(WS, "bin", "tiel-fork-probe12-e9ad0170e", "llama.exe")
# The same branch with layer-centric batches (hypothesis No. 12): LLAMA_LAYER_CENTRIC unset = previous behaviour.
TIEL_FORK_LC_BIN = os.path.join(WS, "bin", "tiel-fork-lc-8ad60f8de", "llama.exe")
# + GGML_SCHED_LAZY_REPLAN: a graph whose allocation plan fits the current buffers does not wait for the previous one
TIEL_FORK_LC2_BIN = os.path.join(WS, "bin", "tiel-fork-lc-5152b82df", "llama.exe")
# llama.cpp takes the last -b, so this overrides the launcher's -b 4096
B32K = ["--llama-arg=-b", "--llama-arg=32768"]
UB2048 = ["--llama-arg=-ub", "--llama-arg=2048"]
F16KV = ["-ctk", "f16", "-ctv", "f16"]  # overrides the profile's q8_0: argparse keeps the last
# 4f66802e1 + c2ead838e: n-gram speculation rolls back through recurrent-state
# snapshots (hypothesis No. 16, fork branch probe/no16-ngram-rs)
TIEL_FORK_NO16_BIN = os.path.join(WS, "bin", "tiel-fork-no16-c2ead838e", "llama.exe")
# c2ead838e + vec FA decode with the Q heads of a GQA group in one block (hypothesis No. 17,
# fork branch probe/no17-fa-vec-gqa): GGML_CUDA_FA_VEC_GQA=<2|4|8>, unset = previous behaviour.
TIEL_FORK_NO17_BIN = os.path.join(WS, "bin", "tiel-fork-no17-e2e6e39ef", "llama.exe")
# The same kernel without the one-off register report on stderr (a6eeaa342) -- the launcher's binary.
TIEL_FORK_NO17_REL_BIN = os.path.join(WS, "bin", "tiel-fork-no17-a6eeaa342", "llama.exe")
# No. 20: a6eeaa342 + hybrid MoE on mid-size batches (LLAMA_MOE_HYBRID=1), branch probe/no20-hybrid;
# without the variable the graph is stock
TIEL_FORK_NO20_BIN = os.path.join(WS, "bin", "tiel-fork-no20-8d87e1a56", "llama.exe")
# No. 26: 8d87e1a56 + chunked GDN prefill (upstream PR #26001), branch probe/no26-chunked-gdn;
# on by default, GGML_CUDA_GDN_CHUNK=0 = stock recurrent kernel
TIEL_FORK_NO26_BIN = os.path.join(WS, "bin", "tiel-fork-no26-303a4cf39", "llama.exe")
# + K > 1 on one sequence (chunked head, recurrent tail for the snapshots): the launcher's ngram config
TIEL_FORK_NO26K_BIN = os.path.join(WS, "bin", "tiel-fork-no26-6e9316a45", "llama.exe")


def NGRAM_MOD(n_min, n_max, n_match=16):
    return ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", str(n_match),
            "--spec-ngram-mod-n-min", str(n_min), "--spec-ngram-mod-n-max", str(n_max)]

SWEEPS = {
    # G2: how many expert layers are worth moving to the GPU. The launcher picks
    # a number from its VRAM plan; this checks the plan against reality, in both
    # directions -- fewer layers than planned, and more than planned until it
    # spills. 30 = every expert layer on the CPU, the Tiel-style configuration.
    "gemma-ncmoe": [
        ("auto", []),
        ("ncmoe30", ["--n-cpu-moe", "30"]),
        ("ncmoe28", ["--n-cpu-moe", "28"]),
        ("ncmoe26", ["--n-cpu-moe", "26"]),
        ("ncmoe24", ["--n-cpu-moe", "24"]),
        ("ncmoe22", ["--n-cpu-moe", "22"]),
    ],
    # G1, speed half only. Whether q8_0 costs quality is a separate question that
    # throughput cannot answer; this measures what it buys, so the trade is known.
    "gemma-kv": [
        ("f16", ["-ctk", "f16", "-ctv", "f16"]),
        ("q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"]),
        ("q4_0", ["-ctk", "q4_0", "-ctv", "q4_0"]),
    ],
    # G4. The Gemma draft head is a separate 225 MiB dense model with no expert
    # tensors at all, so drafting is free of CPU memory traffic -- unlike Tiel,
    # where the head was blk.40 of the MoE itself. Its metadata declares
    # nextn_predict_layers = 4, so 4 is the designed depth rather than an
    # autoregressive stretch, and the sweep brackets it.
    "gemma-mtp": [
        ("A-baseline", []),
        ("nmax1", ["--mtp", "--mtp-n-max", "1"]),
        ("nmax2", ["--mtp", "--mtp-n-max", "2"]),
        ("nmax3", ["--mtp", "--mtp-n-max", "3"]),
        ("nmax4", ["--mtp", "--mtp-n-max", "4"]),
        ("nmax6", ["--mtp", "--mtp-n-max", "6"]),
        ("Z-baseline-repeat", []),
    ],
    # H4 did not transfer from arithmetic on Tiel and there is no reason to
    # assume it transfers across models either. Same mask everywhere so the
    # thread count is the only variable.
    "gemma-threads": [
        ("t6-mask555", []),
        ("t4-maskfff", ["--cpu-mask", "fff", "-t", "4"]),
        ("t6-maskfff", ["--cpu-mask", "fff"]),
        ("t8-maskfff", ["--cpu-mask", "fff", "-t", "8"]),
        # No t12 row. Tiel's sweep has one and it merely ran badly (19.31 t/s);
        # here the same flags reset the display driver mid-measurement, because
        # Gemma keeps 7 expert layers on the GPU and so needs the GPU scheduler
        # to get CPU time on every token. start_server.py now refuses strict
        # pinning above the physical core count, which would defuse it anyway,
        # but a configuration already measured as the worst on this machine is
        # not worth the risk of re-running.
        ("t6-mask555-repeat", []),
    ],
    # G6. Context costs VRAM that would otherwise hold expert layers, so the
    # window is a speed knob here in a way it never was on Tiel.
    "gemma-ctx": [
        ("c16384", ["-c", "16384"]),
        ("c8192", ["-c", "8192"]),
        ("c32768", ["-c", "32768"]),
        ("c65536", ["-c", "65536"]),
        ("c131072", ["-c", "131072"]),
        ("c16384-repeat", ["-c", "16384"]),
    ],
    # The ubatch/expert-layer trade the launcher currently resolves in favour of
    # expert layers. Each doubling above 2048 costs ~636 MiB of compute buffer,
    # which is 1.5 layers. This is the measurement that says whether that policy
    # is right, and it is the one place prefill and generation pull opposite ways.
    "gemma-ub": [
        ("ub2048", ["-ub", "2048"]),
        ("ub1024", ["-ub", "1024"]),
        ("ub4096", ["-ub", "4096"]),
        ("ub6144", ["-ub", "6144"]),
    ],
    # gemma-ncmoe found the frontier but not its shape: 6 layers at ub 2048 runs
    # 33.2 t/s with prefill intact at 800, and 8 layers runs 36.5 t/s with prefill
    # collapsed to 197 because the staging buffer no longer fits and WDDM pages it
    # over PCIe. The interesting question is whether a smaller ubatch buys back
    # the room for the 7th and 8th layer without giving up bulk prefill -- 1024
    # frees 318 MiB, which is most of a layer. Peak VRAM for each row, from the
    # calibrated model:
    #   ncmoe24 ub2048  5478    ncmoe23 ub2048  5887    ncmoe22 ub2048  6295 spill
    #   ncmoe24 ub1024  5160    ncmoe23 ub1024  5569    ncmoe22 ub1024  5977
    "gemma-frontier": [
        ("L6-ub2048", ["--n-cpu-moe", "24", "-ub", "2048"]),
        ("L7-ub2048", ["--n-cpu-moe", "23", "-ub", "2048"]),
        ("L7-ub1024", ["--n-cpu-moe", "23", "-ub", "1024"]),
        ("L8-ub1024", ["--n-cpu-moe", "22", "-ub", "1024"]),
        ("L8-ub512", ["--n-cpu-moe", "22", "-ub", "512"]),
        ("L6-ub2048-repeat", ["--n-cpu-moe", "24", "-ub", "2048"]),
    ],
    # G18. Expert prefetch from the Tiel fork on Gemma: at -ncmoe 23 every ubatch
    # uploads 23 x 408.4 MiB of experts with no overlap, est. ~60 % of a ubatch.
    # The slot is sized from the model's largest expert tensor (ffn_gate_up_exps,
    # 272.25 MiB) against ~258 MiB free, so the -ncmoe 24 rows give one expert
    # layer back to make room. The fork is b11338; b11099 lost 14-17 % generation
    # on Gemma's hybrid layout, so pinned 10826 opens and closes the sweep and the
    # fork's own stock path has a row of its own.
    "gemma-prefetch": [
        ("A-pinned-10826", [], {}),
        ("fork-stock", ["--llama-bin", TIEL_FORK_BIN], {}),
        ("fork-pf1", ["--llama-bin", TIEL_FORK_BIN], {"GGML_SCHED_PREFETCH": "1"}),
        ("fork-pf1-L6", ["--llama-bin", TIEL_FORK_BIN, "--n-cpu-moe", "24"], {"GGML_SCHED_PREFETCH": "1"}),
        ("fork-pf2-L6", ["--llama-bin", TIEL_FORK_BIN, "--n-cpu-moe", "24"], {"GGML_SCHED_PREFETCH": "2"}),
        ("fork-stock-L6", ["--llama-bin", TIEL_FORK_BIN, "--n-cpu-moe", "24"], {}),
        ("Z-pinned-10826", [], {}),
    ],
    # G20. The op offload threshold that cut Tiel's short turn by 41 % (H2-01..06),
    # on Gemma. 23 of Gemma's 30 expert layers live in RAM and take the same
    # upload-then-compute path for batches >= 32; the launcher sets nothing for
    # Gemma, so every row pins the threshold explicitly. Stock 10826 throughout.
    "gemma-offload": [
        ("A-minb32", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
        ("minb64", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "64"}),
        ("minb128", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "128"}),
        ("minb256", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "256"}),
        ("Z-minb32", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
    ],
    # G20 into the launcher: start_server.py now sets 48 for Gemma (crossover at
    # ~47 new tokens, g20-crossover-fine). The launcher row passes no env.
    "gemma-offload-launch": [
        ("launcher-default", [], {}),
        ("A-minb32", [], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
    ],
    # G5. Tiel could never use --mlock:21.7 GiB pinned in a 28 GiB system killed
    # the server part-way through a long prompt. Gemma needs ~12 GiB host-side
    # once the GPU takes its share, so the option exists again.
    "gemma-mlock": [
        ("loadmode-none", ["--load-mode", "none"]),
        ("mlock", ["--mlock"]),
        ("loadmode-none-repeat", ["--load-mode", "none"]),
    ],
    "mtp-pmin": [
        ("A-baseline", ["--no-spec"]),
        ("B-pmin0.0", ["--mtp", "--mtp-n-max", "1", "--mtp-p-min", "0.0"]),
        ("C-pmin0.6", ["--mtp", "--mtp-n-max", "1", "--mtp-p-min", "0.6"]),
        ("D-pmin0.8", ["--mtp", "--mtp-n-max", "1", "--mtp-p-min", "0.8"]),
        ("E-pmin0.9", ["--mtp", "--mtp-n-max", "1", "--mtp-p-min", "0.9"]),
    ],
    # H5. The mask change is isolated in its own step so that a tb result is not
    # really a mask result: the auto mask is 555 (6 physical cores), and tb>6
    # would otherwise be meaningless because the extra threads have nowhere to run.
    "threads-tb": [
        ("tb6-mask555", ["--no-spec"]),
        ("tb6-maskfff", ["--no-spec", "--cpu-mask", "fff"]),
        ("tb8-maskfff", ["--no-spec", "--cpu-mask", "fff", "--threads-batch", "8"]),
        ("tb10-maskfff", ["--no-spec", "--cpu-mask", "fff", "--threads-batch", "10"]),
        ("tb12-maskfff", ["--no-spec", "--cpu-mask", "fff", "--threads-batch", "12"]),
    ],
    # H4. Same mask everywhere so thread count is the only variable.
    "threads-t": [
        ("t6-mask555", ["--no-spec"]),
        ("t4-maskfff", ["--no-spec", "--cpu-mask", "fff", "-t", "4"]),
        ("t6-maskfff", ["--no-spec", "--cpu-mask", "fff"]),
        ("t8-maskfff", ["--no-spec", "--cpu-mask", "fff", "-t", "8"]),
        ("t12-maskfff", ["--no-spec", "--cpu-mask", "fff", "-t", "12"]),
    ],
    # Hypothesis 2 of docs/MOE-VIABLE-HYPOTHESES.md. nsys on the server showed
    # prefill is 55 % host-to-device copies: with op offload every MUL_MAT_ID of
    # a batch >= 32 runs on the GPU after the scheduler uploads every expert the
    # ubatch routes to -- 18.2 GiB per 4096-token ubatch, 6.3 GiB for a 39-token
    # follow-up, at the 6.56 GB/s a Gen3 x8 link gives. The CPU does ~66 t/s of
    # prefill on its own, which beats that upload for short batches. The CUDA
    # backend reads its offload threshold from GGML_OP_OFFLOAD_MIN_BATCH (default
    # 32), so short turns can stay on the CPU while bulk ubatches keep the GPU.
    # Third tuple element = environment for the server process. start_server.py
    # now sets 128 for Tiel by default, so the baselines pin the stock 32 to keep
    # this sweep reproducing H2-01..06.
    "h2-offload": [
        ("A-baseline", ["--no-spec"], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
        ("nopo", ["--no-spec", "--llama-arg=--no-op-offload"], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
        ("minb64", ["--no-spec"], {"GGML_OP_OFFLOAD_MIN_BATCH": "64"}),
        ("minb128", ["--no-spec"], {"GGML_OP_OFFLOAD_MIN_BATCH": "128"}),
        ("minb256", ["--no-spec"], {"GGML_OP_OFFLOAD_MIN_BATCH": "256"}),
        ("Z-baseline-repeat", ["--no-spec"], {"GGML_OP_OFFLOAD_MIN_BATCH": "32"}),
    ],
    # Hypothesis No. 7, overlap variant, own fork only (llama-tiel, branch tiel):
    # GGML_SCHED_PREFETCH=<slots> stages the offloaded expert weights in a ring of
    # slots and uploads them on a second stream from a helper thread, so the upload
    # of the next expert tensor runs while the current one computes. Unset = stock
    # path, so A and Z are the same binary. Needs --llama-bin pointing at the fork.
    # A slot is the largest expert tensor, 210 MiB (Q6_K ffn_down), and replaces the
    # staging the graph allocator kept in the compute buffer -- watch the VRAM
    # column: on WDDM an overcommitted card pages instead of failing.
    # Batches under GGML_SCHED_PREFETCH_MIN (default 1024) keep the stock path, so
    # tg and floor must not move; prefill is where the change shows.
    # The pinned 10826 rows open and close the sweep, so the fork's own stock path
    # (b11338 + patch, env unset) is compared with the journal's build in-sweep:
    # Gemma lost 14-17 % generation on b11099, Tiel on newer builds is unmeasured.
    # start_server.py takes the last --llama-bin, so per-config binaries go in args.
    "tiel-prefetch": [
        ("A-pinned-10826", ["--no-spec"], {}),
        ("fork-stock", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {}),
        ("pf2", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {"GGML_SCHED_PREFETCH": "2"}),
        ("pf3", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {"GGML_SCHED_PREFETCH": "3"}),
        ("pf4", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {"GGML_SCHED_PREFETCH": "4"}),
        ("pf1", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {"GGML_SCHED_PREFETCH": "1"}),
        ("fork-stock-repeat", ["--no-spec", "--llama-bin", TIEL_FORK_BIN], {}),
        ("Z-pinned-10826", ["--no-spec"], {}),
    ],
    # The launcher's own default for Tiel (no -ctk) is f16 KV with -ub 2048, and the
    # planner leaves only ~84 MiB of the card free there -- tiel-prefetch measured
    # prefetch on q8_0 / ub 4096 only, where 2 slots cost +204 MiB. Same binaries,
    # f16 KV: does the default launch page, and which slot count still pays.
    "tiel-launcher-f16": [
        ("A-pinned-10826", ["--no-spec", *F16KV], {}),
        ("fork-pf2", ["--no-spec", "--llama-bin", TIEL_FORK_BIN, *F16KV], {"GGML_SCHED_PREFETCH": "2"}),
        ("fork-pf1", ["--no-spec", "--llama-bin", TIEL_FORK_BIN, *F16KV], {"GGML_SCHED_PREFETCH": "1"}),
        ("fork-stock", ["--no-spec", "--llama-bin", TIEL_FORK_BIN, *F16KV], {}),
        ("Z-pinned-10826", ["--no-spec", *F16KV], {}),
    ],
    # Hypothesis No. 4: early-router L3 prefetch of CPU experts during decode.
    # After layer L-1's experts a CPU node wakes helper threads (SMT siblings
    # 1,5,9) that guess layer L's experts from layer L-1's expert input and
    # read them into the cache while the host waits for the GPU. Generation
    # only (batch 1): tg is the row that matters, floor and prefill must not
    # move. The gain llama bench saw is ~1.7-2 ms of experts per token minus
    # the wake-up cost, so off/on alternate three times to see it through the
    # within-sweep TG noise; 10826 opens and closes as the yardstick.
    "tiel-l3": [
        ("A-pinned-10826", ["--no-spec"], {}),
        ("fork-off-1", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {}),
        ("l3-m4-1", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {"LLAMA_MOE_PREFETCH": "4"}),
        ("fork-off-2", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {}),
        ("l3-m4-2", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {"LLAMA_MOE_PREFETCH": "4"}),
        ("fork-off-3", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {}),
        ("l3-m4-3", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], {"LLAMA_MOE_PREFETCH": "4"}),
        ("Z-pinned-10826", ["--no-spec"], {}),
    ],
    # More off/on pairs of the same server, tg only (run with --tests tg): one tg
    # row is a single 400-token run, and the effect (~+3 %) is the size of the
    # within-sweep TG noise, so tiel-l3's three pairs could not settle it.
    "tiel-l3-ab": [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN], env)
        for i in range(1, 7)
        for tag, env in (("off", {}), ("l3m4", {"LLAMA_MOE_PREFETCH": "4"}))
    ],
    # The same server path with the patch's per-layer timing on (stderr): proves
    # the prefetch runs under llama-server and shows window / expert times there.
    # STATS adds a mark node per layer, so these TG numbers are not comparable
    # with tiel-l3; run with --tests tg.
    # Hypothesis No. 12, step 1: ceiling of layer-centric prefill. Layer-centric
    # uploads each layer's experts once per prompt instead of once per ubatch; its
    # best case is the prefetch path with no uploads at all after the first ubatch,
    # which is what the probe does (the answer is garbage, the timing is not: the
    # slots keep real expert weights of the last uploaded layer). Run with
    # --tests prefill. The timing log shows the per-graph host cost that layer-
    # centric would pay ~40 times per ubatch instead of once.
    "tiel-lc-probe": [
        (f"{tag}-{rep}", ["--no-spec", "--llama-bin", TIEL_FORK_PROBE12_BIN],
         {"GGML_SCHED_PREFETCH": k, "LLAMA_UBATCH_TIMING": "1", **({"GGML_SCHED_PREFETCH_PROBE": "1"} if probe else {})})
        for rep in (1, 2)
        for tag, k, probe in (("pf3", "3", False), ("pf3-noupload", "3", True),
                              ("pf2", "2", False), ("pf2-noupload", "2", True))
    ],
    # Hypothesis No. 12 itself: LLAMA_LAYER_CENTRIC=<layers per pass> runs every
    # ubatch of a llama_decode batch through one layer before the next, so a slot
    # of the prefetch ring keeps the layer's expert tensor for all ubatches and it
    # crosses the bus once per batch. The server hands llama_decode at most -b
    # tokens, so the rows that test it raise -b to 32768 (the 28K prefill becomes
    # one batch of 7 ubatches). b32768 without the env var separates the effect of
    # -b itself. A slot must hold a whole layer (gate, up, down), so K=3.
    "tiel-lc": [
        ("pf3", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN], {"GGML_SCHED_PREFETCH": "3"}),
        ("pf3-b32k", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN, *B32K], {"GGML_SCHED_PREFETCH": "3"}),
        ("lc1-pf3-b32k", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN, *B32K],
         {"GGML_SCHED_PREFETCH": "3", "LLAMA_LAYER_CENTRIC": "1"}),
        ("lc1-pf2-b32k", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN, *B32K],
         {"GGML_SCHED_PREFETCH": "2", "LLAMA_LAYER_CENTRIC": "1"}),
        ("lc1-pf3-b32k-repeat", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN, *B32K],
         {"GGML_SCHED_PREFETCH": "3", "LLAMA_LAYER_CENTRIC": "1"}),
        ("pf3-repeat", ["--no-spec", "--llama-bin", TIEL_FORK_LC_BIN], {"GGML_SCHED_PREFETCH": "3"}),
    ],
    # tiel-lc found layer-centric slower than plain K=3 prefetch: every pass over an
    # attention layer waited for the GPU in the allocator (the graph shape changes),
    # then built the KQ mask with the GPU idle. Lazy replan skips that wait when the
    # plan fits the buffers. Interleaved pairs, same binary; lazy replan is also
    # tried alone since it touches every graph.
    "tiel-lc2": [
        ("pf3-1", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN], {"GGML_SCHED_PREFETCH": "3"}),
        ("lc1-lazy-1", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN, *B32K],
         {"GGML_SCHED_PREFETCH": "3", "LLAMA_LAYER_CENTRIC": "1", "GGML_SCHED_LAZY_REPLAN": "1"}),
        ("pf3-lazy", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN], {"GGML_SCHED_PREFETCH": "3", "GGML_SCHED_LAZY_REPLAN": "1"}),
        ("lc1-lazy-2", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN, *B32K],
         {"GGML_SCHED_PREFETCH": "3", "LLAMA_LAYER_CENTRIC": "1", "GGML_SCHED_LAZY_REPLAN": "1"}),
        ("pf3-2", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN], {"GGML_SCHED_PREFETCH": "3"}),
    ],
    # tiel-lc2 compared layer-centric (-b 32768) with plain K=3 at -b 4096, but -b
    # alone changes how the server cuts the prompt (its checkpoint breaks), and a
    # reference run of plain K=3 at -b 32768 was as fast as layer-centric. Same -b,
    # same lazy replan, interleaved three times; run with --tests prefill.
    "tiel-lc3": [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_LC2_BIN, *B32K], env)
        for i in (1, 2, 3)
        for tag, env in (("pf3-b32k-lazy", {"GGML_SCHED_PREFETCH": "3", "GGML_SCHED_LAZY_REPLAN": "1"}),
                         ("lc1-b32k-lazy", {"GGML_SCHED_PREFETCH": "3", "LLAMA_LAYER_CENTRIC": "1", "GGML_SCHED_LAZY_REPLAN": "1"}))
    ],
    # Hypothesis No. 15, step 1 (filter): a hot set in VRAM pays only if the CPU
    # part of a layer gets cheaper in proportion to the experts it no longer reads.
    # Fewer experts per token is the same CPU work without a patch: k=7 / k=6 read
    # 1/8 and 2/8 fewer bytes. Output changes, so tg only (--tests tg); linear
    # would be 17.5 / 8 = 2.19 ms per dropped expert (step-0 profile).
    "tiel-no15-k": [
        (f"k{k}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_BIN,
                       *([] if k == 8 else ["--llama-arg=--override-kv", f"--llama-arg=qwen35moe.expert_used_count=int:{k}"])], {})
        for i in (1, 2, 3)
        for k in (8, 7, 6)
    ],
    # Hypothesis No. 15 itself: LLAMA_MOE_HOT=<slots per layer> keeps the experts
    # most used in the last LLAMA_MOE_HOT_WINDOW decode tokens in VRAM (6 slots =
    # 440 MiB, 8 = 586 MiB); the GPU computes them, the CPU skips them. Decode only,
    # so tg only (--tests tg). -ub 2048 in every row frees ~630 MiB of compute
    # buffer for the set (generation does not depend on -ub). Same binary off/on.
    "tiel-no15-hot": [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_NO15_BIN, *UB2048],
         {**env, "LLAMA_MOE_HOT_STATS": "1"} if env else {})
        for i in (1, 2, 3)
        for tag, env in (("off", {}),
                         ("hot6", {"LLAMA_MOE_HOT": "6"}),
                         ("hot8", {"LLAMA_MOE_HOT": "8"}),
                         ("hot6-w128", {"LLAMA_MOE_HOT": "6", "LLAMA_MOE_HOT_WINDOW": "128"}))
    ],
    # Hypothesis No. 16, step 1 (filter, no build): draftless n-gram speculation.
    # Short drafts on purpose: verifying k drafted tokens reads the union of their
    # experts from DDR4. Without the n_rs_seq patch every draft round snapshots the
    # recurrent state (62.8 MiB) and every partial accept restores it and replays
    # the accepted tokens -- this sweep measures that stock path. LLAMA_TRACE=1
    # logs one line per round ("accepted x/y", "(restore checkpoint)"); set on
    # every row so the baseline pays the same. Run with --tests free,copy.
    "tiel-no16": [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_BIN, *[f"--llama-arg={a}" for a in spec]],
         {"LLAMA_TRACE": "1"})
        for i in (1, 2)
        for tag, spec in (
            ("off", []),
            ("mod-m16-x4", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "16",
                            "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "4"]),
            ("mod-m16-x8", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "16",
                            "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "8"]),
            ("mod-m16-x16", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "16",
                             "--spec-ngram-mod-n-min", "4", "--spec-ngram-mod-n-max", "16"]),
            ("mapk-n12-m8", ["--spec-type", "ngram-map-k", "--spec-ngram-map-k-size-n", "12",
                             "--spec-ngram-map-k-size-m", "8", "--spec-ngram-map-k-min-hits", "1"]),
        )
    ],
    # No. 16, step 2: the same n-gram drafts with rollback through in-VRAM snapshots
    # of the recurrent state (fork branch probe/no16-ngram-rs: need_n_rs_seq() covers
    # the n-gram types, n_rs_seq = n_max). Each snapshot is 62.8 MiB, n-max 16 adds
    # ~1 GiB, so every row runs -ub 2048 (frees ~630 MiB of compute buffer; decode
    # does not depend on -ub). Stock-binary rows of the same draft settings
    # interleave with the patched ones; run with --tests free,copy.
    "tiel-no16-rs": [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", binary, *UB2048, *[f"--llama-arg={a}" for a in spec]],
         {"LLAMA_TRACE": "1"})
        for i in (1, 2)
        for tag, binary, spec in (
            ("off", TIEL_FORK_NO16_BIN, []),
            ("x4-stock", TIEL_FORK_BIN, NGRAM_MOD(1, 4)),
            ("x4-rs", TIEL_FORK_NO16_BIN, NGRAM_MOD(1, 4)),
            ("x8-rs", TIEL_FORK_NO16_BIN, NGRAM_MOD(2, 8)),
            ("x16-stock", TIEL_FORK_BIN, NGRAM_MOD(4, 16)),
            ("x16-rs", TIEL_FORK_NO16_BIN, NGRAM_MOD(4, 16)),
        )
    ],
    # No. 16: what the snapshots cost off the decode path. n_rs_seq > 0 keeps the
    # last n_rs_seq + 1 tokens of a sequence in one ubatch and writes per-token
    # states; floor and prefill against off, with expert prefetch on as the
    # launcher would run it. pf2 on top of 16 snapshots is the VRAM edge (paging?).
    # Run with --tests floor,prefill.
    "tiel-no16-rs-pp": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO16_BIN, *UB2048, *[f"--llama-arg={a}" for a in spec]],
         {"GGML_SCHED_PREFETCH": pf})
        for tag, spec, pf in (
            ("off-pf1", [], "1"),
            ("x8-rs-pf1", NGRAM_MOD(2, 8), "1"),
            ("x16-rs-pf1", NGRAM_MOD(4, 16), "1"),
            ("x16-rs-pf2", NGRAM_MOD(4, 16), "2"),
            ("off-pf2", [], "2"),
            ("off-pf1-repeat", [], "1"),
        )
    ],
    # NG1: snapshots in the working configuration (q8_0, -ub 4096, prefetch K=2),
    # where the planner leaves ~236 MiB after the two slots. n-max 4 = 4 x 62.8 MiB
    # is at that edge, n-max 8 past it -- a paging card shows as a prefill collapse
    # (PF-05: 143 t/s), not as an error. K=1 frees 210 MiB as the fallback.
    # Run with --tests floor,free,copy,prefill.
    "tiel-ng1": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO16_BIN, *[f"--llama-arg={a}" for a in spec]],
         {"GGML_SCHED_PREFETCH": pf, "LLAMA_TRACE": "1"})
        for tag, spec, pf in (
            ("off-pf2-1", [], "2"),
            ("x4-rs-pf2-1", NGRAM_MOD(1, 4), "2"),
            ("x8-rs-pf2", NGRAM_MOD(2, 8), "2"),
            ("x4-rs-pf1", NGRAM_MOD(1, 4), "1"),
            ("off-pf2-2", [], "2"),
            ("x4-rs-pf2-2", NGRAM_MOD(1, 4), "2"),
        )
    ],
    # NG1 on the launcher's own Tiel default: f16 KV, -b/-ub 2048, where the planner
    # leaves ~84 MiB and the launcher picks K=1 (LF-03). n-max 4 snapshots (251 MiB)
    # do not fit that headroom on paper; n-max 2 (126 MiB) is the smaller step, and
    # snapshots without prefetch (K=0) give the 210-MiB-free fallback.
    # Run with --tests floor,free,copy,prefill.
    "tiel-ng1-f16": [
        # the launcher itself picks -b/-ub 2048 for f16 (same launch line as LF-03)
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO16_BIN, *F16KV, *[f"--llama-arg={a}" for a in spec]],
         {**({"GGML_SCHED_PREFETCH": pf} if pf else {}), "LLAMA_TRACE": "1"})
        for tag, spec, pf in (
            ("off-pf1-1", [], "1"),
            ("x4-rs-pf1-1", NGRAM_MOD(1, 4), "1"),
            ("x2-rs-pf1", NGRAM_MOD(1, 2), "1"),
            ("x4-rs-pf0", NGRAM_MOD(1, 4), None),
            ("off-pf1-2", [], "1"),
            ("x4-rs-pf1-2", NGRAM_MOD(1, 4), "1"),
        )
    ],
    # The f16 rows of tiel-ng1-f16 ran on ~3K tokens of context, and n-max 4 fit
    # only on paper-overcommit (84 MiB headroom, 251 MiB of snapshots). Same rows
    # at ~85K tokens: does a written KV cache start paging? Run with --tests deep;
    # the deep reply also logs its prompt size. q8_0 pair for comparison.
    "tiel-ng1-deep": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO16_BIN, *kv, *[f"--llama-arg={a}" for a in spec]],
         {"GGML_SCHED_PREFETCH": pf, "LLAMA_TRACE": "1"})
        for tag, kv, spec, pf in (
            ("f16-off-pf1", F16KV, [], "1"),
            ("f16-x4-rs-pf1", F16KV, NGRAM_MOD(1, 4), "1"),
            ("q8-off-pf2", [], [], "2"),
            ("q8-x4-rs-pf2", [], NGRAM_MOD(1, 4), "2"),
        )
    ],
    # No. 17: q8_0 FA decode re-reads K/V per Q head (vec kernel, 68 MiB/ms at 65K);
    # GGML_CUDA_FA_VEC_GQA groups the heads. TG at ~90K on the server, --tests deep.
    # f16 row = the launcher's previous default, for the KV1 comparison.
    "tiel-no17-deep": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO17_BIN, *kv, *[f"--llama-arg={a}" for a in spec]],
         {"GGML_SCHED_PREFETCH": pf, "LLAMA_TRACE": "1", **({"GGML_CUDA_FA_VEC_GQA": g} if g else {})})
        for tag, kv, spec, pf, g in (
            ("q8-off-1", [], [], "2", None),
            ("q8-gqa8-1", [], [], "2", "8"),
            ("q8-gqa4", [], [], "2", "4"),
            ("f16-off-pf1", F16KV, [], "1", None),
            ("q8-x4-rs-gqa8", [], NGRAM_MOD(1, 4), "2", "8"),
            ("q8-off-2", [], [], "2", None),
            ("q8-gqa8-2", [], [], "2", "8"),
        )
    ],
    # No. 17 into the launcher default "if nothing is lost": the launcher's own Tiel
    # config (q8_0, ub 4096, K=2, ngram-mod n-max 4 with snapshots), grouping off/on
    # interleaved. Run with --tests floor,tg,free,copy,deep,prefill.
    "tiel-no17-launch": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO17_REL_BIN, *[f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)]],
         {"GGML_SCHED_PREFETCH": "2", "LLAMA_TRACE": "1", **({"GGML_CUDA_FA_VEC_GQA": g} if g else {})})
        for tag, g in (("off-1", None), ("gqa8-1", "8"), ("off-2", None), ("gqa8-2", "8"))
    ],
    # No. 20: hybrid MoE on mid-size batches in the launcher's config (q8_0, ub 4096, K=2, ngram-mod
    # n-max 4, grouped FA, threshold 184), off/on interleaved; the variable leaves batches below 32 and
    # from 1024 tokens on alone. Run with --tests floor,tg,free,copy,deep,prefill.
    "tiel-no20-launch": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO20_BIN, *[f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)]],
         {"GGML_SCHED_PREFETCH": "2", "LLAMA_TRACE": "1", "GGML_CUDA_FA_VEC_GQA": "8", "GGML_OP_OFFLOAD_MIN_BATCH": "184",
          **({"LLAMA_MOE_HYBRID": "1"} if h else {})})
        for tag, h in (("off-1", False), ("hyb-1", True), ("off-2", False), ("hyb-2", True))
    ],
    # No. 26: chunked GDN prefill kernel from upstream PR #26001 (fork branch probe/no26-chunked-gdn).
    # The kernel takes K == 1 only, and ngram snapshots make K = n_rs_seq + 1 = 5, so the rows run
    # without speculation; otherwise the launcher's config. GGML_CUDA_GDN_CHUNK=0 is the stock path.
    # Run with --tests floor,tg,prefill.
    "tiel-no26": [
        (tag, ["--no-spec", "--llama-bin", TIEL_FORK_NO26_BIN],
         {"GGML_SCHED_PREFETCH": "2", "GGML_CUDA_FA_VEC_GQA": "8", "GGML_OP_OFFLOAD_MIN_BATCH": "184",
          "LLAMA_MOE_HYBRID": "1", "GGML_CUDA_GDN_CHUNK": c})
        for tag, c in (("off-1", "0"), ("chunk-1", "1"), ("off-2", "0"), ("chunk-2", "1"))
    ],
    # No. 26 in the launcher's own config: ngram-mod n-max 4 with snapshots (K = 5), which the
    # 6e9316a45 build covers. The first row only warms the page cache (tiel-no26 off-1 ran cold).
    # Run with --tests floor,tg,prefill.
    "tiel-no26-launch": [
        ("warm", ["--no-spec", "--llama-bin", TIEL_FORK_NO26K_BIN], {"GGML_SCHED_PREFETCH": "2"}),
    ] + [
        (f"{tag}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_NO26K_BIN, *[f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)]],
         {"GGML_SCHED_PREFETCH": "2", "GGML_CUDA_FA_VEC_GQA": "8", "GGML_OP_OFFLOAD_MIN_BATCH": "184",
          "LLAMA_MOE_HYBRID": "1", "LLAMA_TRACE": "1", "GGML_CUDA_GDN_CHUNK": c})
        for i in (1, 2, 3)
        for tag, c in (("off", "0"), ("chunk", "1"))
    ],
    "tiel-l3-stats": [
        ("nodes-only-stats", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN],
         {"LLAMA_MOE_PREFETCH": "4", "LLAMA_MOE_PREFETCH_MODE": "3", "LLAMA_MOE_PREFETCH_STATS": "1"}),
        ("l3-m4-stats", ["--no-spec", "--llama-bin", TIEL_FORK_L3_BIN],
         {"LLAMA_MOE_PREFETCH": "4", "LLAMA_MOE_PREFETCH_STATS": "1"}),
    ],
    # Flash attention on/off. Without FA llama.cpp refuses a quantized V cache and materializes
    # KQ as f32 [n_kv, n_ubatch, n_head]: at 131K / ub 2048 that is a 17.6 GiB compute buffer, so
    # Tiel runs where both fit: -c 32768, -b/-ub 512, f16 KV, the launcher's env. Interleaved.
    # Run with --tests floor,tg,prefill.
    "tiel-fa": [
        (f"fa{fa}-{i}", ["--no-spec", "--llama-bin", TIEL_FORK_NO20_BIN, "--cache-type", "f16", "-fa", fa,
                         *[f"--llama-arg={a}" for a in ("-c", "32768", "-b", "512", "-ub", "512",
                                                        "-ctk", "f16", "-ctv", "f16")]],
         {"GGML_SCHED_PREFETCH": "1", "GGML_CUDA_FA_VEC_GQA": "8", "GGML_OP_OFFLOAD_MIN_BATCH": "184",
          "LLAMA_MOE_HYBRID": "1"})
        for i in (1, 2)
        for fa in ("on", "off")
    ],
    # Gemma in its launcher config (-c 16384, ub 2048, f16 KV, threshold 48), FA on/off interleaved.
    "gemma-fa": [
        (f"fa{fa}-{i}", ["-fa", fa]) for i in (1, 2) for fa in ("on", "off")
    ],
    # Without FA the launcher config above needs a 2527 MiB compute buffer (7788 MiB total) and
    # pages; at ub 512 both fit.
    "gemma-fa-ub512": [
        (f"fa{fa}-{i}", ["-fa", fa, *[f"--llama-arg={a}" for a in ("-b", "512", "-ub", "512")]])
        for i in (1, 2) for fa in ("on", "off")
    ],
}


DFLASH_DIR = os.path.join(WS, "models", "Ornith-1.5-35B-A3B-DFlash")


def dflash_args(n_max, quant="q8_0", extra=()):
    """G16. The DFlash drafter published for the *base* Ornith-1.5-35B-A3B, run
    against Tiel. It reads Tiel's hidden states from eight layers and borrows its
    token embeddings and output head, so only the 386M-parameter body is its own.
    Draft KV is q8_0 to match the target's, since the full-attention layer of the
    drafter is sized by -c like everything else.
    """
    path = os.path.join(DFLASH_DIR, f"Ornith-1.5-35B-A3B-DFlash-{quant}.gguf")
    args = ["--spec-type", "draft-dflash", "-md", path,
            "--spec-draft-n-max", str(n_max), "-ctkd", "q8_0", "-ctvd", "q8_0", *extra]
    return ["--llama-arg=" + a for a in args]


SWEEPS["dflash"] = [
    ("A-baseline", ["--no-spec"]),
    ("nmax1", dflash_args(1)),
    ("nmax2", dflash_args(2)),
    ("nmax3", dflash_args(3)),
    ("nmax4", dflash_args(4)),
    ("nmax8", dflash_args(8)),
    ("nmax15", dflash_args(15)),
    ("Z-baseline-repeat", ["--no-spec"]),
]
# Where the drafter fits. The draft context inherits the target's n_ubatch and
# reserves logits through Tiel's 248320-row head for it, so at the working
# -ub 4096 it asks for a 4071 MiB compute buffer and fails even with -ngld 0;
# at -ub 2048 it loads but overcommits the card (5949 MiB, paging).
SWEEPS["dflash-smoke"] = [
    ("base-ub1024", ["--no-spec", "--llama-arg=-ub", "--llama-arg=1024"]),
    ("nmax4-ub1024", dflash_args(4, extra=("-ub", "1024"))),
    ("nmax4-ub1024-ngld0", dflash_args(4, extra=("-ub", "1024", "-ngld", "0"))),
    ("nmax4-ub512", dflash_args(4, extra=("-ub", "512"))),
]


# G17. On the fork the draft context reserves logits for n_max + 1 outputs, not
# for the whole ubatch (common_speculative_init_result + llama_context::graph_reserve),
# so the 4071 MiB buffer that kept DFlash off -ub 4096 on 10826 should be gone
# without a build of our own. What is left: the drafter's weights, its KV (sized
# by the target's -c, 6 full-attention layers) and its compute buffer -- -lv 4
# prints each. Smoke first (--tests tg), the launcher's -c given twice so its
# ubatch planner sees the reduced window; no prefetch, so the slots do not muddy
# the buffer picture.
def _g17_dflash(n_max, ctx):
    return ["-c", str(ctx), "--llama-bin", TIEL_FORK_NO17_REL_BIN, "--llama-arg=-lv", "--llama-arg=4",
            *dflash_args(n_max)]


SWEEPS["tiel-g17-smoke"] = [
    ("base-c131072", ["--no-spec", "--llama-bin", TIEL_FORK_NO17_REL_BIN, "--llama-arg=-lv", "--llama-arg=4"], {}),
    ("df2-c131072", ["--no-spec", *_g17_dflash(2, 131072)], {}),
    ("base-c32768", ["--no-spec", "-c", "32768", "--llama-bin", TIEL_FORK_NO17_REL_BIN,
                     "--llama-arg=-lv", "--llama-arg=4"], {}),
    ("df2-c32768", ["--no-spec", *_g17_dflash(2, 32768)], {}),
]


# G17 smoke showed the draft context still reserving logits for the whole ubatch
# (n_outputs_max = 8192, an 8080 MiB buffer); the cap went to the target context.
# c2c47bb0f caps the draft's outputs at n_max + 1 and selects out_ids before the
# head; 828444e40 keeps that cap for DFlash, which counts as an encoder model.
# The launcher picks -ub 8192 at -c 32768, so b/ub are pinned to 4096.
TIEL_FORK_G17_BIN = os.path.join(WS, "bin", "tiel-fork-g17-828444e40", "llama.exe")
B4096 = ["--llama-arg=-b", "--llama-arg=4096", "--llama-arg=-ub", "--llama-arg=4096"]
LV4 = ["--llama-arg=-lv", "--llama-arg=4"]
SWEEPS["tiel-g17-patch-smoke"] = [
    ("base-c32768", ["--no-spec", "-c", "32768", "--llama-bin", TIEL_FORK_G17_BIN, *LV4, *B4096], {}),
    ("df2-c32768", ["--no-spec", "-c", "32768", "--llama-bin", TIEL_FORK_G17_BIN, *LV4, *dflash_args(2), *B4096], {}),
    ("df2-c131072", ["--no-spec", "--llama-bin", TIEL_FORK_G17_BIN, *LV4, *dflash_args(2), *B4096], {}),
]
# G17 itself, at the only window the drafter fits (-c 32768, b/ub 4096): DFlash
# n-max 2 against no speculation and against the launcher's ngram-mod n-max 4,
# plus both at once (--spec-type takes a list). K=1 everywhere: the drafter
# leaves ~575 MiB. The launcher's real default at 131072 closes the sweep as
# the yardstick. Run with --tests floor,tg,free,copy,prefill.
_G17_ENV = {"GGML_SCHED_PREFETCH": "1", "GGML_CUDA_FA_VEC_GQA": "8", "LLAMA_TRACE": "1"}
_G17_C32 = ["--no-spec", "-c", "32768", "--llama-bin", TIEL_FORK_G17_BIN]
_DF2_NG4 = ["--llama-arg=" + a for a in (
    "--spec-type", "ngram-mod,draft-dflash", "-md",
    os.path.join(DFLASH_DIR, "Ornith-1.5-35B-A3B-DFlash-q8_0.gguf"),
    "--spec-draft-n-max", "2", "-ctkd", "q8_0", "-ctvd", "q8_0",
    "--spec-ngram-mod-n-match", "16", "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "4")]
SWEEPS["tiel-g17"] = [
    ("off-1", [*_G17_C32, *B4096], _G17_ENV),
    ("ngram4", [*_G17_C32, *[f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)], *B4096], _G17_ENV),
    ("df2-1", [*_G17_C32, *dflash_args(2), *B4096], _G17_ENV),
    ("df2-ngram4", [*_G17_C32, *_DF2_NG4, *B4096], _G17_ENV),
    ("off-2", [*_G17_C32, *B4096], _G17_ENV),
    ("df2-2", [*_G17_C32, *dflash_args(2), *B4096], _G17_ENV),
    ("launcher-c131072", ["--no-spec", "--llama-bin", TIEL_FORK_NO17_REL_BIN,
                          *[f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)]],
     {**_G17_ENV, "GGML_SCHED_PREFETCH": "2"}),
]
# tiel-g17: DFlash took prefill 883 -> 531 t/s at 5665 MiB idle VRAM, while on
# 10826 (ub 1024) its prefill tax was 2 %. Paging or the drafter's own work?
# Without prefetch slots (frees ~210 MiB) against off at the same K=0.
# Run with --tests prefill.
SWEEPS["tiel-g17-pp"] = [
    ("off-pf0", [*_G17_C32, *B4096], {"GGML_CUDA_FA_VEC_GQA": "8"}),
    ("df2-pf0", [*_G17_C32, *dflash_args(2), *B4096], {"GGML_CUDA_FA_VEC_GQA": "8"}),
    ("df2-pf0-ub1024", [*_G17_C32, *dflash_args(2), "--llama-arg=-b", "--llama-arg=4096",
                        "--llama-arg=-ub", "--llama-arg=1024"], {"GGML_CUDA_FA_VEC_GQA": "8"}),
    ("off-pf0-ub1024", [*_G17_C32, "--llama-arg=-b", "--llama-arg=4096",
                        "--llama-arg=-ub", "--llama-arg=1024"], {"GGML_CUDA_FA_VEC_GQA": "8"}),
]


# No. 24: is the MTP prefill tax just the draft context overflowing VRAM?
# Smoke on the launcher's fork binary, -lv 4 for the buffer sizes, no prefetch
# slots so they do not muddy the picture. --n-cpu-moe 40 pins the experts of
# blk.0-39 only, so blk.40 (the MTP block, loaded only with speculation) may
# land its experts on CUDA0 -- the -lv 4 load log says.
_NO24 = ["--llama-bin", TIEL_FORK_NO17_REL_BIN, *LV4]
_NO24_ENV = {"GGML_CUDA_FA_VEC_GQA": "8"}
SWEEPS["tiel-no24-smoke"] = [
    ("base", ["--no-spec", *_NO24], _NO24_ENV),
    ("mtp1", ["--mtp", "--mtp-n-max", "1", *_NO24], _NO24_ENV),
]
# The smoke showed it: with speculation on, blk.40 loads and its experts go to
# CUDA0 (model buffer 2038 -> 2896 MiB), the draft KV is f16 (256 MiB at 131K)
# and the recurrent state doubles -- the card is over before the draft compute
# buffer. On the fork the draft context then dies in sched_reserve. So the
# decisive A/B runs on 10826, where every MTP journal row was taken: stock MTP
# (as MTP-14) against MTP with blk.40's experts kept in RAM (--n-cpu-moe 41) and
# a q8_0 draft KV, at 131K and at 32K (b/ub pinned to 4096, the launcher would
# pick 8192 there). If the tax is overflow, the "fit" rows lose only 4-8 %.
B10826 = os.path.join(WS, "bin", "llama-b10826.exe")
_MTP1 = ["--mtp", "--mtp-n-max", "1"]
_FIT = ["--n-cpu-moe", "41", "--llama-arg=-ctkd", "--llama-arg=q8_0", "--llama-arg=-ctvd", "--llama-arg=q8_0"]
_C32 = ["-c", "32768", *B4096]
SWEEPS["tiel-no24"] = [
    ("A-base", ["--no-spec", "--llama-bin", B10826, *LV4]),
    ("mtp1", [*_MTP1, "--llama-bin", B10826, *LV4]),
    ("mtp1-fit", [*_MTP1, *_FIT, "--llama-bin", B10826, *LV4]),
    ("base-c32k", ["--no-spec", *_C32, "--llama-bin", B10826, *LV4]),
    ("mtp1-c32k", [*_MTP1, *_C32, "--llama-bin", B10826, *LV4]),
    ("mtp1-fit-c32k", [*_MTP1, *_FIT, *_C32, "--llama-bin", B10826, *LV4]),
    ("Z-base", ["--no-spec", "--llama-bin", B10826, *LV4]),
]
# tiel-no24: at 131K both MTP rows die in the draft's sched_reserve; at 32K MTP
# n-max 1 costs prefill 3.5 % (720 vs 747), so the tax was the overflow. What is
# MTP worth now? On the launcher's fork at 32K, against no speculation and the
# launcher's ngram-mod n-max 4, interleaved twice. No prefetch slots: the MTP row
# sits at 5693 MiB. Run with --tests floor,tg,free,copy,prefill.
_NO24F = ["--no-spec", "-c", "32768", "--llama-bin", TIEL_FORK_NO17_REL_BIN, *B4096]
_MTPF = lambda n: ["--llama-arg=--spec-type", "--llama-arg=draft-mtp",
                   "--llama-arg=--spec-draft-n-max", f"--llama-arg={n}"]
_NG4 = [f"--llama-arg={a}" for a in NGRAM_MOD(1, 4)]
_NO24F_ENV = {"GGML_CUDA_FA_VEC_GQA": "8", "LLAMA_TRACE": "1"}
SWEEPS["tiel-no24-fork"] = [
    ("off-1", _NO24F, _NO24F_ENV),
    ("mtp1-1", [*_NO24F, *_MTPF(1)], _NO24F_ENV),
    ("mtp2-1", [*_NO24F, *_MTPF(2)], _NO24F_ENV),
    ("ngram4-1", [*_NO24F, *_NG4], _NO24F_ENV),
    ("off-2", _NO24F, _NO24F_ENV),
    ("mtp1-2", [*_NO24F, *_MTPF(1)], _NO24F_ENV),
    ("mtp2-2", [*_NO24F, *_MTPF(2)], _NO24F_ENV),
    ("ngram4-2", [*_NO24F, *_NG4], _NO24F_ENV),
]
# On the fork an MTP server dies silently on the first copy request (a new
# prompt after the free ones), so tiel-no24-fork lost its MTP prefill. Prefill
# alone, on a fresh server per row. Run with --tests floor,prefill.
SWEEPS["tiel-no24-fork-pp"] = [
    ("off", _NO24F, _NO24F_ENV),
    ("mtp1", [*_NO24F, *_MTPF(1)], _NO24F_ENV),
    ("mtp2", [*_NO24F, *_MTPF(2)], _NO24F_ENV),
    ("off-2", _NO24F, _NO24F_ENV),
]


def nmax_sweep(p_min):
    cfgs = [("A-baseline", ["--no-spec"])]
    for n in (1, 2, 3, 4, 5):
        args = ["--mtp", "--mtp-n-max", str(n)]
        if p_min is not None:
            args += ["--mtp-p-min", str(p_min)]
        cfgs.append((f"nmax{n}", args))
    # Order control. The p-min sweep saw prefill fall from 745 to 94-190 t/s, but
    # the MTP configs all ran after the baseline, so a thermal or drift effect
    # would look identical. Re-running the identical baseline last separates the
    # two: if it still reports ~745 t/s, the penalty belongs to MTP.
    cfgs.append(("Z-baseline-repeat", ["--no-spec"]))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", help="one of: " + ", ".join(SWEEPS) + ", or nmax")
    ap.add_argument("--model", default=None, choices=sorted(MODELS),
                    help="which model to launch (default: inferred from the sweep name, "
                         "gemma-* -> gemma, everything else -> tiel)")
    ap.add_argument("--p-min", type=float, default=None, help="p-min to hold fixed for the nmax sweep")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--llama-bin", default=None,
                    help="path to the llama binary, passed through to start_server.py "
                         "(default: llama.exe, the stock build -- not start_server.py's own "
                         "default, which for Tiel is the fork with expert prefetch). "
                         "Use it to sweep a pinned build from bin/ without touching the "
                         "system one -- the binary is recorded in results.json")
    ap.add_argument("--logdir", default=None,
                    help="where the per-config .log files go (default: bench-logs/<sweep>/). "
                         "Give it a fresh directory when repeating a sweep: --out only names "
                         "the results file, so without this a repeat overwrites the previous "
                         "run's logs, and bench-logs/**/*.log is gitignored")
    ap.add_argument("--extra", default=None,
                    help="extra start_server.py args appended to EVERY config in the sweep, "
                         "baselines included, so the comparison stays honest. Quoted and "
                         "shell-split, e.g. --extra \"--n-cpu-moe 27 --ngld 99\"")
    ap.add_argument("--tests", default="tg,floor,prefill",
                    help="comma-separated subset of tg,floor,free,copy,prefill to run (order is fixed). "
                         "For smoke runs: a config that spills VRAM would sit for 20 minutes "
                         "in the 28k-token prefill")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    extra_all = shlex.split(a.extra) if a.extra else []

    model_key = a.model or ("gemma" if a.sweep.startswith("gemma") else "tiel")
    profile = MODELS[model_key]
    launch = list(profile["launch"])
    # Always explicit: start_server.py now defaults Tiel to the fork with expert
    # prefetch, and every A/Z baseline row in the journal means stock llama.exe.
    # Per-config --llama-bin in the sweep args comes later and still wins.
    launch += ["--llama-bin", a.llama_bin or "llama.exe"]

    configs = nmax_sweep(a.p_min) if a.sweep == "nmax" else SWEEPS[a.sweep]
    # (name, args) or (name, args, env)
    configs = [(c[0], list(c[1]) + extra_all, c[2] if len(c) > 2 else {}) for c in configs]
    if a.dry_run:
        print(f"model: {model_key}  launch: {' '.join(launch)}  "
              f"alias: {profile['alias']}  tg prompt: {profile['lang']}")
        for name, extra, env in configs:
            envs = " ".join(f"{k}={v}" for k, v in env.items())
            print(f"{name:20s} {envs + ' ' if envs else ''}{' '.join(extra)}")
        return

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WS,
                            capture_output=True, text=True).stdout.strip()
    logdir = a.logdir or os.path.join(WS, "bench-logs", a.sweep)
    os.makedirs(logdir, exist_ok=True)
    out_path = a.out or os.path.join(logdir, "results.json")

    prompts = load_prompts(profile["lang"], profile.get("prefill_chars"))
    results = {"sweep": a.sweep, "model": model_key, "commit": commit,
               "repeats": a.repeats, "launch": launch,
               "llama_bin": a.llama_bin or "llama.exe", "logdir": logdir,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": []}

    for name, extra, env in configs:
        print(f"\n{'='*70}\n[{name}] {' '.join(extra)} {env or ''}\n{'='*70}", flush=True)
        srv = Server(extra, os.path.join(logdir, f"{name}.log"), launch, env)
        entry = {"name": name, "extra_args": extra}
        if env:
            entry["env"] = env
        try:
            srv.start()
            load_s = srv.wait_ready()
            entry["load_s"] = round(load_s, 1)
            # Snapshot after warmup. On WDDM an overcommitted card does not fail,
            # it pages over PCIe and every number in the row turns to garbage --
            # the first DFlash smoke run read 5949 of 6144 MiB and ran 11 t/s.
            entry["vram_used_mib"] = vram_used_mib()
            entry["executing"] = srv.executing_line()
            entry.update(measure(srv, prompts, a.repeats, profile["alias"],
                                 tuple(t.strip() for t in a.tests.split(","))))
            tg = entry.get("tg", {})
            pf = entry.get("prefill", {})
            fl = entry.get("floor", {})
            print(f"  -> TG {tg.get('tg_tps','?')} t/s | prefill {pf.get('pp_tps','?')} t/s "
                  f"| floor {fl.get('wall_s','?')} s | load {entry['load_s']} s "
                  f"| VRAM {entry['vram_used_mib']} MiB", flush=True)
            for t in ("free", "copy", "deep"):
                if t in entry:
                    b = entry[t]
                    print(f"     {t}: {b.get('tg_tps','?')} t/s, {b.get('tg_tokens','?')} tok, "
                          f"rounds {b.get('rounds','?')}, restores {b.get('restores','?')}, "
                          f"md5 {b.get('reply_md5','?')[:8]}", flush=True)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"  !! {entry['error']}", flush=True)
        finally:
            srv.stop()
        results["configs"].append(entry)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
