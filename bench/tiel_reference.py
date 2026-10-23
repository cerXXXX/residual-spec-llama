#!/usr/bin/env python3
"""
tiel_reference.py - reference answers for Tiel, so a fork cannot quietly break it.

Every hypothesis worked on in the Tiel fork (llama-tiel/, branch `tiel`) is
supposed to leave the model's output alone: it moves bytes or reorders work,
it does not change what is computed. "Supposed to" needs a check, and plain
text equality is the wrong one -- a different summation order on the GPU
legitimately flips a greedy choice where the top two tokens are a near tie.

So each run stores, per generated token, the chosen token and its top-5
log-probabilities (`/completion` with `n_probs`). Comparing two runs gives:

  - the length of the identical token prefix and where it first diverges;
  - the reference's top1 - top2 margin at that position. A divergence at a
    margin of a few hundredths of a nat is numerical noise; a divergence at a
    margin of several nats is a broken computation;
  - the largest |delta logprob| of the chosen token over the common prefix.
    On an unchanged computation this is ~0; on a reordered one it is small
    (1e-3..1e-2); anything near 1 means the logits are wrong.

Prompts cover both prefill paths the fork can touch. With the launcher's
GGML_OP_OFFLOAD_MIN_BATCH=128 a batch under 128 tokens is prefilled on the
CPU, and Tiel's chat template alone is ~200 tokens -- so every chat case goes
through the GPU with the expert upload, and the CPU path is covered by one raw
completion without the template (`raw-short`, ~20 tokens). The chat cases
span one ubatch (~230 tokens), two (~7K) and seven (prompt_real.txt, 28K
tokens, the same text every prefill number in the journal was taken on).

    python bench/tiel_reference.py run --label ref-b10826
    python bench/tiel_reference.py run --label ref-b10826-control
    python bench/tiel_reference.py run --label fork --llama-bin llama-tiel\\build\\bin\\Release\\llama.exe
    python bench/tiel_reference.py compare ref-b10826 fork
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PORT = 9095
API_KEY = "sk-localllama-rtx3060-securekey"
OUTDIR = os.path.join(WS, "bench-logs", "tiel-reference")

SHORT = [
    ("py-intervals",
     "Напиши на Python функцию merge_intervals(intervals), которая объединяет "
     "пересекающиеся отрезки [a, b]. Добавь три assert-теста."),
    ("c-bug",
     "Найди ошибку в этом коде на C и исправь её:\n\n"
     "int sum(int *a, int n) {\n    int s;\n    for (int i = 0; i <= n; i++) s += a[i];\n    return s;\n}"),
    ("sql",
     "Есть таблицы orders(id, user_id, total, created_at) и users(id, name). "
     "Напиши SQL-запрос: пять пользователей с наибольшей суммой заказов за 2025 год."),
    ("ru-math",
     "В корзине 17 яблок. Треть отдали соседям, затем купили ещё 9. "
     "Сколько яблок стало? Покажи вычисления."),
    ("cpp-raii",
     "Explain in five sentences why RAII matters in C++ and show a minimal "
     "example with std::unique_ptr."),
]

# No chat template: the only case whose prefill stays under 128 tokens.
RAW = [
    ("raw-short",
     "# Python 3\nimport heapq\n\n\ndef dijkstra(graph, source):\n"
     '    """Shortest distances from source; graph is {node: [(neighbour, weight), ...]}."""\n'),
]

MAX_SHORT = 400
MAX_LONG = 300


def read(path):
    with open(os.path.join(WS, path), encoding="utf-8") as f:
        return f.read()


def long_prompts():
    # Fixtures only: a prompt read from a file that gets edited (the first
    # version used bench/sweep_bench.py) silently changes between runs.
    wiki = read("prompts/prompt_ru_wiki.txt")[:24000]
    real = read("prompts/prompt_real.txt")
    return [
        ("long-wiki",
         wiki + "\n\nПерескажи этот текст в пяти коротких пунктах."),
        ("long-real",
         real + "\n\nSummarise in a short list which command-line flags "
         "start_server.py passes to llama.exe for the Tiel model, and why."),
    ]


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def generate(text, max_tokens, raw=False):
    prompt = text if raw else post(
        "/apply-template", {"messages": [{"role": "user", "content": text}]})["prompt"]
    t0 = time.perf_counter()
    r = post("/completion", {
        "prompt": prompt, "n_predict": max_tokens, "n_probs": 5,
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 1234,
        "cache_prompt": False, "stream": False,
    })
    wall = time.perf_counter() - t0
    toks = []
    for p in r["completion_probabilities"]:
        toks.append({
            "id": p["id"], "tok": p["token"], "lp": p["logprob"],
            "top": [[t["id"], t["logprob"]] for t in p["top_logprobs"]],
        })
    tm = r.get("timings", {})
    return {
        "content": r["content"], "tokens": toks,
        "n_prompt": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
        "n_gen": tm.get("predicted_n"), "gen_ms": tm.get("predicted_ms"),
        "wall_s": round(wall, 2),
    }


class Server:
    def __init__(self, extra, log_path):
        self.extra, self.log_path, self.lines, self.proc = extra, log_path, [], None

    def __enter__(self):
        cmd = [sys.executable, os.path.join(WS, "start_server.py"), "-m", "tiel",
               "-ctk", "q8_0", "-ctv", "q8_0", "--host", "127.0.0.1",
               "--port", str(PORT),
               # stock 10826 unless the caller names a binary; start_server.py
               # takes the last --llama-bin, and would otherwise pick the fork
               "--llama-bin", "llama.exe"] + self.extra
        print("[*] launching:", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(
            cmd, cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        self.log = open(self.log_path, "w", encoding="utf-8")

        def pump():
            for line in self.proc.stdout:
                self.lines.append(line.rstrip("\n"))
                self.log.write(line)
                self.log.flush()
        threading.Thread(target=pump, daemon=True).start()

        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}")
            if any("[*] Warmup:" in l or "[!] Warmup:" in l for l in self.lines):
                time.sleep(1.0)
                return self
            time.sleep(1.0)
        raise TimeoutError("server never became ready")

    def header(self):
        keep = ("[+] Env:", "[+] Executing:", "device_info", "system_info")
        return [l for l in self.lines if any(k in l for k in keep)]

    def __exit__(self, *exc):
        subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                       capture_output=True)
        for _ in range(40):
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        time.sleep(3)
        self.log.close()


def cmd_run(a):
    os.makedirs(OUTDIR, exist_ok=True)
    extra = list(a.extra or [])
    if a.llama_bin:
        extra += ["--llama-bin", a.llama_bin]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WS,
                            capture_output=True, text=True).stdout.strip()
    cases = [(n, t, MAX_SHORT, True) for n, t in RAW] + \
            [(n, t, MAX_SHORT, False) for n, t in SHORT] + \
            [(n, t, MAX_LONG, False) for n, t in long_prompts()]
    path = os.path.join(OUTDIR, f"{a.label}.json")
    out = {"label": a.label, "commit": commit, "llama_bin": a.llama_bin or "llama.exe",
           "extra": extra, "env_offload": os.environ.get("GGML_OP_OFFLOAD_MIN_BATCH"),
           "env_prefetch": os.environ.get("GGML_SCHED_PREFETCH"),
           "env_prefetch_min": os.environ.get("GGML_SCHED_PREFETCH_MIN"), "cases": {}}
    if a.only:
        # re-run some cases into an existing run; cases no longer defined are dropped
        only = set(a.only.split(","))
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        names = {n for n, *_ in cases}
        out["cases"] = {k: v for k, v in out["cases"].items() if k in names}
        out.setdefault("rerun", []).append({"cases": sorted(only), "commit": commit})
        cases = [c for c in cases if c[0] in only]
    with Server(extra, os.path.join(OUTDIR, f"{a.label}{'-rerun' if a.only else ''}.log")) as srv:
        out["header" if not a.only else "header_rerun"] = srv.header()
        for name, text, mx, raw in cases:
            r = generate(text, mx, raw)
            out["cases"][name] = r
            print(f"  {name:12s} prompt {r['n_prompt']:6d} tok {r['prompt_ms'] / 1000:7.2f} s"
                  f"  gen {r['n_gen']:4d} tok {r['n_gen'] / (r['gen_ms'] / 1000):6.2f} t/s",
                  flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}")


def compare(ref, oth):
    """Per-case comparison of two runs; see the module docstring for the metrics."""
    rows = []
    for name, r in ref["cases"].items():
        o = oth["cases"].get(name)
        if o is None:
            continue
        ta, tb = r["tokens"], o["tokens"]
        n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i]["id"] != tb[i]["id"]), None)
        common = div if div is not None else n
        dmax = max((abs(ta[i]["lp"] - tb[i]["lp"]) for i in range(common)), default=0.0)
        margin = None
        if div is not None:
            top = ta[div]["top"]
            margin = top[0][1] - top[1][1] if len(top) > 1 else None
        rows.append({"case": name, "len_ref": len(ta), "len_oth": len(tb),
                     "diverge_at": div, "ref_margin_at_div": margin,
                     "max_dlogprob_common": dmax})
    return rows


def cmd_compare(a):
    load = lambda l: json.load(open(os.path.join(OUTDIR, f"{l}.json"), encoding="utf-8"))
    rows = compare(load(a.ref), load(a.other))
    print(f"{'case':12s} {'tokens':>11s} {'diverge@':>9s} {'ref margin':>11s} {'max|dlp|':>10s}")
    for r in rows:
        div = "-" if r["diverge_at"] is None else str(r["diverge_at"])
        m = "-" if r["ref_margin_at_div"] is None else f"{r['ref_margin_at_div']:.4f}"
        print(f"{r['case']:12s} {r['len_ref']:5d}/{r['len_oth']:<5d} {div:>9s} {m:>11s}"
              f" {r['max_dlogprob_common']:10.5f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--llama-bin")
    r.add_argument("--only", help="comma-separated cases to re-run into the existing <label>.json")
    r.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="further start_server.py arguments, must come last")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
