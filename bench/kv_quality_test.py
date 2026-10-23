#!/usr/bin/env python3
"""
kv_quality_test.py - does a quantised KV cache change what the model says?

G1. The workspace runs -ctk q8_0 -ctv q8_0 everywhere because on Tiel it was
measured to be nearly free. A published KL-divergence benchmark puts Gemma 4
26B-A4B at 0.377 against f16 at q8_0, versus 0.108 for the dense Gemma 4 31B --
the MoE variant degrading ~3.5x worse than its own dense sibling. That is
someone else's number on someone else's hardware, and the launcher currently
defaults to f16 on the strength of it. This measures it locally.

Method: greedy decoding, identical prompts, identical seed, one KV type per
server launch. Under temperature 0 the sampler is argmax, so the output is a
deterministic function of the logits -- any divergence at all means the
quantisation flipped a top-1 choice. Two numbers come out:

  - the share of prompts whose answers are not byte-identical to the f16 run
  - where the first difference falls, in characters

Neither is a KL divergence. They are better suited to the actual decision, which
is not "how large is the error" but "does it change the reply I get". The answers
are written out side by side so the quality question -- whether the changed reply
is worse, not merely different -- stays with the reader.

    python kv_quality_test.py
    python kv_quality_test.py --types f16 q8_0 --max-tokens 300
"""

import argparse
import difflib
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
ALIAS = "gemma-4-26B-A4B-it"

# Spread across the things a conversation partner actually gets asked: recall,
# explanation, reasoning, instructions, translation, arithmetic, and writing.
# Long-tail proper nouns and numbers are deliberate -- those are the tokens with
# the thinnest probability mass, and the first place a quantisation error shows.
PROMPTS = [
    "Объясни простыми словами, чем отличается вирус от бактерии, и почему антибиотики помогают только в одном из этих случаев.",
    "Перечисли пять причин, по которым в средневековых городах случались крупные пожары, и коротко поясни каждую.",
    "Что произойдёт, если вынуть один кирпич из нижнего ряда кирпичной арки? Рассуждай по шагам.",
    "Напиши короткое письмо соседу с вежливой просьбой не сверлить стены по выходным утром.",
    "Переведи на английский: «Он вышел из дома затемно, чтобы успеть на первый поезд, и всю дорогу думал о вчерашнем разговоре».",
    "В корзине 17 яблок. Треть отдали соседям, затем купили ещё 9. Сколько яблок стало? Покажи вычисления.",
    "Расскажи, как работает обычный термос, и почему чай в нём остывает медленнее.",
    "Какие ошибки чаще всего делают новички при выпечке хлеба на закваске?",
    "Опиши в трёх предложениях, чем Байкал отличается от других крупных озёр мира.",
    "Придумай начало рассказа о смотрителе маяка, который однажды утром обнаружил, что море отступило на километр.",
    "Объясни разницу между «одеть» и «надеть» и приведи по два примера на каждое.",
    "Составь план на день для человека, который работает из дома и хочет больше двигаться.",
]


def chat(messages, max_tokens, timeout=600):
    body = json.dumps({
        "model": ALIAS, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 1234, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


class Server:
    def __init__(self, extra):
        self.extra, self.lines, self.proc = extra, [], None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(WS, "start_server.py"), "-m", "gemma"] + self.extra,
            cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        def pump():
            for line in self.proc.stdout:
                self.lines.append(line.rstrip("\n"))
        threading.Thread(target=pump, daemon=True).start()

        t0 = time.time()
        while time.time() - t0 < 420:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}")
            if any("[*] Warmup:" in l for l in self.lines):
                time.sleep(1.0)
                return self
            time.sleep(1.0)
        raise TimeoutError("server never became ready")

    def __exit__(self, *exc):
        subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                       capture_output=True)
        for _ in range(40):
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        time.sleep(3)


def first_diff(a, b):
    """Character index of the first difference, or None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["f16", "q8_0", "q4_0"])
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--no-control", action="store_true",
                    help="skip the reference-vs-itself run")
    ap.add_argument("--outdir", default=os.path.join(WS, "bench-logs", "gemma-kv-quality"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    answers = {}

    # The reference config, run a second time in its own server process. Without
    # this the whole test is uninterpretable: "every answer changed" only means
    # the KV type did it if the same KV type twice gives the same answers. MoE
    # routing and batching can make greedy decoding non-reproducible across
    # processes, and that would produce exactly the same 12/12 result.
    runs = list(a.types)
    if not a.no_control:
        runs.append(a.types[0] + "#control")

    for label in runs:
        t = label.split("#")[0]
        print(f"\n=== KV {label} ===", flush=True)
        with Server(["-ctk", t, "-ctv", t]):
            answers[label] = []
            for i, p in enumerate(PROMPTS):
                out = chat([{"role": "user", "content": p}], a.max_tokens)
                answers[label].append(out)
                print(f"  {i + 1:2d}/{len(PROMPTS)}  {len(out):5d} chars", flush=True)

    ref = a.types[0]
    report = {"reference": ref, "max_tokens": a.max_tokens, "prompts": PROMPTS,
              "answers": answers, "comparison": {}}

    print(f"\n{'=' * 70}\nAgainst {ref}, greedy decoding, identical seed\n{'=' * 70}")
    for t in [x for x in runs if x != ref]:
        diffs, positions, ratios = 0, [], []
        for p_ref, p_t in zip(answers[ref], answers[t]):
            d = first_diff(p_ref, p_t)
            ratios.append(difflib.SequenceMatcher(None, p_ref, p_t).ratio())
            if d is not None:
                diffs += 1
                positions.append(d)
        n = len(PROMPTS)
        mean_pos = sum(positions) / len(positions) if positions else None
        report["comparison"][t] = {
            "differing": diffs, "of": n,
            "mean_first_diff_chars": mean_pos,
            "mean_similarity": sum(ratios) / n,
        }
        print(f"  {t:6s}  {diffs}/{n} answers changed"
              + (f", first difference at char {mean_pos:.0f} on average" if positions else "")
              + f", mean similarity {sum(ratios) / n:.3f}")

    with open(os.path.join(a.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Side-by-side text, so the reader can judge whether "different" is "worse".
    with open(os.path.join(a.outdir, "answers.md"), "w", encoding="utf-8") as f:
        for i, p in enumerate(PROMPTS):
            f.write(f"## {i + 1}. {p}\n\n")
            for t in runs:
                same = "" if t == ref or answers[t][i] != answers[ref][i] else "  (identical to %s)" % ref
                f.write(f"### KV {t}{same}\n\n{answers[t][i]}\n\n")
            f.write("---\n\n")
    print(f"\nwrote {a.outdir}\\report.json and answers.md")


if __name__ == "__main__":
    main()
