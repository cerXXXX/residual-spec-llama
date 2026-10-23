#!/usr/bin/env python3
"""Hypothesis 2, server side: profile `llama.exe serve` under Nsight Systems.

`llama.exe bench` and the server disagree on Tiel's prefill (bench pp4096 ran
421 t/s, the server's journal is ~745), so the copy/compute split has to be
taken from the thing that is actually used. This launches the server under
nsys with the working Tiel flags, sends the sweep_bench requests -- one warmup
chunk, the floor follow-ups, then the 28k prefill once and last, same order and
prompts as the harness -- and then stops the collection.

Every request is tagged with its wall-clock window, so prefill_xfer_profile's
analysis can be run per request: copies, kernels and their overlap.

    python bench/prefill_xfer_server.py --out bench-logs/h2-prefill-xfer/server-nopo0.json
    python bench/prefill_xfer_server.py --nopo --out ...
    python bench/prefill_xfer_server.py --env GGML_OP_OFFLOAD_MIN_BATCH=128 --out ...
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_kernel_profile import MODELS, DEFAULT_BIN, find_nsys, export_sqlite  # noqa: E402
from prefill_xfer_profile import union, total, intersect  # noqa: E402
import sweep_bench as sb  # noqa: E402

_COMMON = ["-ngl", "99", "-fa", "on", "-t", "6", "-tb", "6", "--prio", "2", "--poll", "50",
           "--backend-sampling", "-np", "1", "--host", "127.0.0.1", "--port", str(sb.PORT),
           "-C", "555", "--cpu-strict", "1", "--api-key", sb.API_KEY, "--jinja", "--load-mode", "none"]
# Working flags per model, as start_server.py resolves them (journal: PF-03, GP-02).
SERVE = {
    "tiel": ["-c", "131072", "-b", "4096", "-ub", "4096", "-ctk", "q8_0", "-ctv", "q8_0",
             "--n-cpu-moe", "40"] + _COMMON,
    "gemma": ["-c", "16384", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
              # no --chat-template-kwargs: nsys mangles its JSON quoting and the server
              # exits before opening its log; replies here are one token anyway
              "--n-cpu-moe", "23"] + _COMMON,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiel", choices=sorted(SERVE))
    ap.add_argument("--llama-bin", default=DEFAULT_BIN)
    ap.add_argument("--nopo", action="store_true", help="add --no-op-offload")
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for the server process")
    ap.add_argument("--duration", type=int, default=260, help="nsys collection seconds")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--no-graph-trace", action="store_true", help="drop --cuda-graph-trace=node (CUPTI crashed with it on layer-centric prefill)")
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra llama-server argument, repeatable; later values override SERVE (e.g. -b 32768)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = os.path.splitext(a.out)[0]
    rep = base + ".nsys-rep"
    cmd = [find_nsys(), "profile", "-o", base, "--force-overwrite", "true",
           "--trace=cuda", "--sample=none", "--cpuctxsw=none",
           *([] if a.no_graph_trace else ["--cuda-graph-trace=node"]),
           "--duration", str(a.duration),
           a.llama_bin, "serve", "-m", MODELS[a.model],
           "--alias", sb.MODELS[a.model]["alias"]] + SERVE[a.model]
    if a.nopo:
        cmd.append("--no-op-offload")
    cmd += a.server_arg
    # nsys on Windows swallows the target's stdout, so the timings come from a log file
    srv_log = os.path.abspath(base + ".server.log")
    if os.path.exists(srv_log):
        os.remove(srv_log)
    cmd += ["--log-file", srv_log]
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    print("[+] Executing: " + " ".join(cmd), flush=True)
    if a.env:
        print("[+] env: " + " ".join(a.env), flush=True)

    lines = []
    log = open(base + ".log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)

    def pump():
        for l in proc.stdout:
            lines.append(l.rstrip("\n"))
            log.write(l)
            log.flush()
    threading.Thread(target=pump, daemon=True).start()

    def srv_lines():
        try:
            with open(srv_log, encoding="utf-8", errors="replace") as f:
                return f.read().splitlines()
        except OSError:
            return []

    t_start = time.time()
    while True:
        if proc.poll() is not None or time.time() - t_start > 200:
            raise SystemExit("server did not come up")
        try:
            with urllib.request.urlopen(sb.BASE + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(1.0)
    print(f"[*] ready after {time.time() - t_start:.1f} s", flush=True)

    prof = sb.MODELS[a.model]
    prompts = sb.load_prompts(prof["lang"], prof.get("prefill_chars"))
    marks = []

    def req(label, messages, n):
        mark = len(srv_lines())
        t0 = time.time()
        sb.chat(messages, n)
        t1 = time.time()
        time.sleep(0.5)
        blk = sb.parse_block(srv_lines()[mark:])
        marks.append({"label": label, "t0": t0, "t1": t1, **blk})
        print(f"  {label:10s} {t1 - t0:7.2f} s  pp {blk.get('pp_tokens')} tok "
              f"{blk.get('pp_tps')} t/s", flush=True)

    # warmup: one full chunk so expert pages are hot (start_server.py does the same)
    req("warmup", [{"role": "user", "content": "Here is a file:\n\n" +
                    open(os.path.join(sb.WS, "prompts", "prompt_real.txt"), encoding="utf-8",
                         errors="replace").read()[:15000] + "\n\nReply OK."}], 1)
    req("tg-prime", prompts["tg"], 1)
    for i in range(3):
        follow = prompts["tg"] + [
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content":
             f"Now restate point number {i + 1} of your answer in one short sentence, "
             f"and mention the identifier QX{i:02d}{i * 7:03d} verbatim."}]
        req(f"floor{i}", follow, 1)
    if not a.skip_prefill:
        req("prefill", prompts["prefill"], 1)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"executing": " ".join(cmd), "env": a.env, "marks": marks}, f, indent=2)
    print("[*] waiting for nsys to end the collection", flush=True)
    proc.wait(timeout=a.duration + 400)
    time.sleep(2)

    # nsys timestamps are ns since session start; align by the CUDA trace itself:
    # map wall-clock windows onto the trace using the sqlite's session start time.
    db = export_sqlite(rep)
    con = sqlite3.connect(db)
    try:
        sess0 = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    except sqlite3.OperationalError:
        sess0 = None
    ker = list(con.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
    mem = list(con.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    # Raw timeline of the bulk request, for looking at when the bus sits idle:
    # every HtoD of 1 MiB or more with its stream, and the merged kernel spans.
    big = list(con.execute("select start, end, bytes, streamId from CUPTI_ACTIVITY_KIND_MEMCPY "
                           "where copyKind = 1 and bytes >= 1048576 order by start"))
    out = {"executing": " ".join(cmd), "env": a.env, "requests": []}
    print(f"\n{'request':10s} {'pp tok':>7} {'pp ms':>9} {'kernels':>9} {'HtoD':>9} {'overlap':>8} "
          f"{'neither':>9} {'HtoD GiB':>9} {'GB/s':>6}")
    for m in marks:
        # session start is wall-clock ns (unix epoch); fall back to nothing if absent
        if sess0 is None:
            raise SystemExit("no TARGET_INFO_SESSION_START_TIME in the export")
        w0 = int(m["t0"] * 1e9) - sess0
        w1 = int(m["t1"] * 1e9) - sess0
        k = union([(s, e) for s, e in ker if w0 <= s < w1])
        hv = [(s, e, b) for s, e, b, c in mem if c == 1 and w0 <= s < w1]
        h = union([(s, e) for s, e, _ in hv])
        kt, ht, ov = total(k), total(h), intersect(k, h)
        span = (w1 - w0)
        hb = sum(b for *_, b in hv)
        row = {**m, "kernels_ms": kt / 1e6, "h2d_ms": ht / 1e6, "overlap_ms": ov / 1e6,
               "neither_ms": (span - kt - ht + ov) / 1e6, "h2d_GiB": hb / 2**30,
               "h2d_GBps": hb / ht if ht else 0.0, "wall_ms": span / 1e6}
        out["requests"].append(row)
        if m["label"] == "prefill":
            out["prefill_timeline"] = {
                "copies": [(s - w0, e - w0, b, sid) for s, e, b, sid in big if w0 <= s < w1],
                "kernels": [(s - w0, e - w0) for s, e in k],
            }
        print(f"{m['label']:10s} {str(m.get('pp_tokens')):>7} {m.get('pp_ms', 0):9.1f} {row['kernels_ms']:9.1f} "
              f"{row['h2d_ms']:9.1f} {row['overlap_ms']:8.1f} {row['neither_ms']:9.1f} "
              f"{row['h2d_GiB']:9.2f} {row['h2d_GBps']:6.2f}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[+] {a.out}")


if __name__ == "__main__":
    sys.exit(main())
