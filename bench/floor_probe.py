#!/usr/bin/env python3
"""Hypothesis No. 18: a fixed host cost on every request that grows with -c / -ub.

Logged, not explained: the 4-token recompute after a checkpoint restore costs
390-500 ms at -c 131072 -ub 4096 and 170-235 ms at -c 32768 -ub 1024, while
the compute of 4 tokens is ~90 ms. This sends the same short turns to one
server per configuration, many times, and reads llama.cpp's own prompt-eval
time for each:

- same : the previous prompt again, max_tokens 1. The slot holds exactly that
         prompt, but a hybrid model cannot drop one token, so the server
         restores the checkpoint 4 tokens before the end and recomputes 4.
- floor: the sweep_bench floor turn (prompt + "Done." + a new user line with a
         distinct suffix), ~39 new tokens, each after the previous one.

    python bench/floor_probe.py --set grid --logdir bench-logs/no18-floor
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_bench as sb  # noqa: E402

BIN = sb.TIEL_FORK_NO17_REL_BIN  # the launcher's binary
ENV = {"GGML_OP_OFFLOAD_MIN_BATCH": "128", "GGML_SCHED_PREFETCH": "2", "GGML_CUDA_FA_VEC_GQA": "8"}


def la(*args):
    return [f"--llama-arg={a}" for a in args]


def cfg(c, ub, *more):
    # -c through the launcher (it sizes the plan from it); -b/-ub passed through last,
    # because for small windows the launcher picks -b/-ub 8192 itself (No. 42)
    return ["-c", str(c), *la("-b", "4096", "-ub", str(ub)), *more]


SETS = {
    "grid": [
        ("c131k-ub4096-1", cfg(131072, 4096)),
        ("c32k-ub4096", cfg(32768, 4096)),
        ("c131k-ub1024", cfg(131072, 1024)),
        ("c32k-ub1024", cfg(32768, 1024)),
        ("c131k-ub4096-2", cfg(131072, 4096)),
    ],
    "diag": [
        ("c131k-ub4096-lv4", cfg(131072, 4096, *la("-lv", "4"))),
        ("c32k-ub1024-lv4", cfg(32768, 1024, *la("-lv", "4"))),
    ],
    # step 2: the same servers, requests with and without backend sampling
    # (the request field, so both modes share one server); run with --modes
    "ab": [
        ("ab-c131k-ub4096-1", cfg(131072, 4096)),
        ("ab-c32k-ub1024", cfg(32768, 1024)),
        ("ab-c131k-ub4096-2", cfg(131072, 4096)),
    ],
    # does a request with tools (grammar) pay the reserve? -lv 4 shows it
    "tools": [
        ("tools-c131k-ub4096-lv4", cfg(131072, 4096, *la("-lv", "4"))),
    ],
    # trace of the no-backend-sampling path without a grammar
    "nobs-diag": [
        ("nobs-c131k-ub4096-lv4", cfg(131072, 4096, *la("-lv", "4"))),
    ],
}

# Request-body variants. "bs" = what the harness always sent (backend sampling from
# the launcher's --backend-sampling); "nobs" turns it off per request; "tools"
# attaches a tool definition the way an agent client does.
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
MODES = {"bs": {}, "nobs": {"backend_sampling": False}, "tools": {"tools": TOOLS}}


def chat(messages, max_tokens, extra):
    body = {"model": sb.MODELS["tiel"]["alias"], "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "top_k": 1, "seed": 1234, "stream": False, **extra}
    req = sb.urllib.request.Request(
        sb.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {sb.API_KEY}"})
    t0 = time.time()
    with sb.urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read().decode())
    return time.time() - t0, payload


def probe(srv, tg, n_same, n_floor, mode="bs", n_tg=0):
    extra = MODES[mode]
    out = {"same": [], "floor": [], "tg": []}

    def one(msgs, max_tokens=1):
        mark = len(srv.lines)
        wall, payload = chat(msgs, max_tokens, extra)
        time.sleep(0.3)  # let the timing lines land
        b = sb.parse_block(srv.lines[mark:])
        b["wall_ms"] = round(wall * 1000, 1)
        b["n_prompt"] = (payload.get("usage") or {}).get("prompt_tokens")
        b["reserves"] = sum("reserve took" in l for l in srv.lines[mark:])
        return b

    # TG first: same prompt each time, so after the first one it is pure generation
    for _ in range(n_tg):
        out["tg"].append(one(tg, 300))
    out["prime"] = one(tg)
    for _ in range(n_same):
        out["same"].append(one(tg))
    for i in range(n_floor):
        out["floor"].append(one(tg + [
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content":
             f"Now restate point number {i + 1} of your answer in one short sentence, "
             f"and mention the identifier QX{i:02d}{i * 7:03d} verbatim."}]))
    return out


def summary(rows, skip=1):
    pp = [r["pp_ms"] for r in rows[skip:] if "pp_ms" in r]
    wall = [r["wall_ms"] for r in rows[skip:] if "pp_ms" in r]
    if not pp:
        return {}
    return {"pp_med": round(statistics.median(pp), 1), "pp_min": round(min(pp), 1),
            "pp_max": round(max(pp), 1), "wall_med": round(statistics.median(wall), 1),
            "tokens": sorted({r["pp_tokens"] for r in rows if "pp_tokens" in r})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--set", default="grid", choices=sorted(SETS))
    ap.add_argument("--same", type=int, default=6)
    ap.add_argument("--floor", type=int, default=6)
    ap.add_argument("--tg", type=int, default=0, help="300-token generations before the probe, first one unscored")
    ap.add_argument("--modes", default="bs", help="comma-separated request variants per server: " + ", ".join(MODES))
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",")]
    os.makedirs(a.logdir, exist_ok=True)

    tg = sb.load_prompts("en")["tg"]
    launch = sb.MODELS["tiel"]["launch"] + ["--no-spec", "--llama-bin", BIN]
    commit = sb.subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sb.WS,
                               capture_output=True, text=True).stdout.strip()
    res = {"set": a.set, "commit": commit, "launch": launch, "env": ENV, "configs": []}
    out_path = os.path.join(a.logdir, f"results-{a.set}.json")
    for name, extra in SETS[a.set]:
        print(f"\n=== [{name}] {' '.join(extra)}", flush=True)
        srv = sb.Server(extra, os.path.join(a.logdir, f"{name}.log"), launch, ENV)
        entry = {"name": name, "extra": extra}
        try:
            srv.start()
            entry["load_s"] = round(srv.wait_ready(), 1)
            entry["vram_used_mib"] = sb.vram_used_mib()
            entry["executing"] = srv.executing_line()
            for k, mode in enumerate(modes):
                r = probe(srv, tg, a.same, a.floor, mode, a.tg)
                r["mode"] = mode
                # first "same" can still carry the prime's leftovers, first floor has 49 tokens
                r["same_sum"] = summary(r["same"])
                r["floor_sum"] = summary(r["floor"])
                tgs = [b["tg_tps"] for b in r["tg"][1:] if "tg_tps" in b]
                r["tg_tps"] = tgs
                if len(modes) == 1:
                    entry.update(r)
                else:
                    entry[f"{mode}-{k}"] = r
                print(f"  [{mode}] same  {r['same_sum']}\n  [{mode}] floor {r['floor_sum']}"
                      f"\n  [{mode}] tg {tgs} | reserves/request "
                      f"{[b['reserves'] for b in r['same'] + r['floor']]}", flush=True)
            print(f"  VRAM {entry['vram_used_mib']} MiB", flush=True)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"  !! {entry['error']}", flush=True)
        finally:
            srv.stop()
        res["configs"].append(entry)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
