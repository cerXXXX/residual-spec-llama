#!/usr/bin/env python3
"""Hypothesis No. 19: does an agent turn re-process the previous assistant answer?

The slot cache after turn k holds prompt_k + the tokens the model generated.
Turn k+1 re-renders the whole history through the chat template; if the
template draws the previous assistant message with different tokens (tool
calls re-serialised from JSON, reasoning trimmed or dropped), the common
prefix ends inside that message, and a hybrid model cannot roll back the
recurrent state token by token: the server restores the last checkpoint
(4 tokens before the end of prompt_k, or earlier) and recomputes from there.

This plays a scripted agent session against one launcher-default server:
OpenAI-style tool loop, real read-only tools on this repo, edit_file
simulated (nothing is written). Per request it records, from the response
and from /slots (LLAMA_SERVER_SLOTS_DEBUG=1):

- held   : tokens in the slot after the previous turn (prompt + generated)
- cache_n: tokens the server reused for this turn
- waste  : held - cache_n, tokens of the previous context recomputed
- new    : prompt_tokens - held, tokens that are genuinely new
- the first character where the re-rendered prompt (/apply-template) departs
  from previous prompt + raw generated text, with context.

Client modes: "echo" sends reasoning_content back, "strip" drops it (what a
plain OpenAI SDK client does).

    python bench/prefix_audit.py --modes echo,strip --logdir bench-logs/no19-prefix
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_bench as sb  # noqa: E402

ENV = {"LLAMA_SERVER_SLOTS_DEBUG": "1", "LLAMA_SERVER_SLOTS_N_DIFF": "16"}
SKIP_DIRS = {".git", "llama-tiel", "llamacpp-src", "models", "bin", "bench-logs", "bench-data",
             ".cache", "__pycache__", "node_modules", ".repowise"}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List the entries of a directory in the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory, relative to the workspace root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the workspace, with line numbers. "
                                            "At most 200 lines per call.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File, relative to the workspace root."},
            "offset": {"type": "integer", "description": "First line to read, 1-based."},
            "limit": {"type": "integer", "description": "Number of lines to read."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search files for a regular expression. Returns file:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory to search, default the root."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace, must be unique."},
            "new_string": {"type": "string", "description": "Replacement text."}},
            "required": ["path", "old_string", "new_string"]}}},
]

SYSTEM = ("You are a coding agent working in the repository at the workspace root "
          "(a llama.cpp tuning project, Python scripts and Markdown notes). Use the tools to "
          "look at the code before answering. Keep answers short.")

TASKS = [
    "Find where start_server.py decides how many expert prefetch slots Tiel gets "
    "(GGML_SCHED_PREFETCH) and explain the rule in two sentences.",
    "Add a one-line comment directly above that function saying the 210 MiB slot size is "
    "the largest expert tensor (Q6_K ffn_down). Use edit_file.",
    "Which value of GGML_OP_OFFLOAD_MIN_BATCH does the launcher set for Gemma, and why that "
    "number? Check the code, then answer in one sentence.",
]

# --tasks long: the three above plus more lookups, for the frequency of token-split mismatches
TASKS_LONG = TASKS + [
    "In bench/sweep_bench.py, what does wait_ready() treat as 'ready', and what is the fallback? "
    "Read the code first.",
    "List the bench/ directory and name three scripts that profile the GPU. Check each one's "
    "docstring before naming it.",
    "In bench/floor_probe.py, which request field turns backend sampling off? Then rename the "
    "MODES key 'nobs' to 'no_backend_sampling' with edit_file.",
    "Find every place in start_server.py that reads an environment variable with os.environ.get "
    "and list the variable names.",
    "Read the first 60 lines of docs/IGPU-PLAN.md and summarise the blocker in one sentence.",
]


# ------------------------------------------------------------------ tools

def _resolve(path):
    p = os.path.normpath(os.path.join(sb.WS, path or "."))
    if not p.startswith(sb.WS):
        raise ValueError("path outside the workspace")
    return p


def t_list_dir(path="."):
    p = _resolve(path)
    out = []
    for e in sorted(os.listdir(p)):
        out.append(e + ("/" if os.path.isdir(os.path.join(p, e)) else ""))
    return "\n".join(out[:200])


def t_read_file(path, offset=1, limit=200):
    p = _resolve(path)
    offset = max(1, int(offset or 1))
    limit = max(1, min(200, int(limit or 200)))
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    chunk = lines[offset - 1:offset - 1 + limit]
    body = "\n".join(f"{offset + i:6d}\t{l}" for i, l in enumerate(chunk))
    if offset - 1 + limit < len(lines):
        body += f"\n... ({len(lines)} lines total)"
    return body


def t_grep(pattern, path="."):
    rx = re.compile(pattern)
    root = _resolve(path)
    files = [root] if os.path.isfile(root) else []
    if not files:
        for d, dirs, fs in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for f in sorted(fs):
                if f.endswith((".py", ".md", ".bat", ".ps1", ".json", ".txt")):
                    files.append(os.path.join(d, f))
    hits = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, l in enumerate(fh, 1):
                    if rx.search(l):
                        hits.append(f"{os.path.relpath(f, sb.WS)}:{i}: {l.rstrip()[:200]}")
                        if len(hits) >= 40:
                            return "\n".join(hits) + "\n... (truncated at 40 matches)"
        except OSError:
            pass
    return "\n".join(hits) or "no matches"


def t_edit_file(path, old_string, new_string):
    p = _resolve(path)
    with open(p, encoding="utf-8", errors="replace") as f:
        text = f.read()
    n = text.count(old_string)
    if n != 1:
        return f"error: old_string found {n} times in {path}, must be exactly once"
    return f"Edited {path}: replaced 1 occurrence ({len(old_string)} -> {len(new_string)} chars)."


IMPL = {"list_dir": t_list_dir, "read_file": t_read_file, "grep": t_grep, "edit_file": t_edit_file}


def run_tool(call):
    fn = call["function"]
    try:
        args = json.loads(fn.get("arguments") or "{}")
        return IMPL[fn["name"]](**args)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ http

def post(path, body, timeout=900):
    req = urllib.request.Request(
        sb.BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {sb.API_KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path):
    req = urllib.request.Request(sb.BASE + path, headers={"Authorization": f"Bearer {sb.API_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def body_for(messages, max_tokens):
    return {"model": sb.MODELS["tiel"]["alias"], "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "seed": 1234, "stream": False}


def first_diff(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ------------------------------------------------------------------ session

def session(srv, mode, max_requests, max_tokens, tasks):
    messages = [{"role": "system", "content": SYSTEM}]
    reqs = []
    prev = None  # {"held", "text"}: slot after the previous request
    for task_i, task in enumerate(tasks):
        messages.append({"role": "user", "content": task})
        while len(reqs) < max_requests:
            rendered = post("/apply-template", body_for(messages, max_tokens)).get("prompt", "")
            mark = len(srv.lines)
            t0 = time.time()
            resp = post("/v1/chat/completions", body_for(messages, max_tokens))
            wall = time.time() - t0
            time.sleep(0.3)
            log = srv.lines[mark:]
            tm = resp.get("timings") or {}
            usage = resp.get("usage") or {}
            msg = resp["choices"][0]["message"]
            slot = get("/slots")[0]
            r = {"task": task_i, "i": len(reqs), "wall_s": round(wall, 2),
                 "prompt_tokens": usage.get("prompt_tokens"), "cache_n": tm.get("cache_n"),
                 "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                 "predicted_n": tm.get("predicted_n"), "finish": resp["choices"][0].get("finish_reason"),
                 "tool_calls": [(c["function"]["name"], c["function"].get("arguments"))
                                for c in msg.get("tool_calls") or []],
                 "reasoning_chars": len(msg.get("reasoning_content") or ""),
                 "content": (msg.get("content") or "")[:300],
                 "checkpoint_lines": [l.strip() for l in log if "checkpoint" in l and (
                     "restored" in l or "forcing" in l or "erased" in l)][:12],
                 "slots_debug": [l for l in log if "old: ..." in l or "new: ..." in l][:4],
                 # token ids around the mismatch: old (cache) line, then new (prompt) line
                 "slots_ids": [l.split("|", 2)[-1].strip() for l in log
                               if re.search(r"\|(\s+\d+){8,}\s*$", l)][:2]}
            if prev:
                r["held"] = prev["held"]
                r["new"] = r["prompt_tokens"] - prev["held"]
                r["waste"] = prev["held"] - r["cache_n"]
                d = first_diff(prev["text"], rendered)
                r["text_match"] = d == len(prev["text"])
                r["diff_at_char"] = d
                r["diff_old"] = prev["text"][max(0, d - 120):d + 120]
                r["diff_new"] = rendered[max(0, d - 120):d + 120]
                r["diff_from_end_chars"] = len(prev["text"]) - d
            prev = {"held": slot.get("n_prompt_tokens"),
                    "text": (slot.get("prompt") or "") + (slot.get("generated") or "")}
            r["held_after"] = prev["held"]
            reqs.append(r)
            print(f"  [{mode}] #{r['i']} task {task_i}: prompt {r['prompt_tokens']} cache {r['cache_n']} "
                  f"pp {r['prompt_n']} ({r['prompt_ms'] and round(r['prompt_ms'])} ms) gen {r['predicted_n']} | "
                  f"new {r.get('new')} waste {r.get('waste')} text_match {r.get('text_match')} "
                  f"(diff {r.get('diff_from_end_chars')} ch from end) | calls {[c[0] for c in r['tool_calls']]}",
                  flush=True)

            amsg = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                amsg["tool_calls"] = msg["tool_calls"]
            if mode == "echo" and msg.get("reasoning_content"):
                amsg["reasoning_content"] = msg["reasoning_content"]
            messages.append(amsg)
            if not msg.get("tool_calls"):
                break
            for c in msg["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": run_tool(c)})
    return reqs, messages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--modes", default="echo,strip")
    ap.add_argument("--max-requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--tasks", default="short", choices=["short", "long"])
    ap.add_argument("--extra", action="append", default=[], help="extra launcher args, e.g. --extra=--no-spec")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    launch = ["-m", "tiel", "--llama-arg=-lv", "--llama-arg=4"] + a.extra  # launcher default otherwise
    commit = sb.subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sb.WS,
                               capture_output=True, text=True).stdout.strip()
    res = {"commit": commit, "launch": launch, "env": ENV, "sessions": []}
    out_path = os.path.join(a.logdir, "results.json")
    for mode in [m.strip() for m in a.modes.split(",")]:
        print(f"\n=== [{mode}]", flush=True)
        srv = sb.Server([], os.path.join(a.logdir, f"{mode}.log"), launch, ENV)
        entry = {"mode": mode}
        try:
            srv.start()
            entry["load_s"] = round(srv.wait_ready(), 1)
            entry["executing"] = srv.executing_line()
            entry["env_line"] = next((l for l in srv.lines if "[+] Env:" in l), "")
            reqs, messages = session(srv, mode, a.max_requests, a.max_tokens,
                                     TASKS_LONG if a.tasks == "long" else TASKS)
            entry["requests"] = reqs
            entry["messages"] = messages
            later = [r for r in reqs if "waste" in r]
            entry["sum_new"] = sum(r["new"] for r in later)
            entry["sum_waste"] = sum(r["waste"] for r in later)
            entry["sum_prompt_n"] = sum(r["prompt_n"] for r in later)
            entry["sum_prompt_ms"] = round(sum(r["prompt_ms"] for r in later))
            print(f"  [{mode}] turns 2+: new {entry['sum_new']}  waste {entry['sum_waste']}  "
                  f"processed {entry['sum_prompt_n']}  prompt {entry['sum_prompt_ms']} ms", flush=True)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"  !! {entry['error']}", flush=True)
        finally:
            srv.stop()
        res["sessions"].append(entry)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
