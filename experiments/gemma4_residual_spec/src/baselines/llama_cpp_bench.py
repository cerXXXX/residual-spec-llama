import argparse
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model")
    ap.add_argument("--spec-type", default="draft-mtp")
    ap.add_argument("--draft-gpu-layers", type=int, default=99)
    ap.add_argument("--draft-n-max", type=int, nargs="*", default=[])
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--fit-target-mib", type=int, default=1536)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=900)
    return ap.parse_args()


def load_prompts(path, key):
    with open(path) as f:
        items = json.load(f)
    return [p if isinstance(p, str) else p[key] for p in items]


def build_configs(args):
    common = ["-m", args.model, "-fit", "on", "-fitt", str(args.fit_target_mib), "-fa", "on", "-c", str(args.ctx_size), "-t", str(args.threads),
              "--port", str(args.port), "--host", args.host, "-np", "1"]
    configs = []
    if not args.skip_baseline:
        configs.append(("baseline", common))
    if args.draft_model:
        spec = ["-md", args.draft_model, "--spec-type", args.spec_type, "-ngld", str(args.draft_gpu_layers)]
        if args.draft_n_max:
            for k in args.draft_n_max:
                configs.append((f"spec_n{k}", common + spec + ["--spec-draft-n-max", str(k)]))
        else:
            configs.append(("spec", common + spec))
    return configs


def get(url, timeout=5):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def chat(base_url, content, max_tokens, timeout):
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": max_tokens, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r, time.perf_counter() - t


def save(res, out):
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts, args.prompt_key)
    base_url = f"http://{args.host}:{args.port}"
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, server_args in build_configs(args):
        log_path = log_dir / f"llama_{name}.log"
        logf = open(log_path, "w")
        t0 = time.time()
        srv = subprocess.Popen([args.server_bin] + server_args, stdout=logf, stderr=subprocess.STDOUT)
        ok = False
        for _ in range(args.startup_timeout):
            time.sleep(1)
            if srv.poll() is not None:
                break
            try:
                ok = get(f"{base_url}/health").get("status") == "ok"
            except Exception:
                ok = False
            if ok:
                break
        load_s = time.time() - t0
        if not ok:
            res[name] = {"error": "server did not start", "log_tail": log_path.read_text()[-2000:]}
            srv.kill()
            logf.close()
            save(res, out)
            continue
        chat(base_url, "Hello.", args.max_tokens, args.request_timeout)
        rows = []
        for i, p in enumerate(prompts):
            r, wall = chat(base_url, p, args.max_tokens, args.request_timeout)
            tm = r.get("timings", {})
            rows.append({"i": i, "wall_s": wall, "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
                         "predicted_ms": tm.get("predicted_ms"), "tok_s": tm.get("predicted_per_second"), "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted")})
            print(f"[{name}] {i + 1}/{len(prompts)}: {tm.get('predicted_n')} tok, {tm.get('predicted_per_second')} tok/s", flush=True)
        srv.terminate()
        try:
            srv.wait(60)
        except Exception:
            srv.kill()
        logf.close()
        log = log_path.read_text()
        m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
        pn = sum(r["predicted_n"] or 0 for r in rows)
        pms = sum(r["predicted_ms"] or 0 for r in rows)
        dn = sum(r["draft_n"] or 0 for r in rows)
        da = sum(r["draft_n_accepted"] or 0 for r in rows)
        res[name] = {"gpu_layers": m[-1] if m else None, "load_s": load_s, "tokens": pn, "tok_s_total": pn / (pms / 1000) if pms else None,
                     "tok_s_median": sorted(r["tok_s"] for r in rows if r["tok_s"])[len(rows) // 2] if rows else None,
                     "draft_acceptance": (da / dn) if dn else None, "rows": rows}
        save(res, out)
        print(f"== {name}: GPU layers {res[name]['gpu_layers']}, {pn} tok, {res[name]['tok_s_total']:.2f} tok/s (median {res[name]['tok_s_median']}), acceptance {res[name]['draft_acceptance']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
