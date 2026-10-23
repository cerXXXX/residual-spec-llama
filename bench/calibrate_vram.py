#!/usr/bin/env python3
"""
calibrate_vram.py - solve the VRAM planner's constants by measuring, one term
at a time.

The Tiel constants in start_server.py were found by trial and error against
nvidia-smi. This does the same thing deliberately: launch the model with one
term varied and everything else pinned, read peak VRAM, and take the slope.

    peak = intercept                      dense weights + CUDA context
         + kv_global_per_tok  * ctx       the blocks that scale with -c
         + kv_fixed                       the window-capped sliding blocks
         + compute_per_tok    * ubatch    activation buffers
         + moe_layer          * layers    expert layers pushed to the GPU

The intercept, kv_fixed and the dense weights are not separable by this method
and do not need to be: the planner only ever uses their sum.

VRAM is sampled after the launcher's warmup prefill, not at "listening": the
compute buffer is allocated on the first batch, so sampling earlier misses the
single largest ubatch-dependent term.

    python calibrate_vram.py -m gemma
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    return int(out.strip().splitlines()[0])


class Run:
    def __init__(self, model, ctx, ub, ncmoe, extra=()):
        # Pinned to the stock build the constants were calibrated on; without it
        # start_server.py would pick the Tiel fork (an extra in `extra` still wins).
        self.args = ["-m", model, "-c", str(ctx), "-ub", str(ub),
                     "--n-cpu-moe", str(ncmoe), "--llama-bin", "llama.exe", *extra]
        self.ctx, self.ub, self.ncmoe = ctx, ub, ncmoe
        self.lines = []

    def measure(self, timeout=420):
        idle = vram_used_mib()
        proc = subprocess.Popen(
            [sys.executable, os.path.join(WS, "start_server.py")] + self.args,
            cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        def pump():
            for line in proc.stdout:
                self.lines.append(line.rstrip("\n"))
        threading.Thread(target=pump, daemon=True).start()

        peak, t0 = idle, time.time()
        try:
            warm = False
            while time.time() - t0 < timeout:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited rc={proc.returncode}")
                if any("[*] Warmup:" in l for l in self.lines):
                    warm = True
                    # A couple of extra samples: the allocator settles a moment
                    # after the first batch completes.
                    for _ in range(4):
                        peak = max(peak, vram_used_mib())
                        time.sleep(0.75)
                    break
                peak = max(peak, vram_used_mib())
                time.sleep(0.5)
            if not warm:
                raise TimeoutError("warmup never reported")
        finally:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
            for _ in range(40):
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            time.sleep(3)
        return peak - idle


def fit(points, key, others_fixed):
    """Least-squares slope of vram against `key` over points sharing the rest."""
    sel = [p for p in points if all(p[k] == v for k, v in others_fixed.items())]
    if len(sel) < 2:
        return None, sel
    xs = [p[key] for p in sel]
    ys = [p["vram"] for p in sel]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, sel
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="gemma")
    ap.add_argument("--n-layer", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(WS, "bench-logs", "gemma-vram", "calib.json"))
    a = ap.parse_args()

    base_ctx, base_ub, base_ncmoe = 4096, 1024, a.n_layer
    plan = [
        # (ctx, ub, ncmoe) -- one term off the base at a time
        (base_ctx, base_ub, base_ncmoe),
        (16384, base_ub, base_ncmoe),
        (32768, base_ub, base_ncmoe),
        (65536, base_ub, base_ncmoe),
        (base_ctx, 2048, base_ncmoe),
        (base_ctx, 4096, base_ncmoe),
        (base_ctx, base_ub, a.n_layer - 2),
        (base_ctx, base_ub, a.n_layer - 4),
    ]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    points = []
    for ctx, ub, ncmoe in plan:
        gpu_layers = a.n_layer - ncmoe
        print(f"[ctx {ctx:6d} ub {ub:5d} ncmoe {ncmoe:3d} -> {gpu_layers} expert layers on GPU] ",
              end="", flush=True)
        try:
            v = Run(a.model, ctx, ub, ncmoe).measure()
            print(f"{v} MiB", flush=True)
            points.append({"ctx": ctx, "ub": ub, "gpu_layers": gpu_layers, "vram": v})
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        with open(a.out, "w") as f:
            json.dump(points, f, indent=2)

    print("\n-- fitted slopes " + "-" * 50)
    s_ctx, _ = fit(points, "ctx", {"ub": base_ub, "gpu_layers": 0})
    s_ub, _ = fit(points, "ub", {"ctx": base_ctx, "gpu_layers": 0})
    s_l, _ = fit(points, "gpu_layers", {"ctx": base_ctx, "ub": base_ub})
    if s_ctx is not None:
        print(f"  KV per context token (global blocks)  {s_ctx * 1024:8.4f} KiB/tok")
    if s_ub is not None:
        print(f"  compute buffer per ubatch token       {s_ub:8.4f} MiB/tok")
    if s_l is not None:
        print(f"  VRAM per expert layer                 {s_l:8.1f} MiB")
    base = [p for p in points
            if p["ctx"] == base_ctx and p["ub"] == base_ub and p["gpu_layers"] == 0]
    if base and s_ctx is not None and s_ub is not None:
        icept = base[0]["vram"] - s_ctx * base_ctx - s_ub * base_ub
        print(f"  intercept (dense + context + fixed KV){icept:8.1f} MiB")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
