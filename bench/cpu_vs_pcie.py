"""Hypothesis No. 20: does a PCIe upload slow a CPU that touches no DDR4 to speak of?

Six threads of a small fp32 matmul that stays in L2 (pure compute) run in
phases: alone, with bench/h2d_load.exe (pinned, DMA only) in the background,
alone again. GFLOPS per phase; run bench/clock_log.ps1 beside it for clocks.
If a DMA-only upload slows L2-resident compute, the cause is power or clocks,
not memory bandwidth.

    python bench/cpu_vs_pcie.py [--phase-s 20]
"""
import argparse
import os
import subprocess
import threading
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker(stop, out, i):
    a = np.random.rand(128, 128).astype(np.float32)
    b = np.random.rand(128, 128).astype(np.float32)
    n = 0
    while not stop.is_set():
        a @ b
        n += 1
    out[i] = n * 2 * 128 ** 3


MEM = [np.ones(64 << 20, np.float32) for _ in range(6)]  # 256 MiB per thread, far past L3


def mem_worker(stop, out, i):
    n = 0
    while not stop.is_set():
        MEM[i].sum()
        n += 1
    out[i] = n * MEM[i].nbytes


def phase(seconds, fn):
    stop, out = threading.Event(), [0] * 6
    ts = [threading.Thread(target=fn, args=(stop, out, i)) for i in range(6)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in ts:
        t.join()
    return sum(out) / (time.perf_counter() - t0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-s", type=float, default=20)
    ap.add_argument("--mode", choices=("compute", "mem"), default="compute",
                    help="compute: L2-resident matmul (GFLOPS); mem: streaming reads of 256 MiB (GB/s)")
    a = ap.parse_args()
    fn, unit = (worker, "GFLOPS") if a.mode == "compute" else (mem_worker, "GB/s read")
    exe = os.path.join(ROOT, "bench", "h2d_load.exe")
    for name in ("alone", "pinned DMA", "alone", "pageable", "alone"):
        h = None
        if name != "alone":
            h = subprocess.Popen([exe, str(a.phase_s + 5), "2", "1" if name.startswith("pinned") else "0", "1"],
                                 stdout=subprocess.PIPE, text=True)
            time.sleep(3)
        g = phase(a.phase_s, fn)
        bw = ""
        if h:
            bw = h.communicate()[0].strip().splitlines()[-1]
        print(f"{time.strftime('%H:%M:%S')}  {name:10s}  {g:7.1f} {unit}   {bw}", flush=True)


if __name__ == "__main__":
    main()
