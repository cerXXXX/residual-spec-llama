import argparse
import ctypes
import os
import pathlib
import time

import numpy as np

LIB = ctypes.CDLL(str(pathlib.Path(__file__).resolve().parent / "libresidual.so"))
LIB.residual_i8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
LIB.residual_i8.restype = ctypes.c_int
LIB.stream_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
LIB.stream_read.restype = ctypes.c_int64
rng = np.random.default_rng(0)


def tile_int8(Wq, S):
    d_out, d_in = Wq.shape
    Wt = np.ascontiguousarray(Wq.reshape(d_out // 16, 16, d_in).transpose(0, 2, 1))
    St = np.ascontiguousarray(S.reshape(d_out // 16, 16, d_in // 128).transpose(0, 2, 1)).astype(np.float16)
    return Wt, St


def run(Wt, St, d_out, d_in, x, n, nth):
    y = np.zeros((n, d_out), np.float32)
    rc = LIB.residual_i8(Wt.ctypes.data, St.ctypes.data, d_out, d_in, x.ctypes.data, n, y.ctypes.data, nth)
    assert rc == 0
    return y


def check_correctness(nth):
    d_out, d_in = 256, 1024
    Wq = rng.integers(-127, 128, (d_out, d_in), dtype=np.int8)
    S = rng.uniform(0.001, 0.01, (d_out, d_in // 128)).astype(np.float32)
    Wt, St = tile_int8(Wq, S)
    Sf = St.astype(np.float32).transpose(0, 2, 1).reshape(d_out, d_in // 128)
    Wd = Wq.astype(np.float64) * np.repeat(Sf.astype(np.float64), 128, axis=1)
    for n in (1, 4, 16, 32):
        x = rng.standard_normal((n, d_in)).astype(np.float32)
        y = run(Wt, St, d_out, d_in, x, n, nth)
        ref = x.astype(np.float64) @ Wd.T
        print(f"correctness n={n:2d}: max relative |err| {np.abs(y - ref).max() / np.abs(ref).max():.2e}", flush=True)


def bench_throughput(nth):
    d_out, d_in = 196608, 5376
    mats = []
    for k in range(4):
        Wq = rng.integers(-127, 128, (d_out, d_in), dtype=np.int8)
        S = rng.uniform(0.001, 0.01, (d_out, d_in // 128)).astype(np.float32)
        mats.append(tile_int8(Wq, S))
        del Wq
    total = sum(Wt.nbytes + St.nbytes for Wt, St in mats)
    print(f"weight set: {total / 1e9:.2f} GB, threads {nth}", flush=True)
    buf = np.concatenate([Wt.reshape(-1) for Wt, _ in mats])
    LIB.stream_read(buf.ctypes.data, buf.nbytes, nth)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        LIB.stream_read(buf.ctypes.data, buf.nbytes, nth)
        best = min(best, time.perf_counter() - t)
    print(f"pure streaming read (DRAM wall): {buf.nbytes / best / 1e9:.1f} GB/s", flush=True)
    del buf
    for n in (1, 2, 4, 8, 12, 16, 24, 32):
        x = rng.standard_normal((n, d_in)).astype(np.float32)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for Wt, St in mats:
                run(Wt, St, d_out, d_in, x, n, nth)
            best = min(best, time.perf_counter() - t)
        print(f"n={n:2d}: {best * 1e3:7.1f} ms per pass  ->  {total / best / 1e9:5.1f} GB/s  ({n * total / best / 1e9:6.0f} G weight*token/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()
    check_correctness(args.threads)
    bench_throughput(args.threads)


if __name__ == "__main__":
    main()
