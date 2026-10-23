import argparse
import ctypes
import os
import pathlib
import time

import numpy as np

LIB = ctypes.CDLL(str(pathlib.Path(__file__).resolve().parent / "libresidual.so"))
LIB.residual_packed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
LIB.residual_packed.restype = ctypes.c_int
LIB.residual_i8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
LIB.residual_i8.restype = ctypes.c_int
rng = np.random.default_rng(0)


def pack_bbit(Q, S, b):
    d_out, d_in = Q.shape
    u = (Q.astype(np.int32) + 2 ** (b - 1)).astype(np.uint16)
    T = u.reshape(d_out // 16, 16, d_in).transpose(0, 2, 1)
    bits = ((T[..., None] >> np.arange(b, dtype=np.uint16)) & 1).astype(np.uint8)
    P = np.packbits(bits.reshape(d_out // 16, d_in, 16 * b), axis=-1, bitorder="little")
    St = np.ascontiguousarray(S.reshape(d_out // 16, 16, d_in // 128).transpose(0, 2, 1)).astype(np.float16)
    return np.concatenate([P.reshape(-1), np.zeros(16, np.uint8)]), St


def run_packed(P, St, d_out, d_in, b, x, n, nth):
    y = np.zeros((n, d_out), np.float32)
    rc = LIB.residual_packed(P.ctypes.data, St.ctypes.data, d_out, d_in, b, x.ctypes.data, n, y.ctypes.data, nth)
    assert rc == 0, rc
    return y


def check_correctness():
    for b in (5, 6, 7, 8):
        qm = 2 ** (b - 1) - 1
        d_out, d_in = 64, 512
        Q = rng.integers(-qm, qm + 1, (d_out, d_in))
        S = rng.uniform(0.001, 0.01, (d_out, d_in // 128)).astype(np.float32)
        P, St = pack_bbit(Q, S, b)
        Sf = St.astype(np.float64).transpose(0, 2, 1).reshape(d_out, d_in // 128)
        Wd = Q.astype(np.float64) * np.repeat(Sf, 128, axis=1)
        for n in (1, 16):
            x = rng.standard_normal((n, d_in)).astype(np.float32)
            y = run_packed(P, St, d_out, d_in, b, x, n, 1)
            ref = x.astype(np.float64) @ Wd.T
            print(f"b={b} n={n:2d}: max relative |err| {np.abs(y - ref).max() / np.abs(ref).max():.2e}  (bytes/weight {(P.nbytes - 16) / Q.size:.3f})", flush=True)


def bench_throughput(nth):
    d_out, d_in = 196608, 5376
    for b in (6, 7, 8):
        qm = 2 ** (b - 1) - 1
        mats = []
        for k in range(4):
            Q = rng.integers(-qm, qm + 1, (d_out, d_in), dtype=np.int8)
            S = rng.uniform(0.001, 0.01, (d_out, d_in // 128)).astype(np.float32)
            mats.append(pack_bbit(Q, S, b))
            del Q
        total = sum(P.nbytes + St.nbytes for P, St in mats)
        for n in (1, 8, 16):
            x = rng.standard_normal((n, d_in)).astype(np.float32)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                for P, St in mats:
                    run_packed(P, St, d_out, d_in, b, x, n, nth)
                best = min(best, time.perf_counter() - t)
            n_weights = 4 * d_out * d_in
            print(f"b={b} n={n:2d}: {best * 1e3:7.1f} ms / {n_weights / 1e9:.2f} G weights ({total / 1e9:.2f} GB) -> {total / best / 1e9:5.1f} GB/s, {n_weights / best / 1e9:5.1f} G weights/s", flush=True)
        del mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()
    check_correctness()
    if args.bench:
        bench_throughput(args.threads)


if __name__ == "__main__":
    main()
