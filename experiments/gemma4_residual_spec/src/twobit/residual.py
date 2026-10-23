import torch

from twobit.base import GROUP, unpack
from twobit.hadamard import SEED
from twobit.quantizers import lloyd_max

LEVEL_SAMPLE = 2_000_000


def symmetric_quant(W, bits, group=None):
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        return (W / s).round().clamp(-qmax, qmax) * s
    d_out, d_in = W.shape
    G = W.reshape(d_out, d_in // group, group)
    s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return ((G / s).round().clamp(-qmax, qmax) * s).reshape(d_out, d_in)


def rotated_base(entry, levels, device="cuda"):
    codes = unpack(entry["P"].to(device)).long()
    scales = entry["s"].to(device).float().repeat_interleave(GROUP, 1)
    return levels.to(device)[codes] * scales


def rms_normalized(R, group=GROUP):
    G = R.reshape(-1, group)
    return (G / G.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)).flatten()


def residual_levels(grid, bits, sample, cache, seed=SEED):
    if grid == "uni":
        qmax = 2 ** (bits - 1) - 1
        return torch.arange(-qmax, qmax + 1, device=sample.device, dtype=torch.float32)
    if bits not in cache:
        g = torch.Generator(device=sample.device)
        g.manual_seed(seed)
        v = sample[torch.randperm(sample.numel(), device=sample.device, generator=g)[:LEVEL_SAMPLE]]
        cache[bits] = lloyd_max(v, L=2 ** bits, iters=80)
    return cache[bits]


def group_scale(W1, grid, qmax):
    if grid == "lm":
        s = W1.pow(2).mean(1, keepdim=True).sqrt()
    else:
        s = W1.abs().amax(1, keepdim=True) / qmax
    return s.clamp_min(1e-12).half().float()


@torch.no_grad()
def quantize_residual(Rt, H, grid, levels, use_gptq, damp=0.01, group=GROUP):
    W = Rt.clone()
    d_out, d_in = W.shape
    thr = (levels[1:] + levels[:-1]) / 2
    qmax = (levels.numel() - 1) / 2
    if not use_gptq:
        G = W.view(d_out, d_in // group, group)
        sc = group_scale(G.reshape(-1, group), grid, qmax).view(d_out, d_in // group, 1)
        return (levels[torch.bucketize(G / sc, thr)] * sc).view(d_out, d_in)
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(d_in, device=H.device)
    Lc = torch.linalg.cholesky(H)
    del H
    Hi = torch.cholesky_inverse(Lc)
    del Lc
    Hinv = torch.linalg.cholesky(Hi, upper=True)
    del Hi
    Q = torch.zeros_like(W)
    for i1 in range(0, d_in, group):
        i2 = i1 + group
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        sc = group_scale(W1, grid, qmax)
        for i in range(group):
            w = W1[:, i:i + 1]
            q = levels[torch.bucketize(w / sc, thr)] * sc
            Q1[:, i:i + 1] = q
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err @ Hb[i:i + 1, i:]
            E1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        if i2 < d_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q
