import math

import torch


def int8_row(W):
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
    return (W / s).round().clamp(-127, 127) * s


def rtn(W, bits, group):
    d_out, d_in = W.shape
    Wg = W.reshape(d_out, d_in // group, group)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    q = 2 ** bits - 1
    s = ((mx - mn) / q).clamp_min(1e-8)
    z = (-mn / s).round()
    return (((Wg / s + z).round().clamp(0, q) - z) * s).reshape(d_out, d_in)


def lloyd_max(v, L=4, iters=60):
    lev = torch.quantile(v, torch.linspace(0.5 / L, 1 - 0.5 / L, L, device=v.device))
    for _ in range(iters):
        thr = (lev[1:] + lev[:-1]) / 2
        idx = torch.bucketize(v, thr)
        lev = torch.stack([v[idx == i].mean() if (idx == i).any() else lev[i] for i in range(L)])
    return (lev - lev.flip(0)) / 2


def rht_lm2(W, group, lev, rot, kappa=False):
    Wt = rot.fwd(W)
    d_out, d_in = Wt.shape
    G = Wt.reshape(d_out, d_in // group, group)
    s = G.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    thr = (lev[1:] + lev[:-1]) / 2
    for _ in range(2):
        idx = torch.bucketize(G / s, thr)
        Q = lev[idx]
        s = ((G * Q).sum(-1, keepdim=True) / (Q * Q).sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-8)
    Q = lev[torch.bucketize(G / s, thr)]
    if kappa:
        D = ((G - Q * s) ** 2).sum(-1, keepdim=True) / (G ** 2).sum(-1, keepdim=True).clamp_min(1e-12)
        s = s / (1 - D).clamp_min(0.5)
    return rot.inv((Q * s).reshape(d_out, d_in))


def quant_res(R, fmt):
    if fmt == "bf16":
        return R
    if fmt == "int8":
        return int8_row(R)
    if fmt == "int4":
        s = R.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7
        return (R / s).round().clamp(-7, 7) * s
    raise ValueError(fmt)


class CorrLinear(torch.nn.Module):
    cfg = {"k_frac": 0.0, "basis": "orig", "static": False, "first_tokens": 0}

    def __init__(self, lin, R_cpu, R_rot_cpu, rot, static_order=None):
        super().__init__()
        self.lin, self.R_cpu, self.R_rot_cpu, self.rot = lin, R_cpu, R_rot_cpu, rot
        self.static_order = static_order

    def forward(self, x):
        y = self.lin(x)
        kf = CorrLinear.cfg["k_frac"]
        if kf <= 0:
            return y
        d_in = x.shape[-1]
        k = max(1, int(math.ceil(kf * d_in)))
        if kf >= 1:
            R = self.R_cpu.to(x.device, non_blocking=True)
            return y + (x.float() @ R.float().T).to(y.dtype)
        if CorrLinear.cfg["basis"] == "rot":
            xx = self.rot.rot_in(x)
            R = self.R_rot_cpu.to(x.device, non_blocking=True)
        else:
            xx = x.float()
            R = self.R_cpu.to(x.device, non_blocking=True)
        if CorrLinear.cfg["static"] and self.static_order is not None:
            sel = self.static_order[:k].to(x.device)
            xs = torch.zeros_like(xx)
            xs[..., sel] = xx[..., sel]
        else:
            idx = xx.abs().topk(k, dim=-1).indices
            xs = torch.zeros_like(xx).scatter_(-1, idx, xx.gather(-1, idx))
        t0 = CorrLinear.cfg["first_tokens"]
        if t0 > 0 and xx.dim() == 3:
            xs[:, :t0, :] = xx[:, :t0, :]
        return y + (xs @ R.float().T).to(y.dtype)
