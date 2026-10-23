import torch

from twobit.hadamard import layer_rot

GROUP = 128


def pack(codes, bits):
    w = (1 << torch.arange(8, device=codes.device, dtype=torch.int64))
    return torch.stack([(((codes >> j) & 1).view(codes.shape[0], -1, 8).to(torch.int64) * w).sum(-1).to(torch.uint8) for j in range(bits)])


def unpack(P):
    bits, d_out, nb = P.shape
    sh = torch.arange(8, device=P.device, dtype=torch.uint8)
    codes = torch.zeros(d_out, nb * 8, dtype=torch.uint8, device=P.device)
    for j in range(bits):
        codes |= (((P[j].unsqueeze(-1) >> sh) & 1).view(d_out, -1) << j).to(torch.uint8)
    return codes


def reconstruct(entry, levels, name, device="cuda"):
    d_out, d_in = entry["shape"]
    codes = unpack(entry["P"].to(device)).long()
    scales = entry["s"].to(device).float().repeat_interleave(GROUP, 1)
    rot = layer_rot(name, d_out, d_in, device)
    return rot.inv(levels.to(device)[codes] * scales)


@torch.no_grad()
def gptq_codes(Wt, H, levels, damp=0.01, group=GROUP):
    thr = (levels[1:] + levels[:-1]) / 2
    W = Wt.clone()
    d_out, d_in = W.shape
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
    codes = torch.zeros(d_out, d_in, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(d_out, d_in // group, dtype=torch.float16, device=W.device)
    for i1 in range(0, d_in, group):
        i2 = i1 + group
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i1:i2, i1:i2]
        sc = W1.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8).half()
        scales[:, i1 // group] = sc[:, 0]
        sc = sc.float()
        for i in range(group):
            w = W1[:, i:i + 1]
            idx = torch.bucketize(w / sc, thr)
            q = levels[idx] * sc
            Q1[:, i:i + 1] = q
            codes[:, i1 + i] = idx[:, 0].to(torch.uint8)
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err @ Hb[i:i + 1, i:]
            E1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        if i2 < d_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, codes, scales
