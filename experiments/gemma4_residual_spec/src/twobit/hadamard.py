import math
import zlib

import torch

SEED = 20261008
_CACHE = {}


def hadamard(b, device):
    key = (b, str(device))
    if key not in _CACHE:
        H = torch.ones(1, 1)
        while H.shape[0] < b:
            H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
        _CACHE[key] = (H / math.sqrt(b)).to(device=device, dtype=torch.float32)
    return _CACHE[key]


def block_size(n, cap=4096):
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


def had_last(X, b):
    s = X.shape
    return (X.reshape(-1, b) @ hadamard(b, X.device)).reshape(s)


def signs(n, seed, device):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(device=device, dtype=torch.float32)


def layer_seed(name, seed=SEED):
    return seed + zlib.crc32(name.encode()) % 100000


class Rot:
    def __init__(self, d_out, d_in, seed, device):
        self.bo, self.bi = block_size(d_out), block_size(d_in)
        self.so, self.si = signs(d_out, seed, device), signs(d_in, seed + 1, device)

    def fwd(self, W):
        W = had_last(W * self.si, self.bi)
        return had_last((W * self.so[:, None]).T.contiguous(), self.bo).T

    def fwd_in(self, W):
        return had_last(W * self.si, self.bi)

    def inv(self, Wt):
        W = (had_last(Wt.T.contiguous(), self.bo).T) * self.so[:, None]
        return had_last(W, self.bi) * self.si

    def rot_in(self, x):
        return had_last(x.float() * self.si, self.bi)


def layer_rot(name, d_out, d_in, device, seed=SEED):
    return Rot(d_out, d_in, layer_seed(name, seed), device)


def rotate_hessian(H, rot):
    H = H * rot.si[:, None] * rot.si[None, :]
    H = had_last(H, rot.bi)
    return had_last(H.T.contiguous(), rot.bi).T
