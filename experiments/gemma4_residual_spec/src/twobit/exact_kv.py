from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from twobit.base import reconstruct


class ExactKVAttention:
    def __init__(self):
        self.mode = "original"
        self.lens = None
        self.band = 1
        self.batch = 0
        self.store = {}
        self.sdpa = None

    def install(self):
        self.sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.attention
        return self

    @contextmanager
    def active(self, mode, lens, band=1, batch=0):
        self.mode = mode
        self.lens = lens
        self.band = band
        self.batch = batch
        yield self
        self.mode = "original"

    def attention(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None, **kwargs):
        if self.mode == "original":
            return self.sdpa(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, sliding_window=sliding_window, **kwargs)
        n = query.shape[2]
        layer = module.layer_idx
        groups = module.num_key_value_groups
        device = query.device
        pos = torch.arange(n, device=device)
        lens = self.lens.to(device)
        key_valid = pos[None, :] < lens[:, None]
        window = sliding_window or 10 ** 9
        if self.mode in ("custom", "capture"):
            if self.mode == "capture":
                self.store[(self.batch, layer)] = (key.detach().to("cpu"), value.detach().to("cpu"))
            allowed = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - window)
            mask = allowed[None, None] & key_valid[:, None, None, :]
            keys, values = key, value
        else:
            exact_key, exact_value = (t.to(device) for t in self.store[(self.batch, layer)])
            d = self.band
            keys = torch.cat([exact_key, key], 2)
            values = torch.cat([exact_value, value], 2)
            in_window = pos[None, :] > pos[:, None] - window
            exact = (pos[None, :] < pos[:, None] - d + 1) & in_window
            band = (pos[None, :] >= pos[:, None] - d + 1) & (pos[None, :] <= pos[:, None]) & in_window
            mask = torch.cat([exact, band], 1)[None, None] & torch.cat([key_valid, key_valid], 1)[:, None, None, :]
        self_only = torch.cat([torch.zeros(n, mask.shape[-1] - n, dtype=torch.bool, device=device), torch.eye(n, dtype=torch.bool, device=device)], 1)
        mask = mask | self_only[None, None]
        out = F.scaled_dot_product_attention(query, keys.repeat_interleave(groups, 1), values.repeat_interleave(groups, 1), attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None


def _to_gpu(module, args, kwargs):
    module.to("cuda")


def _to_cpu(module, args, output):
    module.to("cpu")


def stream_layers(layers):
    for layer in layers:
        layer.register_forward_pre_hook(_to_gpu, with_kwargs=True)
        layer.register_forward_hook(_to_cpu)


def decoder_linears(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model.layers." in name}


@torch.no_grad()
def restore_weights(linears, originals):
    for name, m in linears.items():
        m.weight.data.copy_(originals[name])


@torch.no_grad()
def apply_base_weights(linears, checkpoint, levels):
    for name, m in linears.items():
        m.weight.data.copy_(reconstruct(checkpoint["layers"][name], levels, name).to("cpu", torch.bfloat16))


def load_base(path):
    checkpoint = torch.load(path, weights_only=False)
    return checkpoint, torch.tensor(checkpoint["meta"]["levels"])


def load_texts(path):
    state = torch.load(path, weights_only=False)
    return state["texts"], state["prompt_lens"], state["sources"]


def pad_right(texts, idx, device="cuda"):
    n = max(texts[i].numel() for i in idx)
    ids = torch.zeros(len(idx), n, dtype=torch.long)
    mask = torch.zeros(len(idx), n, dtype=torch.long)
    for row, i in enumerate(idx):
        ids[row, :texts[i].numel()] = texts[i]
        mask[row, :texts[i].numel()] = 1
    return ids.to(device), mask.to(device), torch.tensor([texts[i].numel() for i in idx])


@torch.no_grad()
def forward_texts(model, attention, texts, idx, mode, band=1, batch=0, **kwargs):
    ids, mask, lens = pad_right(texts, idx)
    with attention.active(mode, lens, band, batch):
        out = model.model(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs)
    return out
