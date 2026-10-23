import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ


def text_model(model):
    return getattr(model.model, "language_model", model.model)


def model_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors"])


def load_mtp_tensors(path, prefix="mtp."):
    with open(os.path.join(path, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    tensors = {}
    for key, file in weight_map.items():
        if key.startswith(prefix):
            with safe_open(os.path.join(path, file), "pt") as fh:
                tensors[key] = fh.get_tensor(key)
    return tensors


class QwenMTPHead:
    def __init__(self, model, path, device="cuda", dtype=torch.bfloat16):
        tm = text_model(model)
        cfg = tm.config
        self.config = cfg
        self.device = device
        self.embed_tokens = tm.embed_tokens
        self.lm_head = model.lm_head
        self.rotary = tm.rotary_emb
        tensors = load_mtp_tensors(path)
        self.num_tensors = len(tensors)
        layer_prefix = "mtp.layers.0."
        layer = MQ.Qwen3_5DecoderLayer(cfg, cfg.layer_types.index("full_attention"))
        layer_state = {k[len(layer_prefix):]: v for k, v in tensors.items() if k.startswith(layer_prefix)}
        layer.load_state_dict(layer_state, strict=True)
        self.layer = layer.to(device, dtype).eval()
        self.attn = self.layer.self_attn
        self.head_dim = self.attn.head_dim
        self.kv_groups = self.attn.num_key_value_groups
        d = cfg.hidden_size
        self.pre_fc_norm_embedding = self._rms_norm(tensors["mtp.pre_fc_norm_embedding.weight"], d, dtype)
        self.pre_fc_norm_hidden = self._rms_norm(tensors["mtp.pre_fc_norm_hidden.weight"], d, dtype)
        self.norm = self._rms_norm(tensors["mtp.norm.weight"], d, dtype)
        fc = torch.nn.Linear(2 * d, d, bias=False)
        fc.weight.data = tensors["mtp.fc.weight"].clone()
        self.fc = fc.to(device, dtype).eval()
        self.order = "eh"

    def _rms_norm(self, weight, d, dtype):
        m = MQ.Qwen3_5RMSNorm(d, eps=self.config.rms_norm_eps)
        m.weight.data = weight.clone()
        return m.to(self.device, dtype).eval()

    def positions(self, p0, n):
        return (torch.arange(n, device=self.device) + p0).view(1, 1, -1).expand(3, 1, -1)

    @torch.no_grad()
    def inputs(self, h, tokens, order=None):
        e = self.pre_fc_norm_embedding(self.embed_tokens(tokens))
        hh = self.pre_fc_norm_hidden(h)
        order = order or self.order
        return self.fc(torch.cat([e, hh], -1) if order == "eh" else torch.cat([hh, e], -1))

    @torch.no_grad()
    def step(self, x, p0, k_cache=None, v_cache=None):
        att = self.attn
        hd = self.head_dim
        n = x.shape[1]
        resid = x
        hs = self.layer.input_layernorm(x)
        shp = hs.shape[:-1]
        q, gate = torch.chunk(att.q_proj(hs).view(*shp, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(*shp, -1)
        q = att.q_norm(q.reshape(*shp, -1, hd)).transpose(1, 2)
        k = att.k_norm(att.k_proj(hs).view(*shp, -1, hd)).transpose(1, 2)
        v = att.v_proj(hs).view(*shp, -1, hd).transpose(1, 2)
        cos, sin = self.rotary(hs, self.positions(p0, n))
        q, k = MQ.apply_rotary_pos_emb(q, k, cos, sin)
        keys = k if k_cache is None else torch.cat([k_cache, k], 2)
        values = v if v_cache is None else torch.cat([v_cache, v], 2)
        o = F.scaled_dot_product_attention(
            q,
            keys.repeat_interleave(self.kv_groups, 1),
            values.repeat_interleave(self.kv_groups, 1),
            is_causal=(k_cache is None and n > 1),
            scale=att.scaling,
        )
        o = o.transpose(1, 2).reshape(*shp, -1) * torch.sigmoid(gate)
        hs = resid + att.o_proj(o)
        hs = hs + self.layer.mlp(self.layer.post_attention_layernorm(hs))
        return hs, self.norm(hs), k, v

    @torch.no_grad()
    def argmax_tokens(self, h, chunk=512):
        return torch.cat([self.lm_head(h[i:i + chunk]).argmax(-1) for i in range(0, h.shape[0], chunk)])
