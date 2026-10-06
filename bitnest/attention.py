"""Decode attention backend: replaces the SDPA decode path (StaticCache / PlanesCache, q_len <= DECODE_MAX_Q) with the Triton
flash-decoding kernels (bf16 KV: decode_attn; nested KV4/KV8 planes: kv_planes). Prefill / cache-less calls fall back
to the original path.
  install(arch)          repository modeling (rotated build path): monkeypatch Llama / Qwen2 attention
  install_stock_wrapper  transformers' native Llama / Qwen2 classes (cold-load path, fp16 baseline)"""
import torch.nn.functional as F

from bitnest.decode_attn import DECODE_MAX_Q, decode_attention
from bitnest.kv_planes import decode_attention_planes

MODE_REF = {"get": lambda: "target"}   # injected by the runner: returns the current draft / target mode
DRAFT_LO = {"mode": False}              # draft KV read: False = KV4 (high plane only) / True = KV8 / ("window", N) = KV8 for the last N positions; target always KV8
TARGET_LO = {"mode": True}             # target KV read: KV8 by default; False only for the "KV4 everywhere (no nesting)" ablation


def _read_lo():
    return TARGET_LO["mode"] if MODE_REF["get"]() == "target" else DRAFT_LO["mode"]


def _q_r3(q, cache):
    if not getattr(cache, "r3", False):
        return q.contiguous()
    from bitnest.had_compat import hadamard_transform
    return hadamard_transform(q.contiguous(), scale=q.shape[-1] ** -0.5)


def _is_planes(c):
    from bitnest.kv_cache_planes import PlanesCache
    return isinstance(c, PlanesCache)


def _prefill_sdpa(q, k, v, scaling):
    g = q.shape[1] // k.shape[1]; k = k.repeat_interleave(g, dim=1); v = v.repeat_interleave(g, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling)


def install(arch="llama"):
    if arch != "llama":
        # The repository's modeling_qwen2 uses the new ALL_ATTENTION_FUNCTIONS interface but does not pass
        # position_ids / cache_position to it -> wrap forward to stash cache_position[0] in module._bn_pos (a static
        # tensor view, safe under graph replay); the interface function falls back to reading it.
        from eval_utils import modeling_qwen2 as MQ
        orig_q = MQ.Qwen2Attention.forward

        def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, past_key_value=None, cache_position=None, **kw):
            self._bn_pos = cache_position[0] if cache_position is not None else (position_ids[0, 0] if position_ids is not None else None)
            self._bn_cache = past_key_value
            return orig_q(self, hidden_states, position_embeddings, attention_mask, position_ids, past_key_value, cache_position, **kw)
        MQ.Qwen2Attention.forward = fwd
        return install_hf_stock()
    from eval_utils import modeling_llama as M
    cls = M.LlamaSdpaAttention; apply_rope = M.apply_rotary_pos_emb
    orig = cls.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kw):
        bsz, q_len, _ = hidden_states.size()
        from transformers import StaticCache
        if past_key_value is None or not isinstance(past_key_value, StaticCache) or cache_position is None or (q_len > DECODE_MAX_Q and not _is_planes(past_key_value)):
            return orig(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kw)
        if q_len > DECODE_MAX_Q:   # prefill into a planes cache: quantize into the planes, attend with this call's bf16 K/V
            q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(v, position_ids)
            q, k = apply_rope(q, k, cos, sin); past_key_value.update(k, v, self.layer_idx, {"cache_position": cache_position})
            out = _prefill_sdpa(q, k, v, None).transpose(1, 2).reshape(bsz, q_len, -1)
            return self.o_proj(out), None, past_key_value
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        k_full, v_full = past_key_value.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})   # StaticCache: returns the whole cache [B,Hk,Lmax,D]
        if _is_planes(past_key_value):
            out = decode_attention_planes(_q_r3(q, past_key_value), *past_key_value.planes(self.layer_idx), cache_position[0], read_lo=_read_lo(), sm_scale=self.scaling if hasattr(self, "scaling") else None)
            if past_key_value.v_off_h[self.layer_idx] is not None:
                out = out + past_key_value.v_off_h[self.layer_idx]
        else:
            out = decode_attention(q.contiguous(), k_full, v_full, cache_position[0])          # seq0 = position of the first new token (device scalar, graph-replay safe)
        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out), None, past_key_value
    cls.forward = forward
    return cls


def install_hf_stock():
    """Register the same decode attention for transformers' native Llama / Qwen2 through ALL_ATTENTION_FUNCTIONS:
    StaticCache + q_len <= DECODE_MAX_Q goes to the kernel, everything else to sdpa."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def bitnest_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, cache_position=None, position_ids=None, **kw):
        # HF pops cache_position before calling the attention interface but keeps position_ids [B, M]:
        # seq0 = position_ids[0, 0] (device scalar, graph-replay safe)
        pos = cache_position[0] if cache_position is not None else (position_ids[0, 0] if position_ids is not None else getattr(module, "_bn_pos", None))
        cache = getattr(module, "_bn_cache", None)
        if cache is not None and _is_planes(cache):
            if query.shape[2] <= DECODE_MAX_Q:
                out = decode_attention_planes(_q_r3(query, cache), *cache.planes(module.layer_idx), pos, read_lo=_read_lo(), sm_scale=scaling)
                if cache.v_off_h[module.layer_idx] is not None:
                    out = out + cache.v_off_h[module.layer_idx]
            else:
                out = _prefill_sdpa(query, key, value, scaling)   # prefill: key / value are this call's bf16 K/V (whole prefix)
            return out.transpose(1, 2).contiguous(), None
        if pos is not None and query.shape[2] <= DECODE_MAX_Q and key.shape[2] > query.shape[2] + 8:   # key is the whole static cache
            out = decode_attention(query.contiguous(), key, value, pos, sm_scale=scaling)
            return out.transpose(1, 2).contiguous(), None
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    ALL_ATTENTION_FUNCTIONS["bitnest_decode"] = bitnest_attn
    return "bitnest_decode"


def install_stock_wrapper():
    """transformers' native LlamaAttention / Qwen2Attention: wrap forward to stash cache_position[0] and past_key_value
    on the module (_bn_pos / _bn_cache) so the registered kernel (bitnest_attn) can reach the PlanesCache (the attention
    interface itself never sees the cache object). Used by the cold-load (--pkg) path."""
    import inspect
    from transformers.models.llama import modeling_llama as ML
    from transformers.models.qwen2 import modeling_qwen2 as MQ
    for cls in (ML.LlamaAttention, MQ.Qwen2Attention):
        if getattr(cls, "_bn_wrapped", False):
            continue
        orig = cls.forward; sig = list(inspect.signature(orig).parameters)

        def make(orig, sig):
            def fwd(self, *args, **kw):
                b = dict(zip(sig[1:], args)); b.update(kw)
                cp = b.get("cache_position"); pv = b.get("past_key_value", b.get("past_key_values"))
                self._bn_pos = cp[0] if cp is not None else None; self._bn_cache = pv
                return orig(self, *args, **kw)
            return fwd
        cls.forward = make(orig, sig); cls._bn_wrapped = True
    return install_hf_stock()
