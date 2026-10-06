"""PlanesCache: dual-plane KV cache (draft reads the KV4 high plane, target reads both planes as KV8). Compatible with the
HF StaticCache interface (mask construction / graph capture) without allocating a full bf16 cache.
update() quantizes the new tokens' K/V ("4-bit first, then residual") into the planes at cache_position and returns the
new tokens' bf16 K/V (prefill computes attention directly from them)."""
import torch
from transformers import StaticCache

from bitnest.kv_planes import quant_kv_planes, quant_kv_planes_into  # noqa: F401


class PlanesCache(StaticCache):
    def __init__(self, config, max_batch_size, max_cache_len, device, dtype=torch.bfloat16):
        super().__init__(config=config, max_batch_size=max_batch_size, max_cache_len=8, device=device, dtype=dtype)   # placeholder: tiny 8-slot bf16 cache, unused
        self.planes_len = int(max_cache_len); B = max_batch_size; Hk = self.num_key_value_heads; D = self.head_dim; L = config.num_hidden_layers
        mk = lambda dt, last: [torch.zeros((B, Hk, self.planes_len) + last, dtype=dt, device=device) for _ in range(L)]  # noqa: E731
        self.khi = mk(torch.uint8, (D // 2,)); self.klo = mk(torch.uint8, (D // 2,)); self.ks = mk(torch.float32, ())
        self.vhi = mk(torch.uint8, (D // 2,)); self.vlo = mk(torch.uint8, (D // 2,)); self.vs = mk(torch.float32, ())
        self.seen = 0
        self.k_off = [None] * L; self.v_off = [None] * L; self.v_off_h = [None] * L   # per-(kv head, channel) constant offsets (see set_offsets)
        self.r3 = False   # True: per-head Hadamard on K before quantization (same transform on q, q.k unchanged) spreads channel outliers (SpinQuant R3)

    def set_r3(self, on=True):
        self.r3 = bool(on)
        if self.r3:
            from bitnest.had_compat import hadamard_transform
            sc = self.head_dim ** -0.5
            for li in range(len(self.k_off)):
                if self.k_off[li] is not None:
                    self.k_off[li] = hadamard_transform(self.k_off[li].contiguous(), scale=sc).contiguous()   # H(k - off) = Hk - H.off

    def set_offsets(self, model):
        """Per-(kv head, channel) constant offsets taken from the k_proj / v_proj biases (Qwen2: k-bias outlier channels
        reach |400| vs a median of 0.4, so a per-token scale would wipe out every normal channel).
        K: quantize (K - off_k); the score q.off_k is the same for all positions -> softmax shift invariance, exact.
        V: quantize (V - off_v) and add off_v back after attention (sum p = 1, exact).
        Models without bias (Llama) keep None."""
        Hk = self.num_key_value_heads; D = self.head_dim; H = model.config.num_attention_heads; n = 0
        for li, layer in enumerate(model.model.layers):
            at = layer.self_attn; bk = getattr(at.k_proj, "bias", None); bv = getattr(at.v_proj, "bias", None)
            if bk is not None:
                self.k_off[li] = bk.detach().to(torch.bfloat16).reshape(Hk, D).contiguous(); n += 1
            if bv is not None:
                self.v_off[li] = bv.detach().to(torch.bfloat16).reshape(Hk, D).contiguous()
                self.v_off_h[li] = self.v_off[li].repeat_interleave(H // Hk, 0)[None, :, None, :].contiguous()   # [1,H,1,D] for adding back
        return n

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def reset(self):
        for lst in (self.khi, self.klo, self.vhi, self.vlo, self.ks, self.vs):
            for t in lst:
                t.zero_()
        self.seen = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cp = cache_kwargs["cache_position"]
        # fused kernel: quantize + pack + scatter, one launch each for K and V
        k_q = key_states.contiguous()
        if self.r3:   # rotate only the copy written to the planes; the returned K must stay the original (prefill attends with it)
            from bitnest.had_compat import hadamard_transform
            k_q = hadamard_transform(k_q, scale=self.head_dim ** -0.5)
        quant_kv_planes_into(k_q, cp, self.khi[layer_idx], self.klo[layer_idx], self.ks[layer_idx], self.k_off[layer_idx])
        quant_kv_planes_into(value_states.contiguous(), cp, self.vhi[layer_idx], self.vlo[layer_idx], self.vs[layer_idx], self.v_off[layer_idx])
        if key_states.shape[2] > 64 and layer_idx == 0:
            self.seen = int(cp.shape[0])   # update the python-side counter on prefill only (never inside a captured graph)
        return key_states, value_states

    def planes(self, layer_idx):
        return (self.khi[layer_idx], self.klo[layer_idx], self.ks[layer_idx], self.vhi[layer_idx], self.vlo[layer_idx], self.vs[layer_idx])

    def bytes_kv(self):
        return sum(t.numel() * t.element_size() for lst in (self.khi, self.klo, self.vhi, self.vlo, self.ks, self.vs) for t in lst)
