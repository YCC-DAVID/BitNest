"""Cold load: build the BitNest model directly from a weight package (bitnest_<model>/) without materializing the bf16
model or re-running the rotation -> build-phase peak ~ planes + embed/lm_head (7-9 GiB for 7B), which is what makes
16/24 GB devices (Jetson) feasible.
The model uses transformers' native Llama / Qwen2 classes (instantiated on the meta device); decoder Linears are
replaced by DualPlaneLinear and attention goes through the registered decode kernel."""
import json
import os
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cold_load(pkg, attn_impl=None, device="cuda", DualPlaneLinear=None):
    import transformers
    if DualPlaneLinear is None:   # must be the caller's class (the running generate module), otherwise the MODE switch / isinstance checks point at another module copy
        from bitnest.generate import DualPlaneLinear
    meta = json.load(open(f"{pkg}/meta.json")); cfg = transformers.AutoConfig.from_pretrained(pkg); cfg.tie_word_embeddings = False
    if attn_impl:
        cfg._attn_implementation = attn_impl
    cls = {"llama": transformers.LlamaForCausalLM, "qwen2": transformers.Qwen2ForCausalLM}[meta["arch"]]
    with torch.device("meta"):
        model = cls(cfg).to(dtype=torch.bfloat16)   # cast on meta first: avoids materializing embed/lm_head in fp32 (two fp32 copies of a 152K vocab = 4.4 GB)
    hd = cfg.hidden_size // cfg.num_attention_heads
    # 1) planes -> DualPlaneLinear (layer by layer from safetensors straight to the GPU, never through bf16 weights)
    with safe_open(f"{pkg}/planes.safetensors", framework="pt", device=device) as f, safe_open(f"{pkg}/aux.safetensors", framework="pt", device=device) as fa:
        had_layers = meta["online_hadamard"]["layers"]; n = 0
        for name, lm in meta["layers"].items():
            parent = model; parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            old = getattr(parent, parts[-1]); assert isinstance(old, nn.Linear) and (old.out_features, old.in_features) == tuple(lm["shape"]), name
            if name == "lm_head" and getattr(model.config, "tie_word_embeddings", False):
                model.config.tie_word_embeddings = False
            ent = dict(hi=f.get_tensor(f"{name}.hi"), lo=f.get_tensor(f"{name}.lo"), scale=f.get_tensor(f"{name}.scale"), bias=(f.get_tensor(f"{name}.bias") if lm["has_bias"] else None), shape=tuple(lm["shape"]))
            h = had_layers.get(name); had = dict(K=h["K"], had_K=(fa.get_tensor(f"had_K.{h['K']}") if f"had_K.{h['K']}" in fa.keys() else None), fp32=h["fp32"]) if h else None
            gs = hd if parts[-1] == "o_proj" else -1
            setattr(parent, parts[-1], DualPlaneLinear(ent, had, gs).to(device)); del old; n += 1
        # 2) embed / lm_head use the tensors read from safetensors directly as parameters (to_empty + copy_ would keep two
        #    bf16 copies on the GPU at once); the remaining small modules (norms / rotary) are materialized and filled
        model.model.embed_tokens.weight = nn.Parameter(fa.get_tensor("embed_tokens").to(torch.bfloat16), requires_grad=False)
        if "lm_head" not in meta["layers"]:
            model.lm_head.weight = nn.Parameter(fa.get_tensor("lm_head").to(torch.bfloat16), requires_grad=False)   # package without nested lm_head: bf16 lm_head
        for nm, mod in list(model.named_modules()):
            if any(p.is_meta for p in mod.parameters(recurse=False)) or any(b.is_meta for b in mod.buffers(recurse=False)):
                mod.to_empty(device=device)
        model.model.norm.weight.data.copy_(fa.get_tensor("final_norm"))
        keys = set(fa.keys())
        for i, l in enumerate(model.model.layers):
            for nm in ("input_layernorm", "post_attention_layernorm"):
                k = f"layers.{i}.{nm}"; getattr(l, nm).weight.data.copy_(fa.get_tensor(k) if k in keys else torch.ones_like(getattr(l, nm).weight))
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.__class__(config=cfg, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    left = [nm for nm, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in nm]; assert not left, left[:5]
    from bitnest.generate import drop_page_cache
    for _f in ("planes.safetensors", "aux.safetensors"):
        drop_page_cache(f"{pkg}/{_f}")   # unified-memory devices: drop the weight files' page cache
    # cast the remaining float tensors to bf16. Note: this also casts DualPlaneLinear's fp16 group scales to bf16 (the
    # uint8 planes are untouched); all published --pkg numbers were measured with this behavior.
    model = model.to(torch.bfloat16).eval()
    return model, n, meta
