"""Turn a nested export (outputs/nested_<model>_gs128.pt) into a self-contained, framework-agnostic weight package:
  bitnest_<model>/
    planes.safetensors   every decoder Linear: <name>.hi / .lo (uint8 [N,K/2], unsigned nibbles u=q+8; byte j holds column 2j
                         in the low nibble and 2j+1 in the high nibble), <name>.scale (fp16 [N,K/128], GPTQ group scale s4;
                         target scale = s4/16), <name>.bias (fp16, if any)
    aux.safetensors      embed_tokens (bf16, 8-bit RTN dequantized, R1-rotated), lm_head (bf16, R1-rotated, final norm fused),
                         final_norm (bf16, after fusion), had_K.<K> (R4 blocked-Hadamard matrices)
    meta.json            architecture / formulas / layout / activation quantization / R4 layers / KV offsets / RoPE
    config.json + tokenizer*  from the source model (tie_word_embeddings set to false)
    README.md, checksums.sha256
Usage: MODEL=<key> CUDA_VISIBLE_DEVICES=<g> python bitnest/export_package.py [nested.pt] [out_dir]"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitnest.generate import build_rotated  # noqa: E402
from eval_utils.model_registry import get_entry  # noqa: E402


def _norm(k):
    return k[:-7] if k.endswith(".module") else k


def main():
    ent = get_entry(); mk = ent["name"]
    src = sys.argv[1] if len(sys.argv) > 1 else f"outputs/nested_{mk}_gs128.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else f"release/bitnest_{mk}"
    os.makedirs(out, exist_ok=True)
    pack = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    assert pack.get("pack") == "unsigned+8", pack.get("pack")
    arch = pack.get("arch") or ent["arch"]; gs = int(pack["gs"])
    print(f"[pkg] {mk} arch={arch} gs={gs} nested={len(pack['nested'])} -> {out}", flush=True)
    # ---- planes ----
    planes = {}; layer_meta = {}
    for k, e in pack["nested"].items():
        n = _norm(k); N, K = e["shape"]
        planes[f"{n}.hi"] = e["hi"].contiguous(); planes[f"{n}.lo"] = e["lo"].contiguous(); planes[f"{n}.scale"] = e["scale"].to(torch.float16).contiguous()
        if e.get("bias") is not None:
            planes[f"{n}.bias"] = e["bias"].to(torch.float16).contiguous()
        layer_meta[n] = dict(shape=[N, K], groups=int(e["scale"].shape[1]), has_bias=e.get("bias") is not None)
    save_file(planes, f"{out}/planes.safetensors", metadata={"format": "bitnest-planes-v1"})
    del planes
    # ---- aux: rotated lm_head / norms (rebuild the rotated model once and keep only these tensors) ----
    model = build_rotated(pack["input_model"], pack["rot"], arch)
    aux = {"embed_tokens": pack["embed"].to(torch.bfloat16).contiguous(), "lm_head": model.lm_head.weight.data.to(torch.bfloat16).cpu().contiguous(),
           "final_norm": model.model.norm.weight.data.to(torch.bfloat16).cpu().contiguous()}
    norms_identity = all(bool(torch.allclose(l.input_layernorm.weight.float().cpu(), torch.ones_like(l.input_layernorm.weight.float().cpu()))) and
                         bool(torch.allclose(l.post_attention_layernorm.weight.float().cpu(), torch.ones_like(l.post_attention_layernorm.weight.float().cpu()))) for l in model.model.layers)
    if not norms_identity:   # safety net: after fusion they should all be 1; otherwise store them per layer
        for i, l in enumerate(model.model.layers):
            aux[f"layers.{i}.input_layernorm"] = l.input_layernorm.weight.data.to(torch.bfloat16).cpu().contiguous()
            aux[f"layers.{i}.post_attention_layernorm"] = l.post_attention_layernorm.weight.data.to(torch.bfloat16).cpu().contiguous()
    had = {_norm(k): v for k, v in pack["had"].items()}; had_layers = {}
    for n, h in had.items():
        had_layers[n] = dict(K=int(h["K"]), fp32=bool(h["fp32"]))
        if h.get("had_K") is not None:
            aux[f"had_K.{int(h['K'])}"] = h["had_K"].to(torch.float32).cpu().contiguous()
    save_file(aux, f"{out}/aux.safetensors", metadata={"format": "bitnest-aux-v1"})
    cfg = model.config; hd = cfg.hidden_size // cfg.num_attention_heads
    kv_bias = any(("k_proj" in n or "v_proj" in n) and m["has_bias"] for n, m in layer_meta.items())
    del model; torch.cuda.empty_cache()
    # ---- config / tokenizer ----
    import transformers
    c = transformers.AutoConfig.from_pretrained(pack["input_model"]); c.tie_word_embeddings = False; c.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(pack["input_model"]).save_pretrained(out)
    meta = dict(
        format="bitnest-package-v1", model=mk, arch=arch, source_model=pack["input_model"], rotation=os.path.basename(pack["rot"]), group_size=gs, pack="unsigned+8",
        weight_planes=dict(hi="GPTQ-W4 code q4 in [-8,7] stored as u=q4+8", lo="signed residual qr in [-8,7] (in [0,7] where q4=-8) stored as u=qr+8",
                           byte_layout="uint8 [N, K/2]: low 4 bits of byte j = column 2j, high 4 bits = column 2j+1",
                           draft="W4 = (hi-8) * scale  (scale: fp16 [N,K/128] group scales, one group per 128 columns along K)",
                           target="W8 = (16*(hi-8) + (lo-8)) * scale/16", int8_code="q8 = 16*q4 + qr in [-128,119]"),
        activation=dict(scheme="symmetric int8, clip 1.0, per token (amax/127); o_proj inputs per token in groups of head_dim=%d" % hd,
                        per_layer_group={n: (hd if n.endswith("o_proj") else -1) for n in layer_meta}),
        online_hadamard=dict(note="R4: down_proj inputs go through a blocked Hadamard first (K x 2^k structure, hadK in aux.had_K.<K>, normalized by 1/sqrt(dim)); "
                                  "R1/R2 are fused offline into the weights / embed / lm_head", layers=had_layers),
        norms=dict(fused=True, decoder_norms_identity=norms_identity,
                   note="RMSNorm weights are fused into the adjacent Linears (fuse_layer_norms), decoder norm weights = 1; the final norm is fused into lm_head, final_norm is saved as well"),
        kv=dict(nesting="when K/V are written to the cache: per-token scale s=max|x|/7, q4=clip(round(x/s)), qr=clip(round((x-q4 s)/(s/16))); draft reads q4 (KV4), target reads 16q4+qr (KV8)",
                per_channel_offset=("k_proj/v_proj have bias: subtract it before quantization (K side needs no correction, V side is added back after attention)" if kv_bias else "no bias, no offset needed"),
                r3="optional: per-head normalized Hadamard on K/Q before writing / querying (recommended for Qwen)"),
        rope=dict(theta=getattr(cfg, "rope_theta", None), scaling=getattr(cfg, "rope_scaling", None), max_position_embeddings=cfg.max_position_embeddings),
        layers=layer_meta, embed_note="embed_tokens: 8-bit RTN dequantized to bf16 (same as the GPU engine), R1 rotation included",
        reference=dict(loader="bitnest/reference_model.py (pure PyTorch dequantizing loader, runs on CPU)", gpu_runner="bitnest/generate.py"))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)
    readme = f"""# BitNest weight package · {mk}

Self-contained nested W4/W8 weight package (8 bits/weight in total; the draft reads only the hi plane = 4 bits/weight). Layout and formulas: `meta.json`.
- `planes.safetensors`: hi/lo planes + group scales (+bias) of every decoder Linear.
- `aux.safetensors`: embed_tokens / lm_head / final_norm (offline rotation and norm fusion applied) and the R4 Hadamard matrices.
- `config.json`, tokenizer: from the source model.
- Reference loader: `python bitnest/reference_model.py --pkg <this dir> --mode target|draft [--device cpu|cuda] [--ppl prompts.pt --a8]`.
- GPU engine: `python bitnest/generate.py --pkg <this dir> --prompts prompts.pt --v2_attn --kv_planes --out result.json`.
"""
    open(f"{out}/README.md", "w").write(readme)
    with open(f"{out}/checksums.sha256", "w") as f:
        for fn in sorted(os.listdir(out)):
            if fn == "checksums.sha256":
                continue
            h = hashlib.sha256(open(f"{out}/{fn}", "rb").read()).hexdigest(); f.write(f"{h}  {fn}\n")
    tot = sum(os.path.getsize(f"{out}/{fn}") for fn in os.listdir(out)) / 2**30
    print(f"[pkg] done: {out} ({tot:.2f} GiB; planes {os.path.getsize(out+'/planes.safetensors')/2**30:.2f} GiB, aux {os.path.getsize(out+'/aux.safetensors')/2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
