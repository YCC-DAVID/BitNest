"""Pure-PyTorch reference loader for BitNest weight packages (no Triton, no repository modeling; runs on CPU):
dequantizes the hi/lo planes into bf16 weights (mode=target -> W8, mode=draft -> W4) inside transformers' native
Llama / Qwen2 model, attaches the R4 online Hadamard to down_proj inputs and, optionally, per-token A8 fake quantization.
Useful for numerical alignment on other runtimes and as a starting point for format conversions.
Usage: python bitnest/reference_model.py --pkg bitnest_llama2 --mode target --device cuda --ppl prompts/wiki2.pt --n 3 [--a8]"""
import argparse
import json
import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def dequant(hi, lo, scale, mode):
    q4 = torch.stack([(hi & 0xF).to(torch.int16) - 8, (hi >> 4).to(torch.int16) - 8], -1).reshape(hi.shape[0], -1)   # columns 2j, 2j+1
    if mode == "draft":
        q = q4.float(); s = scale.float()
    else:
        qr = torch.stack([(lo & 0xF).to(torch.int16) - 8, (lo >> 4).to(torch.int16) - 8], -1).reshape(lo.shape[0], -1)
        q = (16 * q4 + qr).float(); s = scale.float() / 16.0
    g = q.shape[1] // s.shape[1]
    return (q.view(q.shape[0], s.shape[1], g) * s[:, :, None]).reshape(q.shape).to(torch.bfloat16)


def hadamard_blocked(x, hadK, K):
    """Blocked Hadamard: last dim n = K * 2^k; Walsh-Hadamard butterfly over the 2^k dim, then multiply by hadK;
    normalized by 1/sqrt(n). Same as utils/hadamard_utils.matmul_hadU."""
    n = x.shape[-1]; shp = x.shape; xf = x.reshape(-1, n).float()
    if K > 1:
        xf = xf.view(-1, K, n // K)          # [B, K, m]
        m = n // K
    else:
        xf = xf.view(-1, 1, n); m = n
    h = 1; y = xf
    while h < m:   # butterfly over the m dim
        y = y.view(-1, K, m // (2 * h), 2, h); a = y[:, :, :, 0, :]; b = y[:, :, :, 1, :]; y = torch.stack([a + b, a - b], 3).view(-1, K, m); h *= 2
    if K > 1:
        y = torch.einsum("ij,bjm->bim", hadK.float().to(y.device), y)
    return (y.reshape(shp) / math.sqrt(n)).to(x.dtype)


def act_quant_a8(x, group):
    x2 = x.float(); shp = x2.shape; x2 = x2.reshape(-1, shp[-1])
    if group == -1:
        s = (x2.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-8); xq = torch.round(x2 / s).clamp(-128, 127) * s
    else:
        xg = x2.view(x2.shape[0], -1, group); s = (xg.abs().amax(2, keepdim=True) / 127.0).clamp_min(1e-8); xq = (torch.round(xg / s).clamp(-128, 127) * s).reshape(x2.shape)
    return xq.reshape(shp).to(x.dtype)


def load_reference(pkg, mode="target", device="cpu", a8=False):
    import transformers
    meta = json.load(open(f"{pkg}/meta.json")); cfg = transformers.AutoConfig.from_pretrained(pkg); cfg.tie_word_embeddings = False
    cls = {"llama": transformers.LlamaForCausalLM, "qwen2": transformers.Qwen2ForCausalLM}[meta["arch"]]
    with torch.device("meta"):
        model = cls(cfg)
    model = model.to_empty(device=device).to(torch.bfloat16)
    planes = load_file(f"{pkg}/planes.safetensors", device="cpu"); aux = load_file(f"{pkg}/aux.safetensors", device="cpu")
    sd = {}
    for n, lm in meta["layers"].items():
        w = dequant(planes[f"{n}.hi"], planes[f"{n}.lo"], planes[f"{n}.scale"], mode); sd[f"{n}.weight"] = w
        if lm["has_bias"]:
            sd[f"{n}.bias"] = planes[f"{n}.bias"].to(torch.bfloat16)
    sd["model.embed_tokens.weight"] = aux["embed_tokens"]; sd["model.norm.weight"] = aux["final_norm"]
    if "lm_head" not in meta["layers"]:
        sd["lm_head.weight"] = aux["lm_head"]
    for i in range(cfg.num_hidden_layers):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            sd[f"model.layers.{i}.{nm}.weight"] = aux.get(f"layers.{i}.{nm}", torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.__class__(config=cfg, device=device)   # rebuild inv_freq (empty after meta init)
    # R4: Hadamard on down_proj inputs; optional A8 fake quantization of every nested Linear input
    had = meta["online_hadamard"]["layers"]; groups = meta["activation"]["per_layer_group"]
    for n, mod in model.named_modules():
        if n in meta["layers"]:
            h = had.get(n); g = groups.get(n, -1); hk = aux.get(f"had_K.{h['K']}") if h else None

            def pre(m, args, _h=h, _hk=hk, _g=g):
                x = args[0]
                if _h is not None:
                    x = hadamard_blocked(x, _hk, _h["K"]) if _hk is not None else hadamard_blocked(x, None, 1)
                if a8:
                    x = act_quant_a8(x, _g)
                return (x,)
            mod.register_forward_pre_hook(pre)
    return model.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", required=True)
    p.add_argument("--mode", default="target", choices=["target", "draft"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ppl", default=None, help="token file [N, L] (tools/make_prompts.py); prints teacher-forced PPL")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--a8", action="store_true", help="per-token int8 activation fake quantization (matches the GPU engine)")
    a = p.parse_args()
    torch.set_grad_enabled(False); m = load_reference(a.pkg, a.mode, a.device, a.a8)
    if a.ppl:
        X = torch.load(a.ppl)[: a.n]; nll = nt = 0.0
        for i in range(X.shape[0]):
            ids = X[i:i + 1].to(a.device); lg = m(ids).logits[0, :-1].float(); nll += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item(); nt += ids.shape[1] - 1
        print(f"REF_PPL mode={a.mode} a8={a.a8} n={X.shape[0]} ppl={math.exp(nll / nt):.4f}", flush=True)
    else:
        tok = __import__("transformers").AutoTokenizer.from_pretrained(a.pkg); ids = tok("The capital of France is", return_tensors="pt").input_ids.to(a.device)
        out = m.generate(ids, max_new_tokens=16, do_sample=False); print(tok.decode(out[0]))
