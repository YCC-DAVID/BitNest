"""Replace the bf16 lm_head of a weight package by dual planes (RTN W4 high plane + residual low plane, gs128), read the
same way as the decoder (draft reads W4 / target reads W8). For large-vocabulary models (Llama-3 128K / Qwen 152K) the
lm_head is 23-36% of the bytes a draft step reads; leaving it in bf16 eats most of the speculative gain.
Usage: python bitnest/nest_lm_head.py <pkg_dir> [out_dir]   (default: in place; the original aux/planes/meta are kept as *.pre_lmhead)"""
import hashlib
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitnest.w4a8_planes import pack_planes  # noqa: E402

GS = 128


def main():
    src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src
    if out != src:
        shutil.copytree(src, out, dirs_exist_ok=True)
    meta = json.load(open(f"{out}/meta.json")); assert "lm_head" not in meta["layers"], "lm_head is already nested"
    with safe_open(f"{out}/aux.safetensors", framework="pt", device="cpu") as fa:
        aux = {k: fa.get_tensor(k) for k in fa.keys()}
    w = aux.pop("lm_head").float(); V, H = w.shape; assert H % GS == 0
    wg = w.view(V, H // GS, GS); s4 = (wg.abs().amax(dim=2) / 7.0).clamp_min(1e-8)                       # [V, H/GS]
    q4 = torch.round(wg / s4[:, :, None]).clamp(-8, 7); r = wg - q4 * s4[:, :, None]
    qr = torch.round(r / (s4[:, :, None] / 16.0)).clamp(-8, 7); qr = torch.where(q4 <= -8, qr.clamp(min=0), qr)
    sat = ((torch.round(r / (s4[:, :, None] / 16.0)) != qr).float().mean().item())
    w8 = (q4 * 16 + qr) * s4[:, :, None] / 16.0; err8 = (w8 - wg).abs().max().item() / (wg.abs().max().item() + 1e-9); err4 = ((q4 * s4[:, :, None]) - wg).norm().item() / wg.norm().item()
    hi, lo = pack_planes(q4.view(V, H).to(torch.int8), qr.view(V, H).to(torch.int8))
    with safe_open(f"{out}/planes.safetensors", framework="pt", device="cpu") as f:
        planes = {k: f.get_tensor(k) for k in f.keys()}
    planes["lm_head.hi"] = hi; planes["lm_head.lo"] = lo; planes["lm_head.scale"] = s4.to(torch.float16).contiguous()
    for fn in ("aux.safetensors", "planes.safetensors", "meta.json"):
        if out == src and not os.path.exists(f"{out}/{fn}.pre_lmhead"):
            shutil.copy2(f"{out}/{fn}", f"{out}/{fn}.pre_lmhead")
    save_file(planes, f"{out}/planes.safetensors"); save_file({k: v.contiguous() for k, v in aux.items()}, f"{out}/aux.safetensors")
    meta["layers"]["lm_head"] = dict(shape=[V, H], groups=H // GS, has_bias=False)
    meta["lm_head_note"] = (f"lm_head as RTN dual planes (W4 high + residual, gs{GS}, no GPTQ): draft reads W4 / target reads W8, per-token A8 inputs; "
                            f"residual saturation {sat:.4%}, W8 max relative error {err8:.2e}, W4 relative Frobenius error {err4:.3f}; R1 rotation and final norm already fused")
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{out}/checksums.sha256", "w") as cf:
        for fn in sorted(os.listdir(out)):
            if fn.endswith(".pre_lmhead") or fn == "checksums.sha256" or os.path.isdir(f"{out}/{fn}"):
                continue
            cf.write(f"{hashlib.sha256(open(f'{out}/{fn}', 'rb').read()).hexdigest()}  {fn}\n")
    print(f"[nest_lm_head] {out}: lm_head [{V},{H}] -> planes {hi.numel()*2/2**30:.2f} GiB (bf16 was {V*H*2/2**30:.2f} GiB); "
          f"residual saturation {sat:.3%}, W8 rel. error {err8:.1e}, W4 rel. F-norm error {err4:.3f}")


if __name__ == "__main__":
    main()
