"""Export BitNest nested integer codes for every decoder Linear (stage 2 of the pipeline).

For each GPTQ-W4 layer: q4 = int4 codes, s4 = group scales [N, K/GS]; qr = closed-form residual (bitnest/build.py).
  hi plane = q4, lo plane = qr, each packed to uint8 [N, K/2] (unsigned nibble u = q + 8);
  target int8 code q8 = 16*q4 + qr with scale s4/16, draft code q4 with scale s4.
The 8-bit RTN embedding is saved dequantized (bf16); lm_head / norms / rotations are rebuilt by export_package.py.

Usage: MODEL=llama2 torchrun --nproc_per_node=1 bitnest/export_nested.py [outputs/nested_llama2_gs128.pt]
"""
import datetime
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitnest.build import GS, INPUT_MODEL, ROT, TAG, build_model, residual_codes, rotated_w0  # noqa: E402
from bitnest.w4a8_planes import pack_planes  # noqa: E402
from utils import quant_utils  # noqa: E402


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    out = sys.argv[1] if len(sys.argv) > 1 else f"outputs/nested_{os.environ['MODEL']}_gs{GS}.pt"
    print(">>> building W16 (rotated) for W0 ...", flush=True)
    W0 = rotated_w0()
    print(">>> building W4 (GPTQ) ...", flush=True)
    model, _, _ = build_model(4, to_cuda=False)
    ql = quant_utils.find_qlayers(model, layers=[torch.nn.Linear, torch.nn.Embedding])
    nested, embed, n_sat, n_tot, skipped = {}, None, 0, 0, []
    for name, m in ql.items():
        if "lm_head" in name:
            continue
        if not (hasattr(m, "int_weight") and hasattr(m, "scale")):
            continue
        q4 = m.int_weight.to(torch.int8)
        if q4.abs().max().item() > 8:                 # 8-bit module (embedding): keep the dequantized weight
            skipped.append(name)
            if "embed_tokens" in name:
                embed = m.weight.data.detach().to(torch.bfloat16).cpu()
            continue
        s4 = m.scale.float(); s4 = s4 if s4.dim() == 2 else s4.view(-1, 1)
        if s4.shape == q4.shape:                      # ptq stores group scales expanded to full size: fold back to [N, K/GS]
            g = int(GS); s4g = s4.view(s4.shape[0], -1, g)
            assert (s4g.max(dim=2).values == s4g.min(dim=2).values).all(), (name, "scale is not constant within a group")
            s4 = s4g[:, :, 0].contiguous()
        assert s4.shape[1] == q4.shape[1] // int(GS), (name, s4.shape, q4.shape)
        s4f = s4.repeat_interleave(q4.shape[1] // s4.shape[1], dim=1)
        qr, qr_raw = residual_codes(q4.float(), s4f, W0[name].float())
        n_sat += int((qr != qr_raw).sum()); n_tot += qr.numel()
        hi, lo = pack_planes(q4, qr.to(torch.int8))
        name = name[:-7] if name.endswith(".module") else name   # strip ActQuantWrapper's ".module" -> native HF layer name
        nested[name] = dict(hi=hi.contiguous(), lo=lo.contiguous(), scale=s4.to(torch.float16).contiguous(),
                            bias=(m.bias.detach().to(torch.bfloat16).cpu() if getattr(m, "bias", None) is not None else None),
                            shape=tuple(q4.shape))
        del q4, s4, s4f, qr_raw, qr
    print(f"[export] nested {len(nested)} layers, residual saturation {n_sat/max(n_tot,1):.4%} ({n_sat}/{n_tot}), "
          f"skipped 8-bit modules {skipped}", flush=True)
    assert nested and n_tot > 0
    had = {}
    for name, m in quant_utils.find_qlayers(model).items():   # ActQuantWrapper layers: record the online Hadamard (R4) config
        if getattr(m, "online_full_had", False):
            had[name] = dict(K=int(m.K), had_K=(m.had_K.cpu() if torch.is_tensor(m.had_K) else None), fp32=bool(m.fp32_had))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(dict(model=os.environ["MODEL"], tag=TAG, input_model=INPUT_MODEL, rot=ROT, gs=int(GS), nested=nested, embed=embed, had=had,
                    a_bits=8, a_sym=True, a_clip=1.0, k_bits=8, v_bits=8, pack="unsigned+8",
                    note="hi=q4 lo=qr q8=16*q4+qr s8=s4/16"), out)
    print(f"[export] saved {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
