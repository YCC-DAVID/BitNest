"""Nested KV cache ("4-bit first, then residual"), isomorphic to the weight nesting.
Quantization (at write time, one scale per token per head): s4 = max|x| / 7;  q4 = clip(round(x/s4), -8, 7)  (draft KV4);
   qr = clip(round((x - q4*s4) / (s4/16)), -8, 7), qr in [0, 7] where q4 = -8;  target KV8 = (16*q4 + qr) * s4/16.
Storage: high plane (q4+8) / low plane (qr+8), each [B, Hk, Lmax, D//2] uint8 (byte j: low nibble = column 2j,
   high nibble = column 2j+1), plus scale [B, Hk, Lmax] fp32.
Attention kernel: same structure as decode_attn, decoding the planes on the fly: the draft reads only the high plane
   (KV4, half the bytes), the target reads both (KV8); the per-token scale is folded into the scores on the K side and
   into P on the V side."""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quant_kv_planes(x: torch.Tensor):
    """Reference quantizer: x [B, Hk, L, D] (bf16/fp16/fp32) -> (hi, lo) uint8 [B,Hk,L,D//2], scale fp32 [B,Hk,L]."""
    xf = x.float(); s4 = (xf.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)                      # [B,Hk,L]
    q4 = torch.round(xf / s4[..., None]).clamp(-8, 7)
    r = xf - q4 * s4[..., None]; qr = torch.round(r / (s4[..., None] / 16.0)).clamp(-8, 7)
    qr = torch.where(q4 <= -8, qr.clamp(min=0), qr)

    def pk(q):
        u = (q.to(torch.int16) + 8).to(torch.uint8); return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    return pk(q4), pk(qr), s4.contiguous()


def dequant_kv_planes(hi, lo, scale, read_lo: bool):
    """Reference dequantizer: planes -> fp32 [B,Hk,L,D]."""
    def unpk(p):
        l4 = (p & 0xF).to(torch.int16) - 8; h4 = (p >> 4).to(torch.int16) - 8
        return torch.stack([l4, h4], dim=-1).reshape(*p.shape[:-1], -1).float()
    q4 = unpk(hi)
    if read_lo:
        return (16 * q4 + unpk(lo)) * (scale[..., None] / 16.0)
    return q4 * scale[..., None]


@triton.jit
def _attn_planes_kernel(q_ptr, khi_ptr, klo_ptr, ks_ptr, vhi_ptr, vlo_ptr, vs_ptr, o_ptr, m_ptr, l_ptr, seq_ptr, SPLIT, sm_scale, M_REAL, LMAX, WIN,
                        stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kl, stride_sb, stride_sh,
                        stride_ob, stride_oh, stride_os, stride_or, stride_mb, stride_mh, stride_ms,
                        READ_LO: tl.constexpr, READ_LO_V: tl.constexpr, LO_MODE: tl.constexpr, LO_MODE_V: tl.constexpr, HK: tl.constexpr, GROUP: tl.constexpr, M: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr):
    # GQA-grouped: one program = one KV head x the GROUP query heads of that group x M queries
    # (row r: query head kvh*GROUP + r//M, query r%M); each plane segment is loaded and unpacked once
    pid = tl.program_id(0); pid_s = tl.program_id(1)
    b = pid // HK; kvh = pid % HK
    seq0 = tl.load(seq_ptr).to(tl.int32)
    L_total = tl.minimum(seq0 + M_REAL, LMAX)
    per = (L_total + SPLIT - 1) // SPLIT; l0 = pid_s * per; l1 = tl.minimum(l0 + per, L_total)
    r = tl.arange(0, R); dh = tl.arange(0, D // 2)
    rr = tl.minimum(r, GROUP * M - 1); g = rr // M; i = rr % M; h = kvh * GROUP + g
    qb = q_ptr + b * stride_qb + h[:, None] * stride_qh + i[:, None] * stride_qm
    qe = tl.load(qb + 2 * dh[None, :]).to(tl.bfloat16); qo = tl.load(qb + 2 * dh[None, :] + 1).to(tl.bfloat16)     # [R, D/2] even / odd columns (matches the nibble layout)
    qi = tl.minimum(i, M_REAL - 1)
    m_i = tl.full((R,), -1e30, tl.float32); l_i = tl.zeros((R,), tl.float32)
    acc_e = tl.zeros((R, D // 2), tl.float32); acc_o = tl.zeros((R, D // 2), tl.float32)
    for s in range(l0, l1, BLOCK_L):
        ls = s + tl.arange(0, BLOCK_L); kmask = ls < l1
        base = b * stride_kb + kvh * stride_kh + ls[:, None] * stride_kl + dh[None, :]
        bh_ = tl.load(khi_ptr + base, mask=kmask[:, None], other=0)
        ke = (bh_ & 0xF).to(tl.int8) - 8; ko = (bh_ >> 4).to(tl.int8) - 8
        rl = (LO_MODE == 1) | ((LO_MODE == 2) & (ls >= seq0 - WIN))                      # read the low plane at this position? (1 always / 2 residual window only / 0 never)
        ks = tl.load(ks_ptr + b * stride_sb + kvh * stride_sh + ls, mask=kmask, other=0.0).to(tl.float32)
        if READ_LO:
            ks = ks / 16.0; ke = ke * 16; ko = ko * 16
            if LO_MODE == 1:                                                                 # KV8: always read the low plane
                bl_ = tl.load(klo_ptr + base, mask=kmask[:, None], other=136)
                ke = ke + ((bl_ & 0xF).to(tl.int8) - 8); ko = ko + ((bl_ >> 4).to(tl.int8) - 8)
            elif tl.sum(rl.to(tl.int32), axis=0) > 0:                                       # window: only issue the load if some position of the block is inside it
                bl_ = tl.load(klo_ptr + base, mask=(kmask & rl)[:, None], other=136)      # unread positions get 0x88 (= two zero residuals)
                ke = ke + ((bl_ & 0xF).to(tl.int8) - 8); ko = ko + ((bl_ >> 4).to(tl.int8) - 8)
        sc = tl.dot(qe, tl.trans(ke.to(tl.bfloat16)), out_dtype=tl.float32) + tl.dot(qo, tl.trans(ko.to(tl.bfloat16)), out_dtype=tl.float32)
        sc = sc * (ks[None, :] * sm_scale)
        valid = (ls[None, :] <= (seq0 + qi[:, None])) & kmask[None, :]
        sc = tl.where(valid, sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=1)); alpha = tl.exp(m_i - m_new); p = tl.where(valid, tl.exp(sc - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vh_ = tl.load(vhi_ptr + base, mask=kmask[:, None], other=0)
        ve = (vh_ & 0xF).to(tl.int8) - 8; vo = (vh_ >> 4).to(tl.int8) - 8
        vs = tl.load(vs_ptr + b * stride_sb + kvh * stride_sh + ls, mask=kmask, other=0.0).to(tl.float32)
        if READ_LO_V:
            rlv = (LO_MODE_V == 1) | ((LO_MODE_V == 2) & (ls >= seq0 - WIN))
            vs = vs / 16.0; ve = ve * 16; vo = vo * 16
            if LO_MODE_V == 1:
                vl_ = tl.load(vlo_ptr + base, mask=kmask[:, None], other=136)
                ve = ve + ((vl_ & 0xF).to(tl.int8) - 8); vo = vo + ((vl_ >> 4).to(tl.int8) - 8)
            elif tl.sum(rlv.to(tl.int32), axis=0) > 0:
                vl_ = tl.load(vlo_ptr + base, mask=(kmask & rlv)[:, None], other=136)
                ve = ve + ((vl_ & 0xF).to(tl.int8) - 8); vo = vo + ((vl_ >> 4).to(tl.int8) - 8)
        pv = (p * vs[None, :]).to(tl.bfloat16)                                             # per-token scale folded into P
        acc_e = acc_e * alpha[:, None] + tl.dot(pv, ve.to(tl.bfloat16), out_dtype=tl.float32)
        acc_o = acc_o * alpha[:, None] + tl.dot(pv, vo.to(tl.bfloat16), out_dtype=tl.float32)
        m_i = m_new
    ob = o_ptr + b * stride_ob + kvh * stride_oh + pid_s * stride_os + r[:, None] * stride_or
    tl.store(ob + 2 * dh[None, :], acc_e); tl.store(ob + 2 * dh[None, :] + 1, acc_o)
    tl.store(m_ptr + b * stride_mb + kvh * stride_mh + pid_s * stride_ms + r, m_i); tl.store(l_ptr + b * stride_mb + kvh * stride_mh + pid_s * stride_ms + r, l_i)


def decode_attention_planes(q, khi, klo, ks, vhi, vlo, vs, seq0, read_lo, sm_scale=None, split=None, block_l=64):
    """read_lo: False -> KV4 (high plane only); True -> KV8; ("window", N) -> KV8 for the most recent N positions and
    KV4 elsewhere (draft residual window); dict(k=.., v=..) -> separate settings for K and V.
    q [B,H,M,D]; planes [B,Hk,Lmax,D/2] uint8; scales [B,Hk,Lmax] fp32."""
    from bitnest.decode_attn import _attn_merge_kernel, split_for, _rows

    def _parse(m):
        if isinstance(m, tuple):
            return 2, int(m[1]), True
        return (1 if m else 0), 0, bool(m)
    if isinstance(read_lo, dict):
        (lo_mode, win, READ), (lo_mode_v, win_v, READ_V) = _parse(read_lo["k"]), _parse(read_lo["v"]); win = max(win, win_v)
    else:
        lo_mode, win, READ = _parse(read_lo); lo_mode_v, READ_V = lo_mode, READ
    B, H, M, D = q.shape; Hk = khi.shape[1]; GROUP = H // Hk; Lmax = khi.shape[2]
    if sm_scale is None:
        sm_scale = D ** -0.5
    if split is None:
        split = split_for(B, Hk, Lmax, block_l)
    R = _rows(GROUP, M)
    if not torch.is_tensor(seq0):
        seq0 = torch.tensor(int(seq0), device=q.device, dtype=torch.int32)
    q = q.contiguous()
    o = torch.empty((B, Hk, split, R, D), device=q.device, dtype=torch.float32); m = torch.empty((B, Hk, split, R), device=q.device, dtype=torch.float32); l = torch.empty_like(m)
    _attn_planes_kernel[(B * Hk, split)](q, khi, klo, ks, vhi, vlo, vs, o, m, l, seq0, split, sm_scale, M, Lmax, win,
        q.stride(0), q.stride(1), q.stride(2), khi.stride(0), khi.stride(1), khi.stride(2), ks.stride(0), ks.stride(1),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), m.stride(0), m.stride(1), m.stride(2),
        READ_LO=READ, READ_LO_V=READ_V, LO_MODE=lo_mode, LO_MODE_V=lo_mode_v, HK=Hk, GROUP=GROUP, M=M, R=R, D=D, BLOCK_L=block_l, num_warps=4)
    out = torch.empty((B, Hk, R, D), device=q.device, dtype=q.dtype)
    _attn_merge_kernel[(B * Hk,)](o, m, l, out, split, o.stride(0), o.stride(1), o.stride(2), o.stride(3), m.stride(0), m.stride(1), m.stride(2),
        out.stride(0), out.stride(1), out.stride(2), HK=Hk, R=R, D=D, num_warps=4)
    return out[:, :, :GROUP * M].reshape(B, H, M, D)


# ---- write side: one fused launch does per-token per-head 4-bit quantization + residual + nibble packing and scatters
# the planes / scales to cache_position
@triton.jit
def _quant_kv_planes_kernel(x_ptr, pos_ptr, hi_ptr, lo_ptr, s_ptr, off_ptr, stride_xb, stride_xh, stride_xm, stride_pb, stride_ph, stride_pl, stride_sb, stride_sh,
                            HK: tl.constexpr, D: tl.constexpr, HAS_OFF: tl.constexpr):
    # grid = (B*HK, M): axis0 = b*HK + h, axis1 = m
    bh = tl.program_id(0); mi = tl.program_id(1); b = bh // HK; h = bh % HK
    d = tl.arange(0, D)
    xb = x_ptr + b * stride_xb + h * stride_xh + mi * stride_xm
    x = tl.load(xb + d).to(tl.float32)
    dh = tl.arange(0, D // 2)
    xe = tl.load(xb + 2 * dh).to(tl.float32); xo = tl.load(xb + 2 * dh + 1).to(tl.float32)     # even / odd columns (avoids reshape layout ambiguity)
    if HAS_OFF:   # subtract a per-(kv head, channel) constant offset (Qwen k/v_proj bias outlier channels): exact for K (softmax shift invariance), added back after attention for V
        ob = off_ptr + h * D
        x = x - tl.load(ob + d).to(tl.float32); xe = xe - tl.load(ob + 2 * dh).to(tl.float32); xo = xo - tl.load(ob + 2 * dh + 1).to(tl.float32)
    s4 = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), 7.0), 1e-8)
    q4e = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.div_rn(xe, s4)), -8.0), 7.0); q4o = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.div_rn(xo, s4)), -8.0), 7.0)
    qre = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.div_rn(xe - q4e * s4, tl.div_rn(s4, 16.0))), -8.0), 7.0); qro = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.div_rn(xo - q4o * s4, tl.div_rn(s4, 16.0))), -8.0), 7.0)
    qre = tl.where(q4e <= -8.0, tl.maximum(qre, 0.0), qre); qro = tl.where(q4o <= -8.0, tl.maximum(qro, 0.0), qro)
    hi_b = (q4e + 8.0).to(tl.int32) + (q4o + 8.0).to(tl.int32) * 16
    lo_b = (qre + 8.0).to(tl.int32) + (qro + 8.0).to(tl.int32) * 16
    pos = tl.load(pos_ptr + mi).to(tl.int64)
    pb = b * stride_pb + h * stride_ph + pos * stride_pl
    tl.store(hi_ptr + pb + dh, hi_b.to(tl.uint8)); tl.store(lo_ptr + pb + dh, lo_b.to(tl.uint8))
    tl.store(s_ptr + b * stride_sb + h * stride_sh + pos, s4)


def quant_kv_planes_into(x, cache_position, hi, lo, s, off=None):
    """Quantize x [B,Hk,M,D] (bf16) and write it into planes hi/lo [B,Hk,Lmax,D/2] and scales s [B,Hk,Lmax] at
    cache_position (one launch); optional off [Hk,D]: quantize (x - off)."""
    B, Hk, M, D = x.shape
    _quant_kv_planes_kernel[(B * Hk, M)](x, cache_position, hi, lo, s, off if off is not None else x, x.stride(0), x.stride(1), x.stride(2), hi.stride(0), hi.stride(1), hi.stride(2), s.stride(0), s.stride(1), HK=Hk, D=D, HAS_OFF=off is not None, num_warps=1)
