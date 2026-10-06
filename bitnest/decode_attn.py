"""Flash-decoding style decode attention (GQA-grouped): one program handles one KV head x all GROUP query heads of that
group x M queries (rows = GROUP*M, padded to >= 16 so tl.dot can be used), so each K/V segment is read once. The
sequence is split into parallel segments with a deterministic merge; seq0 is a device scalar (CUDA-graph replay safe)."""
import torch
import triton
import triton.language as tl


def split_for(B, Hk, Lmax, block_l=64):
    """Number of programs B*Hk*split should fill the SMs (A6000: 84): split ~ 168/(B*Hk) as a power of 2 in [4, 64],
    capped at Lmax/(2*block_l)."""
    s = triton.next_power_of_2(max(1, (168 + B * Hk - 1) // (B * Hk)))
    cap = triton.next_power_of_2(max(4, Lmax // (2 * block_l)))
    return int(min(64, cap, max(4, s)))


@triton.jit
def _attn_part_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, l_ptr, seq_ptr, SPLIT, sm_scale, M_REAL, LMAX,
                      stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kl, stride_vb, stride_vh, stride_vl,
                      stride_ob, stride_oh, stride_os, stride_or, stride_mb, stride_mh, stride_ms,
                      HK: tl.constexpr, GROUP: tl.constexpr, M: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr):
    # grid: (B*HK, SPLIT); row r in [0, R): query head = kvh*GROUP + r//M, query = r%M
    # (padding rows r >= GROUP*M map onto the last real row; their results are discarded)
    pid = tl.program_id(0); pid_s = tl.program_id(1)
    b = pid // HK; kvh = pid % HK
    seq0 = tl.load(seq_ptr).to(tl.int32)
    L_total = tl.minimum(seq0 + M_REAL, LMAX)
    per = (L_total + SPLIT - 1) // SPLIT; l0 = pid_s * per; l1 = tl.minimum(l0 + per, L_total)
    r = tl.arange(0, R); do_ = tl.arange(0, D)
    rr = tl.minimum(r, GROUP * M - 1); g = rr // M; i = rr % M; h = kvh * GROUP + g
    q = tl.load(q_ptr + b * stride_qb + h[:, None] * stride_qh + i[:, None] * stride_qm + do_[None, :]).to(tl.bfloat16)   # [R, D]
    qi = tl.minimum(i, M_REAL - 1)
    m_i = tl.full((R,), -1e30, tl.float32); l_i = tl.zeros((R,), tl.float32); acc = tl.zeros((R, D), tl.float32)
    for s in range(l0, l1, BLOCK_L):
        ls = s + tl.arange(0, BLOCK_L); kmask = ls < l1
        k = tl.load(k_ptr + b * stride_kb + kvh * stride_kh + ls[:, None] * stride_kl + do_[None, :], mask=kmask[:, None], other=0.0)   # [BL, D] bf16
        sc = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale                                                            # [R, BL]
        valid = (ls[None, :] <= (seq0 + qi[:, None])) & kmask[None, :]
        sc = tl.where(valid, sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=1)); alpha = tl.exp(m_i - m_new); p = tl.where(valid, tl.exp(sc - m_new[:, None]), 0.0)
        v = tl.load(v_ptr + b * stride_vb + kvh * stride_vh + ls[:, None] * stride_vl + do_[None, :], mask=kmask[:, None], other=0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1); acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new
    tl.store(o_ptr + b * stride_ob + kvh * stride_oh + pid_s * stride_os + r[:, None] * stride_or + do_[None, :], acc)
    tl.store(m_ptr + b * stride_mb + kvh * stride_mh + pid_s * stride_ms + r, m_i); tl.store(l_ptr + b * stride_mb + kvh * stride_mh + pid_s * stride_ms + r, l_i)


@triton.jit
def _attn_merge_kernel(o_ptr, m_ptr, l_ptr, out_ptr, SPLIT, stride_ob, stride_oh, stride_os, stride_or, stride_mb, stride_mh, stride_ms,
                       stride_outb, stride_outh, stride_outr, HK: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0); b = pid // HK; kvh = pid % HK
    r = tl.arange(0, R); do_ = tl.arange(0, D)
    m_g = tl.full((R,), -1e30, tl.float32)
    for s in range(0, SPLIT):
        m_g = tl.maximum(m_g, tl.load(m_ptr + b * stride_mb + kvh * stride_mh + s * stride_ms + r))
    l_g = tl.zeros((R,), tl.float32); acc = tl.zeros((R, D), tl.float32)
    for s in range(0, SPLIT):
        m_s = tl.load(m_ptr + b * stride_mb + kvh * stride_mh + s * stride_ms + r); l_s = tl.load(l_ptr + b * stride_mb + kvh * stride_mh + s * stride_ms + r)
        w = tl.exp(m_s - m_g); l_g += l_s * w
        acc += tl.load(o_ptr + b * stride_ob + kvh * stride_oh + s * stride_os + r[:, None] * stride_or + do_[None, :]) * w[:, None]
    out = acc / tl.maximum(l_g, 1e-30)[:, None]
    tl.store(out_ptr + b * stride_outb + kvh * stride_outh + r[:, None] * stride_outr + do_[None, :], out.to(out_ptr.dtype.element_ty))


def _rows(GROUP, M):
    return int(max(16, triton.next_power_of_2(GROUP * M)))


def decode_attention(q, k_cache, v_cache, seq0, sm_scale=None, split=None, block_l=64):
    """q [B, H, M, D] (M new queries at positions seq0..seq0+M-1); k/v cache [B, H_kv, Lmax, D] (first seq0+M valid).
    Returns [B, H, M, D]."""
    B, H, M, D = q.shape; Hk = k_cache.shape[1]; GROUP = H // Hk; Lmax = k_cache.shape[2]
    if sm_scale is None:
        sm_scale = D ** -0.5
    if split is None:
        split = split_for(B, Hk, Lmax, block_l)
    R = _rows(GROUP, M)
    if not torch.is_tensor(seq0):
        seq0 = torch.tensor(int(seq0), device=q.device, dtype=torch.int32)
    q = q.contiguous()
    o = torch.empty((B, Hk, split, R, D), device=q.device, dtype=torch.float32); m = torch.empty((B, Hk, split, R), device=q.device, dtype=torch.float32); l = torch.empty_like(m)
    _attn_part_kernel[(B * Hk, split)](q, k_cache, v_cache, o, m, l, seq0, split, sm_scale, M, Lmax,
        q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), m.stride(0), m.stride(1), m.stride(2), HK=Hk, GROUP=GROUP, M=M, R=R, D=D, BLOCK_L=block_l, num_warps=4)
    out = torch.empty((B, Hk, R, D), device=q.device, dtype=q.dtype)
    _attn_merge_kernel[(B * Hk,)](o, m, l, out, split, o.stride(0), o.stride(1), o.stride(2), o.stride(3), m.stride(0), m.stride(1), m.stride(2),
        out.stride(0), out.stride(1), out.stride(2), HK=Hk, R=R, D=D, num_warps=4)
    return out[:, :, :GROUP * M].reshape(B, H, M, D)
