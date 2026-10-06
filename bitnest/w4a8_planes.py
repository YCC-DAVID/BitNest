"""BitNest dual-plane W4A8 / W8A8 Triton kernels (weight group size 128).

Storage (literal nibble split): the 8-bit weight code is u8 = 16*(q4+8) + (qr+8). The high nibble q4+8 is the draft's
uint4 code (zero point 8) and the low nibble qr+8 is the refinement code. The two nibbles live in two separate uint8
planes [N, K//2] (byte j: low half = column 2j, high half = column 2j+1); the draft reads only the high plane, the
target reads both.
  values: w4 = q4 * s4;   w8 = (16*q4 + qr) * s8,  s8 = s4/16   (q4, qr in [-8, 7]; qr in [0, 7] where q4 = -8, so 16*q4+qr >= -128)
Activations: symmetric int8, SpinQuant convention: per token (groupsize=-1) or per 128-group (o_proj, head_dim=128);
  the scale is always passed as sx [M, K//128].
Compute: int8 x int8 -> int32 inside each 128-group (tl.dot), multiplied by the fp32 (weight-group scale x activation
  scale) when leaving the group; split-K partial sums are written to [SPLIT_K, M, N] and reduced deterministically.
  Bit-exact with fake-quant semantics (no saturation / approximation)."""
import os

import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _w4a8_planes_kernel(xq_ptr, sx_ptr, hi_ptr, lo_ptr, sc_ptr, y_ptr, M, N, K,
                        stride_xm, stride_sxm, stride_sn, stride_ym, stride_yk,
                        READ_LO: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); m = tl.arange(0, BLOCK_M)
    n_mask = n < N; m_mask = m < M
    Kh = K // 2; NG = K // GROUP
    gper = (NG + SPLIT_K - 1) // SPLIT_K; g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for g in range(g0, g1):
        j = g * (GROUP // 2) + tl.arange(0, GROUP // 2)
        bh = tl.load(hi_ptr + n[:, None] * Kh + j[None, :], mask=n_mask[:, None], other=0)
        q4_e = (bh & 0xF).to(tl.int8) - 8            # column 2j   (uint4 code - zero point 8)
        q4_o = (bh >> 4).to(tl.int8) - 8             # column 2j+1
        if READ_LO:
            bl = tl.load(lo_ptr + n[:, None] * Kh + j[None, :], mask=n_mask[:, None], other=0)
            w_e = q4_e * 16 + ((bl & 0xF).to(tl.int8) - 8)    # q8 in [-128, 119]
            w_o = q4_o * 16 + ((bl >> 4).to(tl.int8) - 8)
        else:
            w_e = q4_e; w_o = q4_o
        ke = g * GROUP + 2 * tl.arange(0, GROUP // 2)
        xe = tl.load(xq_ptr + m[:, None] * stride_xm + ke[None, :], mask=m_mask[:, None], other=0)
        xo = tl.load(xq_ptr + m[:, None] * stride_xm + ke[None, :] + 1, mask=m_mask[:, None], other=0)
        acc_g = tl.dot(xe, tl.trans(w_e), out_dtype=tl.int32) + tl.dot(xo, tl.trans(w_o), out_dtype=tl.int32)
        s = tl.load(sc_ptr + n * stride_sn + g, mask=n_mask, other=0.0).to(tl.float32)
        if READ_LO:
            s = s / 16.0
        sx = tl.load(sx_ptr + m * stride_sxm + g, mask=m_mask, other=0.0).to(tl.float32)   # activation scale of this group [BM]
        acc += acc_g.to(tl.float32) * (sx[:, None] * s[None, :])
    tl.store(y_ptr + pid_k * stride_yk + m[:, None] * stride_ym + n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _act_quant_kernel(x_ptr, xq_ptr, sx_ptr, K, NG, stride_xm, clip, PER_GROUP: tl.constexpr, GROUP: tl.constexpr, BLOCK_K: tl.constexpr):
    # one program per row (token): per-token or per-128-group absmax -> int8; scales written to sx [M, NG]
    m = tl.program_id(0)
    if PER_GROUP:
        for g in range(0, NG):
            k = g * GROUP + tl.arange(0, GROUP)
            x = tl.load(x_ptr + m * stride_xm + k).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(x), axis=0) * clip / 127.0, 1e-8)
            q = tl.extra.cuda.libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(xq_ptr + m * K + k, q.to(tl.int8)); tl.store(sx_ptr + m * NG + g, s)
    else:
        amax = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + m * stride_xm + k, mask=k < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        s = tl.maximum(tl.max(amax, axis=0) * clip / 127.0, 1e-8)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + m * stride_xm + k, mask=k < K, other=0.0).to(tl.float32)
            q = tl.extra.cuda.libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(xq_ptr + m * K + k, q.to(tl.int8), mask=k < K)
        for g in range(0, NG):
            tl.store(sx_ptr + m * NG + g, s)


@triton.jit
def _reduce_kernel(y_ptr, out_ptr, bias_ptr, M, N, SK, stride_yk, stride_ym, stride_om, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    # out[m, n] = sum_k y[k, m, n] (+ bias[n]); split-K reduction + cast + bias in a single launch
    pid = tl.program_id(0); m = tl.program_id(1)
    n = pid * BLOCK + tl.arange(0, BLOCK); mask = n < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(0, SK):
        acc += tl.load(y_ptr + k * stride_yk + m * stride_ym + n, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + n, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + m * stride_om + n, acc.to(out_ptr.dtype.element_ty), mask=mask)


_CFG = {}   # (N, K, read_lo, BLOCK_M) -> (BLOCK_N, SPLIT_K, num_warps, num_stages); chosen once by calibrate()
_CANDS = [(bn, sk, w, st) for bn in (64, 128) for sk in (4, 8, 16) for w in (4, 8) for st in (2, 3)]


def _cfg(N, K, read_lo, BLOCK_M):
    return _CFG.get((N, K, read_lo, BLOCK_M), (128 if N >= 8192 else 64, 16 if K >= 8192 or N <= 4096 else 8, 4, 2))


def _launch(xq, sx, hi, lo, scale, read_lo, M, N, K, BLOCK_M, cfg, y):
    BN, SK, W, ST = cfg; grid = (triton.cdiv(N, BN), SK)
    _w4a8_planes_kernel[grid](xq, sx, hi, lo, scale, y, M, N, K, xq.stride(0), sx.stride(0), scale.stride(0), y.stride(1), y.stride(0),
                              READ_LO=read_lo, BLOCK_M=BLOCK_M, BLOCK_N=BN, GROUP=GROUP, SPLIT_K=SK, num_warps=W, num_stages=ST)


# calibration workspace cap: candidates whose [SK, M, N] fp32 workspace exceeds this are skipped
# (a large-vocab lm_head at M=64 / SK=16 needs 622 MB per block, too much for 8 GB devices)
CALIB_WS_MB = float(os.environ.get("BITNEST_CALIB_WS_MB", "256"))


def calibrate(N, K, read_lo, M, hi, lo, scale, iters=20):
    """Pick the fastest launch config for (N, K, read_lo, M) by timing on the real weight tensors; stored in _CFG.
    Called by the runner before CUDA-graph capture. The workspace is allocated once for the largest candidate and sliced."""
    import time
    BLOCK_M = 16 if M <= 16 else triton.next_power_of_2(M); key = (N, K, read_lo, BLOCK_M)
    if key in _CFG:
        return _CFG[key]
    xq = torch.randint(-128, 128, (M, K), device=hi.device, dtype=torch.int8); sx = torch.rand(M, K // GROUP, device=hi.device)
    cands = [c for c in _CANDS if c[1] * M * N * 4 <= CALIB_WS_MB * 2**20] or [min(_CANDS, key=lambda c: c[1])]
    ws = torch.empty((max(c[1] for c in cands) * M * N,), device=hi.device, dtype=torch.float32)
    best = None
    for cfg in cands:
        y = ws[: cfg[1] * M * N].view(cfg[1], M, N)
        try:
            _launch(xq, sx, hi, lo, scale, read_lo, M, N, K, BLOCK_M, cfg, y); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _launch(xq, sx, hi, lo, scale, read_lo, M, N, K, BLOCK_M, cfg, y)
            torch.cuda.synchronize(); t = (time.perf_counter() - t0) / iters
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    del ws, xq, sx; _CFG[key] = best[1]; return best[1]


# ---- M=1 GEMV path (on small GPUs such as 8-SM Orin the GEMM kernel above, whose BLOCK_M=16 tl.dot treats a GEMV as
# a GEMM, is instruction-bound at ~45% of the bandwidth roofline).
# Each program: BLOCK_N rows x one K segment (SPLIT_K segments); every iteration loads BLOCK_G 128-groups (BLOCK_G*64
# bytes per row, coalesced), rebuilds int8 codes with bit ops, integer multiply-add along K, and applies
# (group scale x activation scale) once per group.
@triton.jit
def _w4a8_planes_gemv_kernel(xq_ptr, sx_ptr, hi_ptr, lo_ptr, sc_ptr, y_ptr, N, K, stride_sn, stride_yk,
                             READ_LO: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr, GROUP: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = n < N
    Kh = K // 2; NG = K // GROUP; HG: tl.constexpr = GROUP // 2
    gper = (NG + SPLIT_K - 1) // SPLIT_K; g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    jj = tl.arange(0, BLOCK_G * HG)                      # byte columns within one iteration (BLOCK_G groups x 64 bytes)
    gg = tl.arange(0, BLOCK_G)
    for g in range(g0, g1, BLOCK_G):
        col = g * HG + jj; cmask = col < g1 * HG          # groups past the end of this segment are masked to 0 (x too)
        bh = tl.load(hi_ptr + n[:, None] * Kh + col[None, :], mask=n_mask[:, None] & cmask[None, :], other=0)
        q4_e = (bh & 0xF).to(tl.int8) - 8; q4_o = (bh >> 4).to(tl.int8) - 8
        if READ_LO:
            bl = tl.load(lo_ptr + n[:, None] * Kh + col[None, :], mask=n_mask[:, None] & cmask[None, :], other=0)
            w_e = q4_e.to(tl.int32) * 16 + ((bl & 0xF).to(tl.int8) - 8).to(tl.int32)
            w_o = q4_o.to(tl.int32) * 16 + ((bl >> 4).to(tl.int8) - 8).to(tl.int32)
        else:
            w_e = q4_e.to(tl.int32); w_o = q4_o.to(tl.int32)
        ke = 2 * col
        xe = tl.load(xq_ptr + ke, mask=cmask, other=0).to(tl.int32); xo = tl.load(xq_ptr + ke + 1, mask=cmask, other=0).to(tl.int32)
        prod = w_e * xe[None, :] + w_o * xo[None, :]                                   # [BLOCK_N, BLOCK_G*64] int32
        pg = tl.sum(tl.reshape(prod, (BLOCK_N, BLOCK_G, HG)), axis=2)                  # [BLOCK_N, BLOCK_G] per-group integer dot
        gi = g + gg; gmask = gi < g1
        s = tl.load(sc_ptr + n[:, None] * stride_sn + gi[None, :], mask=n_mask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
        if READ_LO:
            s = s / 16.0
        sx = tl.load(sx_ptr + gi, mask=gmask, other=0.0).to(tl.float32)
        acc += tl.sum(pg.to(tl.float32) * (s * sx[None, :]), axis=1)
    tl.store(y_ptr + pid_k * stride_yk + n, acc, mask=n_mask)


_CFG_GEMV = {}   # (N, K, read_lo) -> (BLOCK_N, BLOCK_G, SPLIT_K, num_warps)
_CANDS_GEMV = [(bn, bg, sk, w) for bn in (16, 32, 64) for bg in (1, 2, 4) for sk in (1, 2, 4, 8) for w in (2, 4)]
GEMV_M1 = {"on": os.environ.get("BITNEST_GEMV_M1", "0") == "1"}   # off by default; BITNEST_GEMV_M1=1 for edge devices


def _launch_gemv(xq, sx, hi, lo, scale, read_lo, N, K, cfg, y):
    BN, BG, SK, W = cfg; grid = (triton.cdiv(N, BN), SK)
    _w4a8_planes_gemv_kernel[grid](xq, sx, hi, lo, scale, y, N, K, scale.stride(0), y.stride(0),
                                   READ_LO=read_lo, BLOCK_N=BN, BLOCK_G=BG, GROUP=GROUP, SPLIT_K=SK, num_warps=W, num_stages=2)


def calibrate_gemv(N, K, read_lo, hi, lo, scale, iters=20):
    """M=1 GEMV config, timed with CUDA-graph replay (GEMV + reduction); stored in _CFG_GEMV."""
    import time
    key = (N, K, read_lo)
    if key in _CFG_GEMV:
        return _CFG_GEMV[key]
    xq = torch.randint(-128, 128, (1, K), device=hi.device, dtype=torch.int8); sx = torch.rand(1, K // GROUP, device=hi.device); best = None
    cands = [c for c in _CANDS_GEMV if c[2] * N * 4 <= CALIB_WS_MB * 2**20] or [min(_CANDS_GEMV, key=lambda c: c[2])]
    ws = torch.empty((max(c[2] for c in cands) * N,), device=hi.device, dtype=torch.float32); out = torch.empty((1, N), device=hi.device, dtype=torch.bfloat16)
    for cfg in cands:
        y = ws[: cfg[2] * N].view(cfg[2], 1, N)

        def fn():
            _launch_gemv(xq, sx, hi, lo, scale, read_lo, N, K, cfg, y)
            _reduce_kernel[(triton.cdiv(N, 1024), 1)](y, out, out, 1, N, cfg[2], y.stride(0), y.stride(1), out.stride(0), HAS_BIAS=False, BLOCK=1024, num_warps=4)
        try:
            fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(iters):
                g.replay()
            torch.cuda.synchronize(); t = (time.perf_counter() - t0) / iters
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    del ws, out, xq, sx; _CFG_GEMV[key] = best[1]; return best[1]


def act_quant_sym_int8_triton(x2, groupsize=-1, clip=1.0):
    """Fused activation quantization: x2 [M, K] -> xq [M, K] int8, sx [M, K//128] fp32 (one launch)."""
    M, K = x2.shape; NG = K // GROUP; xq = torch.empty((M, K), device=x2.device, dtype=torch.int8); sx = torch.empty((M, NG), device=x2.device, dtype=torch.float32)
    _act_quant_kernel[(M,)](x2, xq, sx, K, NG, x2.stride(0), clip, PER_GROUP=(groupsize == GROUP), GROUP=GROUP, BLOCK_K=1024, num_warps=4)
    return xq, sx


def w4a8_planes(xq, sx, hi, lo, scale, read_lo: bool, out_dtype=torch.bfloat16, bias=None):
    """xq [M,K] int8, sx [M, K//128] fp32, hi/lo [N,K//2] uint8, scale [N,K//128] -> y [M,N] out_dtype (+bias).
    read_lo=False: draft (W4, high plane only); read_lo=True: target (W8, both planes)."""
    M, K = xq.shape; N = hi.shape[0]; NG = K // GROUP
    assert K % GROUP == 0 and hi.shape[1] == K // 2 and scale.shape == (N, NG) and sx.shape == (M, NG), (xq.shape, sx.shape, hi.shape, scale.shape)
    if M == 1 and GEMV_M1["on"]:
        cfg = _CFG_GEMV.get((N, K, read_lo)) or calibrate_gemv(N, K, read_lo, hi, lo, scale); SK = cfg[2]
        y = torch.empty((SK, 1, N), device=xq.device, dtype=torch.float32); _launch_gemv(xq, sx, hi, lo, scale, read_lo, N, K, cfg, y)
    else:
        BLOCK_M = 16 if M <= 16 else triton.next_power_of_2(M); assert BLOCK_M <= 64, "the kernel path is for decode / verify only (M <= 64)"
        cfg = _cfg(N, K, read_lo, BLOCK_M); SK = cfg[1]
        y = torch.empty((SK, M, N), device=xq.device, dtype=torch.float32)      # every slot is fully written, no zeroing needed
        _launch(xq, sx, hi, lo, scale, read_lo, M, N, K, BLOCK_M, cfg, y)
    out = torch.empty((M, N), device=xq.device, dtype=out_dtype)
    _reduce_kernel[(triton.cdiv(N, 1024), M)](y, out, bias if bias is not None else out, M, N, SK, y.stride(0), y.stride(1), out.stride(0), HAS_BIAS=bias is not None, BLOCK=1024, num_warps=4)
    return out


def pack_planes(q4: torch.Tensor, qr: torch.Tensor):
    """q4, qr: int8 [N,K] in [-8,7] -> unsigned nibbles (u = q+8) packed into (hi, lo) uint8 [N,K//2]."""
    def pk(q):
        u = (q.to(torch.int16) + 8).to(torch.uint8); assert int(u.max()) <= 15 and int(u.min()) >= 0
        return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return pk(q4), pk(qr)


def act_quant_sym_int8(x: torch.Tensor, groupsize: int = -1, clip: float = 1.0):
    """Reference (PyTorch) symmetric int8 activation quantization, SpinQuant convention: groupsize=-1 per token,
    128 per group (o_proj). Returns xq [M,K] int8, sx [M, K//128] fp32 (broadcast over groups when per token)."""
    x2 = x.reshape(-1, x.shape[-1]).float(); M, K = x2.shape; NG = K // GROUP
    if groupsize == -1:
        s = (x2.abs().amax(dim=1, keepdim=True) * clip / 127.0).clamp_min(1e-8)          # [M,1]
        xq = torch.round(x2 / s).clamp(-128, 127).to(torch.int8); sx = s.expand(M, NG).contiguous()
    else:
        assert groupsize == GROUP, f"only groupsize -1 or {GROUP} is supported (got {groupsize})"
        xg = x2.view(M, NG, GROUP); s = (xg.abs().amax(dim=2, keepdim=True) * clip / 127.0).clamp_min(1e-8)   # [M,NG,1]
        xq = torch.round(xg / s).clamp(-128, 127).to(torch.int8).view(M, K); sx = s.view(M, NG).contiguous()
    return xq, sx


@triton.jit
def _dequant_planes_kernel(hi_ptr, lo_ptr, s_ptr, w_ptr, N, KH, NG, stride_hn, stride_sn, stride_wn, READ_LO: tl.constexpr, BLOCK: tl.constexpr, OUT_FP16: tl.constexpr = False):
    # grid (N, cdiv(KH, BLOCK)): each program handles BLOCK bytes (= 2*BLOCK columns) of one row
    n = tl.program_id(0); jb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK); m = jb < KH
    bh = tl.load(hi_ptr + n * stride_hn + jb, mask=m, other=0)
    qe = (bh & 0xF).to(tl.int32) - 8; qo = (bh >> 4).to(tl.int32) - 8
    g = (2 * jb) // 128; sc = tl.load(s_ptr + n * stride_sn + g, mask=m, other=0.0).to(tl.float32)
    if READ_LO:
        bl = tl.load(lo_ptr + n * stride_hn + jb, mask=m, other=0)
        qe = qe * 16 + ((bl & 0xF).to(tl.int32) - 8); qo = qo * 16 + ((bl >> 4).to(tl.int32) - 8); sc = sc / 16.0
    if OUT_FP16:   # fp16 (11-bit mantissa) holds the 8-bit code x scale exactly; bf16 (8-bit mantissa) drops the last bit of W8 codes
        tl.store(w_ptr + n * stride_wn + 2 * jb, (qe.to(tl.float32) * sc).to(tl.float16), mask=m)
        tl.store(w_ptr + n * stride_wn + 2 * jb + 1, (qo.to(tl.float32) * sc).to(tl.float16), mask=m)
    else:
        tl.store(w_ptr + n * stride_wn + 2 * jb, (qe.to(tl.float32) * sc).to(tl.bfloat16), mask=m)
        tl.store(w_ptr + n * stride_wn + 2 * jb + 1, (qo.to(tl.float32) * sc).to(tl.bfloat16), mask=m)


def dequant_planes(hi, lo, scale, read_lo: bool, out_dtype=torch.bfloat16):
    """hi/lo [N,K/2] uint8, scale [N,K/128] -> [N,K] out_dtype (bf16 or fp16; draft: q4*s, target: (16q4+qr)*s/16).
    One bandwidth-bound launch. Prefill uses fp16, rounded directly from fp32 in the kernel (going through bf16 first
    would lose the last bit irrecoverably)."""
    assert out_dtype in (torch.bfloat16, torch.float16), out_dtype
    N, KH = hi.shape; w = torch.empty((N, 2 * KH), device=hi.device, dtype=out_dtype)
    _dequant_planes_kernel[(N, triton.cdiv(KH, 1024))](hi, lo, scale, w, N, KH, scale.shape[1], hi.stride(0), scale.stride(0), w.stride(0), READ_LO=read_lo, BLOCK=1024, OUT_FP16=(out_dtype == torch.float16), num_warps=4)
    return w
