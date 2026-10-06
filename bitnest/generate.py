"""BitNest end-to-end GPU engine: self-speculative decoding where the W4A8 draft and the W8A8 target share one nested
weight tensor (dual-plane Triton kernels) and one KV cache (bf16, or nested KV4/KV8 planes with --kv_planes).
Loop: draft gamma steps (high plane only) -> one batched verify of gamma+1 tokens (both planes) -> accept the longest
matching prefix (greedy) -> continue; every step is a captured CUDA graph.

Protocol: prompts from a token file [N, L] (tools/make_prompts.py), batch 1, greedy generation of --gen tokens (EOS
ignored); decode tok/s = generated tokens / wall clock of the generation phase (prefill excluded). Three paths are
measured in the same process: fp16 AR (original HF model), W8A8 AR (target only), BitNest speculative decoding.

Usage (weight package, recommended):
  python bitnest/generate.py --pkg bitnest_llama2 --prompts prompts/llama2/sharegpt.pt --v2_attn --kv_planes --kv_r3 \
      --gamma 4 --gen 256 --n 10 --out results/llama2_sharegpt.json
Usage (nested export, rebuilds the rotated model):
  MODEL=llama2 python bitnest/generate.py --nested outputs/nested_llama2_gs128.pt --prompts ... --out ..."""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from bitnest import w4a8_planes as _wp  # noqa: E402
from bitnest.w4a8_planes import act_quant_sym_int8_triton, calibrate, dequant_planes, w4a8_planes  # noqa: E402
import transformers  # noqa: E402
from transformers import StaticCache  # noqa: E402

MODE = {"m": "target"}  # global mode: "draft" (high plane only) / "target" (both planes)


class DualPlaneLinear(nn.Module):
    """Replaces a nested decoder Linear: per-token A8 (symmetric, clip 1.0) -> Triton dual-plane kernel; down_proj inputs
    first go through the SpinQuant online Hadamard (R4)."""

    def __init__(self, ent, had=None, a_groupsize=-1):
        super().__init__(); self.a_groupsize = a_groupsize
        self.register_buffer("hi", ent["hi"]); self.register_buffer("lo", ent["lo"]); self.register_buffer("scale", ent["scale"])
        self.bias = None if ent["bias"] is None else ent["bias"].float().contiguous().cuda()
        self.N, self.K = ent["shape"]; self.had = had
        if had is not None and had["had_K"] is not None:
            self.register_buffer("had_K", had["had_K"])
        self.n_calls = 0

    def _had(self, x):
        if self.had is None:
            return x
        hk = getattr(self, "had_K", None)
        if os.environ.get("BITNEST_HAD_TORCH"):   # pure torch path when fast_hadamard_transform is unavailable (slow, functional only)
            from bitnest.reference_model import hadamard_blocked
            return hadamard_blocked(x.float(), hk, self.had["K"]).to(torch.float16)
        from utils import hadamard_utils
        if not self.had["fp32"]:
            return hadamard_utils.matmul_hadU_cuda(x.to(torch.float16), hk, self.had["K"])
        return hadamard_utils.matmul_hadU_cuda(x.float(), hk, self.had["K"]).to(torch.float16)

    def forward(self, x):
        shp = x.shape; x2 = x.reshape(-1, shp[-1]); x2 = self._had(x2); read_lo = (MODE["m"] == "target"); self.n_calls += 1
        if x2.shape[0] <= 64:   # decode / verify: fused Triton activation quantization -> dual-plane kernel -> fused reduction (+bias)
            xq, sx = act_quant_sym_int8_triton(x2.contiguous(), self.a_groupsize)
            y = w4a8_planes(xq, sx, self.hi, self.lo, self.scale, read_lo, x.dtype, self.bias)
        else:
            # prefill (M > 64, compute bound): one Triton dequantization of the planes + cuBLAS GEMM; activations get the
            # same int8 fake quantization, so the semantics match the decode path.
            # GEMM precision defaults to fp16 (11-bit mantissa) rather than bf16 (8-bit): W8 codes have 8 bits and bf16
            # drops exactly the last one, which makes prefill PPL of small deep models wobble with the GEMM shape
            # (Qwen2.5-3B: 9.13 ~ 10.75); fp16 is stable and only ~11% slower (fp32 is 2.8x slower).
            xq, sx = act_quant_sym_int8_triton(x2.contiguous(), self.a_groupsize)
            pdt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("BITNEST_PREFILL_DTYPE", "fp16")]
            if pdt in (torch.float16, torch.bfloat16):
                wd = dequant_planes(self.hi, self.lo, self.scale, read_lo, out_dtype=pdt)
            else:
                wd = dequant_planes(self.hi, self.lo, self.scale, read_lo, out_dtype=torch.float16).float()
            b_ = None if self.bias is None else self.bias.to(pdt)
            CH = int(os.environ.get("BITNEST_PREFILL_CHUNK", "512"))   # dequantize activations in row chunks (3K x 11008 fp32 = 135 MB per layer otherwise)
            M_ = xq.shape[0]
            if M_ <= CH:
                xd = (xq.float().view(M_, -1, 128) * sx[:, :, None]).view(xq.shape).to(pdt)
                y = F.linear(xd, wd, b_).to(x.dtype)
            else:
                y = torch.empty((M_, self.N), device=x.device, dtype=x.dtype)
                for i in range(0, M_, CH):
                    j = min(i + CH, M_); xc = xq[i:j]
                    xd = (xc.float().view(j - i, -1, 128) * sx[i:j, :, None]).view(xc.shape).to(pdt)
                    y[i:j] = F.linear(xd, wd, b_).to(x.dtype)
                    del xd
            del wd
        return y.reshape(*shp[:-1], self.N)


def build_rotated(input_model, rot, arch, attn_impl=None):
    """Same fuse + rotate as the quantization build (R1/R2 fused offline; R4 applied online by DualPlaneLinear), no
    quantization -> rotated bf16 model."""
    from utils.process_args import process_args_ptq
    from eval_utils import rotation_utils
    from utils import fuse_norm_utils
    from eval_utils.model_registry import get_model_class
    sys.argv = ["ptq.py", "--input_model", input_model, "--optimized_rotation_path", rot, "--w_bits", "16", "--a_bits", "16", "--rotate"]
    model_args, _, ptq_args = process_args_ptq()
    ModelClass = get_model_class(arch)
    config = transformers.AutoConfig.from_pretrained(input_model); tie = bool(config.tie_word_embeddings)
    if tie:
        config.tie_word_embeddings = False
    model = ModelClass.from_pretrained(input_model, config=config, torch_dtype=torch.bfloat16, **({"attn_implementation": attn_impl} if attn_impl else {}))
    if tie:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda().eval(); fuse_norm_utils.fuse_layer_norms(model); rotation_utils.rotate_model(model, ptq_args)
    model.cuda()   # rotate_head / rotate_embeddings write lm_head / embedding back to CPU
    return model


def _norm(name):
    return name[:-7] if name.endswith(".module") else name   # exported names carry ActQuantWrapper's ".module" suffix


def swap_nested(model, pack):
    had = {_norm(k): v for k, v in pack["had"].items()}; n = 0; head_dim = model.config.hidden_size // model.config.num_attention_heads
    for raw, ent in pack["nested"].items():
        name = _norm(raw); parent = model; parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1]); assert isinstance(old, nn.Linear), (name, type(old))
        assert (old.out_features, old.in_features) == tuple(ent["shape"]), (name, old.weight.shape, ent["shape"])
        gs = head_dim if parts[-1] == "o_proj" else -1            # SpinQuant: o_proj activations grouped by head_dim, others per token
        assert gs in (-1, 128), f"{name}: head_dim={gs} != 128 is not supported by the kernel"
        newm = DualPlaneLinear({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in ent.items()}, had.get(name), gs).cuda()
        setattr(parent, parts[-1], newm); del old; n += 1
    if pack.get("embed") is not None:
        model.model.embed_tokens.weight.data = pack["embed"].cuda().to(model.model.embed_tokens.weight.dtype)
    left = [nm for nm, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in nm]
    assert not left, f"Linear layers left unreplaced: {left[:5]}"
    torch.cuda.empty_cache(); return n


def nested_modules(model):
    return [m for m in model.modules() if isinstance(m, DualPlaneLinear)]


@torch.no_grad()
def ppl_check(model, X, mode):
    """Teacher-forced (prefill-path) PPL of the target (W8A8) or draft (W4A8)."""
    MODE["m"] = mode; nll = ntok = 0.0
    ch = int(os.environ.get("BITNEST_PPL_LOGIT_CHUNK", "256"))   # chunk lm_head + CE: full [L, V] fp32 logits are 1.2 GB for a 152K vocab
    for i in range(X.shape[0]):
        ids = X[i:i + 1].cuda(); h = model.model(ids, use_cache=False)[0][0]
        for s_ in range(0, ids.shape[1] - 1, ch):
            e_ = min(s_ + ch, ids.shape[1] - 1); lg = model.lm_head(h[s_:e_]).float()
            nll += F.cross_entropy(lg, ids[0, s_ + 1:e_ + 1], reduction="sum").item(); ntok += e_ - s_; del lg
        del h
    MODE["m"] = "target"; return math.exp(nll / ntok)


if not hasattr(StaticCache, "get_max_length"):   # the repository's modeling_llama / qwen2 use the old API name
    StaticCache.get_max_length = lambda self: self.get_max_cache_shape()


class GraphStep:
    """Capture 'feed L tokens into the static cache at a given position' as a CUDA graph (removes the Python launch
    overhead of hundreds of kernels; the fp16 baseline is captured the same way)."""

    def __init__(self, model, cache, L, capture_pos):
        self.L = L; self.inp = torch.zeros((1, L), dtype=torch.long, device="cuda"); self.cp = torch.arange(capture_pos, capture_pos + L, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(self.inp, past_key_values=cache, cache_position=self.cp, use_cache=True)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.logits = model(self.inp, past_key_values=cache, cache_position=self.cp, use_cache=True).logits

    def __call__(self, tokens, pos):   # tokens [1, L] long (device), pos int
        self.inp.copy_(tokens); self.cp.copy_(torch.arange(pos, pos + self.L, device="cuda")); self.g.replay(); return self.logits


class EagerStep:
    def __init__(self, model, cache, L, mode):
        self.model, self.cache, self.L, self.mode = model, cache, L, mode

    def __call__(self, tokens, pos):
        MODE["m"] = self.mode; cp = torch.arange(pos, pos + self.L, device="cuda")
        return self.model(tokens, past_key_values=self.cache, cache_position=cp, use_cache=True).logits


def prefill_last(model, ids, cache):
    """Prefill, returning only the last position's logits [B, V] (model.model + lm_head(h[:, -1:]); avoids [P, V] fp32
    logits at long context)."""
    P = ids.shape[1]
    h = model.model(ids, past_key_values=cache, cache_position=torch.arange(P, device="cuda"), use_cache=True)[0]
    return model.lm_head(h[:, -1:]).float()[:, -1]


class ChunkedPrefillMLP(torch.nn.Module):
    """The MLP is token-independent: chunk rows during prefill to bound intermediates; decode / verify unchanged."""

    def __init__(self, inner, chunk):
        super().__init__(); self.inner = inner; self.chunk = chunk

    def forward(self, x):
        if x.numel() // x.shape[-1] <= self.chunk:
            return self.inner(x)
        flat = x.reshape(-1, x.shape[-1]); out = torch.empty_like(flat)
        for a in range(0, flat.shape[0], self.chunk):
            b = min(a + self.chunk, flat.shape[0]); out[a:b] = self.inner(flat[a:b])
        return out.reshape_as(x)


def drop_page_cache(path):
    """Unified-memory devices (Jetson): drop the page cache of weight files after reading them (clean pages, lossless);
    otherwise 3-6 GB of file cache competes with GPU allocations."""
    import glob as _g
    try:
        fs = [path] if os.path.isfile(path) else _g.glob(os.path.join(path, "*.safetensors"))
        for f in fs:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    except Exception:
        pass


KVP = {"on": False}


def make_cache(model, max_len):
    if KVP["on"]:
        from bitnest.kv_cache_planes import PlanesCache
        c = PlanesCache(model.config, 1, max_len, "cuda", torch.bfloat16); n = c.set_offsets(model) if KVP.get("offsets", True) else 0
        if KVP.get("r3"):
            c.set_r3(True)
        if not KVP.get("_said"):
            KVP["_said"] = True; print(f"[bitnest] KV planes: per-channel bias offsets on {n} layers (K shifted exactly / V added back)", flush=True)
        return c
    return StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def ar_generate(model, ids, gen, ctx):
    """ctx = dict(cache, step1): eager prefill, then one graph replay per token. Returns (tokens, generation seconds)."""
    P = ids.shape[1]; cache = ctx["cache"]; cache.reset()
    nxt = prefill_last(model, ids, cache).argmax(-1, keepdim=True); toks = [nxt]
    torch.cuda.synchronize(); t0 = time.perf_counter(); pos = P
    for _ in range(gen - 1):
        lg = ctx["step1"](nxt, pos); nxt = lg[:, -1].argmax(-1, keepdim=True); toks.append(nxt); pos += 1
    torch.cuda.synchronize(); return [t.item() for t in toks], time.perf_counter() - t0


@torch.no_grad()
def spec_generate(model, ids, gen, gamma, ctx):
    """Draft gamma steps (graph step1, draft mode) -> verify gamma+1 tokens at the same positions (graph stepG, target
    mode; it overwrites the draft's KV) -> accept the matching prefix. Draft and target share one static cache."""
    P = ids.shape[1]; cache = ctx["cache"]; cache.reset(); MODE["m"] = "target"
    nxt = prefill_last(model, ids, cache).argmax(-1, keepdim=True); toks = [nxt]
    n_prop = n_acc = rounds = 0; pos = P; k_hist = [0] * (gamma + 1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while len(toks) < gen:
        tok = nxt; drafts = []
        for i in range(gamma):
            lg = ctx["draft1"](tok, pos + i); tok = lg[:, -1].argmax(-1, keepdim=True); drafts.append(tok)
        seq = torch.cat([nxt] + drafts, dim=1)                   # [1, gamma+1], written at positions pos..pos+gamma
        lg = ctx["verifyG"](seq, pos); tgt = lg[0].argmax(-1)      # [gamma+1]
        d = torch.cat(drafts, dim=1)[0]; k = int((d == tgt[:gamma]).cumprod(0).sum().item())
        toks += [t.view(1, 1) for t in d[:k]] + [tgt[k].view(1, 1)]; nxt = tgt[k].view(1, 1)
        n_prop += gamma; n_acc += k; rounds += 1; pos += k + 1; k_hist[k] += 1
    torch.cuda.synchronize(); t = time.perf_counter() - t0
    return [t.item() for t in toks[:gen]], t, dict(accept=n_acc / max(n_prop, 1), tokens_per_round=(min(len(toks), gen) - 1) / max(rounds, 1), rounds=rounds, k_hist=k_hist)


EAGER = {"on": False}


def build_ctx(model, P, gen, gamma, modes):
    """Static cache + graphs for prompt length P; modes: ('step1',) or ('step1', 'draft1', 'verifyG'). Graphs are
    captured at positions past the region used for decoding, so capture never pollutes the cache."""
    max_len = P + gen + gamma + 8; cache = make_cache(model, max_len + 32); ctx = {"cache": cache}
    cap = max_len
    if EAGER["on"]:
        mk = lambda L, mode: EagerStep(model, cache, L, mode)  # noqa: E731
    else:
        mk = lambda L, mode: (MODE.__setitem__("m", mode), GraphStep(model, cache, L, cap))[1]  # noqa: E731
    if "step1" in modes:
        ctx["step1"] = mk(1, "target")
    if "draft1" in modes:
        ctx["draft1"] = mk(1, "draft")
    if "verifyG" in modes:
        ctx["verifyG"] = mk(gamma + 1, "target")
    MODE["m"] = "target"; return ctx


def calibrate_kernels(model, gamma):
    """Pick kernel launch configs for every (N, K) x {draft, target} x M in {1, gamma+1} before graph capture."""
    mods = nested_modules(model); shapes = {}
    for m_ in mods:
        shapes.setdefault((m_.N, m_.K), m_)
    t0 = time.time()
    for (N_, K_), m_ in shapes.items():
        for rl in (False, True):
            for M_ in (1, gamma + 1):
                calibrate(N_, K_, rl, M_, m_.hi, m_.lo, m_.scale)
            if _wp.GEMV_M1["on"]:   # M=1 GEMV (BITNEST_GEMV_M1=1)
                _wp.calibrate_gemv(N_, K_, rl, m_.hi, m_.lo, m_.scale)
    torch.cuda.empty_cache()
    print(f"[bitnest] kernel calibrated for {len(shapes)} shapes x 2 modes x 2 M in {time.time()-t0:.0f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("model")
    src.add_argument("--pkg", default=None, help="weight package directory (cold load: no bf16 model, no rotation); requires --v2_attn")
    src.add_argument("--nested", default=None, help="nested export .pt (rebuilds the rotated bf16 model, then swaps in the planes)")
    src.add_argument("--fp16_model", default=None, help="fp16 baseline weights (default: source_model in the package meta)")
    run = p.add_argument_group("protocol")
    run.add_argument("--prompts", required=True, help="token file [N, L] (tools/make_prompts.py)")
    run.add_argument("--out", required=True, help="result JSON")
    run.add_argument("--gamma", type=int, default=None, help="draft length (default: 4 for llama2, 3 otherwise)")
    run.add_argument("--gen", type=int, default=256)
    run.add_argument("--n", type=int, default=10, help="number of prompts")
    run.add_argument("--warmup", type=int, default=1, help="warm-up runs per path before timing")
    run.add_argument("--only", default="all", choices=["all", "fp16", "w8a8", "spec"], help="run a single path")
    run.add_argument("--skip_fp16", action="store_true")
    run.add_argument("--accept_only", action="store_true", help="speculative run only (acceptance statistics, no W8A8 AR timing)")
    run.add_argument("--eager", action="store_true", help="no CUDA graphs")
    run.add_argument("--mem_cap_gib", type=float, default=0, help="cap this process' CUDA memory (GiB) to emulate a smaller device")
    eng = p.add_argument_group("engine")
    eng.add_argument("--v2_attn", action="store_true", help="Triton flash-decoding attention")
    eng.add_argument("--kv_planes", action="store_true", help="nested KV cache (draft KV4 / target KV8); requires --v2_attn")
    eng.add_argument("--kv_draft", default="kv4", help="draft KV read: kv4 | kv8 | k4v8 | k8v4 | win:N (KV8 for the last N positions); target is always KV8")
    eng.add_argument("--kv_r3", action="store_true", help="per-head Hadamard on K before quantization (SpinQuant R3, exact)")
    eng.add_argument("--kv_no_offsets", action="store_true", help="ablation: disable the K/V bias offsets")
    chk = p.add_argument_group("quality checks")
    chk.add_argument("--ppl_check", default=None, help="token file for teacher-forced prefill-path PPL of target and draft")
    chk.add_argument("--ppl_n", type=int, default=10)
    chk.add_argument("--ppl_decode_n", type=int, default=0, help="PPL along the decode path (single-step target, and the BitNest draft+verify path)")
    chk.add_argument("--ppl_tail", type=int, default=0, help="prefill P-N tokens, then teacher-force the last N tokens on the target decode path (long-context check)")
    return p.parse_args()


def main():
    torch.set_grad_enabled(False)
    a = parse_args(); EAGER["on"] = a.eager
    if a.mem_cap_gib:
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30; torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gib / tot))
        print(f"[bitnest] CUDA memory cap {a.mem_cap_gib:.0f} GiB (device total {tot:.0f} GiB)", flush=True)
    assert (a.pkg is None) != (a.nested is None), "pass exactly one of --pkg / --nested"
    if a.pkg:   # cold load: only meta.json is needed to describe the pack (nested weights never enter CPU memory)
        assert a.v2_attn, "--pkg requires --v2_attn"
        _meta = json.load(open(f"{a.pkg}/meta.json"))
        pack = dict(model=_meta["model"], arch=_meta["arch"], input_model=_meta["source_model"], pack="unsigned+8", nested={k: None for k in _meta["layers"]}, embed=None, had={})
    else:
        pack = torch.load(a.nested, map_location="cpu", weights_only=False)
    if a.kv_planes:
        assert a.v2_attn, "--kv_planes requires --v2_attn"
    assert pack.get("pack") == "unsigned+8", f"unsupported packing {pack.get('pack')}"
    mk = pack["model"]; gamma = a.gamma or {"llama2": 4}.get(mk, 3)
    X = torch.load(a.prompts)[: a.n]; res = dict(model=mk, prompts=a.prompts, gamma=gamma, gen=a.gen, n=int(X.shape[0]))
    arch = pack.get("arch") or ("qwen2" if "qwen" in mk else "llama")

    # ---- fp16 AR baseline (original HF model, not rotated, not quantized)
    if a.only in ("w8a8", "spec"):
        a.skip_fp16 = True
    if a.fp16_model:
        pack["input_model"] = a.fp16_model
    if not a.skip_fp16:
        kw16 = {}
        if a.v2_attn:   # same decode attention kernel for the baseline (engine parity); the repository modeling uses the old
            # ATTENTION_CLASSES dict, so the baseline uses transformers' native classes, which accept registered interfaces
            from bitnest.attention import install_hf_stock
            kw16["attn_implementation"] = install_hf_stock(); print("[bitnest] fp16 baseline uses the Triton decode attention too", flush=True)
            cls16 = {"llama": transformers.LlamaForCausalLM, "qwen2": transformers.Qwen2ForCausalLM}[arch]
        else:
            from eval_utils.model_registry import get_model_class
            cls16 = get_model_class(arch)
        m16 = cls16.from_pretrained(pack["input_model"], torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True, **kw16).cuda().eval()
        drop_page_cache(pack["input_model"])
        _ch16 = int(os.environ.get("BITNEST_FP16_MLP_CHUNK", "0"))   # optional row-chunked baseline MLP during prefill (numerically identical; off by default)
        if _ch16 > 0:
            for _l in m16.model.layers:
                _l.mlp = ChunkedPrefillMLP(_l.mlp, _ch16)
        c16 = build_ctx(m16, X.shape[1], a.gen, gamma, ("step1",))
        for _ in range(a.warmup):
            ar_generate(m16, X[0:1].cuda(), 8, c16)
        tps = []
        for i in range(X.shape[0]):
            _, t = ar_generate(m16, X[i:i + 1].cuda(), a.gen, c16); tps.append((a.gen - 1) / t)
        res["ar_fp16_tok_s"] = statistics.median(tps)
        if a.only == "fp16":
            res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print("BITNEST_RESULT " + json.dumps(res), flush=True); return
        del c16
        print(f"[bitnest] fp16 AR {res['ar_fp16_tok_s']:.1f} tok/s", flush=True); del m16; torch.cuda.empty_cache()

    # ---- BitNest model
    impl = None
    if a.v2_attn:
        from bitnest.attention import install as _install_v2
        impl = _install_v2(arch); res["v2_attn"] = True
        print(f"[bitnest] decode attention installed ({'registry:' + impl if isinstance(impl, str) else 'llama monkeypatch'})", flush=True)
    if a.pkg:
        from bitnest.cold_load import cold_load
        from bitnest.attention import install_stock_wrapper
        impl = install_stock_wrapper()
        model, n, _ = cold_load(a.pkg, impl, DualPlaneLinear=DualPlaneLinear); res["cold_load"] = True
        print(f"[bitnest] cold-load from {a.pkg}: {n} DualPlaneLinear, no bf16 model / no rotation", flush=True)
    else:
        model = build_rotated(pack["input_model"], pack["rot"], arch, impl if isinstance(impl, str) else None); n = swap_nested(model, pack)
    assert n == len(pack["nested"]), (n, len(pack["nested"]))
    print(f"[bitnest] swapped {n}/{len(pack['nested'])} nested linears (all decoder Linear replaced)", flush=True)
    if a.ppl_check:
        Xc = torch.load(a.ppl_check)[: a.ppl_n]; p8 = ppl_check(model, Xc, "target"); p4 = ppl_check(model, Xc, "draft")
        print(f"[bitnest] PPL check on {a.ppl_check} (n={Xc.shape[0]}): target(W8A8)={p8:.4f} draft(W4A8)={p4:.4f}", flush=True)
        res["ppl_target"] = p8; res["ppl_draft"] = p4
    if a.v2_attn:
        from bitnest import attention as _v2
        _v2.MODE_REF["get"] = lambda: MODE["m"]
    if a.kv_planes:
        KVP["on"] = True; KVP["offsets"] = not a.kv_no_offsets; KVP["r3"] = a.kv_r3
        res.update(kv_planes=True, kv_draft=a.kv_draft, kv_offsets=not a.kv_no_offsets, kv_r3=a.kv_r3)
        dm = {"kv4": False, "kv8": True, "k4v8": dict(k=False, v=True), "k8v4": dict(k=True, v=False)}.get(
            a.kv_draft, ("window", int(a.kv_draft.split(":")[1])) if a.kv_draft.startswith("win:") else None)
        assert dm is not None, f"--kv_draft {a.kv_draft} not recognized"
        _v2.DRAFT_LO["mode"] = dm; print(f"[bitnest] KV planes cache on (draft {a.kv_draft} / target KV8)", flush=True)
    mods = nested_modules(model); c0 = sum(m.n_calls for m in mods)
    calibrate_kernels(model, gamma)
    # warm up both shapes (M=1 and M=gamma+1), then capture graphs
    ids0 = X[0:1].cuda(); MODE["m"] = "draft"; model(ids0[:, :1], use_cache=False); MODE["m"] = "target"; model(ids0[:, :gamma + 1], use_cache=False)
    assert sum(m.n_calls for m in mods) - c0 == 2 * len(mods), "DualPlaneLinear was not called in the forward pass"
    res["peak_mem_build_gib"] = torch.cuda.max_memory_allocated() / 2**30
    ctx = build_ctx(model, X.shape[1], a.gen, gamma, ("step1", "draft1", "verifyG")); torch.cuda.reset_peak_memory_stats()
    print(f"[bitnest] build-phase peak {res['peak_mem_build_gib']:.2f} GiB (decode-phase peak reported separately)", flush=True)
    for _ in range(a.warmup):
        ar_generate(model, ids0, 8, ctx); spec_generate(model, ids0, 8, gamma, ctx)
    if a.only == "spec":
        a.accept_only = True
    kv_desc = "KV8 planes" if a.kv_planes else "bf16 KV"
    if a.ppl_decode_n:
        nd = min(a.ppl_decode_n, X.shape[0])
        # (1) target decode path: single-step teacher forcing
        nll = ntok = 0.0; MODE["m"] = "target"
        for i in range(nd):
            ids = X[i:i + 1].cuda(); ctx["cache"].reset(); P_ = ids.shape[1]
            for t in range(P_ - 1):
                lg = ctx["step1"](ids[:, t:t + 1], t); nll += F.cross_entropy(lg[0, -1].float()[None], ids[0, t + 1][None], reduction="sum").item(); ntok += 1
        res["ppl_decode_target"] = math.exp(nll / ntok)
        print(f"[bitnest] decode-path PPL (target, single step, {kv_desc}) n={nd}: {res['ppl_decode_target']:.4f}", flush=True)
        # (2) the real BitNest path: each round, gamma draft single steps write the KV, then verifyG recomputes the same
        #     gamma+1 positions and overwrites them -> exactly the cache state of speculative decoding; NLL from verify logits
        nll = ntok = 0.0
        for i in range(nd):
            ids = X[i:i + 1].cuda(); ctx["cache"].reset(); P_ = ids.shape[1]; t = 0
            while t + gamma + 1 < P_:
                MODE["m"] = "draft"
                for j in range(gamma):
                    ctx["draft1"](ids[:, t + j:t + j + 1], t + j)
                MODE["m"] = "target"; lg = ctx["verifyG"](ids[:, t:t + gamma + 1], t)   # [1, gamma+1, V]: positions t..t+gamma predict t+1..t+gamma+1
                nll += F.cross_entropy(lg[0].float(), ids[0, t + 1:t + gamma + 2], reduction="sum").item(); ntok += gamma + 1; t += gamma + 1
        res["ppl_decode_bitnest"] = math.exp(nll / ntok)
        print(f"[bitnest] BitNest-path PPL (draft single steps write KV + verify batch M=gamma+1 overwrites, {kv_desc}, gamma={gamma}) n={nd}: {res['ppl_decode_bitnest']:.4f}", flush=True)
    if a.ppl_tail:
        nll = ntok = 0.0; MODE["m"] = "target"; N_ = a.ppl_tail
        for i in range(X.shape[0]):
            ids = X[i:i + 1].cuda(); ctx["cache"].reset(); P_ = ids.shape[1] - N_
            lg = prefill_last(model, ids[:, :P_], ctx["cache"]); nll += F.cross_entropy(lg.float(), ids[0, P_][None], reduction="sum").item(); ntok += 1
            for t in range(P_, ids.shape[1] - 1):
                lg = ctx["step1"](ids[:, t:t + 1], t); nll += F.cross_entropy(lg[0, -1].float()[None], ids[0, t + 1][None], reduction="sum").item(); ntok += 1
        res["ppl_tail"] = math.exp(nll / ntok); res["ppl_tail_n"] = N_
        print(f"[bitnest] tail-PPL (prefill {X.shape[1]-N_} + teacher-force last {N_} tokens, target decode path, {kv_desc}, n={X.shape[0]}): {res['ppl_tail']:.4f}", flush=True)

    # ---- timing: W8A8 AR vs BitNest speculative decoding on the same prompts
    ar8, sp, same, accs, tpr, first_div, tok_log = [], [], [], [], [], [], []; k_hist_all = [0] * (gamma + 1)
    for i in range(X.shape[0]):
        ids = X[i:i + 1].cuda()
        if a.accept_only:
            t8, tt = [], 1.0
        else:
            t8, tt = ar_generate(model, ids, a.gen, ctx)
        ar8.append((a.gen - 1) / tt)
        if a.only == "w8a8":
            ts, tsp, st = t8, tt, dict(accept=0.0, tokens_per_round=1.0, rounds=0, k_hist=[0] * (gamma + 1))
        else:
            ts, tsp, st = spec_generate(model, ids, a.gen, gamma, ctx)
        sp.append((a.gen - 1) / tsp); same.append(ts == t8); accs.append(st["accept"]); tpr.append(st["tokens_per_round"])
        for j, c in enumerate(st["k_hist"]):
            k_hist_all[j] += c
        fd = next((j for j, (x, y) in enumerate(zip(ts, t8)) if x != y), len(t8)); first_div.append(fd); tok_log.append(dict(spec=ts, ar=t8, first_div=fd))
        print(f"[bitnest] prompt {i}: W8A8 AR {ar8[-1]:.1f} tok/s | spec {sp[-1]:.1f} tok/s | accept {st['accept']:.3f} | tok/round {st['tokens_per_round']:.2f} | same_as_AR={same[-1]}", flush=True)
    res.update(ar_w8a8_tok_s=statistics.median(ar8), spec_tok_s=statistics.median(sp), accept=sum(accs) / len(accs), tokens_per_round=sum(tpr) / len(tpr),
               identical_to_w8a8_ar=sum(same) / len(same), prefix_agree_mean=sum(first_div) / len(first_div) / a.gen, first_div_all=first_div,
               peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30, ar8_all=ar8, spec_all=sp, tokens_all=tok_log)
    print(f"[bitnest] spec vs W8A8-AR: identical {res['identical_to_w8a8_ar']:.1f}, mean first divergence {sum(first_div)/len(first_div):.1f}/{a.gen}", flush=True)
    R_ = max(sum(k_hist_all), 1); res["round_k_hist"] = k_hist_all   # histogram of rounds accepting k draft tokens (k=0..gamma)
    res["accept_first_k"] = {str(k): sum(k_hist_all[k:]) / R_ for k in range(1, gamma + 1)}   # P(first k draft tokens all accepted)
    print("[bitnest] per-position acceptance P(first k accepted): " + ", ".join(f"k={k}: {v:.3f}" for k, v in res["accept_first_k"].items()) + f"  (rounds={R_})", flush=True)
    if "ar_fp16_tok_s" in res:
        res["speedup_vs_fp16"] = res["spec_tok_s"] / res["ar_fp16_tok_s"]
    res["speedup_vs_w8a8"] = res["spec_tok_s"] / res["ar_w8a8_tok_s"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print("BITNEST_RESULT " + json.dumps({k: res[k] for k in res if not k.endswith("_all")}), flush=True)


if __name__ == "__main__":
    main()
