"""Model construction shared by the BitNest export / evaluation scripts.

build_model(w_bits) rebuilds the SpinQuant-rotated model (R1/R2 fused offline, R4 online Hadamard on down_proj inputs)
and quantizes its decoder Linears with GPTQ at `w_bits` (group size GS, A8 per-token activations). With w_bits=16 it
returns the rotated full-precision weights W0 that the residual plane is fitted against.

BitNest residual nesting ("resid"): GPTQ-W4 codes (q4, s4) are locked as the high nibble; the low nibble is the
closed-form rounding of the residual against W0 on the finer grid s8 = s4/16:
    qr = clamp(round((W0 - q4*s4) / s8), lo, 7),   lo = 0 if q4 == -8 else -8   (keeps 16*q4 + qr inside int8)
    draft  W4 = q4 * s4                 (literally the native GPTQ-W4 model)
    target W8 = (16*q4 + qr) * s8
"""
import os
import sys

import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval_utils.main import ptq_model  # noqa: E402
from eval_utils.model_registry import get_entry, get_model_class  # noqa: E402
from utils import data_utils, quant_utils  # noqa: E402
from utils.process_args import process_args_ptq  # noqa: E402

ENTRY = get_entry()
GS = os.environ.get("GS", "128")
ROT = os.environ.get("ROT", ENTRY["rot"])
INPUT_MODEL = ENTRY["input_model"]
TAG = ENTRY["tag"]
ModelClass = get_model_class(ENTRY["arch"])


def build_model(w_bits, to_cuda=True, drop_bufs=False):
    """Rotated model with GPTQ W{w_bits} weights (fake-quant), A8/KV8 activation quantizers attached.
    Quantized layers keep `int_weight` (int8 codes) and `scale` buffers; embedding / lm_head use 8-bit RTN."""
    sys.argv = ["ptq.py", "--input_model", INPUT_MODEL,
                "--optimized_rotation_path", ROT,
                "--w_bits", str(w_bits), "--a_bits", "8", "--k_bits", "8", "--v_bits", "8",
                "--w_groupsize", GS, "--w_clip", "--rotate", "--export_to_et"]
    model_args, training_args, ptq_args = process_args_ptq()
    config = transformers.AutoConfig.from_pretrained(model_args.input_model, token=model_args.access_token)
    tie = bool(config.tie_word_embeddings)
    if tie:
        config.tie_word_embeddings = False
    model = ModelClass.from_pretrained(model_args.input_model, config=config,
                                       torch_dtype=torch.bfloat16, token=model_args.access_token)
    if tie:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()
    model = ptq_model(ptq_args, model, model_args)
    # After ptq_model the model lives on CPU and int_weight is stored as float32 (~26 GB for 7B), which would OOM on the
    # GPU: compress int_weight to int8 (lossless, integer values) and GPTQ's fp32 dequantized weights to bf16.
    seen = set(); nbytes = 0
    for mod in model.modules():
        if drop_bufs:  # paths that never read the integer codes can drop them (large-vocab models would not fit otherwise)
            for n in ("int_weight", "scale"):
                if n in mod._buffers:
                    del mod._buffers[n]
        for n, b in list(mod._buffers.items()):
            if b is None:
                continue
            if n == "int_weight" and b.dtype != torch.int8:
                b = b.to(torch.int8); mod._buffers[n] = b
            if b.data_ptr() not in seen:
                seen.add(b.data_ptr()); nbytes += b.numel() * b.element_size()
        for n, p in list(mod._parameters.items()):
            if p is None:
                continue
            if n == "weight" and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr()); nbytes += p.numel() * p.element_size()
    print(f"[build_model w{w_bits}] tensor footprint before moving to GPU (deduplicated) ~ {nbytes/1e9:.1f} GB", flush=True)
    if to_cuda:   # nesting paths do their weight surgery on CPU first and move the model themselves
        model.cuda()
    model.config.use_cache = False
    return model, model_args, ptq_args


def rotated_w0():
    """Rotated (fused + R1/R2) full-precision weights W0 of every quantizable layer except lm_head, as bf16 CPU tensors."""
    m16, _, _ = build_model(16, to_cuda=False)
    ql16 = quant_utils.find_qlayers(m16, layers=[torch.nn.Linear, torch.nn.Embedding])
    W0 = {n: m.weight.data.detach().to(torch.bfloat16).cpu() for n, m in ql16.items() if "lm_head" not in n}
    del m16; torch.cuda.empty_cache()
    return W0


def residual_codes(q4, s4_full, w0):
    """Closed-form low nibble: qr in [-8, 7] (in [0, 7] where q4 == -8). Returns (qr, raw rounding before clamping)."""
    s8 = s4_full / 16.0
    qr_raw = torch.round((w0 - q4 * s4_full) / s8.clamp_min(1e-12))   # GPTQ scales are positive
    lo = torch.where(q4 <= -8, torch.zeros_like(qr_raw), torch.full_like(qr_raw, -8.0))
    qr = torch.maximum(torch.clamp(qr_raw, max=7.0), lo)
    return qr, qr_raw


def stash_resid(model, W0):
    """Fake-quant BitNest model (run on CPU before moving to GPU): for every 4-bit layer store
       m.w8_tensor = target W8 = q4*s4 + qr*s8 and m.w4_tensor = draft W4 = q4*s4 (GPTQ's own write-back, zero copy).
    8-bit RTN modules (embedding) are shared by draft and target and skipped. int_weight / scale buffers are freed."""
    qlayers = quant_utils.find_qlayers(model, layers=[torch.nn.Linear, torch.nn.Embedding])
    n_sat = n_tot = n_skip = 0
    for name, m in qlayers.items():
        if "lm_head" in name:
            continue
        if not (hasattr(m, "int_weight") and hasattr(m, "scale")):
            continue
        q4 = m.int_weight.to(torch.float32)
        if q4.abs().max().item() > 8:                # not a 4-bit module (embedding is 8-bit RTN)
            del m.int_weight, m.scale, q4
            n_skip += 1
            continue
        s4 = m.scale.float()
        s4 = s4 if s4.dim() == 2 else s4.view(-1, 1)
        if s4.shape != q4.shape:                     # per-group scale -> full size
            s4 = s4.repeat_interleave(q4.shape[1] // s4.shape[1], dim=1)
        qr, qr_raw = residual_codes(q4, s4, W0[name].float().to(q4.device))
        n_sat += int((qr != qr_raw).sum()); n_tot += qr.numel()
        m.w8_tensor = (q4 * s4 + qr * (s4 / 16.0)).to(m.weight.dtype)
        m.w4_tensor = m.weight.data
        del m.int_weight, m.scale, q4, s4, qr, qr_raw
    print(f"[resid] residual saturation = {(n_sat/n_tot if n_tot else 0):.4%} ({n_sat}/{n_tot}), skipped non-4-bit modules = {n_skip}", flush=True)
    assert n_tot > 0, "stash_resid: no 4-bit module was nested"


def swap_resid(model, which):
    """which: 'w8' (target) | 'w4' (draft)."""
    for m in model.modules():
        if which == "w8" and hasattr(m, "w8_tensor"):
            m.weight.data = m.w8_tensor
        if which == "w4" and hasattr(m, "w4_tensor"):
            m.weight.data = m.w4_tensor


def build_resid_model():
    """Fake-quant BitNest model on GPU, initially in target (W8) mode; switch with swap_resid(model, 'w4'|'w8')."""
    print(">>> building W16 (rotated) for W0 ...", flush=True)
    W0 = rotated_w0()
    print(">>> building GPTQ-W4 and fitting the residual plane ...", flush=True)
    model, model_args, ptq_args = build_model(4, to_cuda=False)
    stash_resid(model, W0)
    del W0
    model.cuda()
    for m in model.modules():                  # stashed tensors are plain attributes: model.cuda() does not move them
        for n in ("w8_tensor", "w4_tensor"):
            t = getattr(m, n, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, n, t.cuda())
    swap_resid(model, "w8")
    return model, model_args, ptq_args


# ---------------------------------------------------------------- evaluation domains
def _pack_ids(tokenizer, text, seqlen, nseq):
    ids = tokenizer(text, return_tensors="pt").input_ids
    ns = ids.numel() // seqlen
    assert ns >= nseq, f"not enough text: {ids.numel()} tokens give only {ns} segments (< {nseq}x{seqlen})"
    return ids[:, :ns * seqlen].view(ns, seqlen)[:nseq]


def domain_text(name, pg19_json=None):
    """Raw text of an evaluation domain (concatenated in dataset order)."""
    import datasets
    if name == "wiki2":
        return "\n\n".join(datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    if name == "gsm8k":
        ds = datasets.load_dataset("gsm8k", "main")["test"]
        return "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds.select(range(400)))
    if name == "code":       # HumanEval (all) + MBPP test (first 300): docstring + solution
        he = datasets.load_dataset("openai/openai_humaneval")["test"]
        mb = datasets.load_dataset("google-research-datasets/mbpp", "full")["test"]
        parts = [r["prompt"] + r["canonical_solution"] for r in he]
        parts += ['"""%s"""\n%s' % (r["text"], r["code"]) for r in mb.select(range(min(300, len(mb))))]
        return "\n\n".join(parts)
    if name == "sharegpt":   # real multi-turn conversations
        ds = datasets.load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                                   data_files="ShareGPT_V3_unfiltered_cleaned_split.json")["train"]
        parts = []
        for r in ds.select(range(150)):
            conv = "\n".join(f"{'Human' if t.get('from') == 'human' else 'Assistant'}: {t.get('value', '')}"
                             for t in (r["conversations"] or []))
            if conv:
                parts.append(conv)
        return "\n\n".join(parts)
    if name == "longdoc":    # LongBench gov_report long documents
        lb = datasets.load_dataset("THUDM/LongBench", "gov_report", split="test", trust_remote_code=True)
        return "\n\n".join(r["context"] for r in lb.select(range(12)))
    if name in ("pg19", "pg19u"):   # first 20 books of the PG-19 test split (jsonl with a "text" field, as shipped by QuantSpec)
        import json
        assert pg19_json or os.environ.get("PG19_JSON"), "pg19 needs --pg19_json / PG19_JSON=<pg19-test.json>"
        docs = []
        with open(pg19_json or os.environ["PG19_JSON"]) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
                if len(docs) >= 20:
                    break
        return "\n\n".join(docs)
    raise KeyError(f"unknown domain {name} (wiki2|gsm8k|code|sharegpt|longdoc|pg19)")


def get_domain_ids(tokenizer, name, seqlen=2048, nseq=20):
    """Teacher-forced evaluation segments [nseq, seqlen] of a domain (text concatenated, cut into seqlen chunks)."""
    if name == "wiki2":
        enc = data_utils.get_wikitext2(seed=0, seqlen=seqlen, tokenizer=tokenizer, eval_mode=True)
        ids = enc.input_ids
        ns = ids.numel() // seqlen
        return ids[:, :ns * seqlen].view(ns, seqlen)[:nseq]
    return _pack_ids(tokenizer, domain_text(name), seqlen, nseq)
