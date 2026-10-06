"""Side-by-side decoding demo: FP16 autoregressive decoding vs BitNest self-speculative decoding on the same prompt.

Both paths use the same engine (CUDA graphs, Triton decode attention). Every generated token (FP16 AR) and every
speculative round (BitNest) is timestamped after a device synchronization, so the trace shows when each token became
available. The trace (JSON) drives the replay on the project page (docs/).

Usage:
  python examples/compare_decoding.py --pkg release/bitnest_qwen2.5 --gen 192 --out docs/traces/qwen2.5.json
  python examples/compare_decoding.py --pkg release/bitnest_qwen2.5 --prompt "def fibonacci(n):" --gen 128
"""
import argparse
import json
import os
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import bitnest.generate as R  # noqa: E402

GSM8K_SHOTS = 4


def builtin_examples(gsm8k_idx=11, humaneval_idx=156, chat_tokens=136, chat_question="What is the difference between TCP and UDP?"):
    """Three prompts: GSM8K (math), HumanEval (code), and a chat question (cut after chat_tokens tokens).
    The default problems have reference solutions of ~150 tokens, i.e. a few seconds of FP16 decoding."""
    import datasets
    gs = datasets.load_dataset("gsm8k", "main")
    shots = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in gs["train"].select(range(GSM8K_SHOTS)))
    q = gs["test"][gsm8k_idx]["question"]
    he = datasets.load_dataset("openai/openai_humaneval")["test"][humaneval_idx]
    chat = ("A conversation between a curious user and a helpful, knowledgeable assistant.\n\n"
            f"User: {chat_question}\nAssistant:")
    return [
        dict(id="math", title="Math word problem (GSM8K, 4-shot)", context=shots, shown=f"Question: {q}\nAnswer:", stop="\n\nQuestion:"),
        dict(id="code", title="Code completion (HumanEval)", context="", shown=he["prompt"], stop="\ndef "),
        dict(id="chat", title="Chat", context="", shown=chat, stop="\nUser:", max=chat_tokens),
    ]


def encode(tok, text):
    ids = tok(text, add_special_tokens=False).input_ids
    first = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id   # same convention as the evaluation prompts
    return [first] + ids


@torch.no_grad()
def timed_ar(model, ids, gen, ctx):
    """AR decoding; returns tokens, per-token availability times (s, from the start of decoding) and prefill time."""
    cache = ctx["cache"]; cache.reset(); R.MODE["m"] = "target"
    torch.cuda.synchronize(); tp = time.perf_counter()
    nxt = R.prefill_last(model, ids, cache).argmax(-1, keepdim=True); torch.cuda.synchronize(); t0 = time.perf_counter()
    toks = [int(nxt)]; times = [0.0]; pos = ids.shape[1]
    for _ in range(gen - 1):
        lg = ctx["step1"](nxt, pos); nxt = lg[:, -1].argmax(-1, keepdim=True); pos += 1
        toks.append(int(nxt)); times.append(time.perf_counter() - t0)   # int() synchronizes
    return toks, times, t0 - tp


@torch.no_grad()
def timed_spec(model, ids, gen, gamma, ctx):
    """BitNest speculative decoding; returns tokens, per-token times and the rounds (accepted drafts + target token)."""
    cache = ctx["cache"]; cache.reset(); R.MODE["m"] = "target"
    torch.cuda.synchronize(); tp = time.perf_counter()
    nxt = R.prefill_last(model, ids, cache).argmax(-1, keepdim=True); torch.cuda.synchronize(); t0 = time.perf_counter()
    toks = [int(nxt)]; times = [0.0]; rounds = []; pos = ids.shape[1]
    while len(toks) < gen:
        tok = nxt; drafts = []
        for i in range(gamma):
            lg = ctx["draft1"](tok, pos + i); tok = lg[:, -1].argmax(-1, keepdim=True); drafts.append(tok)
        seq = torch.cat([nxt] + drafts, dim=1)
        tgt = ctx["verifyG"](seq, pos)[0].argmax(-1)
        d = torch.cat(drafts, dim=1)[0]; k = int((d == tgt[:gamma]).cumprod(0).sum().item())
        t = time.perf_counter() - t0
        new = d[:k].tolist() + [int(tgt[k])]
        rounds.append(dict(t=t, accepted=k, tokens=new)); toks += new; times += [t] * len(new)
        nxt = tgt[k].view(1, 1); pos += k + 1
    return toks[:gen], times[:gen], t0 - tp, rounds


def cut(tok, toks, stop):
    """Number of generated tokens up to EOS or the first occurrence of the stop string (a token that straddles the
    stop position, e.g. "()\n\n", is kept)."""
    if tok.eos_token_id in toks:   # the benchmark loop ignores EOS; the demo ends the answer there
        toks = toks[: toks.index(tok.eos_token_id)]
    idx = tok.decode(toks, skip_special_tokens=True).find(stop) if stop else -1
    if idx < 0:
        return len(toks)
    for n in range(1, len(toks) + 1):
        if len(tok.decode(toks[:n], skip_special_tokens=True)) >= idx:
            return n
    return len(toks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", required=True); p.add_argument("--fp16_model", default=None)
    p.add_argument("--prompt", default=None, help="custom prompt (default: the three built-in examples)")
    p.add_argument("--gen", type=int, default=192); p.add_argument("--gamma", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3, help="timed runs per path; the median run is kept")
    p.add_argument("--gsm8k_idx", type=int, default=11); p.add_argument("--humaneval_idx", type=int, default=156)
    p.add_argument("--chat_tokens", type=int, default=136, help="the chat example is cut at the last sentence end within this many tokens")
    p.add_argument("--chat_question", default="What is the difference between TCP and UDP?")
    p.add_argument("--out", default=None)
    a = p.parse_args(); torch.set_grad_enabled(False)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    meta = json.load(open(f"{a.pkg}/meta.json")); tok = AutoTokenizer.from_pretrained(a.pkg)
    gamma = a.gamma or {"llama2": 4, "llama2_32k": 4, "llama3.2_3b": 4}.get(meta["model"], 3)
    ex = [dict(id="custom", title="Custom prompt", context="", shown=a.prompt, stop=None)] if a.prompt else builtin_examples(a.gsm8k_idx, a.humaneval_idx, a.chat_tokens, a.chat_question)
    for e in ex:
        e["ids"] = torch.tensor(encode(tok, e["context"] + e["shown"]))[None].cuda()
    pick = lambda runs: sorted(runs, key=lambda r: r[1][-1])[len(runs) // 2]  # noqa: E731  median by total decode time

    from bitnest.attention import install_hf_stock, install_stock_wrapper
    # ---- FP16 autoregressive (original weights)
    m16 = AutoModelForCausalLM.from_pretrained(a.fp16_model or meta["source_model"], torch_dtype=torch.bfloat16, attn_implementation=install_hf_stock(),
                                               device_map="cuda", low_cpu_mem_usage=True).eval()
    for e in ex:   # cache and graphs sized per prompt (the cache length changes the attention split and thus rounding)
        c16 = R.build_ctx(m16, e["ids"].shape[1], a.gen, gamma, ("step1",)); timed_ar(m16, e["ids"], 16, c16)
        e["fp16"] = pick([timed_ar(m16, e["ids"], a.gen, c16) for _ in range(a.repeats)])
        del c16; torch.cuda.empty_cache()
    del m16; torch.cuda.empty_cache()

    # ---- BitNest: W4A8 draft + W8A8 target from one nested weight tensor, nested KV4/KV8 cache
    impl = install_stock_wrapper()
    from bitnest.cold_load import cold_load
    from bitnest import attention as A
    model, _, _ = cold_load(a.pkg, impl, DualPlaneLinear=R.DualPlaneLinear)
    R.KVP.update(on=True, offsets=True, r3=True); A.MODE_REF["get"] = lambda: R.MODE["m"]; A.DRAFT_LO["mode"] = False
    R.calibrate_kernels(model, gamma)
    for e in ex:
        ctx = R.build_ctx(model, e["ids"].shape[1], a.gen, gamma, ("step1", "draft1", "verifyG"))
        timed_ar(model, e["ids"], 16, ctx); timed_spec(model, e["ids"], 16, gamma, ctx)
        e["w8a8"] = pick([timed_ar(model, e["ids"], a.gen, ctx) for _ in range(a.repeats)])
        e["spec"] = pick([timed_spec(model, e["ids"], a.gen, gamma, ctx) for _ in range(a.repeats)])
        del ctx; torch.cuda.empty_cache()

    out = dict(model=meta["model"], source_model=meta["source_model"], gamma=gamma, gen=a.gen,
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, examples=[])
    for e in ex:
        f_t, f_times, f_pre = e["fp16"]; w_t, w_times, w_pre = e["w8a8"]; s_t, s_times, s_pre, rounds = e["spec"]
        nf, ns = cut(tok, f_t, e["stop"]), cut(tok, s_t, e["stop"])
        if e.get("max"):   # cut both lanes at the same sentence end at or before `max` tokens
            n = min(nf, ns, e["max"])
            def clean_end(k):   # ends a sentence, and is not a dangling list number such as "\n\n3."
                txt = tok.decode(s_t[:k], skip_special_tokens=True).rstrip()
                return txt.endswith((".", ":")) and not re.search(r"(^|\n)\s*\d+\.$", txt)
            while n > 1 and not (s_t[:n] == f_t[:n] and clean_end(n)):
                n -= 1
            nf = ns = n if n > 1 else min(nf, ns, e["max"])
        rr = []; n = 0
        for r in rounds:
            if n >= ns:
                break
            take = r["tokens"][: ns - n]; rr.append(dict(t=r["t"], accepted=r["accepted"], tokens=take)); n += len(take)
        tps = lambda toks, times, n: (n - 1) / times[n - 1] if n > 1 else 0.0  # noqa: E731
        rec = dict(id=e["id"], title=e["title"], prompt=e["shown"], context_tokens=int(e["ids"].shape[1]),
                   fp16=dict(text=tok.decode(f_t[:nf], skip_special_tokens=True), pieces=[tok.decode([x]) for x in f_t[:nf]],
                             times=f_times[:nf], prefill_s=f_pre, tok_s=tps(f_t, f_times, nf)),
                   w8a8=dict(tok_s=tps(w_t, w_times, cut(tok, w_t, e["stop"])), prefill_s=w_pre, identical_to_spec=w_t[:ns] == s_t[:ns]),
                   bitnest=dict(text=tok.decode(s_t[:ns], skip_special_tokens=True), pieces=[tok.decode([x]) for x in s_t[:ns]],
                                times=s_times[:ns], prefill_s=s_pre, tok_s=tps(s_t, s_times, ns), rounds=rr,
                                accept=sum(r["accepted"] for r in rr) / (gamma * max(len(rr), 1))))   # over the rounds shown
        rec["speedup"] = rec["bitnest"]["tok_s"] / rec["fp16"]["tok_s"]
        out["examples"].append(rec)
        print(f"\n=== {e['title']} ({rec['context_tokens']} prompt tokens)\n{e['shown']}")
        print(f"--- FP16 AR   {rec['fp16']['tok_s']:6.1f} tok/s ({nf} tokens)\n{rec['fp16']['text']}")
        print(f"--- BitNest   {rec['bitnest']['tok_s']:6.1f} tok/s ({ns} tokens, accept {rec['bitnest']['accept']:.1%}, "
              f"{rec['speedup']:.2f}x vs FP16; W8A8 AR {rec['w8a8']['tok_s']:.1f} tok/s, same output as W8A8 AR: {rec['w8a8']['identical_to_spec']})\n{rec['bitnest']['text']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print(f"\ntrace saved to {a.out}")


if __name__ == "__main__":
    main()
