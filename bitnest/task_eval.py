"""Task-level quality through the real kernel generation path. The target is fixed to W8A8 + KV8 planes; compares draft
KV8 / draft KV4 speculative decoding (plus target-only AR and fp16 AR) on final task scores, and reports acceptance.
Tasks: gsm8k (8-shot greedy, strict last number) | longbench:<qasper|hotpotqa|multifieldqa_en|gov_report|samsum|trec>
       (official LongBench prompts / max_gen / metrics, data/longbench/*.json)
Modes: target (W8A8 AR) | fp16 (HF AR) | spec_kv8 (draft reads KV8) | spec_kv4 (draft reads KV4, nested; the BitNest default)
       | target_kv4 / spec_kv4all (ablation: KV4 everywhere, no nesting)
Usage: python bitnest/task_eval.py --pkg bitnest_llama2_32k --task longbench:qasper --mode spec_kv4 --n 200 --max_len 31500 --out x.json"""
import argparse
import collections
import json
import os
import re
import string
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
p = argparse.ArgumentParser()
p.add_argument("--pkg", required=True)
p.add_argument("--task", required=True)
p.add_argument("--mode", required=True, choices=["target", "spec_kv8", "spec_kv4", "fp16", "target_kv4", "spec_kv4all"])
p.add_argument("--n", type=int, default=200)
p.add_argument("--gamma", type=int, default=4)
p.add_argument("--max_len", type=int, default=31500, help="prompt + generation budget; longer prompts are truncated in the middle")
p.add_argument("--gen", type=int, default=256, help="max new tokens for gsm8k")
p.add_argument("--fp16_model", default=None)
p.add_argument("--mem_cap_gib", type=float, default=float(os.environ.get("MEM_CAP_GIB", "0")), help="cap this process' CUDA memory (0 = no cap)")
p.add_argument("--out", required=True)
a = p.parse_args()
torch.set_grad_enabled(False)
if a.mem_cap_gib:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)); print(f"[task_eval] mem cap {a.mem_cap_gib} GiB", flush=True)
import bitnest.generate as R  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.pkg); meta = json.load(open(f"{a.pkg}/meta.json"))
# ---- model
from bitnest.attention import install_stock_wrapper  # noqa: E402
impl = install_stock_wrapper()
if a.mode == "fp16":
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(a.fp16_model or meta["source_model"], torch_dtype=torch.bfloat16, attn_implementation=impl, device_map="cuda", low_cpu_mem_usage=True).cuda().eval()
else:
    from bitnest.cold_load import cold_load
    model, _, _ = cold_load(a.pkg, impl, DualPlaneLinear=R.DualPlaneLinear)
    R.KVP["on"] = os.environ.get("NO_KVP") != "1"; R.KVP["r3"] = os.environ.get("NO_R3") != "1"; R.KVP["offsets"] = True
    from bitnest import attention as _v2
    _v2.MODE_REF["get"] = lambda: R.MODE["m"]; _v2.DRAFT_LO["mode"] = (True if a.mode == "spec_kv8" else False)
    _v2.TARGET_LO["mode"] = (a.mode not in ("target_kv4", "spec_kv4all"))   # target_kv4 / spec_kv4all: verify and AR also read only the high plane = KV4 everywhere
    R.calibrate_kernels(model, a.gamma)


# ---- data
def truncate_mid(ids, max_len):
    if len(ids) <= max_len:
        return ids
    h = max_len // 2; return ids[:h] + ids[-(max_len - h):]   # the first token (BOS) stays in the head half


if a.task == "gsm8k":
    import datasets
    ds = datasets.load_dataset("gsm8k", "main"); shots = ds["train"].select(range(8)); test = ds["test"].select(range(min(a.n, len(ds["test"]))))
    prefix = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    items = [dict(prompt=prefix + f"Question: {t['question']}\nAnswer:", gold=t["answer"].split("####")[-1].strip().replace(",", "")) for t in test]; max_gen = a.gen; cut_nl = False
else:
    name = a.task.split(":", 1)[1]
    import datasets
    ds = datasets.load_dataset("THUDM/LongBench", name, split="test", trust_remote_code=True); ds = ds.select(range(min(a.n, len(ds))))
    P = json.load(open(f"{ROOT}/data/longbench/dataset2prompt.json"))[name]; max_gen = json.load(open(f"{ROOT}/data/longbench/dataset2maxlen.json"))[name]
    items = [dict(prompt=P.format(context=r["context"], input=r["input"]), gold=r["answers"], classes=r.get("all_classes")) for r in ds]; cut_nl = name not in ("gov_report",)


# ---- metrics (LongBench conventions)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation)); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def f1(pred, gold):
    p_ = norm(pred).split(); g_ = norm(gold).split(); c = collections.Counter(p_) & collections.Counter(g_); ns = sum(c.values())
    if ns == 0:
        return 0.0
    pr = ns / len(p_); rc = ns / len(g_); return 2 * pr * rc / (pr + rc)


def rougeL(pred, gold):
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False).score(gold, pred)["rougeL"].fmeasure


def classification(pred, gold, classes):
    m = [c for c in classes if c.lower() in pred.lower()]; return (1.0 / len(m) if gold in m else 0.0) if m else 0.0


def score(pred, gold, classes=None):
    if a.task == "gsm8k":
        t = pred.split("Question:")[0]; m = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", "")); return float(bool(m) and m[-1].strip(".") == gold)
    name = a.task.split(":", 1)[1]
    if name in ("gov_report", "samsum", "multi_news"):
        return max(rougeL(pred, g) for g in gold)
    if name == "trec":
        return classification(pred, gold[0], classes)
    return max(f1(pred, g) for g in gold)


# ---- generation
BOS = [tok.bos_token_id] if tok.bos_token_id is not None else ([tok.eos_token_id] if os.environ.get("ADD_BOS", "1") == "1" else [])   # Qwen has no BOS: prepend <|endoftext|> like the calibration / prompt files


def encode(t):
    ids = tok(t, add_special_tokens=False).input_ids; return BOS + ids if (tok.bos_token_id is None or ids[:1] != BOS) else ids


enc = [truncate_mid(encode(it["prompt"]), a.max_len - max_gen) for it in items]; Pmax = max(len(e) for e in enc)
print(f"[task_eval] {a.task} mode={a.mode} n={len(items)} Pmax={Pmax} max_gen={max_gen}", flush=True)
ctx = R.build_ctx(model, Pmax, max_gen, a.gamma, ("step1",) if a.mode in ("target", "fp16", "target_kv4") else ("step1", "draft1", "verifyG"))
recs = []; tot = 0.0; accs = []; tprs = []
for i, (it, e) in enumerate(zip(items, enc)):
    ids = torch.tensor(e)[None].cuda()
    if a.mode in ("target", "fp16", "target_kv4"):
        toks, _ = R.ar_generate(model, ids, max_gen, ctx); st = None
    else:
        toks, _, st = R.spec_generate(model, ids, max_gen, a.gamma, ctx); accs.append(st["accept"]); tprs.append(st["tokens_per_round"])
    text = tok.decode(toks, skip_special_tokens=True); pred = text.split("\n")[0] if cut_nl else text
    sc = score(pred, it["gold"], it.get("classes")); tot += sc; recs.append(dict(i=i, pred=pred[:300], score=sc, toks=toks))
    if (i + 1) % 25 == 0:
        print(f"[task_eval] {i+1}/{len(items)} mean {tot/(i+1):.4f}" + (f" accept {sum(accs)/len(accs):.3f}" if accs else ""), flush=True)
res = dict(task=a.task, mode=a.mode, n=len(items), score=tot / len(items), accept=(sum(accs) / len(accs) if accs else None),
           tokens_per_round=(sum(tprs) / len(tprs) if tprs else None), gamma=a.gamma, max_gen=max_gen, max_len=a.max_len, records=recs)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w")); print("TASK_RESULT " + json.dumps({k: v for k, v in res.items() if k != "records"}), flush=True)
