"""Fake-quant quality of BitNest (teacher-forced, no custom kernels): one build gives
  * PPL of the target (W8A8, resid) and of the draft (W4A8 = native GPTQ-W4) on each domain
  * teacher-forced acceptance: top-1 agreement of draft vs target on the same segments
  * optionally GSM8K (lm-eval, 5-shot) of target and draft
MODE=fp16 / w8 give the fp16 and plain GPTQ-W8A8 reference columns (PPL only).

Domains: wiki2 | gsm8k | code | sharegpt | longdoc | pg19 (20 x 2048-token segments each; pg19 needs PG19_JSON),
         wiki2full = every 2048-token segment of the WikiText-2 test set (the standard PPL protocol).
Usage: MODEL=llama2 torchrun --nproc_per_node=1 eval_quality.py
       MODEL=qwen2.5 MODE=resid DATASET=wiki2full,code,sharegpt GSM8K=1 torchrun --nproc_per_node=1 eval_quality.py"""
import datetime
import math
import os

import torch
import torch.distributed as dist
import transformers

from bitnest.build import GS, INPUT_MODEL, TAG, ModelClass, build_model, build_resid_model, get_domain_ids, swap_resid


@torch.no_grad()
def run_domain(model, loader, target_ids=None):
    """Returns (ppl, argmax ids per batch, top-1 agreement with target_ids if given)."""
    model.eval(); nll = ntok = 0.0; ids_out = []; match = tot = 0
    for bi, batch in enumerate(loader):
        ids = batch.cuda(); lg = model(ids).logits.float()
        nll += torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction="sum").item(); ntok += ids[:, 1:].numel()
        am = lg.argmax(-1).cpu(); ids_out.append(am)
        if target_ids is not None:
            match += (am == target_ids[bi]).sum().item(); tot += am.numel()
        del lg
    return math.exp(nll / ntok), ids_out, (match / tot if tot else None)


def gsm8k(model, tokenizer):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    model.config.use_cache = True
    for mod in model.modules():   # the quantizer stores some weights as plain tensor attributes; model.cuda() does not move them
        for n in ("weight", "bias"):
            t = getattr(mod, n, None)
            if torch.is_tensor(t) and not isinstance(t, torch.nn.Parameter) and t.device.type != "cuda":
                setattr(mod, n, t.cuda())
    lim = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    r = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8), tasks=["gsm8k"], num_fewshot=5, limit=lim)["results"]["gsm8k"]
    model.config.use_cache = False
    return r.get("exact_match,strict-match"), r.get("exact_match,flexible-extract")


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    mode = os.environ.get("MODE", "resid")            # resid | w8 | fp16
    ds_names = [s.strip() for s in os.environ.get("DATASET", "wiki2full,wiki2,gsm8k,code,sharegpt,longdoc").split(",") if s.strip()]
    if mode == "fp16":
        model = ModelClass.from_pretrained(INPUT_MODEL, torch_dtype=torch.bfloat16).cuda(); model.config.use_cache = False
    elif mode == "w8":
        model, _, _ = build_model(8, drop_bufs=True)
    elif mode == "resid":
        model, _, _ = build_resid_model()
        pairs = [m for m in model.modules() if hasattr(m, "w8_tensor")]
        dmax = max((m.w8_tensor - m.w4_tensor).abs().max().item() for m in pairs) if pairs else 0.0
        print(f"[resid] {len(pairs)} nested modules, max|W8 - W4| = {dmax:.3e}", flush=True)
        assert pairs and dmax > 0, "nesting is not effective: w8 / w4 tensors missing or identical"
    else:
        raise KeyError(f"unknown MODE={mode} (fp16|w8|resid)")
    tok = transformers.AutoTokenizer.from_pretrained(INPUT_MODEL, use_fast=True, add_eos_token=False, add_bos_token=False)
    loaders = {}
    for n in ds_names:   # "wiki2@4096" = wiki2 cut into 4096-token segments
        d, sl = n.split("@", 1) if "@" in n else (n, "2048")
        ids = get_domain_ids(tok, "wiki2", seqlen=int(sl), nseq=10**9) if d == "wiki2full" else get_domain_ids(tok, d, seqlen=int(sl), nseq=20)
        loaders[n] = torch.utils.data.DataLoader(ids, batch_size=1)

    if mode != "resid":
        cfg = "fp16" if mode == "fp16" else "w8a8"
        for n in ds_names:
            p, _, _ = run_domain(model, loaders[n])
            print(f"QUALITY {TAG} [{cfg}] {n}: ppl={p:.4f}", flush=True)
    else:
        tgt = {}
        swap_resid(model, "w8")
        for n in ds_names:
            p, tgt[n], _ = run_domain(model, loaders[n])
            print(f"QUALITY {TAG} [target W8A8 (resid), gs{GS}] {n}: ppl={p:.4f}", flush=True)
        if os.environ.get("GSM8K") == "1":
            s, f = gsm8k(model, tok); print(f"QUALITY {TAG} [target W8A8 (resid)] gsm8k(lm-eval 5-shot): strict={s:.4f} flexible={f:.4f}", flush=True)
        swap_resid(model, "w4")
        for n in ds_names:
            p, _, acc = run_domain(model, loaders[n], tgt[n])
            print(f"QUALITY {TAG} [draft W4A8, gs{GS}] {n}: ppl={p:.4f} teacher-forced accept vs target={acc:.2%}", flush=True)
        if os.environ.get("GSM8K") == "1":
            s, f = gsm8k(model, tok); print(f"QUALITY {TAG} [draft W4A8] gsm8k(lm-eval 5-shot): strict={s:.4f} flexible={f:.4f}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
