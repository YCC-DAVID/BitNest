"""Build the evaluation prompt files used by bitnest/generate.py: for each domain, the text is concatenated, tokenized,
cut into seq_len chunks (the last partial chunk is dropped) and the first token of every chunk is set to BOS (EOS for
tokenizers without BOS); the first N chunks are saved as a LongTensor [N, seq_len] to <out>/<domain>.pt.
Domains: wiki2 gsm8k code sharegpt longdoc pg19u (pg19u = first 20 books of the PG-19 test split; pass --pg19_json).
Usage: python tools/make_prompts.py --tokenizer meta-llama/Llama-2-7b-hf --out prompts/llama2 [--n 10 --seq_len 2048]"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["wiki2", "gsm8k", "code", "sharegpt", "longdoc", "pg19u"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--pg19_json", default=None, help="PG-19 test split as jsonl with a 'text' field")
    a = p.parse_args()
    os.environ.setdefault("MODEL", "llama2")   # bitnest.build reads the registry at import time
    from bitnest.build import domain_text
    tok = AutoTokenizer.from_pretrained(a.tokenizer); os.makedirs(a.out, exist_ok=True)
    first = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for d in a.domains.split(","):
        if d == "pg19u" and not a.pg19_json:
            print("[prompts] skip pg19u (no --pg19_json)"); continue
        ids = tok.encode(domain_text(d, a.pg19_json), return_tensors="pt")
        chunks = list(ids.split(a.seq_len, dim=-1))[:-1]
        assert chunks, f"{d}: not enough text for one {a.seq_len}-token chunk"
        for c in chunks:
            c[:, 0] = first
        data = torch.cat(chunks, dim=0)
        data = data.repeat(max(1, -(-a.n // len(chunks))), 1)[: a.n]   # repeat when the domain has fewer than n chunks
        torch.save(data, f"{a.out}/{d}.pt")
        print(f"[prompts] {d}: {ids.numel()} tokens -> {len(chunks)} chunks, saved {tuple(data.shape)} to {a.out}/{d}.pt")


if __name__ == "__main__":
    main()
