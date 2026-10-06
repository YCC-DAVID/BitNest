"""Stage BitNest weight packages for the Hugging Face Hub and (optionally) upload them.

For every package a staging directory is created: large files are symlinked, meta.json / config.json get the public
source-model id (instead of a local path), README.md is replaced by a model card, base-model license files are copied
if available, and checksums.sha256 is regenerated.
Usage: python tools/upload_hf.py --release <dir with bitnest_*> --stage <staging dir> --namespace <hf user/org> [--upload] [--private]"""
import argparse
import hashlib
import json
import os
import shutil

PAPER = "https://arxiv.org/abs/2610.02800"
CODE = "https://github.com/YCC-DAVID/BitNest"
# key -> (repo name, public source model, license tag, license name, pretty name, default gamma)
MODELS = {
    "llama2":      ("BitNest-Llama-2-7B",     "meta-llama/Llama-2-7b-hf",        "llama2",     None,           "Llama-2-7B",       4),
    "llama2_32k":  ("BitNest-Llama-2-7B-32K", "togethercomputer/LLaMA-2-7B-32K", "llama2",     None,           "LLaMA-2-7B-32K",   4),
    "llama3":      ("BitNest-Llama-3-8B",     "meta-llama/Meta-Llama-3-8B",      "llama3",     None,           "Meta-Llama-3-8B",  3),
    "qwen2":       ("BitNest-Qwen2-7B",       "Qwen/Qwen2-7B",                   "apache-2.0", None,           "Qwen2-7B",         3),
    "qwen2.5":     ("BitNest-Qwen2.5-7B",     "Qwen/Qwen2.5-7B",                 "apache-2.0", None,           "Qwen2.5-7B",       3),
    "llama3.2_3b": ("BitNest-Llama-3.2-3B",   "meta-llama/Llama-3.2-3B",         "llama3.2",   None,           "Llama-3.2-3B",     4),
    "qwen2.5_3b":  ("BitNest-Qwen2.5-3B",     "Qwen/Qwen2.5-3B",                 "other",      "qwen-research", "Qwen2.5-3B",       3),
}
SMALL = {"meta.json", "config.json", "README.md", "checksums.sha256", "PPL_CHECK.txt"}

CARD = """---
license: {license}{license_name}
base_model: {src}
library_name: bitnest
tags:
- quantization
- speculative-decoding
- self-speculative-decoding
- w4a8
- w8a8
- bitnest
---

# {repo}

**BitNest** nested W4/W8 weight package for [{pretty}](https://huggingface.co/{src}), from the paper
[BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration]({paper}). Code: [{code}]({code}).

A single 8-bit weight tensor serves two precisions: the **target** reads all 8 bits (W8A8, near-lossless) and the
**draft** reads only the high 4-bit nibble (W4A8). The high nibble is a GPTQ-W4 model fitted first; the low nibble is the
closed-form residual back to W8. The two nibbles are stored as separate planes, so a draft step reads half the bytes and
the draft costs **no extra memory**. Speculative decoding verifies every draft token with the target, so the output is
exactly the W8A8 target's greedy output.

| | |
|---|---|
| Base model | [{src}](https://huggingface.co/{src}) |
| Rotation | SpinQuant R1/R2 (learned, W4A8 recipe) fused offline, R4 online Hadamard on `down_proj` inputs |
| Weights | GPTQ-W4 high plane + residual low plane, group size 128 ({lm_head}) |
| Activations | symmetric int8, per token (o_proj: per 128-group) |
| KV cache | nested at runtime: draft reads KV4, target reads KV8 (`--kv_planes`) |
| Default draft length | γ = {gamma} |

## Files
- `planes.safetensors`: per decoder Linear `<name>.hi` / `<name>.lo` (uint8 `[N, K/2]`, unsigned nibbles `u = q + 8`; byte `j` holds column `2j` in its low nibble and `2j+1` in its high nibble), `<name>.scale` (fp16 `[N, K/128]`), `<name>.bias` (if any).
  draft `W4 = (hi - 8) * scale`, target `W8 = (16 * (hi - 8) + (lo - 8)) * scale / 16`.
- `aux.safetensors`: rotated `embed_tokens`, `final_norm`, R4 Hadamard matrices{aux_lm}.
- `meta.json`: layout, formulas, activation / KV quantization and RoPE details. `config.json`, tokenizer: from the base model.
- `rotation/R.bin`: the learned SpinQuant rotations (R1, per-layer R2). With it the package can be rebuilt from the base model without
  rotation training: `ROT=rotation/R.bin bash scripts/build_package.sh {key}`.

## Usage
```bash
git clone {code} && cd BitNest && pip install -r requirements.txt
huggingface-cli download {ns}/{repo} --local-dir bitnest_{key}
python tools/make_prompts.py --tokenizer bitnest_{key} --out prompts/{key} --domains sharegpt,wiki2
# fp16 AR vs W8A8 AR vs BitNest speculative decoding (2K-token prompts, 256 new tokens)
python bitnest/generate.py --pkg bitnest_{key} --prompts prompts/{key}/sharegpt.pt \\
    --v2_attn --kv_planes --kv_r3 --gamma {gamma} --gen 256 --n 10 --out results/{key}_sharegpt.json
# PPL of target / draft through the engine
python bitnest/generate.py --pkg bitnest_{key} --prompts prompts/{key}/wiki2.pt --v2_attn --kv_planes --kv_r3 \\
    --skip_fp16 --accept_only --n 1 --gen 16 --ppl_check prompts/{key}/wiki2.pt --out results/{key}_ppl.json
# pure PyTorch reference loader (CPU or GPU, no Triton)
python bitnest/reference_model.py --pkg bitnest_{key} --mode target --a8
```

## License
These weights are a derivative of [{src}](https://huggingface.co/{src}) and are distributed under the base model's license
({license_text}). {extra}

## Citation
```bibtex
@article{{yang2026bitnest,
  title   = {{BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration}},
  author  = {{Yang, Chence and Cheng, Ningxi and Akbari, Arash and Tan, Qitao and Zhu, Qingchan and Zhang, Ci and Yang, Changdi and Wang, Yanzhi and Niu, Wei and Wang, Jinhui and Lu, Jin and Yuan, Geng}},
  journal = {{arXiv preprint arXiv:2610.02800}},
  year    = {{2026}}
}}
```
"""
LICENSE_TEXT = {"llama2": "Llama 2 Community License", "llama3": "Meta Llama 3 Community License", "llama3.2": "Llama 3.2 Community License",
                "apache-2.0": "Apache 2.0", "other": "Qwen Research License"}
EXTRA = {"llama3": "Built with Meta Llama 3.", "llama3.2": "Built with Llama."}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def stage(key, src_dir, out_dir, ns, license_dir=None, rot=None):
    repo, src, lic, lic_name, pretty, gamma = MODELS[key]
    os.makedirs(out_dir, exist_ok=True)
    old_sums = {}
    if os.path.exists(f"{src_dir}/checksums.sha256"):
        for line in open(f"{src_dir}/checksums.sha256"):
            h, fn = line.split(); old_sums[fn] = h
    for fn in os.listdir(src_dir):
        if fn in SMALL or fn.endswith(".pre_lmhead") or os.path.isdir(f"{src_dir}/{fn}"):
            continue
        dst = f"{out_dir}/{fn}"
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(f"{src_dir}/{fn}"), dst)
    meta = json.load(open(f"{src_dir}/meta.json")); meta["source_model"] = src
    meta.get("reference", {}).update(loader="bitnest/reference_model.py (pure PyTorch dequantizing loader, runs on CPU)", gpu_runner="bitnest/generate.py")
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1, ensure_ascii=False)
    cfg = json.load(open(f"{src_dir}/config.json")); cfg["_name_or_path"] = src
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=2)
    if rot and os.path.exists(rot):
        os.makedirs(f"{out_dir}/rotation", exist_ok=True)
        if not os.path.lexists(f"{out_dir}/rotation/R.bin"):
            os.symlink(os.path.abspath(rot), f"{out_dir}/rotation/R.bin")
    nested_head = "lm_head" in meta["layers"]
    card = CARD.format(license=lic, license_name=(f"\nlicense_name: {lic_name}" if lic_name else ""), src=src, repo=repo, pretty=pretty, paper=PAPER, code=CODE,
                       lm_head=("lm_head nested too, RTN W4 + residual" if nested_head else "embedding 8-bit RTN, lm_head bf16"),
                       aux_lm=("" if nested_head else ", rotated `lm_head` (bf16, final norm fused)"), gamma=gamma, ns=ns, key=key,
                       license_text=LICENSE_TEXT[lic], extra=EXTRA.get(lic, ""))
    open(f"{out_dir}/README.md", "w").write(card)
    if license_dir:
        for fn in ("LICENSE", "USE_POLICY.md"):
            if os.path.exists(f"{license_dir}/{fn}") and not os.path.exists(f"{out_dir}/{fn}"):
                shutil.copy2(f"{license_dir}/{fn}", f"{out_dir}/{fn}")
    with open(f"{out_dir}/checksums.sha256", "w") as f:
        for fn in sorted(os.listdir(out_dir)):
            if fn == "checksums.sha256" or os.path.isdir(f"{out_dir}/{fn}"):
                continue
            p = f"{out_dir}/{fn}"
            h = old_sums[fn] if (os.path.islink(p) and fn in old_sums) else sha(p)
            f.write(f"{h}  {fn}\n")
        if os.path.exists(f"{out_dir}/rotation/R.bin"):
            f.write(f"{sha(f'{out_dir}/rotation/R.bin')}  rotation/R.bin\n")
    return repo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True); p.add_argument("--stage", required=True); p.add_argument("--namespace", required=True)
    p.add_argument("--models", default=",".join(MODELS)); p.add_argument("--license_dirs", default="{}", help='JSON {key: base model dir with LICENSE files}')
    p.add_argument("--rot_root", default=None, help="directory with <key>_w4a8/R.bin (learned rotations, shipped as rotation/R.bin)")
    p.add_argument("--upload", action="store_true"); p.add_argument("--private", action="store_true")
    a = p.parse_args(); lic_dirs = json.loads(a.license_dirs)
    for key in a.models.split(","):
        out = f"{a.stage}/bitnest_{key}"
        repo = stage(key, f"{a.release}/bitnest_{key}", out, a.namespace, lic_dirs.get(key), a.rot_root and f"{a.rot_root}/{key}_w4a8/R.bin")
        print(f"[stage] {key} -> {out} ({a.namespace}/{repo})", flush=True)
        if a.upload:
            from huggingface_hub import HfApi
            api = HfApi(); rid = f"{a.namespace}/{repo}"
            api.create_repo(rid, repo_type="model", private=a.private, exist_ok=True)
            api.upload_large_folder(repo_id=rid, repo_type="model", folder_path=out, private=a.private)
            print(f"[upload] done https://huggingface.co/{rid}", flush=True)


if __name__ == "__main__":
    main()
