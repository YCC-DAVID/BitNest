# BitNest: Bit-Nested Speculative Decoding

Code for **[BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration](https://arxiv.org/abs/2610.02800)**.

BitNest embeds a low-precision draft model *inside* the weights of a higher-precision target model. A single 8-bit
weight tensor serves both: the **target** reads all 8 bits (W8A8), the **draft** reads only the high 4-bit nibble
(W4A8). Instead of truncating a W8 model, BitNest first builds a strong low-precision foundation (GPTQ-W4) and recovers
the higher precision through a closed-form residual refinement. The two nibbles live in separate memory planes, so a
draft step reads half the bytes of a target step and the draft costs no extra memory. The same nesting is applied to
the KV cache (draft KV4 / target KV8) for long contexts. Speculative verification makes the output exactly the W8A8
target's output.

Across 7B–8B models BitNest reaches an average speculative acceptance rate of **95.2%** and a **1.48–1.61× end-to-end
speedup over FP16 autoregressive decoding**.

<p align="center"><i>
target W8 = (16·q4 + qr)·s/16 &nbsp;&nbsp;|&nbsp;&nbsp; draft W4 = q4·s &nbsp;&nbsp;|&nbsp;&nbsp;
q4 = GPTQ-W4 code (high plane), qr = round((W − q4·s)/(s/16)) (low plane)
</i></p>

## Repository layout

| Path | Content |
|---|---|
| `optimize_rotation.py`, `train_utils/` | Stage 1: learned rotations R1/R2 (SpinQuant, Cayley SGD on the Stiefel manifold), W4A8 recipe |
| `ptq.py`, `eval_utils/`, `utils/` | Rotation fusion, GPTQ, activation / KV quantizers, modeling (based on SpinQuant / QwenSpinQuant) |
| `bitnest/build.py` | Rotated + GPTQ model construction and the closed-form residual nesting |
| `bitnest/export_nested.py` | Stage 2: export the nested integer codes (hi/lo planes + group scales) |
| `bitnest/export_package.py`, `bitnest/nest_lm_head.py` | Stage 3: self-contained weight package (safetensors + meta.json); optional nested lm_head |
| `bitnest/w4a8_planes.py` | Triton kernels: dual-plane W4A8/W8A8 GEMM (decode/verify), M=1 GEMV, fused int8 activation quantization, plane dequantization (prefill) |
| `bitnest/decode_attn.py`, `bitnest/kv_planes.py`, `bitnest/kv_cache_planes.py` | Triton flash-decoding attention (bf16 KV and nested KV4/KV8 planes), fused KV quantize-and-scatter, `PlanesCache` |
| `bitnest/attention.py` | Hooks the Triton decode attention into HF Llama / Qwen2 |
| `bitnest/generate.py` | End-to-end engine: CUDA-graph draft/verify loop, fp16 / W8A8 / BitNest benchmarks, PPL checks |
| `bitnest/cold_load.py`, `bitnest/reference_model.py` | Package loaders: GPU cold load (no bf16 model materialized) and a pure-PyTorch reference (CPU) |
| `bitnest/task_eval.py`, `eval_quality.py` | Task-level evaluation through the engine (GSM8K, LongBench) and fake-quant PPL / acceptance |
| `tools/make_prompts.py` | Evaluation prompt files (wiki2, gsm8k, code, sharegpt, longdoc, pg19) |
| `scripts/` | Pipeline: `train_rotation.sh`, `build_package.sh`, `run_benchmark.sh`, `eval_quality.sh` |

## Installation

```bash
conda create -n bitnest python=3.10 -y && conda activate bitnest
pip install -r requirements.txt
```
Tested with PyTorch 2.10 / Triton 3.6 / transformers 4.49 on NVIDIA RTX A6000 (CUDA 12.8) and Jetson Orin.
`fast-hadamard-transform` provides the online Hadamard; without it, set `BITNEST_HAD_TORCH=1` (slower PyTorch fallback).

## Quick start with released weight packages

| Model | Package | γ |
|---|---|:---:|
| Llama-2-7B | [YccHugAi/BitNest-Llama-2-7B](https://huggingface.co/YccHugAi/BitNest-Llama-2-7B) | 4 |
| Meta-Llama-3-8B | [YccHugAi/BitNest-Llama-3-8B](https://huggingface.co/YccHugAi/BitNest-Llama-3-8B) | 3 |
| Qwen2-7B | [YccHugAi/BitNest-Qwen2-7B](https://huggingface.co/YccHugAi/BitNest-Qwen2-7B) | 3 |
| Qwen2.5-7B | [YccHugAi/BitNest-Qwen2.5-7B](https://huggingface.co/YccHugAi/BitNest-Qwen2.5-7B) | 3 |
| LLaMA-2-7B-32K (long context) | [YccHugAi/BitNest-Llama-2-7B-32K](https://huggingface.co/YccHugAi/BitNest-Llama-2-7B-32K) | 4 |
| Llama-3.2-3B (edge, nested lm_head) | [YccHugAi/BitNest-Llama-3.2-3B](https://huggingface.co/YccHugAi/BitNest-Llama-3.2-3B) | 4 |
| Qwen2.5-3B (edge, nested lm_head) | [YccHugAi/BitNest-Qwen2.5-3B](https://huggingface.co/YccHugAi/BitNest-Qwen2.5-3B) | 3 |

Each package holds `planes.safetensors` (hi/lo planes + scales), `aux.safetensors` (rotated embedding / lm_head / norms,
Hadamard matrices), `meta.json` (layout and formulas), the tokenizer, and the learned rotation `rotation/R.bin`.

```bash
huggingface-cli download YccHugAi/BitNest-Llama-2-7B --local-dir release/bitnest_llama2
python tools/make_prompts.py --tokenizer release/bitnest_llama2 --out prompts/llama2 --domains sharegpt,wiki2

# fp16 AR vs W8A8 AR vs BitNest speculative decoding: 10 prompts of 2048 tokens, 256 new tokens, batch 1
python bitnest/generate.py --pkg release/bitnest_llama2 --prompts prompts/llama2/sharegpt.pt \
    --v2_attn --kv_planes --kv_r3 --gamma 4 --gen 256 --n 10 --out results/llama2_sharegpt.json
```
The run prints per-prompt W8A8 / speculative tok/s and acceptance, and writes a JSON with `ar_fp16_tok_s`,
`ar_w8a8_tok_s`, `spec_tok_s`, `speedup_vs_fp16`, `accept`, the per-position acceptance and peak memory.

Useful flags of `bitnest/generate.py`:
- `--kv_planes` nested KV cache (draft KV4 / target KV8); `--kv_draft kv8|win:N` changes what the draft reads, `--kv_r3` per-head Hadamard on K (recommended).
- `--ppl_check <prompts.pt>` prefill-path PPL of target and draft; `--ppl_decode_n N` PPL along the decode path and along the exact BitNest draft+verify path; `--ppl_tail N` long-context tail PPL.
- `--only fp16|w8a8|spec`, `--skip_fp16`, `--mem_cap_gib G` (emulate a smaller device), `--eager` (no CUDA graphs).
- Environment: `BITNEST_GEMV_M1=1` M=1 GEMV kernel (faster on small GPUs such as Jetson Orin), `BITNEST_PREFILL_DTYPE=fp16|bf16|fp32`, `BITNEST_PREFILL_CHUNK`, `BITNEST_CALIB_WS_MB`.

Pure-PyTorch reference (no Triton, runs on CPU), e.g. for porting the format to another runtime:
```bash
python bitnest/reference_model.py --pkg release/bitnest_llama2 --mode target --a8 --ppl prompts/llama2/wiki2.pt --n 3
```

## Building packages from scratch

Model keys are listed in `eval_utils/model_registry.py` (`llama2`, `llama3`, `qwen2`, `qwen2.5`, `llama2_32k`,
`llama3.2_3b`, `qwen2.5_3b`); set `INPUT_MODEL=<local dir>` to use a local copy of the base model.

```bash
# 1) learn the rotations (~30 min for 7B on one 48 GB GPU); or reuse rotation/R.bin from a released package via ROT=...
bash scripts/train_rotation.sh llama2                  # -> outputs/llama2_w4a8/R.bin
# 2-3) GPTQ-W4 high plane + residual low plane -> nested export -> weight package -> PPL gate through the engine
bash scripts/build_package.sh llama2                   # -> release/bitnest_llama2/
# speed / acceptance on several domains
bash scripts/run_benchmark.sh release/bitnest_llama2 "sharegpt gsm8k code"
# quality: fake-quant PPL + teacher-forced acceptance per domain, GSM8K through the engine
bash scripts/eval_quality.sh llama2
```
The nested build performs its weight surgery on the CPU; plan for ~80 GB of host RAM for 7–8B models.
Long-context tasks: `python bitnest/task_eval.py --pkg release/bitnest_llama2_32k --task longbench:qasper --mode spec_kv4 --max_len 31500 --out ...`.

## How the engine works

- **Dual-plane linear layers.** Every decoder Linear stores `hi` and `lo` uint8 planes `[N, K/2]` plus fp16 group scales (group 128).
  Decode/verify (M ≤ 64 rows) quantizes activations to int8 in one fused kernel and runs an int8×int8→int32 dot per 128-group,
  reading only `hi` in draft mode and `hi`+`lo` in target mode. Prefill dequantizes the planes once and uses cuBLAS.
- **Online rotation.** R1/R2 are fused offline; R4 (blocked Hadamard) is applied to `down_proj` inputs at runtime.
- **Nested KV cache.** K/V are quantized per token and head as `q4` + residual `qr` into two planes at write time; the attention
  kernel reads one plane (draft) or both (target). Qwen k/v biases are subtracted before quantization (exact for K, added back for V).
- **Speculative loop.** γ draft steps (M=1) → one verify step (M=γ+1) over the same positions, which overwrites the draft's KV →
  greedy prefix acceptance. Each step is a captured CUDA graph; draft and target share one cache.
- **Kernel calibration.** Launch configurations are picked by timing on the real weights before graph capture. Different picks change
  the fp32 split-K reduction order, so decode-path numbers can vary slightly between runs on deep small models.

## Citation

```bibtex
@article{yang2026bitnest,
  title   = {BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration},
  author  = {Yang, Chence and Cheng, Ningxi and Akbari, Arash and Tan, Qitao and Zhu, Qingchan and Zhang, Ci and Yang, Changdi and Wang, Yanzhi and Niu, Wei and Wang, Jinhui and Lu, Jin and Yuan, Geng},
  journal = {arXiv preprint arXiv:2610.02800},
  year    = {2026}
}
```

## Acknowledgements and license

The rotation training, rotation fusion, GPTQ and quantized modeling code build on
[SpinQuant](https://github.com/facebookresearch/SpinQuant) and its Qwen port [QwenSpinQuant](https://github.com/shijiew/QwenSpinQuant),
which in turn build on [QuaRot](https://github.com/spcl/QuaRot) and [GPTQ](https://github.com/IST-DASLab/gptq).
Following SpinQuant, this repository is released under the [CC BY-NC 4.0](LICENSE) license. The released weight packages
follow the licenses of their base models.
