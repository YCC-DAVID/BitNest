#!/bin/bash
# Stages 2-3: GPTQ-W4 high plane + closed-form residual low plane -> nested export -> self-contained weight package,
# then a PPL gate through the Triton engine. Needs outputs/<model>_w4a8/R.bin (scripts/train_rotation.sh) or ROT=<path>.
# Host RAM: the nested build runs its weight surgery on CPU (~80 GB peak for 7-8B models).
# Usage: bash scripts/build_package.sh <model key> [package dir]
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?model key}; PKG=${2:-release/bitnest_$M}
export MODEL=$M PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NESTED=outputs/nested_${M}_gs128.pt
[ -f $NESTED ] || torchrun --nproc_per_node=1 --master_port=${PORT:-29611} bitnest/export_nested.py $NESTED
python bitnest/export_package.py $NESTED $PKG
# small models tie the embedding: nest lm_head too (large-vocab lm_head dominates the draft's bytes)
case $M in llama3.2_3b|qwen2.5_3b) python bitnest/nest_lm_head.py $PKG ;; esac
# PPL gate (prefill path, target W8A8 / draft W4A8) + BitNest-path PPL, on WikiText-2 prompts
PR=prompts/$M; [ -f $PR/wiki2.pt ] || python tools/make_prompts.py --tokenizer $PKG --out $PR --domains wiki2
python bitnest/generate.py --pkg $PKG --prompts $PR/wiki2.pt --v2_attn --kv_planes --kv_r3 --skip_fp16 --accept_only \
    --n 1 --gen 16 --ppl_check $PR/wiki2.pt --ppl_n 10 --ppl_decode_n 1 --out $PKG.pplcheck.json
