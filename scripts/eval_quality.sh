#!/bin/bash
# Quality: (1) fake-quant PPL + teacher-forced acceptance over domains (eval_quality.py),
#          (2) GSM8K through the real engine (bitnest/task_eval.py): fp16 AR, W8A8 target AR, BitNest spec (draft KV4).
# Usage: bash scripts/eval_quality.sh <model key> [package dir]
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?model key}; PKG=${2:-release/bitnest_$M}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p results
MODEL=$M MODE=resid DATASET=${DATASET:-wiki2full,wiki2,gsm8k,code,sharegpt,longdoc} torchrun --nproc_per_node=1 --master_port=${PORT:-29621} eval_quality.py | tee results/${M}_quality.log
G=$([ "$M" = llama2 ] && echo 4 || echo 3)
for MD in fp16 target spec_kv4; do
  python bitnest/task_eval.py --pkg $PKG --task gsm8k --mode $MD --gamma $G --n ${N_GSM8K:-500} --max_len 8192 --out results/${M}_gsm8k_$MD.json
done
