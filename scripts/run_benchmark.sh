#!/bin/bash
# End-to-end decode speed (fp16 AR / W8A8 AR / BitNest speculative decoding), acceptance and memory on 2K-token prompts.
# Usage: bash scripts/run_benchmark.sh <package dir> [domains] [gamma]
#   e.g. bash scripts/run_benchmark.sh release/bitnest_llama2 "sharegpt gsm8k code" 4
set -euo pipefail
cd "$(dirname "$0")/.."
PKG=${1:?package dir}; DOMS=${2:-"sharegpt gsm8k code wiki2 longdoc"}
M=$(python -c "import json;print(json.load(open('$PKG/meta.json'))['model'])")
G=${3:-$([ "$M" = llama2 ] && echo 4 || echo 3)}
PR=prompts/$M; mkdir -p results
for D in $DOMS; do
  [ -f $PR/$D.pt ] || python tools/make_prompts.py --tokenizer $PKG --out $PR --domains $D
  python bitnest/generate.py --pkg $PKG --prompts $PR/$D.pt --v2_attn --kv_planes --kv_r3 --kv_draft kv4 \
      --gamma $G --gen 256 --n 10 --warmup 2 --out results/${M}_${D}.json
done
