#!/bin/bash
# Stage 1: learn the SpinQuant rotations R1/R2 with the W4A8 recipe used in the paper (Cayley SGD on the Stiefel manifold,
# WikiText-2 calibration, 100 steps, RTN weights in the loop). ~30 min for a 7B model on one 48 GB GPU.
# Usage: bash scripts/train_rotation.sh <model key> [num_gpus]     (model keys: eval_utils/model_registry.py)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?model key, e.g. llama2}; NGPU=${2:-1}
IM=${INPUT_MODEL:-$(python -c "import os;os.environ['MODEL']='$M';from eval_utils.model_registry import get_entry;print(get_entry()['input_model'])")}
OUT=outputs/${M}_w4a8; mkdir -p $OUT/logs
torchrun --nnodes=1 --nproc_per_node=$NGPU --master_port=${PORT:-29601} optimize_rotation.py \
    --input_model "$IM" \
    --output_rotation_path $OUT/R.bin \
    --output_dir $OUT/train \
    --logging_dir $OUT/logs \
    --model_max_length 2048 \
    --fp16 False --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 1.5 \
    --max_steps 100 \
    --save_safetensors False \
    --save_strategy "no" \
    --w_bits 4 --a_bits 8 --k_bits 16 --v_bits 16 \
    --w_rtn
echo "rotation saved to $OUT/R.bin"
