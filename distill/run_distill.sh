#!/bin/bash
# Knowledge Distillation: Llama-3.1-8B → Llama-3.2-1B for EG Entailment
# Usage: bash run_distill.sh [GPU_IDs]
# Example: bash run_distill.sh 0        (single GPU — teacher on CPU offload if needed)
#          bash run_distill.sh 0,1      (two GPUs — teacher on GPU 0, student on GPU 1)
#
# Tune distillation:
#   bash run_distill.sh 0 --temperature 4.0 --alpha 0.7
#   bash run_distill.sh 0 --alpha 1.0       (pure KD, no CE loss)

set -e

PYTHON="/home/liang/miniconda3/envs/distll/bin/python"
SCRIPT="$(cd "$(dirname "$0")"; pwd)/distill_train.py"
GPUS="${1:-0}"
shift 1 2>/dev/null || true   # remaining args forwarded to script

echo "========================================"
echo "  EG Entailment Knowledge Distillation"
echo "  Teacher: Llama-3.1-8B-Instruct"
echo "  Student: Llama-3.2-1B-Instruct + LoRA"
echo "  GPUs   : $GPUS"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" "$SCRIPT" \
    --output_dir "$(dirname "$SCRIPT")/saves/llama3.2-1b-distill-EG" \
    "$@"
