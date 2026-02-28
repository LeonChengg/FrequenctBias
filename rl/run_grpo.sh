#!/bin/bash
# GRPO fine-tuning for EG Entailment
# Usage: bash run_grpo.sh [GPU_ID]
# Example: bash run_grpo.sh 0
#
# Optional: start from SFT LoRA checkpoint by passing --sft_lora_path:
#   bash run_grpo.sh 0 --sft_lora_path ../finetune/saves/llama3.2-1b-lora-EG

set -e

PYTHON="/home/liang/miniconda3/envs/tuneEG/bin/python"
SCRIPT="$(cd "$(dirname "$0")"; pwd)/grpo_train.py"
GPU="${1:-0}"
shift 1 2>/dev/null || true  # remaining args forwarded to script

echo "========================================"
echo "  EG Entailment GRPO Training"
echo "  GPU : $GPU"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" \
    --output_dir "$(dirname "$SCRIPT")/saves/llama3.2-1b-grpo-EG" \
    "$@"
