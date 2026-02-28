#!/bin/bash
# Evaluate models on LevyHolt test set
#
# Model choices:
#   standard  – Llama-3.2-1B-Instruct, no fine-tuning
#   sft       – SFT LoRA checkpoint
#   grpo      – GRPO LoRA checkpoint
#   distill   – Distillation LoRA checkpoint
#   all       – all four above
#
# Usage:
#   bash run_eval.sh <GPU> <MODEL> [<MODEL> ...] [--negation]
#
# Examples:
#   bash run_eval.sh 0 standard
#   bash run_eval.sh 0 standard sft
#   bash run_eval.sh 0 all
#   bash run_eval.sh 0 all --negation

set -e

PYTHON="/home/liang/miniconda3/envs/tuneEG/bin/python"
SCRIPT="$(cd "$(dirname "$0")"; pwd)/evaluate.py"
GPU="${1:-0}"
shift 1 2>/dev/null || true   # remaining args forwarded to script

echo "========================================"
echo "  LevyHolt Evaluation"
echo "  GPU    : $GPU"
echo "  Options: $*"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" \
    --output_dir "$(dirname "$SCRIPT")/results" \
    --models "$@"
