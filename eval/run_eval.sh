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
# Metrics reported:
#   accuracy       – classification accuracy on parsed responses
#   auc            – ROC-AUC using P(True) from softmax over {True, False} tokens
#   auc_norm       – ROC-AUC using S_ent score (full-vocab probability formula):
#                      S_ent = 0.5 + 0.5·S_tok  if first token = "True"
#                      S_ent = 0.5 − 0.5·S_tok  if first token = "False"
#                      S_ent = 0.5               if parse failure
#   f1             – F1 score (binary)
#   precision      – Precision
#   recall         – Recall
#   parse_fail_rate– Fraction of responses that are neither "True" nor "False"
#
# Flags:
#   --negation     Also evaluate on negated premise/hypothesis; adds
#                  negation_flip_rate and negation_consistent_rate columns
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
