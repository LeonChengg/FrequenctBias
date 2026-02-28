#!/bin/bash
# Fine-tune Llama-3.2-1B-Instruct on EG entailment dataset with LoRA
# Usage: bash run_train.sh [GPU_IDs]
# Example: bash run_train.sh 0        (single GPU)
#          bash run_train.sh 0,1      (two GPUs)

set -e

LLAMA_FACTORY_DIR="/mnt/nvme0n1/luzhan/finetune-qwen/LLaMA-Factory"
PYTHON="/home/liang/miniconda3/envs/tuneEG/bin/python"
TORCHRUN="/home/liang/miniconda3/envs/tuneEG/bin/torchrun"
CONFIG="$(cd "$(dirname "$0")"; pwd)/train_lora.yaml"
GPUS="${1:-0}"

echo "========================================"
echo "  EG Entailment Fine-tuning (LoRA SFT)"
echo "  Model : Llama-3.2-1B-Instruct"
echo "  Config: $CONFIG"
echo "  GPUs  : $GPUS"
echo "========================================"

export PYTHONPATH="$LLAMA_FACTORY_DIR/src:$PYTHONPATH"

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" "$LLAMA_FACTORY_DIR/src/train.py" "$CONFIG"
else
    # Multi-GPU via torchrun
    CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        "$LLAMA_FACTORY_DIR/src/train.py" "$CONFIG"
fi
