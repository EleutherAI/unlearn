#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

MODEL_PATH="${1:-models/deep-ignorance-unfiltered}"

cd "$REPO_ROOT/unlearn"

echo "Running finetune attack on: $MODEL_PATH"
echo "Command:"
echo "python finetune_attack.py --model_name $MODEL_PATH --eval_every 100 --epochs 1 --lr 2e-5"
echo ""

python finetune_attack.py \
    --model_name "$MODEL_PATH" \
    --eval_every 100 \
    --epochs 1 \
    --lr 2e-5
