#!/bin/bash
#SBATCH --job-name=lens-sft
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/lens-sft-%j.out

# Get repository root
# Use SLURM_SUBMIT_DIR if available, otherwise fall back to script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Arguments: 
NUM_TRAIN_EXAMPLES=${1:-1024}

echo "============================================"
echo "Lens Unlearning - Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "num_train_examples: $NUM_TRAIN_EXAMPLES"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

echo "===== CUDA check ====="
nvidia-smi | head -20

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available in PyTorch. Exiting."
    exit 1
fi

LENS_DIR="$REPO_ROOT/runs/lens_cache/deep-ignorance-unfiltered-lens"

if [ ! -f "$LENS_DIR/params.pt" ]; then
    echo "ERROR: Lens not found at $LENS_DIR. Run lens download first."
    exit 1
fi

NUM_GPUS=4

# Model checkpoint to evaluate
MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_lens_ex1024_sft_sft"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_PATH"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
echo ""

# Run Eval
accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks wmdp_bio_robust \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --batch_size auto


accelerate launch --num_processes 4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks mmlu \
    --include_path "$REPO_ROOT/unlearn/lm_eval_tasks" \
    --batch_size auto

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
