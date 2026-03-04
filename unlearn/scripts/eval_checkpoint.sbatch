#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/eval-ckpt-%j.out

CHECKPOINT_PATH="${1:?Usage: sbatch eval_checkpoint.sbatch CHECKPOINT_PATH OUTPUT_PATH [TASKS]}"
OUTPUT_PATH="${2:?Usage: sbatch eval_checkpoint.sbatch CHECKPOINT_PATH OUTPUT_PATH [TASKS]}"
TASKS="${3:-wmdp_bio_robust}"

echo "============================================"
echo "Eval checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Tasks: $TASKS"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray
module load cuda/12.6

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn

python -m unlearn.scripts.eval_checkpoint \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --output_path="$OUTPUT_PATH" \
    --tasks="$TASKS" \
    --num_gpus=4

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
