#!/bin/bash
#SBATCH --job-name=tamper-random-init
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/tamper-random-init-%j.out

echo "============================================"
echo "Tamper attack on randomly initialized model"
echo "Architecture: deep-ignorance-unfiltered (GPTNeoX 7B)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Started: $(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="0"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

WORK_DIR=/home/a6a/lucia.a6a/unlearn
cd $WORK_DIR

REFERENCE_MODEL=models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0
RANDOM_MODEL=models/EleutherAI/deep-ignorance-unfiltered_random_init
OUTPUT_DIR=runs/tamper_random_init

echo "Creating randomly initialized model..."
python3 -c "
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained('${REFERENCE_MODEL}')
model = AutoModelForCausalLM.from_config(config)

save_path = Path('${RANDOM_MODEL}')
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_path)

ref = Path('${REFERENCE_MODEL}')
for f in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
    src = ref / f
    if src.exists():
        shutil.copy2(src, save_path / f)

print(f'Random init model saved to {save_path}')
"

if [ ! -d "$RANDOM_MODEL" ]; then
    echo "ERROR: Random model not created at $RANDOM_MODEL"
    exit 1
fi

echo "Running tamper attack on random init model..."
echo "python -m unlearn.scripts.run_tamper_attack_with_plot --model_name=$RANDOM_MODEL --output_dir=$OUTPUT_DIR --num_train_examples=512 --epochs=1 --eval_every=10 --lr=2e-5 --title='Tamper Attack: Randomly Initialized Model'"

python -m unlearn.scripts.run_tamper_attack_with_plot \
    --model_name=$RANDOM_MODEL \
    --output_dir=$OUTPUT_DIR \
    --num_train_examples=512 \
    --epochs=1 \
    --eval_every=10 \
    --lr=2e-5 \
    --title="Tamper Attack: Randomly Initialized Model"

PLOT_FILE=$(ls -t ${OUTPUT_DIR}/tamper_results_*.png 2>/dev/null | head -1)
if [ -n "$PLOT_FILE" ]; then
    cp "$PLOT_FILE" experiment_logs/tampering_random_init.png
    echo "Plot copied to experiment_logs/tampering_random_init.png"
else
    echo "WARNING: No plot file found in $OUTPUT_DIR"
fi

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
