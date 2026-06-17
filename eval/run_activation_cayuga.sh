#!/bin/bash
# WS1 axis-B activation arm on Cayuga (SLURM). Submit: sbatch run_activation_cayuga.sh
# Override model: ACT_MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch run_activation_cayuga.sh
#SBATCH --job-name=bge-activation
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=activation_%j.log

set -euo pipefail
module load anaconda3/2023.09-3 cuda/12.1
source venv/bin/activate

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # open-weight default
export ACT_N="${ACT_N:-2000}"
export ACT_CSV="${ACT_CSV:-herg.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$ACT_MODEL  n=$ACT_N  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u "${ACT_SCRIPT:-activation_arm.py}"

# Read: best-layer activation AUROC ~0.8+ -> EXPRESSION gap (signal inside, not surfaced);
#       ~0.57 -> ENCODING gap. Compare SFM-probe 0.91, LLM-output ~0.57.
