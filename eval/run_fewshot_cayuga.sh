#!/bin/bash
# Re-run the FIXED few-shot output control (deep-review bug: query consumed all positives,
# so the "K=10 balanced" prompt was kf negatives + 0 positives). Submit: sbatch run_fewshot_cayuga.sh
# Read: if few-shot output stays ~0.45-0.55 vs activation probe 0.787, the probe advantage
# is NOT supervision and the expression gap is real.
#
# CAYUGA MODULE GOTCHA: /etc/profile.d/lmod.sh NOOPs under SLURM (SLURM_NODELIST set), so
# `module` is undefined in batch jobs. FIX: set MODULEPATH + source lmod init directly.
#SBATCH --job-name=bge-fewshot
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fewshot_%j.log

set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"
source venv/bin/activate

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"
export ACT_N="${ACT_N:-1250}"
export ACT_CSV="${ACT_CSV:-herg.csv}"
export ACT_FEWSHOT="${ACT_FEWSHOT:-10}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$ACT_MODEL  K=$ACT_FEWSHOT  n=$ACT_N  host=$(hostname)"
python -u fewshot_output.py
