#!/bin/bash
# WS3 #4 arm 5: activation/structure/output 3-arm on the WITHDRAWAL endpoint (Qwen3-8B).
# Consistency check: on a fingerprint-weak endpoint the ACTIVATION probe should ALSO be weak
# (~0.6, near the Morgan probe), i.e. there is little structural signal to encode, unlike
# hERG. The withdrawal signal is recalled from the NAME (frontier arm, withdrawn_endpoint.py),
# not encoded from the SMILES. Submit: sbatch run_activation_withdrawn_cayuga.sh
#
# CAYUGA MODULE GOTCHA: lmod.sh NOOPs under SLURM, so init lmod manually (see memory).
#SBATCH --job-name=bge-act-wd
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=act_wd_%j.log

set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"
source venv/bin/activate

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"
export ACT_N="${ACT_N:-2000}"
export ACT_CSV="${ACT_CSV:-withdrawn.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$ACT_MODEL  csv=$ACT_CSV  n=$ACT_N  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u activation_arm.py
