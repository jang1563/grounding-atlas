#!/bin/bash
# DNA/RNA modality rung: 3-arm instrument on the promoter content-property (Qwen3-8B).
# Ceiling = 6-mer LR (gated at 0.898). Submit: sbatch run_activation_dna_cayuga.sh
# Read SUMMARY: activation near the 0.90 ceiling + output at chance = expression-dominant
# (like SMILES); activation near chance = encoding gap (the model never forms promoter-ness
# from the raw sequence, a web-poor modality).
#
# CAYUGA MODULE GOTCHA: lmod.sh NOOPs under SLURM; init lmod manually (see memory).
#SBATCH --job-name=bge-act-dna
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=act_dna_%j.log

set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"
source venv/bin/activate

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"
export ACT_N="${ACT_N:-1500}"
export ACT_CSV="${ACT_CSV:-dna_promoter.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$ACT_MODEL  csv=$ACT_CSV  n=$ACT_N  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u activation_arm_dna.py
