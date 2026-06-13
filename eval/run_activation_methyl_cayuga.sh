#!/bin/bash
#SBATCH --job-name=bge-act-methyl
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=act_methyl_%j.log
set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"; source venv/bin/activate
export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}" ACT_N="${ACT_N:-600}" ACT_K="${ACT_K:-100}" ACT_CSV="${ACT_CSV:-$HOME/bge/methyl_age.csv}" HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"
echo "csv=$ACT_CSV host=$(hostname)"
python -u activation_arm_methyl.py
