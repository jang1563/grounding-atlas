#!/bin/bash
#SBATCH --job-name=bge-act-3d
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=act_3d_%j.log
set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"; source venv/bin/activate
export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}" ACT_N="${ACT_N:-400}" ACT_CSV="${ACT_CSV:-herg_xyz.csv}" HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"
echo "model=$ACT_MODEL csv=$ACT_CSV n=$ACT_N host=$(hostname)"
python -u activation_arm_3d.py
