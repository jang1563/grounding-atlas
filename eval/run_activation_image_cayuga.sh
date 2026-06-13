#!/bin/bash
# Image rung B: open-VLM (Qwen2.5-VL-7B) hidden-state 3-arm on rendered hERG molecules.
# Direct activation measurement (the OCSR proxy could only approximate it).
# Submit: sbatch run_activation_image_cayuga.sh
# CAYUGA MODULE GOTCHA: lmod.sh NOOPs under SLURM; init lmod manually (see memory).
#SBATCH --job-name=bge-act-img
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=act_img_%j.log

set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"
source venv/bin/activate
pip install qwen-vl-utils --quiet 2>&1 | tail -1 || true   # processor helper, harmless if present

export VL_MODEL="${VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export VL_N="${VL_N:-400}"
export VL_CSV="${VL_CSV:-herg.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$VL_MODEL  n=$VL_N  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u activation_arm_image.py
