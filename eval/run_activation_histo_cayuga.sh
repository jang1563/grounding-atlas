#!/bin/bash
#SBATCH --job-name=bge-act-histo
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=act_histo_%j.log
set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"; source venv/bin/activate
export VL_MODEL="${VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" VL_N="${VL_N:-400}" VL_CSV="${VL_CSV:-$HOME/bge/histo/pcam.csv}" HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"
echo "csv=$VL_CSV host=$(hostname)"
python -u activation_arm_histo.py
