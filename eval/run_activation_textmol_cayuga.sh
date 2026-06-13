#!/bin/bash
#SBATCH --job-name=bge-act-tm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=act_tm_%j.log
set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"; source venv/bin/activate
export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}" ACT_N="${ACT_N:-600}" HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"
echo "field=$ACT_TEXT_FIELD csv=$ACT_CSV host=$(hostname)"
python -u activation_arm_textmol.py
