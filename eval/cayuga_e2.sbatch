#!/bin/bash
# Experiment-2 (3-way calibrated LLM<->SFM bridge) GPU arms on Cayuga a40. Runs eval/bridge_arm.py
# (Arm A) or eval/ws3_lora.py (Arm C) on Qwen3-8B; Arm B (orchestrate) is CPU and already done.
# Staged STRUCTURE-PRESERVING under ~/bge (so ROOT=~/bge -> signal/ + results/ resolve), reusing the
# ~/bge/venv (torch+transformers+peft+sklearn) + the Qwen3-8B cache in ~/bge/hf_cache.
# Submit (E2_SCRIPT picks the arm; the arm's own env vars drive it):
#   E2_SCRIPT=bridge_arm.py BRIDGE_MODEL=Qwen/Qwen3-8B BRIDGE_ENDPOINT=herg BRIDGE_MODE=within sbatch cayuga_e2.sbatch
#   E2_SCRIPT=ws3_lora.py LORA_MODEL=Qwen/Qwen3-8B LORA_FOLD=lpo_herg_clearance LORA_ENDPOINT=herg LORA_MODE=within LORA_N=4400 sbatch cayuga_e2.sbatch
#SBATCH --job-name=bge-e2
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=e2_%j.log
set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles            # lmod NOOPs under SLURM; init manually
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"                                          # ROOT=~/bge for the eval/* scripts
source venv/bin/activate
export PYTHONNOUSERSITE=1                               # guard the ~/.local numpy2-vs-sklearn break
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"
echo "E2_SCRIPT=${E2_SCRIPT}  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u "eval/${E2_SCRIPT}"
