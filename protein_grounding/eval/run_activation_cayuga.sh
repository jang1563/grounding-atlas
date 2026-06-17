#!/bin/bash
# Protein-branch axis-B activation arm on Cayuga (SLURM). Submit: sbatch run_activation_cayuga.sh
# Override model: ACT_MODEL=Qwen/Qwen3-32B sbatch run_activation_cayuga.sh
# a100 node (g0001) is DRAIN -> use a40 (scu-gpu, qos=normal). h100 alt below.
#SBATCH --job-name=pge-activation
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=pge_activation_%j.log
# h100 alternative (preempt_gpu, low qos, preemptible):
#   #SBATCH --partition=preempt_gpu
#   #SBATCH --gres=gpu:h100:1
#   #SBATCH --qos=low

set -euo pipefail
# self-contained venv (torch 2.12 / transformers 5.10 / sklearn 1.9; no `kernels` pkg).
# torch bundles CUDA 13.0, so it needs only the GPU node's NVIDIA driver: no `module load`
# (non-interactive sbatch does not inherit Lmod's `module` function, and sourcing lmod.sh
# breaks under `set -u`).
source "$HOME/bge/venv/bin/activate"

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"          # open-weight default
export ESM_MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
export ACT_N="${ACT_N:-2000}"                           # caps to the balanced set size
export ACT_CSV="${ACT_CSV:-protein_meltome.csv}"
export ACT_THINK="${ACT_THINK:-0}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "act_model=$ACT_MODEL  esm=$ESM_MODEL  n=$ACT_N  think=$ACT_THINK  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u activation_arm_protein.py

# Read: ESM2 ceiling high + activation ~ceiling -> EXPRESSION gap (signal inside, not surfaced);
#       activation well below ceiling -> ENCODING gap (the hypothesised protein case).
#       Compare the SMILES branch: probe 0.825 / act 0.79 / out 0.45 (expression-dominant).
