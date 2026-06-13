#!/bin/bash
# Protein-branch ceiling gate on Cayuga (SLURM). Submit: sbatch run_ceiling_cayuga.sh
# ESM2 embedding needs a GPU; the sklearn probes are CPU. Light job.
#SBATCH --job-name=pge-ceiling
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pge_ceiling_%j.log

set -euo pipefail
# self-contained venv: torch 2.12 bundles CUDA 13.0, so it needs only the GPU node's NVIDIA
# driver (no `module load` needed). Non-interactive sbatch does not inherit Lmod's `module`
# function, and sourcing lmod.sh breaks under `set -u`, so module is dropped entirely.
source "$HOME/bge/venv/bin/activate"

export ESM_MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
export PG_CSV="${PG_CSV:-protein_meltome.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "esm=$ESM_MODEL  csv=$PG_CSV  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u ceiling_gate_protein.py

# Read: random ~ cluster AUROC and both high -> signal is genuine sequence content (PASS).
#       Large random->cluster drop -> homolog leakage (the DTI trap), not a clean ceiling.
