#!/bin/bash
#SBATCH --job-name=sfm_act_8b
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/sfm_act_8b_%j.log
# SFM-embedding rung, 8B ACTIVATION arm. Settles encode-vs-not for the ESM-2-embedding-as-text:
# the local 0.5B proxy was INCONCLUSIVE (best-layer 0.71, held-out-layer 0.599, bias +0.11).
# Reads the staged 150M embeddings (signal/sfm_embedding/meltome_esm2.npz) and probes Qwen3-8B
# hidden states with the same nested held-out-layer CV. Compare to ceiling 0.83, output 0.47-0.56.
set -e
REPO=${REPO:-$HOME/grounding-atlas}        # set REPO to your checkout
cd "$REPO"
PY=${PY:-python}                            # your env python: torch(+cuda), transformers, sklearn
# (if a shared conda's ~/.local numpy shadows the env, also: export PYTHONNOUSERSITE=1)
echo "host $(hostname)"; nvidia-smi -L || true
[ -f signal/sfm_embedding/meltome_esm2.npz ] || { echo "ERROR: staged npz missing"; exit 1; }
ACT_MODEL=Qwen/Qwen3-8B SFM_BATCH=8 $PY eval/sfm_embedding_activation.py
echo "DONE -> results/sfm_embedding_activation_Qwen3-8B.json"
echo "held-out AUROC selective and near 0.83 = EXPRESSION gap (8B encodes it); ~0.6 = encoding-limited."
