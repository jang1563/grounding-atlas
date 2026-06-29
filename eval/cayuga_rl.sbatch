#!/bin/bash
# Experiment-3 (RL_ENV_PREREG) Cayuga a40 launcher. GPU compute runs HERE, not locally.
# Stage to ~/bge (structure-preserving) then submit, e.g.:
#   sbatch --export=ALL,E3_SCRIPT=smiles_generator_init.py eval/cayuga_rl.sbatch
#   sbatch --export=ALL,E3_SCRIPT=rl_guidance.py,RL_ENDPOINT=herg eval/cayuga_rl.sbatch
#SBATCH -p scu-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 1:30:00
#SBATCH -J e3rl
#SBATCH -o logs/e3_%x_%j.out
#SBATCH -e logs/e3_%x_%j.out
set -e
cd "$HOME/bge"
export PYTHONNOUSERSITE=1                       # bioguard ~/.local numpy2-vs-sklearn break guard
export HF_HOME="$HOME/.cache/huggingface"       # ChemBERTa-77M-MTR is cached here
mkdir -p logs signal/reward
echo "host=$(hostname) script=${E3_SCRIPT} gpu=${CUDA_VISIBLE_DEVICES} $(date)"
"$HOME/bge/venv/bin/python" -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
"$HOME/bge/venv/bin/python" -u "eval/${E3_SCRIPT}"
echo "done $(date)"
