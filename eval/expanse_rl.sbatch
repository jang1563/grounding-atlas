#!/bin/bash
# Experiment-3 (RL_ENV_PREREG) SDSC Expanse launcher (alt to Cayuga when scu-gpu is contended).
# Stage to ~/bge then submit, e.g.:
#   sbatch --export=ALL,E3_SCRIPT=smiles_generator_init.py,GEN_ENDPOINT=clearance eval/expanse_rl.sbatch
#SBATCH -p nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH -A crl195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J e3rl
#SBATCH -o logs/e3_%x_%j.out
#SBATCH -e logs/e3_%x_%j.out
set -e
cd "$HOME/bge"
export PYTHONNOUSERSITE=1                        # ignore ~/.local (numpy/sklearn hygiene)
export PYTHONUTF8=1
export HF_HOME="$HOME/.cache/huggingface"        # ChemBERTa-77M-MTR pre-cached here
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p logs signal/reward
PY="$HOME/.conda/envs/bioguard/bin/python"
echo "host=$(hostname) script=${E3_SCRIPT} $(date)"
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
$PY -u "eval/${E3_SCRIPT}"
echo "done $(date)"
