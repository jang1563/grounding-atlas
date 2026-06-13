#!/bin/bash
# Variant-branch OPTIONAL activation arm on Cayuga (SLURM). Submit: sbatch run_activation_cayuga.sh
# Override model: ACT_MODEL=Qwen/Qwen3-32B sbatch run_activation_cayuga.sh
# a100 node (g0001) is DRAIN -> a40 (scu-gpu, qos=normal). h100 alt below.
#SBATCH --job-name=vge-activation
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=vge_activation_%j.log
# h100 alternative (preempt_gpu, low qos, preemptible):
#   #SBATCH --partition=preempt_gpu
#   #SBATCH --gres=gpu:h100:1
#   #SBATCH --qos=low

set -euo pipefail
source "$HOME/bge/venv/bin/activate"

export VG_CSV="${VG_CSV:-../data/variant_clinvar.csv}"
export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"          # open-weight, no HF token
export VG_N="${VG_N:-1500}"
export VG_FORMS="${VG_FORMS:-text,seq}"
export VG_CONTROL="${VG_CONTROL:-1}"                    # random-label selectivity control
export ACT_THINK="${ACT_THINK:-0}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "act_model=$ACT_MODEL  csv=$VG_CSV  n=$VG_N  forms=$VG_FORMS  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u activation_arm_variant.py

# Read: per surface form, activation near the ceiling + output at chance -> EXPRESSION gap (the
# signal is encoded, not surfaced); activation well below ceiling -> ENCODING gap. The text-vs-seq
# contrast on activation is the encoding-side web-exposure test (gene GroupKFold removes the
# trivial gene-prior shortcut). Compare the SMILES branch: probe 0.825 / act 0.79 / out 0.45.
