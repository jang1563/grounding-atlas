#!/bin/bash
# Variant-branch ESM-1v secondary ceiling on Cayuga (SLURM). Submit: sbatch run_ceiling_esm1v_cayuga.sh
# a100 node (g0001) is DRAIN -> a40 (scu-gpu, qos=normal). h100 alt below.
#SBATCH --job-name=vge-esm1v
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=vge_esm1v_%j.log
# h100 alternative (preempt_gpu, low qos, preemptible):
#   #SBATCH --partition=preempt_gpu
#   #SBATCH --gres=gpu:h100:1
#   #SBATCH --qos=low

set -euo pipefail
# self-contained venv (torch / transformers / sklearn; no `kernels` pkg). torch bundles CUDA, so
# no `module load` (non-interactive sbatch does not inherit Lmod and sourcing lmod.sh breaks under set -u).
source "$HOME/bge/venv/bin/activate"

export VG_CSV="${VG_CSV:-../data/variant_clinvar_full.csv}"   # full set for star/temporal strata power
export ESM1V_MODEL="${ESM1V_MODEL:-facebook/esm1v_t33_650M_UR90S_1}"
export VG_RAW="${VG_RAW:-../data/raw}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "esm1v=$ESM1V_MODEL  csv=$VG_CSV  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u ceiling_esm1v_variant.py

# Read: ESM-1v (fully unsupervised, no MSA) should separate ClinVar P/B and HOLD on the
# post_2026_01 temporal holdout. That flat specialist AUROC across the holdout is the baseline
# against which the output arm's collapse on the same novel variants is measured.
