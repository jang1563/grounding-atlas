#!/bin/bash
# Decisive surface-vs-chemistry test: activation probe on CANONICAL vs RANDOMIZED SMILES.
# Runs BOTH arms in one job (same node, model, sample, scaffold split) so the only
# difference is the input notation. Submit: sbatch run_activation_randomize_cayuga.sh
#
# Read the two best-layer ACTIVATION AUROCs:
#   randomized ~ canonical  -> the LLM hidden-state signal SURVIVES re-notation, like the
#                              char-n-gram (0.81 on randomized, lipophilicity_control.md);
#                              structural but still surface-decodable (does NOT beat a
#                              substring probe). Keep "encodes chemistry" softened.
#   randomized << canonical -> the canonical run was reading canonical-string orthography.
#
# CAYUGA MODULE GOTCHA: /etc/profile.d/lmod.sh NOOPs when SLURM_NODELIST is set, so inside
# a batch job `module` is undefined (and `#!/bin/bash -l` does not help, shebang args are
# ignored). FIX: set MODULEPATH and source the lmod init directly, bypassing the guard.
#SBATCH --job-name=bge-act-rand
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=act_rand_%j.log

set -eo pipefail
export MODULEPATH=/opt/ohpc/pub/modulefiles
source /opt/ohpc/admin/lmod/lmod/init/bash
module load anaconda3/2023.09-3 cuda/12.1
cd "$HOME/bge"
source venv/bin/activate

export ACT_MODEL="${ACT_MODEL:-Qwen/Qwen3-8B}"   # match the 0.787 baseline arm
export ACT_N="${ACT_N:-2000}"
export ACT_CSV="${ACT_CSV:-herg.csv}"
export HF_HOME="${HF_HOME:-$HOME/bge/hf_cache}"

echo "model=$ACT_MODEL  n=$ACT_N  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "==================== ARM 1: CANONICAL (baseline) ===================="
ACT_RANDOMIZE=0 python -u activation_arm.py

echo "==================== ARM 2: RANDOMIZED (re-notated) ===================="
ACT_RANDOMIZE=1 python -u activation_arm.py
