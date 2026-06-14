#!/bin/bash
#SBATCH --job-name=ws3_lora_cells
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/ws3_lora_cells_%j.log
# WS3 TRAIN-placement empirical nail. LoRA-finetune Qwen3-8B on the two weak-cheap-specialist
# cells; check if finetuned OUTPUT beats the best non-train placement. Predicted strong-second
# (results/ws3_train_placement.md). hERG is already done (LoRA 0.856) so not re-run here.
#   variant_seq: bar = AlphaMissense 0.96 (novel 0.985), seq ceiling ~0.92  -> predicted LOSE
#   spectra_ms : bar = cheap binned-m/z 0.667, retrieve 0.586               -> predicted MARGINAL
set -e
REPO=${REPO:-$HOME/grounding-atlas}        # set REPO to your checkout
cd "$REPO"
PY=${PY:-python}                            # your env python: torch(+cuda), transformers, peft, sklearn
# (if a shared conda's ~/.local numpy shadows the env, also: export PYTHONNOUSERSITE=1)
echo "host $(hostname)"; nvidia-smi -L || true
VARIANT_PROMPT=$'Protein sequence window (variant at center): {rep}\nIs this variant pathogenic? Answer yes or no.\nAnswer:'
MS_PROMPT=$'Mass spectrum peaks (m/z): {rep}\nDoes this molecule block the hERG potassium channel (cardiotoxicity)? Answer yes or no.\nAnswer:'

echo "### variant_seq"
LORA_CELL=variant_seq LORA_PAIRS=signal/variant_seq/pairs.jsonl LORA_PROMPT="$VARIANT_PROMPT" \
  LORA_N=2000 LORA_EPOCHS=3 $PY eval/ws3_lora.py

echo "### spectra_ms"
LORA_CELL=spectra_ms LORA_PAIRS=signal/spectra_ms/pairs.jsonl LORA_PROMPT="$MS_PROMPT" \
  LORA_N=1250 LORA_EPOCHS=5 $PY eval/ws3_lora.py

echo "DONE -> results/ws3_lora.json (cell tags). TRAIN wins only if finetuned_output_auroc strictly"
echo "exceeds cheap AND retrieve AND specialist on the held-out split."
