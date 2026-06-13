# Best-layer selection bias: the encoding claim survives an unbiased layer choice (measured)

*Reviewer-proofing note for the activation arm. 2026-06-09. Companion to `head_to_head.md`, `confound_controls.md`, and `../docs/WS1_BACKLOG.md` (item E). No em dashes.*

## The objection

The activation arm reports `max-over-layers` AUROC (the best of 37 per-layer linear probes) and the code flags it "selection-biased". A reviewer is right to ask: is the encoding signal an artifact of picking the single luckiest layer on the same data it is scored on? An unbiased estimate selects the layer WITHOUT seeing the test labels.

## The fix: a nested-CV held-out-layer protocol, now RUN

`eval/activation_arm.py:heldout_layer_auroc` (wired into all three activation scripts): nested GroupKFold. The inner folds pick the best layer using train rows only; the held-out outer fold is scored at that layer. The mean over outer folds is the unbiased number, and `max - held-out` IS the selection bias. This was run on a direct GPU pass (Expanse H100, Qwen3-8B, same n and leakage-controlled splits as the anchors; raw logs in `results/expanse_logs/`).

## Measured result (max vs nested-CV held-out, one run)

| modality | specialist ceiling | activation `max`@L | **activation held-out (nested CV)** | **selection bias** | output | enc gap (held-out) | exp gap (held-out) |
|---|---|---|---|---|---|---|---|
| SMILES / hERG | 0.837 (Morgan FP) | 0.786 @ L2 | **0.779** | **+0.007** | 0.454 | 0.058 | **+0.325** |
| protein / Tm | 0.680 (ESM2) | 0.605 @ L1 | **0.571** | **+0.034** | 0.486 | 0.109 | **+0.085** |
| variant / text | 0.962 (AlphaMissense) | 0.814 @ L25 | **0.810** | **+0.003** | 0.593 | 0.152 | **+0.217** |
| variant / seq | 0.962 (AlphaMissense) | 0.743 @ L28 | **0.731** | **+0.012** | 0.494 | 0.231 | **+0.237** |

(Held-out layers picked per outer fold: SMILES [22,26,2,2,2]; protein [1,36,24,36,1]; variant-text [25,25,32,34,25]; variant-seq [28,21,28,29,28] - the early-layer protein pick is less stable than the broad variant/SMILES plateau, which is exactly why protein carries the largest bias. This is a fresh H100 run; the max-over-layers values reproduce the original anchors within re-run variance, e.g. SMILES 0.786 vs 0.787, protein 0.605 vs 0.609, variant-seq 0.743 vs 0.740.)

## Reads

- **The bias is small, and it is largest exactly where predicted.** SMILES +0.007, variant-text +0.003, variant-seq +0.012 (broad high plateaus, nested-CV lands in the band). Protein is the outlier at **+0.034**, because its informative signal is an early-layer spike (L1) over a flat 0.55-0.59 tail rather than a plateau, so the held-out pick is less stable. This matches the prior bound analysis, which flagged protein as the one rung where max could overstate the arm by up to ~0.03.
- **The expression gap is immune to layer selection.** Output does not depend on a layer, so the held-out correction leaves every expression gap large and positive: SMILES +0.325, variant-seq +0.237, variant-text +0.217, protein +0.085. The headline (a general LLM encodes the property well above its output yet verbalizes near chance) does not move.
- **Every held-out correction reinforces the rung's regime label, none flips it.** SMILES stays expression-dominant (enc gap 0.058 vs exp gap 0.325). Protein stays encoding-weak and gets MORE so: its held-out encoding gap is 0.109, now clearly larger than its 0.085 expression gap, so protein's deficit is dominated by encoding, not expression (the opposite of SMILES). Variant stays mixed (text and seq both carry an encoding and an expression gap). The regime spectrum that carries the cross-modality story is unchanged.
- **The layer-formation pattern is a robust, selection-free observation.** Chance at L0, a jump by L1-L2 (SMILES, protein) or a late monotone climb to a plateau (variant): the signal forms and persists, it does not appear only at one cherry-picked depth.

## Bottom line

Measured selection bias on the activation arm is +0.003 to +0.034 (largest for protein, as predicted), and the one effect the project rests on, the expression gap, is layer-selection-independent by construction and stays large under the held-out estimate. No regime label flips. The protein selectivity control that was the last open methodology item is now also run (activation selectivity +0.114, structure-probe +0.193; see `confound_controls.md`), so every rung now has both probe defenses (selectivity and an unbiased layer choice).
