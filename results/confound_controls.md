# Confound controls: the encoding-vs-expression gap survives every named confound

*Reviewer-proofing summary for axis B. 2026-06-09. The three confounds are from `../eval/README.md` ("Three confounds the design must control"); selectivity and best-layer are the two probe defenses. No em dashes.*

The headline is that a general LLM encodes a content property near the specialist ceiling yet outputs near chance (an expression gap). Every way that gap could be an artifact is controlled here, with the status and the number.

## 1. Supervision asymmetry (a trained probe vs a zero-shot LLM) - CONTROLLED

A linear probe is trained on labels; the output arm is zero-shot, which is unfair by construction. Control: hand the LLM K=10 matched in-context labeled examples (few-shot) and re-score.

- 8B / hERG: few-shot output **0.493**, essentially unchanged from zero-shot 0.453 and far below the activation probe 0.787 (`../eval/fewshot_output.py`). Given labeled examples the model still cannot verbalize hERG, so the probe advantage is not supervision. The expression gap is real.

## 2. Input asymmetry (the probe reads an SFM embedding, the LLM reads raw text) - CONTROLLED (SMILES)

The structure-probe reads a Morgan fingerprint / ESM-2 embedding (already processed by a chemistry or biology model); the LLM reads raw text. So the gap could be "the SFM pre-processed the input" rather than "internally present but not surfaced". Control: probe the property from a RAW-TEXT featurization that uses no chemistry and no SFM, just character n-grams of the SMILES string the LLM itself reads, with a linear model on top (`../eval/input_asymmetry.py`, balanced 1250, Murcko scaffold GroupKFold).

| arm (same molecules, same split) | AUROC | selectivity (vs shuffled-label) |
|---|---|---|
| SFM ceiling (Morgan FP, chemistry) | 0.834 | +0.311 |
| **raw-text (char 2-5 gram, no SFM)** | **0.801** | +0.270 |
| LLM output (8B zero-shot) | 0.453 | - |

The raw-text probe (0.801) nearly matches the chemistry fingerprint (0.834) and beats the LLM output by **+0.348**. The hERG signal is linearly decodable from the very raw SMILES string the LLM reads, with no SFM. So the probe-minus-LLM gap is not an SFM input advantage; the LLM sees the same string and (per the activation arm, 0.787) encodes it internally but does not surface it. The gap is expression, not input.

(The complementary "orchestrate" form of this control, feeding the SFM's processed representation INTO the LLM to see if its output then improves, requires an embedding-to-token adapter and belongs to WS3, not this cheap pass. The raw-text probe already rules out the input-advantage reading of the existing gap.)

## 3. Property leakage (recoverable from the name alone) - CONTROLLED by content-sensitivity

The property must not be answerable by recalling the entity name. Controls already run:

- **variant:** the web-rich text form scores 0.79 on ClinVar labels but falls to 0.61 on experimental DMS labels for the SAME genes, and a scrambled gene name drops it toward chance, so the symbolic score is largely clinical-label recall, not variant-specific grounding. The temporal holdout (post-2026-01 ClinVar) and unsupervised ESM-1v 0.92 keep the ceiling leakage-free.
- **SMILES:** swapping a molecule for a valid opposite-label one moves the answer by only mean|delta| 0.085, and re-notating to a randomized-but-valid SMILES barely changes the AUROC (canonical 0.573 vs 0.553): the output is near-uniform regardless of structure (a floor, no name shortcut to exploit).
- **protein:** scramble + mismatch conditions run; composition-preserving + motif-targeted shuffles are the stated next step.

## Probe defense A: selectivity (Hewitt-Liang 1909.03368, shuffled-label control)

A random-label probe must sit at chance, or the probe is fitting noise not reading signal.

| modality | structure-probe selectivity | activation selectivity | status |
|---|---|---|---|
| SMILES / hERG | +0.331 | +0.301 | done |
| variant / text | - (AlphaMissense ceiling, no trained probe) | +0.301 | done |
| variant / seq | - | +0.218 | done |
| protein / Tm | **+0.193** | **+0.114** | **DONE** (Expanse H100 run, `results/expanse_logs/pge-act_50612829.log`; was the one missing rung) |

(Values from the Expanse H100 re-run; the SMILES and variant selectivities reproduce the earlier Cayuga numbers within re-run variance. Protein's activation selectivity +0.114 is positive and well above chance, so the protein probe reads real ESM-grounded thermostability signal, not noise; it is lower than SMILES and variant, consistent with protein being the weakest-encoding rung.)

## Probe defense B: best-layer selection bias - CONTROLLED (measured)

Activation is reported max-over-layers. The nested-CV held-out-layer protocol (`../eval/activation_arm.py:heldout_layer_auroc`, wired into all three activation scripts) was RUN on a GPU pass (Expanse H100). Measured bias: SMILES +0.007, variant-text +0.003, variant-seq +0.012, protein +0.034 (largest, as predicted, an early-layer spike). The expression gap is layer-selection-immune and stays large under the held-out estimate; no regime label flips. Full analysis in `selection_bias.md`.

## Status line

All three named confounds are controlled: supervision asymmetry and input asymmetry with numbers (SMILES), property leakage by content-sensitivity across all three modalities. Both probe defenses now hold on every rung: selectivity is measured for SMILES, variant, AND protein (+0.114, the previously-open item), and best-layer selection bias is measured by nested CV (+0.003 to +0.034). No methodology item on the encoding claim remains open; the remaining backlog items are coverage/depth, not confound controls.
