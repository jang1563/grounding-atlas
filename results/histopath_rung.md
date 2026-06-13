# Histopathology rung: the largest expression gap, in a vision model

*Results. 2026-06-12. `eval/activation_arm_histo.py`, data `signal/histo/pcam.csv` (PatchCamelyon, 400 balanced 96x96 H&E patches, tumor vs normal). Open-VLM 3-arm: Qwen2.5-VL-7B. ceiling = cheap color-feature classifier (per-channel statistics), with the real pathology FM CONCH ~0.9 as reference; activation = VLM hidden-state probe on the patch; output = VLM verbalized P(tumor). No em dashes.*

## Result

| arm | AUROC |
|---|---|
| ceiling (cheap color-feature classifier) | 0.746 |
| ceiling (pathology FM CONCH, reference) | ~0.90 |
| ACTIVATION (VLM hidden-state probe) | **0.827** |
| OUTPUT (VLM verbalized P(tumor)) | **0.463** |

- encoding gap (vs CONCH ~0.9): ~0.07 (small)
- expression gap (activation - output): **0.364, the largest measured across all modalities**

## What it shows

A general vision-language model SEES tumor in its features and WILL NOT say it. The hidden-state probe separates tumor from normal tissue at 0.827, above the cheap color ceiling (0.746) and approaching the dedicated pathology foundation model (CONCH ~0.9). But asked directly "estimate the probability this patch contains tumor", the same model answers at chance (0.463, in fact slightly anti-correlated). The information is fully present in the activations; it does not reach the output.

This is the molecular-image result (VLM encodes structure-derived properties it cannot verbalize) but stronger and cleaner, for three reasons:
1. The activation EXCEEDS the cheap specialist ceiling, so the encoded signal is not a trivial color shortcut; it is rich morphological feature content (the same kind CONCH uses).
2. Tumor tissue is GROSS-VISIBLE (nuclear density, architecture), unlike a molecule's hERG-blockade which is not visible in a 2D depiction, so the encoding gap is small here, not the bottleneck.
3. The property is web-DOCUMENTED (pathology is described in text at length), so by the two-factor law the model should ground it. It encodes it. It still does not verbalize it.

So the bottleneck is purely expression, and it is the most severe instance in the project: a web-documented, gross-visible, richly-encoded property that the model refuses to or cannot put in its output. The likely cause is that a diagnostic call ("this is cancer") is outside a general VLM's output distribution (no diagnostic instruction tuning, possibly safety-shaped away), even though the discriminative features are right there in the representation.

## Scale axis: frontier vision PARTIALLY closes the gap, and not by refusal

`eval/frontier_histo_vision.py` sends the same patches to claude opus/sonnet/haiku (n=140, vision API):

| model | verbalized P(tumor) AUROC | refusal/fallback |
|---|---|---|
| Qwen2.5-VL-7B (open, output) | 0.463 | (encodes 0.827) |
| claude-haiku-4-5 | 0.482 | 0/140 |
| claude-sonnet-4-6 | 0.664 | 0/140 |
| claude-opus-4-8 | 0.644 | 0/140 |

Two things. First, scale PARTIALLY closes the gap: from chance (8B 0.463, haiku 0.482) to ~0.65 at sonnet/opus. But it plateaus there, below the open 7B's ENCODED 0.827 and the pathology FM CONCH ~0.9. So general frontier vision verbalizes tumor at 0.65, well above chance but far short of both what a 7B model already encodes and what a specialist achieves. The most striking framing: the open 7B's hidden states (0.827) hold MORE tumor signal than any frontier model will SAY (0.65). Second, the failure is NOT refusal: 0/140 fallbacks at every scale, the models give real numbers and partially succeed. So the cap at ~0.65 is a genuine perception-plus-knowledge limit of general (non-pathology-tuned) vision, not safety shaping.

This places histopath between the two scale regimes already mapped: unlike DNA-promoter (closed fully to 0.82) and single-cell-gene-name (to 0.99), and unlike the scale-invariant web-zero rungs (MS, single-cell-anon, stuck at chance), histopath PARTIALLY closes and then plateaus. The interpretation: the two-factor law predicts grounding (property web-documented, encoding confirms it at 0.827), and scale does move the output, but a residual encoding-to-output gap survives because raw-pixel tumor calling is a hard task that only a pathology-tuned model (LLaVA-Med, CONCH-aligned) would verbalize to ceiling. Web-exposure governs ENCODING; reaching the OUTPUT is a separate axis that scale advances but, for a hard perceptual property, does not finish.

## Caveats

Cheap ceiling (color features 0.746) is a floor specialist, not the true ceiling; CONCH ~0.9 is the right reference and the activation (0.827) sits between them. n=400 (200/200), single VLM (Qwen2.5-VL-7B), central-region label (PatchCamelyon convention). The output-arm prompt asks for a probability; a yes/no or chain-of-thought prompt might lift output somewhat, but the molecular-image rung showed prompt variation does not close a gap of this size. Pilot.

## Reproduce

`VL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct VL_N=400 python eval/activation_arm_histo.py` with `signal/histo/pcam.csv` and patch PNGs under `signal/histo/img/`.
