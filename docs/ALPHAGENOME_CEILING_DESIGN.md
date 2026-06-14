# AlphaGenome ceiling: a regulatory rung with a strong specialist, and a new orchestrate target

*Design doc, 2026-06-14. Replaces the weak 6-mer ceiling of the DNA-promoter rung with AlphaGenome (Cheng et al, Nature 2025), a unified regulatory-variant-effect predictor, and adds AlphaGenome to the orchestrate arm of the decision map. Tests whether the LLM ENCODES regulatory grammar or only NAMES web-documented eQTLs. No em dashes.*

## The question

The DNA-promoter rung has an honest hole: its ceiling is a 6-mer surface probe (0.889), so the near-zero encoding gap means the model encodes surface k-mer statistics, the DNA analog of SMILES not beating a char-n-gram, NOT deep regulatory understanding (`results/dna_promoter.md`, noted in the project's own caveats). To measure encode-vs-verbalize on real regulatory function we need a strong, content-grounded ceiling. AlphaGenome is that ceiling: from a long DNA context it predicts expression, chromatin accessibility, splicing, TF binding, and contact maps, and scores variant effects, integrating the scattered regulatory signals one model was built to unify.

Two questions follow:
1. Is regulatory-variant effect an ENCODING-limited rung (like 3D coordinates and molecular graph for chemistry), where the property needs long-range, cell-type-specific context the model cannot extract from a text sequence and a probe also fails, as opposed to the expression-limited regime?
2. Where the model does score regulatory variants, is it grounding the sequence or recalling web-documented eQTLs (axis A, recognition)?

## The construct

Items: regulatory variants (or short regulatory sequences) with an AlphaGenome-predicted effect on a chosen track (e.g. expression change in a fixed cell type) as the label. The ceiling is AlphaGenome itself (or a thin supervised head on its tracks). Arms: ceiling (AlphaGenome), activation (open-weight probe on the sequence's hidden states), output (LLM verbalizes the regulatory effect from the sequence, 8B and frontier).

## Conditions (the web-exposure contrast at a fixed ceiling)

- **web-rich**: known regulatory variants documented as eQTLs / GWAS hits (the model has seen them named in text).
- **web-poor**: novel or post-cutoff regulatory variants AlphaGenome scores but no literature describes; and a random-regulatory-sequence set with no symbolic identity.
- **scrambled-name control**: the same variant with its rsID / gene symbol removed or scrambled, the existing re-notation control, to separate axis-A recognition from axis-B content.

## Prediction (falsifiable)

This mirrors the variant flagship, which is orchestrate-won (AlphaMissense AUROC 0.96, 0.985 on novel post-cutoff variants, far above any LLM seq-reader; `results/decision_map_placement.md`). Predicted here:

- web-rich eQTL variants: LLM output decent, but it COLLAPSES under the scrambled-name control, revealing recall not grounding.
- web-poor / novel regulatory effects: LLM output at chance while AlphaGenome reads them, a clean **orchestrate-won** cell, extending the decision map into the non-coding / regulatory axis.
- the discriminating result is the activation arm: if the probe also fails on web-poor regulatory effect (cannot recover it from the text sequence), this is the first ENCODING-limited rung in the regulatory modality, the genomics analog of 3D/graph for chemistry; if the probe succeeds but output fails, it is expression-limited like the rest. Either outcome is informative and pre-registered.

## Why it matters

1. Fixes the DNA rung's surface-k-mer ceiling with a content-grounded specialist, so the encode-vs-verbalize claim for regulatory biology stands on a real ceiling.
2. Adds AlphaGenome to the orchestrate arm alongside ESM, AlphaMissense, Evo, and Boltz-2, and lands a measured "call the regulatory specialist" placement, which is precisely the AlphaGenome-era conclusion that models widen the candidate space but do not close causality, the model should route regulatory-effect questions to AlphaGenome rather than answer them.
3. Tests the encoding-limit hypothesis in a new modality, sharpening the claim that the encoding axis is governed by whether discriminative features sit on the surface of the representation.

## Honest limits

- AlphaGenome itself is trained on the same public, cell-line-biased data (K562 and friends), so its ceiling is strong only where it has data. This is the SAME bias the data-war framing names, and it couples this rung to `DATA_DENSITY_RUNG_DESIGN.md`: AlphaGenome supplies the specialist, the data-density map says where that specialist (and therefore the orchestrated answer) is trustworthy. Report the rung stratified by cell-state data density, not pooled.
- a 6-mer ceiling 0.889 was already high; the point is not a higher number but a content-grounded one that survives the scrambled-name control.
- AlphaGenome variant effects are predictions, not measured ground truth; for the web-poor novel set the label is a strong proxy, not gold, so frame it as predicted-structural (the project's third property type), not empirical, and note the proxy-gold gap.
- activation is open-weight-only.

## Implementation entry points

A `signal/regulatory/` generator producing (sequence/variant, AlphaGenome-effect, condition) items via the public AlphaGenome API; ceiling and labels from the API; `eval/activation_arm_dna_cayuga.sh` adapted for the regulatory sequence probe; output arm mirroring `eval/frontier_output_panel.py` with the scrambled-name control from `eval/notation_control.py`. First executable step: pull AlphaGenome scores for a small known-eQTL set plus a novel set, confirm the web-rich vs web-poor split and the scrambled-name control are clean, before scaling.
