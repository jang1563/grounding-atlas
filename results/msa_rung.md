# MSA rung: the positive control for the two-factor law

*Results. 2026-06-12. `eval/activation_arm_msa.py`, data `signal/msa/msa_conservation.csv` (658 alignment columns from 8 Pfam seed alignments: protein kinase, globin, homeobox, RRM, cNMP-binding, 7tm GPCR, Ig I-set, SH3; depth ~40; conserved vs variable by within-family column entropy). Qwen3-8B 3-arm. ceiling = logistic regression on transparent column statistics (gap fraction, number of distinct residues, depth); activation = hidden-state probe on the column text; output = verbalized P(conserved). No em dashes.*

## Result

| arm | AUROC |
|---|---|
| ceiling (LR on column statistics) | 0.999 |
| ACTIVATION (hidden-state probe) | 1.000 |
| OUTPUT (verbalized P(conserved)) | **0.795** |

- encoding gap: ~0 (the model encodes conservation as perfectly as the statistics define it)
- expression gap: 0.205

## What it shows

This is the designed POSITIVE control for the two-factor web-exposure law, the mirror image of the methylation rung. Here BOTH factors are satisfied: the representation is amino-acid LETTERS (web-rich tokens, ubiquitous in text) and the property is CONSERVATION (web-documented: conservation analysis is one of the most-described concepts in molecular biology). The law predicts the model grounds it, and it does, at both stages: the hidden states encode conservation perfectly (1.000) and the verbalized output reads it at 0.795, the HIGHEST output AUROC of any rung in the project (above RNA-coding 0.720, single-cell-gene-name closes only at frontier scale). The model can both encode AND say which alignment column is conserved.

The contrast that makes the law concrete: take two rungs that are formally identical (a vector/string of values per item, mapped to a binary property), differing only in token web-exposure:
- MSA column = web-rich amino-acid letters, web-documented property -> output 0.795 (grounds).
- methylation beta-vector = web-zero numbers, property not bound to those numbers in text -> output at chance (the encoding floor).

Same task shape, opposite outcome, predicted by which factors the representation satisfies. The expression gap here (0.205) is real but the smallest among the harder rungs, because reading "are most of these residues the same letter" is a transparent operation on web-known tokens that the model largely performs in its output, not only in its activations.

## Caveats

Conservation is a near-transparent function of the visible column (gap fraction + residue diversity), so ceiling and activation are both ~perfect by construction; this rung is informative at the OUTPUT stage and as the positive anchor of the contrast, not as an encoding-gap measurement. Labels are within-family entropy terciles (conserved = bottom third, variable = top third, middle dropped), 8 families, n=658. Pilot.

## Reproduce

`ACT_CSV=$HOME/bge/msa_conservation.csv python eval/activation_arm_msa.py` (Qwen3-8B, GPU). Data built by the Pfam seed-alignment fetch in `signal/msa/`.
