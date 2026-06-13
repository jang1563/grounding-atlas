# Ceiling-gate results, variant branch (axis-B candidate screening)

*2026-06-09. CPU (AlphaMissense precomputed + sklearn). Script: `eval/ceiling_gate_variant.py`.*

**Question:** is pathogenicity predictable from the variant CONTENT (protein sequence context +
substitution)? A high specialist ceiling means the signal is there, so a probe-vs-LLM
head-to-head is meaningful. This is the variant analog of `../../results/ceiling_gate.md` (Morgan
FP) and `../../protein_grounding/results/ceiling_gate.md` (ESM2). Here the specialist is
**AlphaMissense** (precomputed pathogenicity, Science 2023); **ESM-1v** WT-marginal LLR is the
secondary, fully-unsupervised ceiling (GPU, `ceiling_esm1v_variant.py`).

**Data:** ClinVar `variant_summary` (pinned 2026-06 release), GRCh38 single-AA missense, germline
`Pathogenic`/`Likely pathogenic` (1) vs `Benign`/`Likely benign` (0), review star >= 1, mapped to
UniProt canonical with a WT-residue consistency QC, AlphaMissense matched on (accession,
substitution). 181,831 variants with a score (98.5% of the mapped set), P=56,653 / B=125,178.

## AlphaMissense ceiling

| stratum | n | %pos | AUROC [95% CI] | AUPRC |
|---|---|---|---|---|
| **ALL** | 181,831 | 31 | **0.960 [0.959, 0.961]** | 0.920 |
| LogReg(score) 5-fold CV | 181,831 | 31 | 0.960 (monotone: same as raw) | - |
| star 1 | 123,720 | 27 | 0.963 [0.962, 0.964] | 0.911 |
| star 2 | 53,782 | 37 | 0.955 [0.953, 0.957] | 0.934 |
| star 3+ (expert/guideline) | 4,329 | 70 | 0.932 [0.925, 0.941] | 0.968 |
| star 2+ | 58,111 | 40 | 0.954 [0.952, 0.956] | 0.937 |
| first seen <= 2025-06 | 156,617 | 33 | 0.956 [0.956, 0.958] | 0.920 |
| first seen 2025-H2 | 15,298 | 15 | 0.982 [0.980, 0.985] | 0.926 |
| **first seen post-2026-01 (strict holdout)** | 9,916 | 24 | **0.965 [0.961, 0.969]** | 0.895 |

## Reads

- **The ceiling is high and PASSES.** AlphaMissense separates ClinVar P/B at AUROC 0.960 on the
  star-1+ set, matching the published ~0.94 (slightly higher here because star-1+ skews toward
  cleaner calls). There is abundant content signal for the LLM to fail to surface, which is what
  the head-to-head needs. Far above the hERG (0.825) and meltome (0.70) ceilings: variant effect
  is the most separable of the three modalities probed so far.
- **The ceiling HOLDS on the temporal holdout (the headline control).** On variants first added
  to ClinVar after 2026-01 (post the Opus-class cutoff boundary, 9,916 variants), AlphaMissense
  is 0.965, essentially flat against the pre-cutoff 0.956. The specialist grounds from content, so
  it generalizes to novel variants by construction. This flat specialist line is the baseline
  against which the LLM output arm's behavior on the SAME novel variants is read
  (`head_to_head.md`): any output-side collapse there is a grounding gap, not a hard-variant
  artifact.
- **Stable across review stars.** 0.963 (star 1) to 0.932 (star 3+); the mild decline at star 3+
  reflects its pathogenic enrichment (70% pos) and harder expert-adjudicated calls, not a ceiling
  failure. The activation arm's structure-probe uses the AlphaMissense AUROC as its ceiling
  reference on the matched set.
- **AlphaMissense is not leakage-free, by design.** It was trained with ClinVar-adjacent
  population-frequency and proxy signals, so a flat temporal-holdout AUROC is expected and is not
  itself evidence about memorization. That is why the **fully unsupervised ESM-1v** is run as the
  secondary ceiling (`ceiling_esm1v_variant.py`): an unsupervised specialist that holds on the
  holdout rules out the "the labels themselves leaked into the specialist" reading.

## ESM-1v secondary ceiling (unsupervised, confirmed)

ESM-1v 650M wild-type-marginal LLR (single forward pass, no MSA, fully unsupervised),
`ceiling_esm1v_variant.py` on Cayuga a40 (jobs 3027001 main / 3027003 holdout; the full-184k run
was too slow for the 2h limit, so it was run on the balanced sets).

| set | stratum | n | AUROC | AUPRC |
|---|---|---|---|---|
| main (`variant_clinvar`) | ALL | 2000 | 0.893 | 0.909 |
| | star 1 | 1300 | 0.886 | 0.889 |
| | star 2+ | 700 | 0.902 | 0.936 |
| | first seen <= 2025-06 | 1762 | 0.890 | 0.911 |
| | first seen 2025-H2 | 139 | 0.843 | 0.839 |
| holdout (balanced post-2026-01) | **post-2026-01** | 2000 | **0.921** | 0.933 |

- ESM-1v separates ClinVar P/B at 0.893 (main set), below AlphaMissense (0.96) but well above
  chance, and **holds on the strict post-2026-01 holdout at 0.921** (dedicated balanced
  2000-variant set; even above its overall 0.893).
- **This is the leakage-free confirmation of the ceiling.** ESM-1v is a masked language model
  trained only on UniRef protein sequences and has never seen a clinical label, so its high
  holdout AUROC cannot be ClinVar-label memorization. A supervised specialist (AlphaMissense
  0.96/0.965) and a fully-unsupervised one (ESM-1v 0.89/0.92) BOTH read pathogenicity from variant
  content and BOTH generalize to novel variants. The ceiling is genuine sequence content, so the
  LLM output collapse on the SAME novel variants (`head_to_head.md`) is a real grounding gap.
- The two ceilings (AlphaMissense 0.96, ESM-1v 0.89) bracket the specialist signal that Qwen3-8B's
  hidden states recover only ~77-83% of internally (`head_to_head.md` activation arm): a real
  encoding gap on top of the expression gap.

## DMS parallel track ceiling (`prepare_dms.py`)

AlphaMissense predicts the experimental DMS damaging label (the memorization-resistant track,
clinically famous genes) at AUROC 0.88 (BRCA1), 0.85 (PTEN), 0.91 (TP53), 0.87 (MSH2). The
specialist holds on experimental functional labels too, so the DMS comparison in `head_to_head.md`
is read against a real ceiling, not noise. (These also confirm the damaging-label orientation:
AlphaMissense pathogenicity is positively associated with low DMS fitness, as it must be.)

## Caveat carried forward

The WT-residue QC keeps only variants whose UniProt canonical residue matches ClinVar's
wild-type AA (93% map; 13,310 dropped on mismatch), so the set is biased toward
canonical-isoform-consistent variants. AlphaMissense and the sequence window use that same
canonical isoform, so the three forms stay self-consistent; the cost is some loss of
non-canonical-isoform variants.

## Next

Run the dual-form output arm (`output_arm_variant.py`): does a general LLM surface this
content from the web-rich text form (gene + HGVS) and from the web-poor sequence form, and how
much of any above-chance signal survives the temporal holdout, the gene-name scramble, and the
DMS track? See `head_to_head.md`.
