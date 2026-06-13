# P1 web-exposure covariate: the cross-modality regression is mis-specified; the within-entity contrast is the valid test

*Reviewer-proofing note for the web-exposure law (P1). 2026-06-09. Companion to `../PROJECT_DESIGN.md` section 7, `../docs/WS1_BACKLOG.md` item D. Instrument: `../eval/p1_webexposure.py`. No em dashes.*

## The objection (and the backlog status it resolves)

P1 says the encoding gap should DECREASE as a modality's content-to-property web co-occurrence increases. The first attempt (`p1_webexposure.py` v1) regressed the measured encoding gap on a PMC co-mention count and FAILED: the correlations came out opposite to P1 (+0.94 on the gap, -0.72 on recovery). A reviewer reading "the law's own falsifiable prediction did not fit" needs an answer. The answer is that the failure is a **covariate mis-specification**, diagnosable and now documented, not a refutation.

## Diagnosis: meta-description is not form-instance (shown live)

The v1 query counted PROSE ABOUT a representation form, not INSTANCES of that form bound to its property. Live PMC counts (`eval/p1_webexposure.py`, Part 1):

| query for the variant raw-sequence form | PMC count |
|---|---|
| meta-description: `"amino acid sequence" AND pathogenic` (the v1 proxy) | **143,853** |
| form-instance: a literal 16-residue window `AND pathogenic` | **0** |

The literature is full of sentences about amino-acid sequences, so the meta-description query is huge; but a raw sequence string actually sitting next to a pathogenicity label is essentially absent from indexed text, so the form-instance count is zero. v1 fed the 143,853 into the regression and therefore ranked the **web-poorest form (raw sequence) as the most exposed**. That single inversion flips the predicted sign. The proxy measured the wrong thing, in the wrong direction, on exactly the rungs the law most depends on.

## Why no cross-modality regression should be fit (three reasons, not one)

1. **The web-poor forms are not faithfully countable.** Raw sequences, database accessions, and randomized SMILES barely exist as indexed text; free-text search either returns ~0 (above) or 500s on the punctuation of the actual notation (HGVS `c.`, literal sequence windows). The covariate you need is the one you cannot reliably measure with a co-mention count. The queryable signal is dominated by name and symbol exposure (gene name + pathogenic = 27,729; SMILES + hERG = 1,552), the web-rich end only.
2. **Ceiling-confound.** The encoding-gap absolute scales with the specialist ceiling, and the five points span ceilings 0.70 to 0.96. A cross-modality regression confounds web-exposure with ceiling.
3. **n = 5.** Five modality-form points cannot support a fitted line whatever the covariate.

So the honest move is the one the backlog already states: do NOT claim a fitted P1 law.

## The valid covariate: within-entity notation contrast (ceiling held fixed)

The law makes a sharper, measurable claim WITHIN a single entity: hold the content, the property, and the specialist ceiling fixed, vary only the surface form, and the web-richer notation should ground better. There the covariate is a robustly orderable notation-exposure RANK, not a flaky absolute count.

| entity (ceiling fixed) | form | web-exposure rank | activation | output |
|---|---|---|---|---|
| variant, ClinVar (0.962), gene GroupKFold | gene + HGVS symbol (web-rich) | 1 | **0.795** | **0.599** |
| variant, ClinVar (0.962), gene GroupKFold | raw aa sequence (web-poor) | 3 | 0.740 | 0.494 |
| SMILES, hERG (0.825), same molecules | canonical SMILES | 2 | - | 0.573 |
| SMILES, hERG (0.825), same molecules | randomized valid SMILES | 3 | - | 0.553 |

- **Variant confirms the predicted direction on BOTH arms.** The web-rich symbol beats the web-poor sequence at activation (0.795 > 0.740) and output (0.599 > 0.494), at one fixed 0.962 ceiling, under a gene GroupKFold that removes the trivial gene-prior shortcut. Same biology, different surface form, web-exposure is the variable that moved.
- **SMILES is a floor, not a counterexample.** Canonical 0.573 vs randomized 0.553 is no effect because there is no surfaced output signal to bind; notation-sensitivity is measurable only where grounding already surfaces. This is what the law predicts at the expression-dominant end, not a refutation.

The notation-exposure rank itself (name/symbol > HGVS/canonical-SMILES/InChIKey > raw sequence/accession/randomized-SMILES) is corpus-grounded and stable, and it is the general-capability form of the FRT recognition gap (axis A: name ~100% vs accession ~2-28%).

## Bottom line

P1 stands as a **within-entity qualitative law** (variant text > seq at one ceiling; SMILES canonical ~ randomized floor) plus the **regime spectrum** across modalities, not a cross-modality fitted regression. The v1 failure is now a documented covariate mis-specification with the inversion mechanism shown live (143,853 vs 0), which preempts the reviewer objection instead of leaving it open. A form-specific exposure covariate that can reach the web-poor rungs (e.g. counting a notation form's frequency in a pretraining-scale corpus rather than in indexed abstracts) remains the open methodological problem; until one exists, claiming a fitted P1 would be the overclaim, not the result.
