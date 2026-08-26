# PBMC68k module comparison: post-hoc discovery localization

> Descriptive post-hoc localization. The preregistered aggregate
> gate is unchanged; gene and subtype results are hypothesis-generating.
> Terminology correction: the immutable primary report's “Primary
> mechanistic result” heading should read “Primary intervention result”;
> no internal mechanism was measured.

## Descriptive localization

The intervention shows two asymmetric token-deletion sensitivities in this mixed-marker subset: a modest, distributed CD3/TCR-category sensitivity (`+0.073`) and a larger frozen cytotoxic/NK-enriched-category sensitivity (`-0.214`).

The localization does not support a generic cytotoxic-program interpretation. Cells whose selected target was `GNLY` descriptively account for `88.5%` of the signed equal-class aggregate, and the `NKG7`-targeted stratum accounts for `14.1%`. Small opposing contributions in the `CCL5`- and `GZMK`-targeted contexts make the signed shares sum above 100%. These are context-confounded strata, not isolated gene effects.

## Bidirectional condition shift

| condition | equal-class P(CD8) |
|---|---:|
| unmasked | 0.530 |
| TCR/CD8 token masked | 0.461 |
| T-specific neutral masked | 0.534 |
| cytotoxic/NK-enriched token masked | 0.735 |
| cytotoxic-specific neutral masked | 0.521 |

## Cytotoxic/NK-enriched target localization

| masked target | cells | equal-class contribution | signed share of aggregate | strong form-level switches |
|---|---:|---:|---:|---:|
| GNLY | 24 | -0.190 | 88.5% | 50/96 |
| NKG7 | 9 | -0.030 | 14.1% | 7/36 |
| CCL5 | 26 | +0.004 | -1.8% | 0/104 |
| GZMK | 5 | +0.002 | -0.9% | 0/20 |
| CTSW | 1 | -0.000 | 0.1% | 0/4 |

## State dependence

| annotation | n | cytotoxic/NK-enriched effect on P(CD8) |
|---|---:|---:|
| CD56+ NK | 10 | -0.320 |
| CD8+ Cytotoxic T | 37 | -0.169 |
| CD8+/CD45RA+ Naive Cytotoxic | 18 | +0.015 |

The `GNLY`-targeted stratum is itself context dependent: CD56+ NK `-0.386` (n=7), cytotoxic CD8 `-0.397` (n=15), and naive CD8 `-0.012` (n=2).

## Cell-level heterogeneity

The TCR/CD8 effect had the expected positive direction in `38`/65 cells (`23` zero, `4` opposite); the cytotoxic/NK-enriched effect had the expected negative direction in `36`/65 (`21` zero, `8` opposite). Only `18`/65 cells had both expected signs, while T-minus-cytotoxic separation was positive in `54`/65. The aggregate and class-mean sign gates therefore do not imply universal per-cell behavior.

## Output regime

Only `7` raw probability values were emitted; `75.2%` of calls were exactly `0.15` or `0.85`. In the `GNLY`-targeted stratum, `50`/`96` prompt-form pairs switched from low (≤0.25) to high (≥0.75) P(CD8) after masking. This looks more like cue-triggered category switching than calibrated additive evidence integration.

## Claim boundary

Post-hoc and context-confounded localization in one label-informed reduced panel, one donor, and one model. GNLY and NKG7 are NK associated but are not NK-lineage-specific, and the selected-target strata do not isolate either gene from its cellular context. The result does not establish individual-gene causality, a biological pathway mechanism, hidden-state activation, latent knowledge, prompt invariance, or a physical law.
