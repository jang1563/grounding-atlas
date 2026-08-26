# PBMC68k within-frame label-blind module comparison

This experiment tests the direction of two recognizable evidence
categories on the same common-support cells. Within the label-defined
CD8/NK task frame, every cell receives both TCR/CD8 and frozen
cytotoxic/NK-enriched marker masks, separately matched neutral masks,
and an unmasked input. Target labels never select common-support
inclusion or a mask.

## Primary mechanistic result

- Cells: `65` (CD8 `55`, NK `10`)
- Unique model calls: `1256`; logical condition-form observations: `1300`; shared-neutral response reuses: `44`; parse rate: `100.0%`
- TCR/CD8 effect on P(CD8): **+0.073 [+0.033, +0.118]**, one-sided Welch p=`0.0044427`
- Cytotoxic/NK-enriched marker effect on P(CD8): **-0.214 [-0.289, -0.132]**, one-sided Welch p=`0.00017631`
- TCR-minus-cytotoxic separation: **+0.287 [+0.194, +0.374]**
- Conjunctive intersection-union gate passed: **true** (IUT p=`0.0044427`)
- Both modules have the expected sign within both annotation classes: **true**
- Full preregistered directional gate passed: **true**

Positive values mean the module pushes the model toward CD8 relative
to its module-specific neutral deletion; negative values mean it
pushes toward NK.

| module | effect in CD8 cells | effect in NK cells | equal-class effect |
|---|---:|---:|---:|
| TCR/CD8 | +0.064 [+0.038, +0.093] | +0.081 [+0.006, +0.167] | +0.073 [+0.033, +0.118] |
| cytotoxic/NK-enriched marker | -0.109 [-0.165, -0.056] | -0.320 [-0.458, -0.163] | -0.214 [-0.289, -0.132] |

## Unmasked and neutral controls

| module | unmasked - module mask | unmasked - neutral mask | adjusted |
|---|---:|---:|---:|
| TCR/CD8 | +0.069 [+0.033, +0.111] | -0.003 [-0.030, +0.025] | +0.073 [+0.033, +0.118] |
| cytotoxic/NK-enriched marker | -0.205 [-0.275, -0.125] | +0.010 [-0.019, +0.038] | -0.214 [-0.289, -0.132] |

## Prompt-factor boundary

| module | order interaction | queried-target interaction | all forms expected sign |
|---|---:|---:|---:|
| TCR/CD8 | +0.012 [-0.026, +0.056] | -0.023 [-0.087, +0.050] | true |
| cytotoxic/NK-enriched marker | +0.047 [-0.026, +0.113] | +0.043 [-0.093, +0.177] | true |

Both modules pass the full ±0.03 prompt-equivalence and sign gate: **false**.

## Matching sensitivity

- Strict subset n=`61` (CD8 `53`, NK `8`)
- Strict TCR/CD8 effect: +0.084 [+0.037, +0.135]
- Strict cytotoxic/NK-enriched marker effect: -0.238 [-0.311, -0.160]
- Strict separation: +0.322 [+0.238, +0.405]

## Interpretation boundary

A passed gate supports opposing token-level evidence use on an
equal-class-weighted average in one model and one external cohort:
highest-ranked TCR/CD8-category symbols push output toward CD8,
while highest-ranked frozen cytotoxic/NK-enriched marker symbols
push it toward NK. The within-class sign guard is descriptive,
not donor-level replication.

The 10 included NK-labeled cells were selected because they also
contain CD3/CD8-category evidence. They are atypical mixed-marker
cells and may include contamination or doublets. This is a paired
single-token two-category comparison crossed with a prompt
factorial, not a biological 2x2 factorial; it cannot estimate
module interaction, additivity, or mediation.
It does not isolate an NK-receptor mechanism because the reduced
feature panel censors that module. It is not a gene knockout,
biological causality, a hidden-state activation route, latent-knowledge
proof, multi-donor generalization, or a physical law.
