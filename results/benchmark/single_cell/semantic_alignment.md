# Single-cell semantic grounding is graded but polarity-asymmetric

This paired reanalysis asks whether replacing anonymous feature IDs with real gene
symbols moves each cell toward its correct class, rather than merely increasing AUROC.
It uses the existing 200-cell predictions for four models on two PBMC contrasts.

## Main result

All 8/8 model-task cells have a positive name-minus-anonymous AUROC difference; 8 remain significant after Holm correction. The probability update is strongly class-asymmetric: 85.3% of the aggregate mean class-separation gain comes from lowering the first-class probability on comparator-class cells (NK or CD16+).

In 7/8 cells, the comparator class contributes at least 65% of the separation gain. In 2 cells, revealing gene names actually lowers the mean probability of the correct first class; AUROC still improves because the comparator-class probability falls more.

After removing between-class differences, 8/8 model-task cells show a Holm-significant correlation between the name-induced probability update and graded out-of-fold bag-of-gene specialist evidence.

The 8 model-task cells reuse the same 200 cells within each biological task; they are not 8 independent datasets. The model-averaged task summaries below therefore provide the cleaner unit for describing the shared pattern.

## Model-averaged task decomposition

| task | ensemble AUROC gap (95% CI) | first-class correct shift | comparator correct shift | comparator share | prior shift | within-class evidence rho | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| CD8+ T vs NK | 0.466 [0.386, 0.547] | -0.022 [-0.051, 0.006] | 0.331 [0.310, 0.351] | 107.3% | -0.177 | 0.497 | 4e-05 |
| CD14+ vs CD16+ monocyte | 0.469 [0.389, 0.548] | 0.122 [0.104, 0.137] | 0.248 [0.210, 0.286] | 67.0% | -0.063 | 0.529 | 4e-05 |

A comparator share above 100% is a signed decomposition, not a variance fraction: it occurs when the first-class mean update points in the wrong direction while the larger comparator update still improves separation.

The out-of-fold reference did not show weaker evidence for the first class. Its mean correct-oriented margins were CD8+ T vs NK: 4.56 (CD8+ T) versus 3.17 (NK); CD14+ vs CD16+ monocyte: 5.18 (CD14+) versus 3.15 (CD16+). The comparator-dominant LLM update therefore is not explained by a weaker first class under this model-independent reference.

The item-level update is also moderately shared across models: the mean within-class pairwise rank correlation of name-minus-anonymous deltas is 0.392 for CD8+ T vs NK and 0.443 for CD14+ vs CD16+ monocyte.

## Per-model decomposition

| task | model | AUROC gap (95% CI) | first-class correct shift | comparator correct shift | comparator share | prior shift | within-class evidence rho | Brier improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CD8+ T vs NK | Haiku 4.5 | 0.326 [0.268, 0.384] | -0.073 [-0.121, -0.030] | 0.487 [0.424, 0.548] | 117.7% | -0.280 | 0.319 | 0.206 |
| CD8+ T vs NK | Sonnet 4.6 | 0.367 [0.277, 0.457] | 0.088 [0.046, 0.130] | 0.168 [0.149, 0.187] | 65.6% | -0.040 | 0.322 | 0.081 |
| CD8+ T vs NK | Opus 4.8 | 0.483 [0.416, 0.549] | 0.036 [0.003, 0.070] | 0.306 [0.290, 0.322] | 89.4% | -0.135 | 0.565 | 0.100 |
| CD8+ T vs NK | GPT-4o | 0.366 [0.301, 0.430] | -0.141 [-0.173, -0.110] | 0.362 [0.348, 0.375] | 164.0% | -0.252 | 0.347 | 0.132 |
| CD14+ vs CD16+ monocyte | Haiku 4.5 | 0.258 [0.203, 0.313] | 0.060 [0.043, 0.070] | 0.221 [0.151, 0.293] | 78.7% | -0.081 | 0.324 | 0.103 |
| CD14+ vs CD16+ monocyte | Sonnet 4.6 | 0.412 [0.329, 0.496] | 0.117 [0.085, 0.147] | 0.350 [0.298, 0.399] | 75.0% | -0.117 | 0.240 | 0.196 |
| CD14+ vs CD16+ monocyte | Opus 4.8 | 0.483 [0.465, 0.501] | 0.170 [0.128, 0.210] | 0.388 [0.369, 0.404] | 69.6% | -0.109 | 0.541 | 0.163 |
| CD14+ vs CD16+ monocyte | GPT-4o | 0.268 [0.215, 0.322] | 0.142 [0.138, 0.146] | 0.031 [-0.015, 0.081] | 18.0% | 0.055 | 0.312 | 0.022 |

## Interpretation

The named-symbol advantage is real in these paired outputs, but it is not a
uniform increase in cell-level biological correctness. Much of the effect is
a one-sided probability update toward the comparator class. For CD8/NK in particular, some
models reduce P(CD8) for both CD8 and NK cells, but reduce it much more for NK.
That improves ranking and aggregate proper scores while harming the mean
direction of the CD8 update.

This separates two hypotheses that AUROC alone conflates:

1. **Class anchoring:** recognizable markers trigger a class prototype or prompt-side
   prior shift.
2. **Graded semantic grounding:** within a class, probability updates track the
   strength of model-independent, out-of-fold bag-of-gene evidence for each cell.

The within-class permutation result reports how much evidence supports the second
hypothesis. Neither result identifies training exposure, latent knowledge, or a
causal activation-to-output route.

## Follow-up intervention

The answer roles and probability orientation were subsequently reversed on the
exact same cells for Haiku 4.5. The pilot identifies a positive graded semantic
correction together with a substantially larger prompt-role prior; see
[the orientation-swap result](orientation_swap/claude-haiku-4-5-20251001.md).

That two-form swap changes both list position and which class is queried. A pure
position test still requires four contemporaneous forms crossing order (A/B, B/A)
with queried target (P(A), P(B)).
