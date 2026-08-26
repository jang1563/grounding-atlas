# PBMC68k lineage-program masking preregistration

Freeze date: 2026-07-27  
Analysis ID: `pbmc68k-cd8-nk-program-mask-v1`  
Model: `claude-haiku-4-5-20251001`

## Discovery question

Does removing a biologically recognizable, reference-label-matched CD8 or NK
gene program reduce the model's correct-class probability more than removing
the same number of expression- and rank-matched non-program genes in an
external PBMC cohort?

This is a causal input-to-output masking question. It is not an unassisted
cell-annotation test, a hidden-state causal test, a physical-law test, or a
test of pretraining exposure.

## Frozen data and provenance

- Feature/program source: Scanpy PBMC3k processed artifact,
  SHA-256 `0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38`.
- External target: Scanpy's reduced, annotated 10x PBMC68k artifact,
  SHA-256 `863e19914ab2d4ba97edc9623ac3a343c0461f1e40b121bfb5fa92638b22e9bd`.
- The target is the 10x Fresh 68k PBMC Donor A cohort. Technical barcode
  suffixes are not donors.
- Source documentation:
  <https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc68k_reduced.html>
- Original 68k study: Zheng et al. 2017, DOI
  <https://doi.org/10.1038/ncomms14049>.

The common PBMC3k/PBMC68k non-housekeeping universe contains 750 genes.
Each target cell is rendered as its 50 highest expressed genes in that frozen
universe. Reference annotations map both `CD8+ Cytotoxic T` and
`CD8+/CD45RA+ Naive Cytotoxic` to class A, and `CD56+ NK` to class B.

There are 128 annotated target cells. Nine CD8 cells have no frozen
CD8-program hit in the top 50 and are excluded before any model call. The
claim-bearing set contains 119 cells: 88 CD8 and 31 NK.

Frozen derived artifacts:

- CSV SHA-256:
  `6f9fae1ecd3cbe03cd2d317f97cf27f26c8070ec7aaf6fb8fdf236f7b3b1a57d`
- Manifest SHA-256:
  `757eec7507b3cf5da410546409c43e2e3cd685b87019e83284c0a81e6076f749`
- Randomized call-plan SHA-256:
  `912cace17d4f2f2c1ffd0b037f7273a175aa296a55f10a3143dca0ca51d5161d`

## Frozen gene programs

The parent marker set was committed in
`signal/single_cell/build_cd8t_nk.py` at commit
`20f94201c60878b83dbae182b8447248666ede29` on 2026-06-19, before the
outputs analyzed here existed. It is partitioned without using any LLM output.

CD8-identity program:

`CD8A CD8B GZMK KLRB1 CD3D CD3E CD3G CD247 IL7R`

NK-identity program:

`GZMB GZMH NKG7 GNLY KLRD1 KLRF1 KLRC1 KLRC2 NCAM1 FCGR3A PRF1 NCR1 FGFBP2 SPON2 XCL1 XCL2 TYROBP`

Shared cytotoxic genes `GZMA CCL5 CST7 CTSW` are excluded from both the
target program and the control pool for this identity-specific test.

For each eligible cell, the target arm replaces up to the three
highest-ranked genes from the program matching its reference label with
`MASKED_GENE`.

The control arm replaces the same number of non-program genes. Controls are
selected by within-cell linear-sum assignment using expression rank,
log-normalized expression, global top-50 prevalence, and token length.
All parent-program genes are excluded. Genes with an absolute CD8-versus-NK
top-50 prevalence difference greater than 0.10 in PBMC3k are also excluded
from the control pool. No target-model output enters matching.

## Prompt factorial and execution

Every cell has two mask conditions:

- label-matched lineage-program mask;
- matched non-program control mask.

Both conditions are crossed with four prompt forms:

| form | answer order | queried probability |
|---|---|---|
| `ab_pa` | CD8 / NK | P(CD8) |
| `ab_pb` | CD8 / NK | P(NK) |
| `ba_pa` | NK / CD8 | P(CD8) |
| `ba_pb` | NK / CD8 | P(NK) |

P(NK) responses are converted to P(CD8) as `1-P(NK)`. The 119 cells × two
mask conditions × four forms produce 952 calls. Their order is frozen by the
call-plan hash above and is executed contemporaneously with deterministic
decoding. Raw strings, parse status, timestamps, prompt hashes, model ID, and
input hashes are checkpointed.

## Primary estimand

For cell \(i\), prompt form \(f\), and label \(y_i\), define
\(s_i=2y_i-1\) and let \(p^A\) denote the probability aligned to CD8:

\[
d_{if}=s_i\left(p^A_{i,\mathrm{control},f}
                 -p^A_{i,\mathrm{program},f}\right).
\]

The per-cell effect is the mean of \(d_{if}\) over the four prompt forms.
The primary population estimand gives CD8 and NK equal weight:

\[
\tau=\frac{1}{2}
\left(E[d_i\mid\mathrm{CD8}]+E[d_i\mid\mathrm{NK}]\right).
\]

Positive \(\tau\) means that masking the frozen lineage program harms the
reference-class probability more than matched information deletion.

Inference uses a class-stratified paired cell bootstrap with 20,000 draws and
a two-sided cell-level sign-flip test with 50,000 permutations. This is one
primary test; the class-specific effects are decomposition, not independent
replications.

## Prompt-robustness gate

The order interaction is the AB-form effect minus the BA-form effect. The
queried-target interaction is the P(A)-form effect minus the P(B)-form
effect. The program effect is described as prompt-robust only if both 95%
bootstrap intervals lie wholly inside the preregistered `±0.03` probability
margin. Otherwise the primary average is reported as prompt-dependent.

## Sensitivities and decision rule

- Strict matching sensitivity retains cells for which every matched pair has
  rank distance at most 10 and expression distance at most 1.0.
- Effects are reported by class, mask dose, prompt form, and technical barcode
  suffix. Barcode suffixes are not biological replicates.
- Orientation-averaged AUROC and Brier scores are secondary because the
  reference label determines which program is masked.

Interpretation:

- primary 95% interval above zero and sign-flip `p<0.05`: evidence of
  transferable program-specific output dependence in this one external cohort;
- interval containing zero: no detected program-specific effect beyond
  matched deletion;
- interval below zero: matched non-program deletion is more damaging;
- prompt-interaction gate failure: the effect is elicitation-dependent, even
  if the four-form average is nonzero.

No result from this experiment alone establishes multi-donor generalization,
latent knowledge, a causal activation route, training exposure, or a physical
law.
