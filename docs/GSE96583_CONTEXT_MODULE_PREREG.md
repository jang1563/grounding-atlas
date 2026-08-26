# GSE96583 donor-aware expression-context module preregistration

Freeze date: 2026-07-27  
Analysis ID: `gse96583-cd8-nk-context-module-donor-replication-v1`  
Model: `claude-haiku-4-5-20251001`  
Status: built and audited; no model calls made

## Question and decision

Across eight donors, does the model use recognizable marker tokens in
context-specific directions?

1. In cells containing both TCR/CD8 and cytotoxic-effector evidence, do
   TCR/CD8 tokens push toward CD8 while cytotoxic tokens push toward NK?
2. In cells containing both NK-receptor and cytotoxic-effector evidence, do
   both categories push toward NK, and how do their magnitudes differ?

This is a causal intervention on the rendered text input. It tests lexical
evidence use at the model interface. It is not a gene perturbation, pathway
intervention, hidden-state intervention, proof of latent knowledge, or
physical-law test.

The broad cytotoxic module is carried forward unchanged. Consequently this is
a module-policy replication and decomposition, not an exact `GNLY/NKG7`
replication. The selected cytotoxic targets are frozen and reported below.

## Authenticated source

The target is the control arm of batch 2 from Kang et al., GEO
[GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) and
DOI [10.1038/nbt.4042](https://doi.org/10.1038/nbt.4042). Batch 2 pooled eight
systemic-lupus-erythematosus donors. The control cells were cultured for six
hours without IFN-beta. They are not healthy controls.

Frozen local source hashes:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `data/raw/gse96583/GSE96583_RAW.tar` | 76,195,840 | `e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80` |
| `data/raw/gse96583/GSM2560248_2.1.mtx.gz` | 28,545,918 | `32add28a0b3397d9ef3f220b7a6a55e98e60fe7b66fe48d0986d634df8ca0013` |
| `data/raw/gse96583/GSM2560248_barcodes.tsv.gz` | 52,904 | `d58d8d55cbe4a12757207784b3bc9227bf200c9100ca15131176e9f8159c955e` |
| `data/raw/gse96583/GSE96583_batch2.genes.tsv.gz` | 277,054 | `93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca` |
| `data/raw/gse96583/GSE96583_batch2.total.tsne.df.tsv.gz` | 756,342 | `1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09` |
| `data/pbmc3k_processed.h5ad` | frozen parent | `0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38` |

The authenticated archive also transitively locks two stimulated-arm members
that are not ingested: `GSM2560249_2.2.mtx.gz` (29,050,932 bytes,
`8aecc98a7ac4957bbc2570f87ebe8ce97332a5bcdbf557d40bd5aabfd287bdc5`)
and `GSM2560249_barcodes.tsv.gz` (52,366 bytes,
`9bb38e080dfae81036fbcd9902c6c6254a4466aa85f70705479be3b2d6679d55`).
Activating a stimulated secondary requires extracting and validating those
members. It would still contain the same eight donors, not eight new
replicates.

The deposited control matrix is 35,635 genes by 14,619 cells with 8,732,747
nonzero integer entries. Its barcode order exactly matches the 14,619 control
metadata rows. The deposited metadata has seven header tokens but eight
fields per row; the first positional field is the unlabeled barcode. The
builder binds and tests that positional schema instead of silently shifting
columns.

The paper assigned cell types from the same expression measurements using a
PBMC3k marker-reference procedure. Therefore the deposited CD8/NK annotations
are not orthogonal biological truth. They define only the upstream binary task
frame and are retained for descriptive and secondary summaries. They do not
select an expression context, sampled cell, module pair, target, control, or
mask within that frame. This experiment is not an annotation-accuracy
benchmark.

## Preprocessing and operational modules

The frame contains unstimulated demuxlet singlets deposited as CD8 T cells or
NK cells. Counts from Ensembl rows sharing a gene symbol are summed before
per-cell total-count normalization to 10,000 and `log1p`. Genes are restricted
to the frozen PBMC3k symbol intersection after the established housekeeping
filter. This leaves 13,503 symbols; 1,445 duplicate-symbol rows are aggregated.
Of 1,998 framed cells, 1,997 have at least 50 positive genes in this universe.
Each rendered input is the top 50 by normalized expression, with lexical
gene-symbol tie breaking.

Frozen modules:

- `T_TCR_CD8`: `CD3D CD3E CD3G CD8A CD8B`
- `NK_receptor_identity`: `FCGR3A KLRC1 KLRC2 KLRD1 KLRF1 NCAM1 NCR1`
- `cytotoxic_effector`:
  `CCL5 CTSW FGFBP2 GNLY GZMA GZMB GZMH GZMK NKG7 PRF1 XCL1 XCL2`

These are operational marker categories, not homogeneous pathways. For each
eligible cell and module, the highest-ranked top-50 member is masked
(`k=1`). No dose-response is claimed.

## Feasibility result and frozen sample

A three-module same-cell comparison is a source-data NO-GO. Donor 107 has no
renderable cell with all three modules, and several other donor/class strata
also have zero or very sparse support. Donors are not dropped to manufacture a
three-way comparison.

The executable design uses two expression-defined contexts in a fixed order:

1. `T_plus_cytotoxic`: require at least one top-50 `T_TCR_CD8` hit and one
   `cytotoxic_effector` hit; select four cells per donor.
2. `NK_receptor_plus_cytotoxic`: require at least one top-50
   `NK_receptor_identity` hit and one `cytotoxic_effector` hit; after excluding
   cells assigned to the first context, select three cells per donor.

Within each donor and context, the lowest hashes of
`SHA256(analysis_id|donor_id|barcode)` are selected. Neither annotation nor
context is in the hash. Minimum eligible support after prior-context exclusion
is seven for the first context and four for the second.

The frozen sample has 56 unique cells, seven per donor:

- 32 `T_plus_cytotoxic` cells;
- 24 `NK_receptor_plus_cytotoxic` cells;
- descriptive deposited labels only: 33 CD8 and 23 NK;
- cross-tabulation: 28/4 CD8/NK in the first context and 5/19 in the second.

The mismatch between expression context and deposited label is retained rather
than corrected. It confirms that labels did not deterministically assign the
context.

Frozen masked-target counts:

- TCR/CD8: `CD3D` 25, `CD8A` 3, `CD8B` 3, `CD3E` 1;
- NK receptor/identity: `KLRD1` 10, `FCGR3A` 6, `KLRC1` 6, `KLRC2` 1,
  `KLRF1` 1;
- cytotoxic effector: `CCL5` 23, `GNLY` 23, `GZMB` 5, `NKG7` 3, `GZMA` 1,
  `GZMK` 1.

The derived CSV is
`signal/single_cell/gse96583_cd8_nk_module_replication.csv`, SHA-256
`f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321`.
Its manifest is
`signal/single_cell/gse96583_cd8_nk_module_replication.manifest.json`.
The manifest SHA-256 and randomized call-plan SHA-256 are frozen after the
runner is generated:

- manifest: `3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f`
- call plan: `bb046113f08eac0e69a12dbcca63ecbbd26fdbea1d70aa1d42db5d6ebd801615`

## Matched controls and prompt factorial

Each module target receives a separately matched neutral deletion from the
same 50-token sentence. Matching uses rank, log-normalized expression,
framed-cell top-50 prevalence, and token length. The eligible control pool
excludes all 30 parent markers and requires an absolute PBMC3k CD8-versus-NK
top-50 prevalence gap at most 0.10.

Every selected cell receives:

1. unmasked input;
2. first context-module mask and its matched-neutral mask;
3. second context-module mask and its matched-neutral mask.

All 56 cells have two distinct neutral inputs. Each input is crossed with the
existing four answer-order by queried-target forms (`ab_pa`, `ab_pb`, `ba_pa`,
`ba_pb`), and P(NK) is aligned to P(CD8) as `1-P(NK)`. This yields 1,120
logical observations and 1,120 unique API calls.

The exact model is `claude-haiku-4-5-20251001`, with temperature 0 and
`max_tokens=16`. Call order is deterministically shuffled. Checkpoints bind
raw output, assignments, donor, expression context, annotation, target and
control genes, timestamps, model, input hashes, preregistration hash, plan
hash, prompt hashes, and execution-code hash. Every probability is recomputed
from the raw string. Confirmatory analysis aborts unless all 1,120 outputs
parse exactly.

## Donor-level estimands

Let \(p^A\) be probability aligned to CD8. For module \(m\), cell \(i\), donor
\(d\), and form \(f\):

\[
r_{mdif}=p^A_{\mathrm{neutral}(m),dif}
          -p^A_{\mathrm{mask}(m),dif}.
\]

Average forms within cell, cells within the relevant donor/context, then give
the eight donors equal weight. Donors—not cells or prompt forms—are the
inferential units.

Gate A, primary replication in `T_plus_cytotoxic`:

- \(\theta_{T|TC}>0\): TCR/CD8 tokens push toward CD8;
- \(\theta_{C|TC}<0\): cytotoxic tokens push toward NK;
- report \(\Delta_{TC}=\theta_{T|TC}-\theta_{C|TC}\) with a two-sided donor
  interval.

Only if Gate A passes, Gate B tests the novel
`NK_receptor_plus_cytotoxic` extension:

- \(\theta_{R|RC}<0\): receptor/identity tokens push toward NK;
- \(\theta_{C|RC}<0\): cytotoxic tokens push toward NK;
- report \(\Delta_{RC}=\theta_{R|RC}-\theta_{C|RC}\) two-sided with no
  preregistered direction.

For every directional component, report its eight donor effects, unweighted
mean, two-sided Student-\(t_7\) 95% interval, one-sided Student-\(t_7\)
p-value, and exact one-sided Rademacher sign-flip p-value over all \(2^8\)
donor sign assignments.

A component passes only if:

1. its mean and entire 95% interval have the registered direction;
2. exact sign-flip `p<0.05`;
3. at least 7/8 donor effects have the registered sign; and
4. all eight leave-one-donor-out means retain the registered direction.

Each context gate is conjunctive; its IUT p-value is the larger exact
sign-flip component p-value. Gate B is interpreted confirmatorily only after
Gate A passes. A cell-level analysis cannot rescue a failed donor-level gate.
All eight donors must have complete outputs.

The T-versus-receptor contrast is not estimable: those modules occur in
disjoint expression-selected cell sets. There is no three-way factorial,
interaction, additivity, or mediation estimand. A cross-context cytotoxic
difference is exploratory because context and sampled cells change together.

## Controls and sensitivities

For each component, also report unmasked-minus-module-mask and
unmasked-minus-neutral-mask donor effects. The latter is a generic deletion
sham. It is considered practically equivalent to zero only if its donor-level
95% interval lies wholly within `±0.03`; failure limits deletion-specificity
language but does not replace the matched-neutral primary contrast.

Prompt robustness is separate from the directional gates. For every registered
context/module component:

1. all four form-specific donor means must retain the expected direction; and
2. donor-level 95% intervals for answer-order and queried-target interactions
   must lie wholly within `±0.03`.

If this fails, report a four-form-averaged, prompt-dependent effect rather than
prompt invariance.

The strict matching sensitivity requires rank distance at most 10 and
expression distance at most 1.0 for both interventions on a cell. It retains
54/56 cells and at least two cells in every donor/context, so donor-level
summaries remain estimable. It is sensitivity analysis only and cannot rescue
the primary gates.

Deposited-label, target-gene, and individual-cell summaries are exploratory.
A target-gene donor summary is shown only if that target occurs in at least
six donors. Even then, target selection is expression-dependent, so no
individual-gene causal attribution is permitted.

## Claim boundary

If both gates pass, the supported conclusion is:

> In one model and the unstimulated control arm of one eight-donor SLE cohort,
> context-recognizable marker names have reproducible donor-level directional
> leverage at the text interface: TCR/CD8 evidence opposes cytotoxic evidence
> in T-plus-cytotoxic cells, while NK-receptor and cytotoxic evidence both
> favor NK in receptor-plus-cytotoxic cells.

This would be an external multi-donor replication/decomposition of lexical
evidence use. It would not establish healthy-population, study, chemistry, or
model-family generalization. It would not prove calibrated biological
probabilities, annotation truth, an exact `GNLY/NKG7` mechanism, individual
gene causality, a biological perturbation, a hidden-state route, stored but
inactive knowledge, a causal latent “activation gap,” or a physical/universal
law.

The operational “activation gap” remains the change in output caused by
removing recognizable evidence relative to a matched token deletion. Calling
it a latent activation gap would require an independent knowledge probe plus a
separate intervention on access or routing while holding the knowledge content
fixed. This experiment does not provide that.

This is a locally hashed preregistration, not an externally authenticated
timestamp.
