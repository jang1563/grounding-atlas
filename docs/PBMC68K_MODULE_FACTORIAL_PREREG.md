# PBMC68k within-frame label-blind module-comparison preregistration

Freeze date: 2026-07-27  
Analysis ID: `pbmc68k-cd8-nk-module-factorial-v1`  
Model: `claude-haiku-4-5-20251001`

## Discovery question

When both types of evidence are present in the same rendered cell, do
TCR/CD8 marker symbols push the model toward CD8 while shared
cytotoxic-effector symbols push it toward NK?

This is a causal text-input ablation question. It is not a biological gene
perturbation, hidden-state intervention, latent-knowledge proof, or physical-law
test.

## Frozen source and target

- Parent marker and feature-universe source:
  `data/pbmc3k_processed.h5ad`, SHA-256
  `0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38`.
- Target: Scanpy PBMC68k-reduced artifact, SHA-256
  `863e19914ab2d4ba97edc9623ac3a343c0461f1e40b121bfb5fa92638b22e9bd`.
- Derived factorial CSV, SHA-256
  `fadd90b9aa1249c2691943287962a381329bcba8f91f0b392636ba80ce9f1d9b`.
- Derived manifest, SHA-256
  `b849e34b94f0004568ca289f1997f39b15f25eb991200b8b3d0f0c3e62869f79`.
- Randomized call plan, SHA-256
  `25d083894f54b17ee0dca72a04b19a30f03e076403dde7f9ae05d05c7c42ae96`.

The target is Scanpy's subsampled and processed PBMC68k dataset:
<https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc68k_reduced.html>.
The original study is Zheng et al. 2017:
<https://doi.org/10.1038/ncomms14049>.

The target artifact is not a full-transcriptome donor study. Scanpy documents
that it retains a reduced processed matrix, `bulk_labels`, and gene rankings
based on those labels. Those labels define the 128-cell CD8/NK task frame and
are used for stratified analysis. The analysis therefore concerns evidence in
this specific, label-informed rendered feature panel. It does not interpret an
absent token as biological non-expression.

## Frozen operational modules

The parent set was present in
`signal/single_cell/build_cd8t_nk.py` at commit
`20f94201c60878b83dbae182b8447248666ede29`, before the outputs in this
experiment.

Executed TCR/CD8 module:

`CD3D CD3E CD3G CD8A CD8B`

Executed shared cytotoxic-effector module:

`GNLY NKG7 PRF1 GZMA GZMB GZMH GZMK CTSW CCL5 FGFBP2 XCL1 XCL2`

Deferred NK-receptor/identity candidate:

`KLRD1 KLRF1 KLRC1 KLRC2 NCAM1 FCGR3A NCR1`

Excluded ambiguous or broad state/adaptor genes:

`CD247 KLRB1 IL7R TYROBP CST7 SPON2`

These are operational marker categories, not purified pathways. The
receptor/identity candidate is not executed: `KLRD1`, `KLRF1`, `NCAM1`, and
`NCR1` are absent from the matrix; `KLRC1` and `KLRC2` never enter a selected
top-50 sentence; and only `FCGR3A` appears (10/128 cells: six CD8 and four NK).
There is no two-gene support. A
receptor-versus-effector claim would therefore be invalid on this artifact.
The distinction between NK lineage and cytotoxic-effector programs is
biologically motivated by single-cell NK studies, including Crinier et al.
2018 (<https://doi.org/10.1016/j.immuni.2018.09.009>).

## Within-frame label-blind common-support population

After `bulk_labels` define the CD8/NK task frame, every framed cell is screened
using only its ranked input genes and the two frozen executed modules. Within
that frame, target labels do not select common-support inclusion, a module, or
a mask. PBMC3k class labels are used only in the pre-existing source-side
neutral-control filter described below. Thus “label-blind” applies to
common-support selection and module assignment within the target task frame,
not to task framing, control construction, or analysis.

The primary population contains all cells with at least one top-50 TCR/CD8 hit
and at least one top-50 cytotoxic-effector hit:

- 65 cells total;
- 55 annotated CD8 and 10 annotated NK;
- 37 `CD8+ Cytotoxic T`;
- 18 `CD8+/CD45RA+ Naive Cytotoxic`;
- 10 `CD56+ NK`.

The 10 NK-labeled cells are deliberately mixed-evidence cells because
common-support inclusion requires a TCR/CD8-category token. Their T targets are
`CD3E` in six cells, `CD3D` in three, and `CD8B` in one. They may include
sorting contamination or doublets and are not representative of the broader
NK population.

The highest-ranked hit from each module is masked (`k=1`). Higher doses are not
run because two-module common support falls to 36 cells at `k=2` and nine at
`k=3`, including only two and one NK cells. No dose-response claim is planned.

## Matched controls and inputs

For each cell and each module, a neutral gene is matched separately within the
same 50-gene sentence using expression rank, log-normalized expression,
global top-50 prevalence, and token length. All 30 frozen parent markers and
genes with an absolute PBMC3k CD8-versus-NK top-50 prevalence gap above 0.10
are excluded from the control pool. Independent matching selects the same
neutral gene for both modules in 11/65 cells. That shared neutral is a valid
common input and is queried once per prompt form; its raw response is reused
for both logical module comparisons rather than treated as two stochastic
replicates.

Each cell has five condition-labeled inputs:

1. unmasked;
2. TCR/CD8 module mask;
3. TCR/CD8-specific matched-neutral mask;
4. cytotoxic-effector module mask;
5. cytotoxic-specific matched-neutral mask.

These comprise five unique gene sequences in 54 cells and four in the 11 cells
with a shared neutral. The separately matched controls are required because
the two module hits occupy different ranks; they need not be different when
the same gene is the best match for both. All masks replace exactly one token
with `MASKED_GENE`.

## Prompt factorial and execution

Each input is crossed with four forms:

| form | answer order | queried probability |
|---|---|---|
| `ab_pa` | CD8 / NK | P(CD8) |
| `ab_pb` | CD8 / NK | P(NK) |
| `ba_pa` | NK / CD8 | P(CD8) |
| `ba_pb` | NK / CD8 | P(NK) |

P(NK) is converted to P(CD8) as `1-P(NK)`. There are
65 cells × five condition labels × four forms = 1,300 logical condition-form
observations. The 44 duplicated neutral condition-form prompts are executed
once and reused, yielding 1,256 unique API calls. Unique calls are
contemporaneously randomized under the frozen plan hash and use deterministic
decoding. Raw strings, assignment mappings, parse status, timestamps, model
ID, prompt hashes, input hashes, preregistration hash, plan hash, and execution
code hash are checkpointed.

Every checkpointed probability and alignment is recomputed from the raw string
during analysis. Confirmatory analysis aborts unless all 1,256 unique responses
satisfy the exact-output parser; an invalid response is not imputed as 0.5.

## Primary estimand and gate

Let \(p^A\) be the probability aligned to CD8. For cell \(i\), module \(m\),
and prompt form \(f\), define

\[
r_{imf}=p^A_{i,\mathrm{neutral}(m),f}
        -p^A_{i,\mathrm{mask}(m),f}.
\]

Average the four forms within cell, then give annotated CD8 and NK cells equal
weight:

\[
\theta_m=\frac{1}{2}
\left\{E[r_{im}\mid CD8]+E[r_{im}\mid NK]\right\}.
\]

Interpretation:

- \(\theta_T>0\): TCR/CD8 tokens push output toward CD8;
- \(\theta_C<0\): cytotoxic-effector tokens push output toward NK.

The primary mechanistic conclusion is conjunctive and passes only if:

1. \(\theta_T>0\) with a one-sided Welch-Satterthwaite `p<0.05`;
2. \(\theta_C<0\) with a one-sided Welch-Satterthwaite `p<0.05`; and
3. the CD8-specific and NK-specific mean effects both have the expected sign
   for both modules.

The first two components form an intersection-union test. Its reported p-value
is the larger component p-value; no additional multiplicity correction is
applied to that conjunctive test. The third component is a directional
heterogeneity guard and does not receive a separate p-value. Report the
separation \(\psi=\theta_T-\theta_C\) with a two-sided 95% interval.

Inference uses a one-sided Welch-Satterthwaite test for the linear contrast
\(\frac{1}{2}(\bar r_{CD8}+\bar r_{NK})\) and 20,000
class-stratified paired cell-bootstrap draws for intervals. Cells are the
inferential units; prompt forms are repeated measurements. This is
cell-sampling inference inside one donor and is not donor-level uncertainty.

## Secondary controls

For each module, report:

\[
a_{im}=p^A_{i,\mathrm{unmasked}}-p^A_{i,\mathrm{mask}(m)}
\]

and

\[
g_{im}=p^A_{i,\mathrm{unmasked}}-p^A_{i,\mathrm{neutral}(m)}.
\]

The adjusted primary effect is \(a_{im}-g_{im}=r_{im}\). These quantities
distinguish module removal from generic one-token deletion; they do not make
the unmasked input a randomized biological baseline.

The strict matching sensitivity retains cells whose two module-specific
matches both have rank distance at most 10 and expression distance at most
1.0. Annotation subtype, masked-gene, and input-condition metrics are
descriptive only and do not support individual-gene claims.

## Prompt-robustness gate

For each module, calculate:

- answer-order interaction: AB effect minus BA effect;
- queried-target interaction: P(A) effect minus P(B) effect.

The result is described as prompt-robust only if, for both modules:

1. both interaction 95% intervals lie wholly inside `±0.03`; and
2. all four form-specific effects have the expected sign.

Failure of this gate means magnitude-level prompt invariance is not
established; it is not itself proof of a nonzero interaction.

## Claim boundary

A passed primary gate supports opposing token-level evidence use on an
equal-class-weighted average in Haiku 4.5 on one atypical common-support subset
of one external PBMC cohort. It would show that the model treats the
highest-ranked TCR/CD8-category token as CD8 evidence and the highest-ranked
frozen cytotoxic/NK-enriched marker-category token as NK evidence, relative to
their separately matched neutral deletions. The cytotoxic category is not a
claim that cytotoxicity is biologically NK-specific.

This is a paired two-category single-token ablation crossed with a four-form
prompt factorial. It is not a biological 2×2 module factorial: there is no
combined TCR/CD8-plus-cytotoxic mask, so the design cannot estimate module
interaction, additivity, or mediation. Target genes also differ across cells
and classes. The result concerns the frozen highest-ranked-member masking
policy, not every module member, a homogeneous pathway, or an individual gene.

It would not establish:

- an NK-receptor mechanism;
- representativeness of the mixed-marker NK subset;
- mediation of the earlier combined marker-mask effect;
- a biological gene knockout or pathway perturbation;
- hidden-state activation causality;
- latent biological knowledge;
- donor, study, or model-family generalization; or
- a physical or universal law.

This is a locally hashed preregistration, not an externally authenticated
timestamp.
