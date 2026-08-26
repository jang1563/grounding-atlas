# GSE96583 held-out GNLY/NKG7/CCL5 factorial preregistration

Freeze date: 2026-07-30  
Analysis ID: `gse96583-sentinel-factorial-holdout-v1`  
Model: `claude-haiku-4-5-20251001`  
Status: inputs built and audited; no model calls made

This is a local content-hash freeze made before execution, not an externally
authenticated timestamp or public preregistration service.

## Question and prospective boundary

The preceding donor-aware experiment localized most of its descriptive
cytotoxic-token leverage to `GNLY`, while `CCL5` was often weak. This
factorial asks a narrower, discriminating question:

> In held-out cells where `GNLY`, `NKG7`, and `CCL5` coexist in the rendered
> top 50, does the model use a sparse `GNLY` sentinel, or do `NKG7` and `CCL5`
> add material NK-directed leverage beyond `GNLY`?

The hypothesis was generated after inspecting the preceding experiment.
However, every one of its 56 queried cells is excluded before the new
eligibility gate and sampling step. The present eight cells and all multi-token
mask combinations are unqueried at freeze time. This is therefore a
prospective held-out-cell test within the same cohort and donors, not an
independent donor, cohort, assay, or model replication.

The intervention erases rendered gene names by replacing them with the frozen
`MASKED_GENE` token. It is causal only at this text interface. It is not a
biological gene perturbation, pathway ablation, hidden-state intervention,
proof of stored knowledge, causal activation-gap test, or physical-law test.

## Authenticated source and held-out sample

The source is the control arm of batch 2 from Kang et al., GEO
[GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583),
DOI [10.1038/nbt.4042](https://doi.org/10.1038/nbt.4042): eight SLE donors,
cultured for six hours without IFN-beta. These are not healthy controls.

The preprocessing, source hashes, CD8/NK task frame, common symbol universe,
normalization, and top-50 rendering are inherited unchanged from
`GSE96583_CONTEXT_MODULE_PREREG.md`. Deposited CD8/NK labels were derived from
the same expression measurements. They define only the upstream task frame
and are descriptive after selection; they do not enter the holdout exclusion,
triple-positive gate, sampling hash, subset assignment, target mask, or
control match.

The frozen parent exclusion is all 56 base-cell barcodes in
`signal/single_cell/gse96583_cd8_nk_module_replication.csv`, SHA-256
`f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321`.
The canonical exclusion-record hash is
`35d2ba1aa04f7e740f037d30b677001b5b05900e75a35da1b82fc2dd1f036950`.

Eligibility requires all three frozen names—`GNLY`, `NKG7`, and `CCL5`—in
the rendered top 50. Support before and after excluding the parent cells is:

| donor | all triple-positive | unused triple-positive |
|---|---:|---:|
| 101 | 39 | 37 |
| 107 | 14 | 11 |
| 1015 | 135 | 132 |
| 1016 | 194 | 193 |
| 1039 | 8 | 5 |
| 1244 | 58 | 55 |
| 1256 | 161 | 157 |
| 1488 | 69 | 66 |

Within each donor, select the eligible unused cell with the lowest
`SHA256(parent_CSV_SHA256|donor_id|barcode)`, using the barcode only as a
deterministic tie break. The salt is the immutable parent CSV hash, which was
frozen before this hypothesis; an investigator-chosen analysis name is not
used to re-randomize the holdout. This is deterministic hash selection, not a
claim of population-random sampling. Exactly one cell is selected per donor.
The selected annotation composition, seven NK and one CD8 T cell, is
descriptive only. There is zero overlap with the parent 56 cells. No donor or
cell may be replaced after model calls.

Triple-positive selection deliberately enriches a cytotoxic context and does
not represent the full CD8/NK population. With one cell per donor, the
experiment estimates recurrence across donors at one frozen cell each; it
cannot estimate within-donor cell heterogeneity.

## Nested target and matched-control lattice

Let `G=GNLY`, `N=NKG7`, and `C=CCL5`. Each selected cell receives all seven
nonempty subsets:

`G`, `N`, `C`, `GN`, `GC`, `NC`, and `GNC`.

The three targets are jointly assigned once per cell to three distinct
PBMC3k reference-prevalence-balanced controls by the frozen linear-sum cost
over rank,
log-normalized expression, framed-cell top-50 prevalence, and token length.
All 24 matches are feasible without relaxation; the maximum rank distance is
13. Controls exclude the 30 parent markers and have a PBMC3k absolute
CD8-versus-NK prevalence gap at most 0.10.

“Reference-prevalence-balanced” does not mean biologically or lexically
neutral. Controls can carry immune semantics, including HLA names, and their
target-frame prevalence distances can be appreciable. The absolute target
surface below is therefore required alongside the matched-control surface;
control identities and distances are reported rather than hidden.

Every subset uses the corresponding subset of this fixed one-to-one control
assignment. Thus if target subset \(S_1\) is nested in \(S_2\), its neutral
counterpart is nested in exactly the same way. Target and control masks always
contain the same number of `MASKED_GENE` tokens. Positions are retained; no
rank-51 backfill or reranking occurs.

Each cell therefore has 15 distinct inputs: one unmasked input, seven target
masks, and seven matched-control masks. Eight cells crossed with the four
frozen answer-order by queried-target forms yield exactly 480 unique model
calls. Calls are deterministically shuffled. No response is reused from the
parent experiment.

Frozen construction artifacts:

- builder:
  `signal/single_cell/build_gse96583_sentinel_factorial.py`,
  SHA-256
  `9336e633763b91c8d0c983d75b67004da9ee6f681c1b4f7dd2d4ba92b07f8992`;
- CSV:
  `signal/single_cell/gse96583_sentinel_factorial.csv`,
  SHA-256
  `673db8c8bcd6ba923e62891de6cd5f04f97967706ac3ade8a6e44ad2d14a4b95`;
- manifest:
  `signal/single_cell/gse96583_sentinel_factorial.manifest.json`,
  SHA-256
  `9f0035469c81852b11e2a36651b7892bf2dad4d30f8049b37cd1f655ca9bf0c4`;
- input-sentence plan:
  SHA-256
  `8e2e22a3e612f3383f6466d79fb6978e1c1bd0117cae87c6f0edff74f0026933`.

The deterministic hash-shuffled request-plan SHA-256 is
`db5af60a5142c6f79cbac7abdba85520077efd75fab732db0bfda80f6843a20e`.

## Prompt policy and raw-response contract

The canonical confirmatory form is `ab_pa`: list CD8 first, NK second, and
directly request P(CD8). It is chosen before these calls because the previous
experiment rejected prompt invariance and because this form needs no
probability complement. `ab_pa` is the pre-existing first canonical form.
Selection occurs after observing parent prompt non-invariance, so conclusions
remain canonical-prompt contingent unless the registered robustness gate
passes.

The other three frozen forms (`ab_pb`, `ba_pa`, `ba_pb`) are run for every
condition as secondary prompt-surface analyses. They cannot rescue or overturn
the canonical-form decision. A four-form average is reported only as a
secondary standardized interface ensemble, not as evidence of prompt
invariance.

The exact system prompt is:

> You are a property predictor. Respond with ONLY a single decimal number
> between 0 and 1 (for example: 0.42). No words, no explanation, no analysis,
> no units. Your entire reply must be just the number.

Its SHA-256 is
`37aa5ee3d2b27cd4ae670f3a064e89de1d25013f939951d3993a6732acd5dacb`.
The exact user template is:

> These are the most distinctively expressed genes of one human immune cell
> (PBMC), highest first:  
> `{genes}`  
> Is it a `{first}` or a `{second}`? Reply with ONLY a number between 0 and 1
> = the probability that it is a `{target}`.

Its SHA-256, including the frozen newline characters and without Markdown
quoting, is
`8da04371ae26646b51f22aaead0815b3625b62be6c537a1788b298f67064433a`.
The class strings are exactly `CD8+ T cell` and `NK cell`; the form definitions
are frozen in the request plan.

The model receives temperature 0 and `max_tokens=16`. Analysis aborts unless
all 480 raw outputs parse exactly as one number in \([0,1]\). P(NK) responses
are aligned to P(CD8) as `1-P(NK)`. Each checkpoint record binds the raw
output and recomputed probability to the model request and returned model,
response ID, token usage, stop reason, retry/fallback state, condition,
subset, target/control names, donor and cell, prompt and input hashes, call
plan, CSV, manifest, parent exclusion, preregistration, runner, parsing
helper, runtime/dependency manifest, decode payload, and UTC timestamps.
The final non-circular runtime manifest is written and audited only after the
runner is complete; it binds the final runner and helper hashes to the already
frozen preregistration and call plan. No code or registered artifact may
change between that no-call freeze and execution.

## Confirmatory estimands

Let \(p^A_{df}(X)\) denote P(CD8) for donor \(d\), prompt form \(f\), and
input state \(X\). For a nonempty target subset \(S\), let \(q(S)\) be its
fixed matched-control subset. Define

\[
a_{df}(S)=p^A_{df}(\varnothing)-p^A_{df}(S\ \mathrm{masked}),
\]

\[
h_{df}(S)=p^A_{df}(\varnothing)-p^A_{df}(q(S)\ \mathrm{masked}),
\]

\[
r_{df}(S)=p^A_{df}(q(S)\ \mathrm{masked})
          -p^A_{df}(S\ \mathrm{masked})
          =a_{df}(S)-h_{df}(S).
\]

Negative \(r\) means that erasing the target names moves P(CD8) upward more
than erasing their matched controls; equivalently, the intact target names
had greater NK-directed leverage relative to matched masking.

The confirmatory vectors use only `ab_pa`; because there is one held-out cell
per donor, its eight cell values are the eight donor values. Define:

\[
A_d=r_d(G),\qquad T_d=r_d(GNC),\qquad
J_d=T_d-A_d,
\]

\[
U_d=r_d(NC),\qquad Q_d=T_d-U_d,\qquad
K_d=T_d-A_d-U_d.
\]

Here \(A\) is GNLY-alone leverage, \(T\) is full triple leverage, \(J\) is
the incremental NKG7/CCL5 leverage after GNLY, \(U\) is their leverage
without GNLY, \(Q\) is GNLY leverage on the NKG7/CCL5-masked background, and
\(K\) is a model-score-scale nonadditivity. The practical margin is frozen at
\(\delta=0.03\) model-reported probability-score units. The numerical value
is inherited from the parent sham/prompt tolerance and is prospectively
repurposed here as the smallest material endpoint difference. Because the
model outputs are coarse and uncalibrated, this is not three percentage
points of biological risk or calibrated probability. A secondary
`±0.05`-unit sensitivity is reported but cannot rescue the primary rule.

## Donor inference

For every registered vector, report all eight donor values, their unweighted
mean, Student-\(t_7\) 95% interval, the relevant one-sided Student-\(t_7\)
p-value, exact one-sided Rademacher sign-flip p-value over all \(2^8\)
assignments, donor sign count, and all eight leave-one-donor-out means.
“Exact” here means exact enumeration conditional on donor-effect sign
symmetry; it is not distribution-free inference. Also report the one-sided
exact binomial sign-test p-value. The 7/8 registered sign rule itself implies
a one-sided binomial p-value at most \(9/256=0.03515625\).

A material negative component \(x<-\delta\) passes only if:

1. its entire 95% interval is below \(-\delta\);
2. the exact one-sided sign-flip test on \(x+\delta\) has `p<0.05`;
3. at least 7/8 donor values are below \(-\delta\); and
4. all leave-one-donor-out means are below \(-\delta\).

Practical equivalence to zero passes only if:

1. its entire 95% interval lies strictly inside
   \([-\delta,+\delta]\);
2. both exact shifted sign-flip tests pass at 0.05:
   \(x+\delta>0\) and \(x-\delta<0\); and
3. every leave-one-donor-out mean lies strictly inside the margin.

The conventional 90% equivalence interval and Student-\(t\) TOST p-values are
also reported. The registered 95% plus exact criterion is deliberately more
conservative and maintains a nonoverlapping region between equivalence and a
material negative effect. Failure of equivalence is inconclusive, not evidence
for distribution.

## Decision hierarchy

1. **Anchor gate.** Both matched-control effects `r(G)` and `r(GNC)` and both
   absolute target-mask effects `a(G)` and `a(GNC)` must pass the
   material-negative rule. Its intersection-union exact p-value is the
   largest of the four shifted exact p-values. If this gate fails, neither
   sparse nor distributed wording is allowed.
2. **Endpoint discriminator.**
   - If both `r(GNC)-r(G)` and `a(GNC)-a(G)` are equivalent to zero, report
     that GNLY captures the triple-mask endpoint within 0.03 model-score
     units.
   - If both increments pass the material-negative rule, report a material
     joint NKG7/CCL5 residual beyond GNLY, supporting a multi-token output
     surface.
   - Otherwise classify the endpoint as inconclusive/hybrid.
3. **Strong sparse-GNLY sentinel.** This wording additionally requires both
   `r(GNC)-r(NC)` and `a(GNC)-a(NC)` to pass the material-negative rule and
   both the matched-control and absolute-target versions of all six vectors
   below to be equivalent within `±0.03`:

\[
r(N),\quad r(C),\quad r(NC),\quad
r(GN)-r(G),\quad r(GC)-r(G),\quad r(GNC)-r(G).
\]

For the absolute version, replace every \(r\) with \(a\). This is a
conjunctive intersection-union gate. Endpoint equivalence by itself does not
establish the full sparse surface.
4. **Distributed localization.** Only after both the matched-control and
   absolute nested increments are material negative, calculate matched-control
   conditional Shapley allocations:

\[
\phi_{N|G}=\frac12\{[r(GN)-r(G)]+[r(GNC)-r(GC)]\},
\]

\[
\phi_{C|G}=\frac12\{[r(GC)-r(G)]+[r(GNC)-r(GN)]\}.
\]

They sum to matched-control `rJ`. Calculate the two corresponding absolute
target-mask allocations by replacing every \(r\) with \(a\); those sum to
absolute `aJ`. An individual NKG7 or CCL5 target-use localization requires
its matched-control and absolute allocations both to pass the
material-negative rule, with Holm correction across all four shifted exact
tests. If relative and absolute allocations disagree, or no individual pair
passes, report only a joint non-G residual and label the four allocations
descriptive.

## Controls and secondary analyses

For every subset in a registered gate, report absolute target-mask effects
\(a(S)\), generic matched-mask shams \(h(S)\), and the registered nested
increments. Sham equivalence uses the same `±0.03` contract. Sham failure
does not erase a result that passes both the absolute and relative gates, but
it prohibits “isolated deletion-specific effect” language. A relative result
without concordant absolute target movement is comparator-driven and cannot
pass the primary hierarchy.

For all four prompt forms, report every subset surface, the endpoint
classification, answer-order and queried-target interactions, and the
uniform four-form average. For any donor vector \(v_f\), define the
answer-order interaction as
\([v_{ab,pa}+v_{ab,pb}]/2-[v_{ba,pa}+v_{ba,pb}]/2\), and the queried-target
interaction as
\([v_{ab,pa}+v_{ba,pa}]/2-[v_{ab,pb}+v_{ba,pb}]/2\), after all probabilities
are aligned to P(CD8). “Prompt robust” requires each of the four forms,
analyzed separately with the same donor rules, to produce the same anchor and
endpoint classification as canonical `ab_pa`, and requires the 95%
interaction intervals for `r(G)`, `r(GNC)`, `rJ`, `a(G)`, `a(GNC)`, and
`aJ` to lie within `±0.03`. Otherwise the result is explicitly
canonical-prompt contingent.

Report the model-score-scale subset nonadditivities

\[
\beta_{GN}=r(GN)-r(G)-r(N),\quad
\beta_{GC}=r(GC)-r(G)-r(C),\quad
\beta_{NC}=r(NC)-r(N)-r(C),
\]

\[
\beta_{GNC}=r(GNC)-r(GN)-r(GC)-r(NC)+r(G)+r(N)+r(C)
\]

descriptively. They are mask-surface interactions, not biological gene
epistasis. Deposited-label stratification is descriptive only and cannot
rescue donor-level inference.

## Interpretation boundary

A positive result supports only a donor-recurrent, held-out-cell
text-interface dependence on these three rendered names in one model and one
SLE control cohort. “Sentinel” means dominance among this frozen three-token
surface. “Distributed” means the joint NKG7/CCL5 mask materially changes the
output beyond GNLY under matched controls.

It does not establish a homogeneous cytotoxic pathway, gene or pathway
causality, calibrated cell-type correctness, annotation truth, knowledge
absence versus failed access, latent stored knowledge, a hidden-state
activation route, an invariant mathematical law, or a physical law.
