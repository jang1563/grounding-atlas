# Coherent binary readout and orthogonal gain/activation-gap identification design

Status: design revision; **confirmatory execution NO-GO** until the Phase-0 cohort
census, state-label lock, full-hierarchy power simulation, and full preregistration
are complete. The separately preregistered, outcome-exposed development Level-0
run is authorized for readout engineering only; updated 2026-08-02; no model
execution reported here

Primary model class: open-weight causal language model with first-token logits and
residual-stream hooks

Confirmatory unit: donor

## Decision in one sentence

The tested scalar probability interface in the GSE96583 Haiku prompt family is
not valid enough to support another biology or activation claim. The next run
must first pass a **Level-0 coherent binary-readout gate** using two logits from
one forward pass; only then may it test
lineage versus cytotoxic-state use in an independently and orthogonally labeled
cohort.

## Primary biology question

> **After holding effector state and other NK cues fixed, does GNLY change
> NK-vs-CD8 lineage evidence or only cytotoxic-state evidence, and can
> independently verified marker knowledge be causally patched into the cell-level
> decision?**

> **효과기 상태와 다른 NK 단서를 고정했을 때, GNLY는 NK-vs-CD8 계통 증거를
> 바꾸는가, 아니면 세포독성 상태 증거만 바꾸는가? 그리고 독립적으로 검증한
> 마커 지식을 세포 수준 판단에 인과적으로 패치할 수 있는가?**

These are three questions, not one: what the output score measures, what rendered
gene-name interventions do, and what a hidden-state intervention does.

## Why Level 0 is mandatory

The eight-donor parent experiment rejected prompt robustness
([summary](../results/benchmark/single_cell/donor_context_factorial/claude-haiku-4-5-20251001.md),
[raw](../results/benchmark/single_cell/donor_context_factorial/claude-haiku-4-5-20251001_raw.jsonl)).
Pairing the two direct queries with the same answer order gives 560 pairs: mean
`|P(A)+P(B)-1| = 0.291`, and only `29.6%` are within `0.03` of complementarity.
The held-out sentinel factorial also failed its anchor and prompt-robustness gates
([summary](../results/benchmark/single_cell/sentinel_factorial/claude-haiku-4-5-20251001.md),
[result](../results/benchmark/single_cell/sentinel_factorial/claude-haiku-4-5-20251001.json),
[raw](../results/benchmark/single_cell/sentinel_factorial/claude-haiku-4-5-20251001_raw.jsonl)).
Its 240 direct-query pairs have pooled mean absolute complement residual `0.492`,
with only `7.5%` within `0.03`.

Those failures do not show that the model lacks biological information. They show
that separately elicited decimal answers cannot be treated as one coherent binary
distribution. Averaging incompatible prompt forms cannot repair that measurement.

## Level 0: coherent open-weight binary readout

### One forward pass, two tokens

Use one frozen, symmetric prompt template for each task:

```text
Classify the cell using exactly one label.
{line_1}
{line_2}
Genes, highest expression rank first: {gene_sentence}
Label:
```

Each mapping line has the form `label {token} means {class}`. Choose two opaque
labels, provisionally `X` and `Y`, only after the frozen model tokenizer confirms
that each label including its leading-space convention is exactly one token at the
answer position. The two labels must have equal character length. Freeze token IDs,
model revision, model-config vocabulary size, chat template, dtype, tokenizer, prompt
bytes, tokenized inputs, donor/source-item topology, and the complete request-ID
registry before inspecting outcomes. The pre-forward plan also binds the fixture,
fixture manifest, preregistration, runner, candidate margin lock, and every expected
record. Natural-language probability generation is forbidden.

Cross two factors for every input:

1. **class-line order:** NK mapping line first versus CD8 mapping line first;
2. **opaque remapping:** NK=`X`, CD8=`Y` versus NK=`Y`, CD8=`X`.

The same four renderings are used for the second readout, cytotoxic-high versus
cytotoxic-low. Only the class strings change. At the first answer-token position,
retain the unrounded logits for the two permitted tokens. After alignment to the
positive biological class, define

\[
\Delta_i=z_{i,+}-z_{i,-},\qquad
q_{i,+}=\frac{e^{z_{i,+}}}{e^{z_{i,+}}+e^{z_{i,-}}},\qquad
s_i=q_{i,+}-q_{i,-}=2q_{i,+}-1.
\]

`s` in `[-1,1]` is the primary normalized token-logit difference. Retain `Delta`
for saturation diagnostics. `q` is a constrained two-token conditional score, not
a calibrated biological probability or, without the format-adherence gate below,
an unconstrained native answer.

### Exact Level-0 gates

Let `o=+/-` denote class-line order and `m=+/-` opaque mapping. For each item,
after biological alignment, define

\[
O_i=\tfrac12(s_{++}+s_{+-}-s_{-+}-s_{--}),
\]

\[
R_i=\tfrac12(s_{++}+s_{-+}-s_{+-}-s_{--}),
\]

\[
I_i=\tfrac12\{(s_{++}-s_{+-})-(s_{-+}-s_{--})\}.
\]

`O`, `R`, and `I` are respectively order, remapping, and order-by-remapping
effects. Aggregate cells to a mean within donor first. Apply the following gates
separately to each readout and to each concrete, manifest-registered input family.
Do not pool target depletion, each matched-control depletion, target add-back,
matched add-back, patch, or control arms under broad family names.

1. **Extraction/coherence:** all logits are finite; both opaque labels are one
   token; every planned record exists once; every source item occurs under every
   registered readout and family; and
   `max |q_positive + q_negative - 1| <= 1e-6`. Any violation is an immediate
   `NO-GO`. The last equality is a software/extraction identity, not evidence of
   model calibration. Save the complete raw, pre-processor next-token vector for
   every call as a hashed float32 sidecar. An independent analyzer must reconstruct
   both retained logits, the full-vocabulary argmax, and log-sum-exp from that
   sidecar; selected-logit attestations alone are insufficient. In addition, the
   greedy first token over the full vocabulary must be one of the two permitted
   labels in at least 95% of item-form records globally, at least 95% within every
   readout by family, and 90% within every donor/readout/family. Record the two-label
   full-vocabulary probability mass descriptively. Failure is
   `READOUT_FORMAT_INVALID`.
2. **Order equivalence:** the donor-vector `O` has a 90% Student-t interval wholly
   inside `[-0.06,+0.06]` score units, both exact shifted sign-flip TOST tests have
   `p<0.05`, and all leave-one-donor-out means remain inside the margin.
3. **Remapping equivalence:** apply the identical rule to `R`.
4. **Interaction equivalence:** apply the identical rule to `I`.
5. **Item guardrail:** at least 95% of items have
   `max(s_o,m)-min(s_o,m) <= 0.20`; for every item whose four-form mean has
   `|mean(s)| >= 0.20`, all four forms must have the same sign.

The candidate `0.06` score margin equals `0.03` on the two-token `q` scale, but it
is not justified by the invalid decimal interface. Phase 0 must freeze it using
development-only negative-control renderings: the retained margin must be no larger
than 30% of the `0.20` minimum material effect and must contain the complete 90%
interval of the donor nuisance contrast. If `0.06` is too narrow, the design stops;
the margin is not widened after seeing confirmatory data. Code accepts stricter
thresholds only: the equivalence, range, and strong-item thresholds cannot exceed
`0.06`, `0.20`, and `0.20`; format and item-pass minima cannot be reduced below
their registered values.

### Implemented Phase-0 donor-power contract

Let `p` be the number of jointly simulated donor-level components. A development
matrix is eligible for covariance estimation only when it has at least
`max(8,p+2)` distinct development donors. Fewer donors stop with
`PILOT_COVARIANCE_UNSTABLE`; covariance is not estimated. After that size gate and
the development-only margin-qualification gate pass, estimate the complete joint
covariance with Ledoit-Wolf shrinkage. No confirmatory donor or outcome may enter
the mean, variance, or correlation estimates.

For a frozen unsigned 64-bit seed and at least 10,000 replicates, generate exactly
one joint `B x 20 x p` simulation cube with NumPy `PCG64DXSM`. Candidate sizes are
the complete ordered set `n=12,...,20`, and every candidate reuses the corresponding
prefix of that same cube. Independent resimulation by candidate size is forbidden;
the common random numbers make candidate comparisons paired. Development observations
must lie inside their registered finite component supports. Simulated support is
then checked both globally across the pooled cube and separately for every component;
the pooled fraction **and every component fraction** outside support must be no
larger than the frozen maximum. Any failure is `SIMULATION_MODEL_OUT_OF_SUPPORT`.

Within each replicate, apply the exact registered donor gate to every component and
form each scenario by logical conjunction of its component masks. Report every
component and every scenario, including the required full conjunction. With `R`
required scenarios and nine candidate donor counts, use a Bonferroni-adjusted
one-sided Clopper-Pearson lower confidence bound with
`alpha_MC=0.05/(9R)`. A candidate qualifies only if every required scenario has a
lower bound of at least `0.80`; the first qualifying candidate is reported. The
simulation must not place an alternative exactly on a decision boundary, where
rejection probability is approximately the test size. Freeze distinct design
alternatives before simulation: `0` for nuisance-equivalence gates, `+0.30` for a
`+0.20` material boundary, `+0.15` for a `+0.10` boundary, and `+0.075` for a
`+0.05` boundary.

The implemented schema is explicitly `power_scope=level0_only`. It contains the
Level-0 `O/R/I` nuisance-equivalence components for every readout by concrete input
family and may populate `candidate_n_selected` only. It must always leave
`selected_n_conf=null`; neither a selected candidate nor a Level-0 development pass
authorizes confirmatory model execution. Final `n_conf` requires a future,
authenticated power configuration covering the complete primary-claim hierarchy:
all directional, equivalence, recurrence, orthogonal-cohort/state, multiplicity,
and stopping gates in the final preregistration. At that future stage define
`r_conf=ceil(0.80*n_conf)`, enumerate all `2^n_conf` sign assignments, and report
the exact donor sign test and donor-mean interval alongside each directional result.
Rademacher sign-flip p-values are exact only under donor-effect sign symmetry; this
assumption is reported, not implied by enumeration. TOST uses two one-sided shifted
tests at `alpha=0.05`, and equality to a margin fails strict equivalence. If the full
hierarchy needs more than 20 donors, stop and write a new inference preregistration;
do not call an approximated sign-flip p-value exact.

Level 0 passes only if every numbered condition passes for both biological
readouts. There is no majority vote and no preferred-form rescue. Only after a pass
may the four forms be averaged as the order/remapping-standardized score
`s_bar=(s_+++s_+-+s_-++s_--)/4`.

## Phase 0 and the independent orthogonal 2x2 cohort

No **confirmatory** model execution is authorized until an accession-level census
proves that the candidate cohort satisfies every donor, quadrant, marker, and cell-
count rule below. Public cohorts and nonoverlapping development donors may be used
for Level-0 engineering, variance estimation, and power simulation behind a
permanent item/donor firewall; their prompts, activations, and outcomes can never be
promoted into confirmation.

If no public cohort passes, the next step is prospective paired RNA plus index-sorted
protein acquisition, not relaxation of the labels. Phase 0 also freezes one state
construct, completes the development-only power simulation, and publishes the
donor/quadrant count table before confirmatory prompts or model outputs are generated.

### Accession census: no qualifying public cohort

The 2026-08-02 primary-source census found **no public accession that passes the
frozen gate**:

| candidate | useful property | disqualifying condition | permitted use |
|---|---|---|---|
| Hao et al. [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) | same-cell RNA plus a 228-antibody CITE panel | 8 donors; ordinary surface CD107a is not a controlled degranulation label; no intracellular GZMB state label | Level-0/pipeline pilot only |
| Stephenson et al. [E-MTAB-10026](https://www.ebi.ac.uk/biostudies/studies/E-MTAB-10026) | over 780,000 cells from 130 individuals with same-cell surface-protein measurements | functional flow is a separate, non-barcode-linked T-cell aliquot; no same-cell NK/CD8-by-state labels | lineage/readout development only |
| Lawlor et al. [PRJEB40448](https://www.ebi.ac.uk/ena/browser/view/PRJEB40448) | paired RNA plus 39 surface proteins under stimulation | 10 donors; CD69/CD25 are generic activation and the reported cytotoxic score is RNA-derived | controlled-activation pilot only |

Therefore `confirmatory cohort = NONE` and confirmatory execution remains `NO-GO`. The
confirmatory experiment requires prospective acquisition from at least
`4+n_conf` donors: at least four development donors plus `n_conf` donor-held-out
confirmatory donors, with same-cell RNA, the frozen surface lineage panel, and
intracellular granzyme-B with the required controls. Public cohorts may develop
software and the Level-0
interface, but none may be promoted into the confirmatory biological test.

Use a cohort independent of GSE96583 and of all hypothesis-generating cells. The
preferred acquisition is donor-resolved CITE-seq or index-sorted PBMC with paired
RNA and protein/functional measurements. RNA gene symbols must not define either
axis.

The four quadrants are:

| orthogonal lineage | orthogonal state | shorthand |
|---|---|---|
| CD8 T | cytotoxic-low | T/low |
| CD8 T | cytotoxic-high | T/high |
| NK | cytotoxic-low | NK/low |
| NK | cytotoxic-high | NK/high |

After live-singlet and CD45-positive gates, exclude myeloid (`CD14`/`CD33`) and B
(`CD19`/`CD20`) cells. Lineage is frozen from protein gates: CD3- and TCRalpha-beta-
positive, CD8-positive T cells versus CD3- and TCRalpha-beta-negative,
CD56- and/or CD16-positive NK cells. Ambiguous, NKT, double-positive, and
double-negative events are excluded by rules frozen before RNA inspection.
Cytotoxic state is one locked construct: intracellular granzyme-B protein abundance,
with high/low thresholds fixed from FMO/isotype and batch controls before RNA
inspection. Perforin, surface CD107a abundance without a functional degranulation
assay, and RNA signatures are not interchangeable substitutes. `GNLY`, `NKG7`,
`CCL5`, and all RNA values are forbidden from lineage and state labels.

Use nonoverlapping development donors for layer/direction selection. Freeze exactly
`n_conf` confirmatory donors by accession/sample ID hash. For the base 2x2 analysis,
select 32 QC-passing cells per quadrant and donor by a label-stratified hash after
all protein gates are frozen. A donor lacking any quadrant is ineligible before
model execution; donors and cells cannot be replaced afterward. Render the same
top-50 RNA-symbol sentence without values.

For natural GNLY deletion, require at least eight cells per donor in both
cytotoxic-high lineage strata with `GNLY` naturally present in the top 50. Select
eight by hash. A lineage-axis gain/activation claim uses those two opposite-lineage
strata. A cytotoxic-state gain/activation claim additionally requires eight naturally
GNLY-positive cells in both high and low state strata within each lineage. If the
relevant support gate fails for any confirmatory donor, that axis is `NO-GO`; a
synthetic GNLY insertion may be reported only as a separately labeled
lexical-sensitivity experiment and cannot rescue the gain/activation claim.

As a task-informativeness control, a simple baseline receives exactly the rendered
top-50 ranked-symbol representation—not the full count matrix—with the registered
intervention genes excluded, and is trained only on development donors. Each axis
must reach confirmatory donor-mean AUROC at least `0.75`, with the 95% interval lower
bound above `0.65`. Failure means that this rendering is not an adequate test
substrate, not that the model lacks knowledge.

## Base 2x2 estimands

Let `sL` be positive toward NK lineage and `sC` positive toward cytotoxic-high.
Average cells within quadrant and donor before any group inference. For donor `d`,

\[
L_d=\tfrac12[(sL_{NK,H}+sL_{NK,L})-(sL_{T,H}+sL_{T,L})],
\]

\[
C_d=\tfrac12[(sL_{NK,H}+sL_{T,H})-(sL_{NK,L}+sL_{T,L})],
\]

\[
LC_d=\tfrac12[(sL_{NK,H}-sL_{NK,L})-(sL_{T,H}-sL_{T,L})].
\]

The lineage readout separates lineage from state only if `L` is materially positive
above `+0.20` score units while `C` and `LC` are equivalent within `+/-0.10`.
Apply the transposed definitions to `sC`: cytotoxic state must be material above
`+0.20`, while lineage leakage and interaction are equivalent within `+/-0.10`.
Directional claims require a 95% t interval beyond the material boundary, exact
shifted sign-flip `p<0.05`, at least `r_conf` donor effects beyond the boundary,
and all leave-one-donor-out means beyond it. Equivalence requires a 90% interval
inside the margin, both exact shifted sign-flip TOST tests at `p<0.05`, and all
leave-one-donor-out means inside the margin.

Classify the base result prospectively. **Separated** requires the intended main
effect to pass while leakage and `LC` are equivalent. **Materially conflated**
requires the intended main effect to pass and at least one leakage or interaction
contrast to be materially nonzero beyond `+/-0.10` under the same donor rule and
multiplicity plan. **Inactive** requires the intended effect, leakage, and
interaction all to pass equivalence. **Reversed/misdirected** requires either a
material intended effect in the wrong direction or an equivalent intended effect
with material leakage/interaction. Every other pattern is **INCONCLUSIVE**. A failed
gate alone is not positive evidence of conflation, absence, or a gap.

## Three interventions that must remain distinct

### 1. Rendered-token depletion and text add-back

For a GNLY-positive cell, replace only the `GNLY` symbol with a tokenizer-verified
neutral nonce selected without running the model. The nonce must match the target
symbol's answer-context token count and UTF-8 byte count. Use an independently
selected matched nonce for every control symbol that has a different budget.
Preserve rank, punctuation, total token count, all other genes, and all
NK/effector cues; do not rerank or backfill. The prompt template contains a
fixed-width additional-evidence slot in **every** condition. That slot contains a
token- and byte-budget-matched neutral nonce when inactive and the registered
gene-plus-rank text when active. Full, depleted, target-add-back, and control-add-
back prompts must be byte-identical outside the two preregistered editable spans.
This same-cell pair therefore holds the orthogonal state and every non-GNLY cue
fixed without introducing a meaningful `MASKED_GENE` word or a length cue.

Register three controls per cell:

- deletion of a distinct, rank/expression/prevalence/token-length-matched gene;
- deletion of a frozen effector set (`NKG7`, `CCL5`, `GZMB`, `PRF1`) while GNLY
  remains intact;
- deletion of a frozen other-NK set (`KLRD1`, `FCGR3A`, `TYROBP`) while GNLY
  remains intact.

No control or nonce may be chosen using model outputs. A text add-back starts from
the GNLY-depleted prompt and supplies `GNLY` in the preallocated evidence slot with
its original rank. A matched-control add-back uses the same syntax and token/byte
budget. Automated tests must compare tokenizer IDs, byte lengths, and the exact
unchanged spans. This is a **causal prompt-token intervention**, not a gene
perturbation and not a knowledge patch.

For each donor and cytotoxic-high lineage stratum `l in {NK,T}`, estimate the
natural GNLY full-minus-depleted change separately:

\[
G^L_{d,l}=mean_l(sL_{full}-sL_{-GNLY}),\qquad
G^C_{d,l}=mean_l(sC_{full}-sC_{-GNLY}).
\]

Subtract the same-cell matched-deletion effect to obtain `GtildeL_{d,l}` and
`GtildeC_{d,l}`, but always report absolute and adjusted effects together. Define
the scaled GNLY-by-lineage heterogeneity contrasts

\[
J^L_d=\tfrac12(\widetilde G^L_{d,NK}-\widetilde G^L_{d,T}),\qquad
J^C_d=\tfrac12(\widetilde G^C_{d,NK}-\widetilde G^C_{d,T}).
\]

An axis is material only if its absolute and adjusted effects exceed `+0.10` in
**each** lineage stratum under the directional donor rule and its `J` contrast is
equivalent within `+/-0.06`. An axis is inactive only if absolute and adjusted
effects are equivalent within `+/-0.06` in each stratum and `J` is also equivalent.
Opposite-signed material stratum effects or a material `J` are reported as
**lineage-heterogeneous**, never averaged into inactivity. Holm-correct the two
axis-specific material families (`GtildeL`, `GtildeC`); equivalence still requires
both one-sided tests for every stratum and estimand. Classify only after these
stratum rules pass:

- **lineage cue:** `GtildeL` material and `GtildeC` equivalent;
- **state cue:** `GtildeC` material and `GtildeL` equivalent;
- **mixed/conflated cue:** both material;
- **inactive at this interface:** both equivalent;
- **lineage-heterogeneous:** a material stratum disagreement or interaction;
- **inconclusive:** every other pattern.

Target add-back must recover at least `0.10` on the same intended axis and exceed
matched add-back by `0.10`; the restored-versus-full difference must be equivalent
within `+/-0.06`. Add-back failure blocks reversible lexical-cue wording but does
not transform depletion into hidden-state evidence.

### 2. Independent marker-knowledge probe

Before cell-level outcomes, freeze two disjoint, source-cited banks independent of
the cohort:

1. a **direction-construction bank**, used only to derive the two relation-specific
   activation directions below; and
2. a **held-out probe bank** with at least 48 balanced fact/foil clusters covering
   T-lineage components, NK-lineage receptors, and shared cytotoxic effectors,
   including at least eight GNLY items from at least four distinct source/concept
   clusters.

Foils preserve gene-name frequency, sentence length, relation syntax, and answer
balance. Paraphrases are nested within a fact/gene cluster, not counted as new
independent facts. For non-GNLY category validation, hold out whole genes from the
construction bank. Because a GNLY-specific direction necessarily shares the gene
name, its probe items must instead hold out the exact source facts, biological
concepts, relation objects, and surface templates used for construction. No
construction item contributes to probe accuracy. The same Level-0 two-token
readout and four order-by-mapping renderings apply.

The probe passes only if cluster-weighted overall accuracy is at least `90%`, the
gene/fact-cluster bootstrap 95% lower bound is at least `80%`, every marker category
and the held-out GNLY subset are at least `80%`, every leave-one-gene/fact-cluster-
out accuracy remains at least `80%`, and its own order/remapping item guardrails
pass. The bootstrap resamples biological fact/gene clusters; a binomial interval
treating paraphrases as independent is forbidden. This shows **explicit lexical
marker-fact access** under the probe. It does not show that the cell prompt
activates, uses, or contains that knowledge.

### 3. Open-weight residual activation patching

Text remains byte-identical during a hidden-state intervention. On development
data only, freeze one layer `ell*`, the final pre-answer token position, direction
normalization, patch norm, and one scale `alpha_k` per biological axis. Construct
two nonexchangeable unit directions from the direction-construction bank:

- `v_GNLY,L`: the independently supported GNLY-to-lineage relation, with
  cytotoxic-state language balanced between true statements and foils; and
- `v_GNLY,C`: the independently supported GNLY-to-cytotoxic-state relation, with
  lineage language balanced between true statements and foils.

If an axis-specific relation cannot be independently sourced, that axis is not
tested. Do not pool the two banks into a generic `v_GNLY`. For each direction,
average all four order/remapping forms, subtract matched non-GNLY relation
contrasts, and project out frozen global truth, answer-label, and gene-identity
directions before normalization. No confirmatory label, logit, activation, or
outcome may choose the layer, direction, sign, or scale. Holm correction covers
the two axis-specific direction families.

### Natural activation alignment is a separate estimand

Output-score units and residual-space alignment are not interchangeable. Let
`h_i(x)` be the residual at the frozen layer/position and let `mu_dev` be the
development-only centering vector. Define

\[
\pi^k_i(x)=cos(h_i(x)-\mu_{dev},v_k)
\]

and, within donor,

\[
A^k_d=mean_i\{[\pi^k_i(full)-\pi^k_i(-GNLY)]-
[\pi^k_i(full)-\pi^k_i(-matched)]\}.
\]

Thus `A` asks whether target depletion removes more relation-aligned activation
than a matched deletion; it is not a probability and does not reuse the `0.10`
output margin. The candidate alignment margin is `+0.05` cosine units, frozen from
development-only matched deletions and norm-matched random/control directions. It
must also exceed the preregistered 95th percentile of their absolute contrasts; if
the controls require a wider margin, the axis stops. Natural alignment requires a
95% donor-mean interval above `+0.05`, the directional donor tests, at least
`r_conf` positive donor contrasts, all leave-one-donor-out means above zero, and the
same result after each order/remapping form is inspected separately.

Passing this test shows that a GNLY-aligned component is already naturally present;
it does **not** show that its amplitude is deficient. Underactivation wording is an
optional stricter construct. Before confirmatory outcomes, freeze a bank of
biologically matched positive-control cues that (i) use the same cell template,
axis, truth strata, and token budget, and (ii) show material native use on
development donors. Freeze `T_A^k`, the lower 10th percentile of their donor-level
alignment amplitudes, and one unedited downstream checkpoint `ell_down > ell*`.
Relative underactivation requires all of the following on confirmation: the control
cues retain material native use and their alignment has a 95% lower bound at or above
`T_A^k`; GNLY `A^k` is above the `+0.05` alignment floor but its one-sided 95% upper
bound is below `T_A^k`; the paired within-donor contrast
`U^k_d=A^k_{control,d}-A^k_{GNLY,d}` has a 95% lower bound above `+0.05`; and the
intact-cue patch moves the alignment at `ell_down` into equivalence with the frozen
positive-control range while rescuing output. If no scientifically matched control
exists, the control amplitude does not transport, or GNLY is not adjudicated below it,
the term **activation gap** is forbidden. The strongest remaining construct is a
causal integration/gain gap.

### Claim-bearing gain patch on an intact cue

The claim-bearing recipient is an **intact GNLY-positive prompt** on an axis that
already has a preregistered, statistically adjudicated native-use shortfall and a
Level-4 decoder gap. Add the same `+alpha_k*v_k` to every eligible cell at
`(ell*, final-context-token)`; the sign cannot depend on the true cell label. A
patch on a GNLY-depleted prompt is run only as a sufficiency control and cannot, by
itself, support an integration/gain or activation-gap claim.

For axis `k`, set `y_i=+1` for NK or cytotoxic-high and `y_i=-1` for CD8 T or
cytotoxic-low, and define the correct-class score `c_i^k=y_i*s_i^k`. With
`delta s_i^k=s_i^k(patch)-s_i^k(native)`, also define donor-level discrimination
and common-bias contrasts:

\[
D^k_d=\tfrac12[mean_{y=+1}(\delta s^k)-mean_{y=-1}(\delta s^k)],\qquad
B^k_d=\tfrac12[mean_{y=+1}(\delta s^k)+mean_{y=-1}(\delta s^k)].
\]

With the frozen balanced sampling, `D` is algebraically the donor-mean improvement
in `c`; it is the single primary margin estimand rather than duplicate evidence.
Cell-level `c` remains the stratum diagnostic. Every native, positive-patch,
reverse-patch, erasure, residual-swap, and control arm must first pass Level-0 finite-
logit/extraction, full-vocabulary format-adherence, item, order, remapping, and
interaction gates. Apply the same nuisance-equivalence gates to each paired
intervention-minus-native effect. An arm that makes another vocabulary token the
native answer is `READOUT_FORMAT_INVALID`, even if its conditional `X/Y` score moves.

The positive patch passes only if donor-mean `D` exceeds `+0.10` under the full
directional rule, donor-mean AUROC improves by more than `+0.05` under the same
rule, and `B` is equivalent within `+/-0.06`. The patched value of the frozen
Level-4 native-adequacy endpoint must additionally reach its adequacy target or
close at least 80% of the same-metric frozen decoder-minus-native gap `G_dec`.
Otherwise an otherwise valid effect is `PARTIAL_CAUSAL_IMPROVEMENT`, not rescue.
Every required true-class-by-lineage stratum must be noninferior with its 90% interval
above `-0.03` and all leave-one-donor-out means above that boundary. The unintended
biological axis must be equivalent within `+/-0.06`. All thresholds are
conjunctive, and the two biological axes are Holm-corrected.

The causal control family contains:

- `-alpha_k*v_k`, which must reduce correct-class score and discrimination by more
  than `0.10` rather than merely reverse a raw class score;
- projection erasure of `v_k` from intact prompts, which must produce the same
  selective reduction;
- the other biological direction (`v_GNLY,L` versus `v_GNLY,C`) as a cross-axis
  specificity control, equivalent on the tested axis while evaluated separately on
  its own registered axis;
- norm-matched random, shuffled-fact, matched-gene, and unrelated-marker
  directions, all equivalent within `+/-0.06`;
- the same vector at frozen early and late control layers; and
- no-op hooks proving that instrumentation changes `s` by at most `1e-6`.

Finally, use the token-budget-aligned intact/depleted pairs for a natural mediation
control. At the frozen layer/position, replace the depleted residual with its
paired intact residual and, in the reverse arm, replace the intact residual with
its paired depleted residual. The forward swap must improve correct-class score by
more than `+0.05` and leave it equivalent to the paired intact output within
`+/-0.06`; the reverse swap must reduce it by more than `0.05`.
Within-cell matched-gene swaps, same-cell nonce-span swaps, and control-layer swaps
must be equivalent within `+/-0.06`. If the native contrast is too small to identify
this registered mediation estimand, the gain/activation claim is `INCONCLUSIVE`;
the ratio or threshold is not changed afterward.

The default Level-5 construct therefore requires natural alignment, intact-cue
patch rescue, selective erasure/reverse-patch damage, and paired bidirectional
mediation beyond every control and is named a **local causal integration/gain gap**.
It may be upgraded to a **relative activation-amplitude gap** only through the
independent positive-control amplitude calibration above. Rescue of a depleted
prompt alone is hidden-state sufficiency or steering.

## Gap definitions and stopping hierarchy

1. **Level 0 — readout validity.** If coherence, order, remapping, or interaction
   fails, stop. Report `READOUT_INVALID`. Do not test or describe biology,
   lexical access, activation, or latent knowledge from that score.
2. **Level 1 — cohort validity.** Require orthogonal labels, all four quadrants,
   frozen donor/cell selection, and the task-informativeness baseline. Failure is
   `TARGET_NOT_ADJUDICABLE`.
3. **Level 2 — biological/output result.** Report the 2x2 separation and GNLY cue
   classification. Token depletion/add-back can support text-interface dependence
   only.
4. **Level 3 — lexical-to-cell integration.** A passing marker probe supports
   explicit fact access. It becomes a **lexical-to-cell integration gap** only if
   the complete 95% interval for the relevant native absolute and matched GNLY-use
   effects lies inside `[-0.06,+0.10)` in every required stratum, both shifted
   boundary tests pass, and leave-one-donor-out estimates remain in that interval.
   This bounded rule establishes non-harm plus sub-usefulness; an upper bound below
   `+0.10` alone is insufficient. A materially negative effect is
   `LEXICAL_MISINTEGRATION`. A mere failure to reject, a boundary-crossing interval,
   or wide uncertainty is
   `FACT_ACCESS_WITH_INCONCLUSIVE_INTEGRATION`, not a gap. This level is not a
   hidden-state claim.
5. **Level 4 — readout gap.** Train a linear hidden-state decoder only on
   development donors and freeze it before confirmatory inference. A
   **supervised readout gap** requires decoder donor-mean AUROC at least `0.75`,
   with its lower 95% bound above `0.65`. Before registration, choose either AUROC
   or correct-class margin as the native-adequacy endpoint and define `G_dec` as
   decoder minus native in that **same metric**. For a margin endpoint, transform
   the decoder probability to the same `[-1,1]` correct-class scale. `G_dec` must
   exceed `+0.10` under the directional donor rule. A genuine native-output
   shortfall additionally requires its 95% upper bound below the frozen adequacy
   target (`0.75` AUROC or `+0.20` correct-class score). Failure to meet an adequacy
   target is not itself proof of shortfall. This establishes held-out decodability
   beyond the coherent native readout, not stored knowledge or natural causal use.
6. **Level 5a — local causal integration/gain gap.** This requires every earlier
   validity gate, Level-3 adjudicated native-use shortfall, a passing supervised
   readout gap with a genuine native-output shortfall, natural alignment, a valid
   intact-cue gain patch that reaches the rescue criterion, selective erasure/
   reverse-patch damage, and bidirectional intact/depleted mediation beyond all
   controls. These results show a boostable, naturally present causal component;
   they do not by themselves identify deficient activation amplitude. If only a
   depleted-prompt patch succeeds, report sufficiency/steering. If a claim-bearing
   gate fails with adequate precision, stop at lexical or readout gap; otherwise
   report `INCONCLUSIVE`.
7. **Level 5b — relative activation-amplitude gap.** Upgrade Level 5a only if the
   separately frozen, same-context positive-control calibration adjudicates native
   GNLY alignment above the null floor but below `T_A^k`, and the intact-cue patch
   restores the unedited downstream checkpoint to the positive-control range while
   rescuing the output. This is relative underactivation for the named relation,
   layer, task, and controls—not proof of a general latent-knowledge store.
8. **Level 6 — replicated causal tier.** Repeat the complete frozen hierarchy in
   both an independent orthogonally labeled cohort and an independent open-weight
   model family. Call the result a replicated integration/gain gap unless Level 5b
   also passes independently in both replications. Neither more cells nor another
   prompt in the same run is replication.

## Required implementation artifacts and verification

Freeze before any confirmatory forward pass:

- Phase-0 accession census, state-label lock, full-conjunction power simulation,
  and the resulting donor target;
- byte and canonical SHA-256 identities for the development/confirmatory fixture
  and its provenance manifest, the applicable preregistration, the exact runner
  source, the immutable pre-forward call-plan/design, and the margin-lock artifact
  and status;
- cohort accession/raw hashes, protein-gate specification, donor/quadrant count
  table, and development/confirmatory cell registry;
- exact top-50 renderer, tokenizer-verified nonce/evidence-slot budgets, editable
  byte spans, and target/control/add-back plan;
- model, tokenizer, chat-template, opaque-token IDs, layer, position, patch
  directions by biological axis, norms, and scales;
- prompt bytes and the complete order-by-remapping call plan;
- estimands, margins, native-adequacy endpoint, exclusions, multiplicity families,
  and exact decision tree; and
- disjoint construction/probe fact banks, truth provenance, activation-swap plan,
  development/confirmatory firewall, and all code/dependency hashes.

For every call, the runner must save the raw model-output next-token vector **before
all logits processors** as a mandatory NumPy sidecar: little-endian float32,
C-contiguous, record by complete model-config vocabulary. The manifest must record
its exact shape, row-to-record order, vocabulary size, byte format, file SHA-256,
matrix SHA-256, and per-row SHA-256 values. The immutable pre-forward plan/design
must hash-bind the fixture bytes, provenance manifest, preregistration, runner
source, call plan, margin lock, model and tokenizer revisions, chat template, dtype,
token IDs, and vocabulary size. The execution manifest must then bind that design
to the records and sidecar; the analysis artifact must bind all three inputs. A
selected-logit-only artifact is invalid.

The runner must also save both opaque-token logits, full-vocabulary format-adherence
diagnostics, aligned `Delta/q/s`, all four form IDs, donor/cell/quadrant,
intervention identity, hook/swap specification, activation hashes, and no-op
verification. An independent analyzer must verify the frozen call plan and sidecar,
then reconstruct both retained logits, the greedy token and logit, full-vocabulary
log-sum-exp, `q`, `s`, and every row and matrix digest. Tests must reconstruct every
request, prove zero donor/cell/fact overlap across firewalls, verify tokenizer and
byte-span invariants, and reproduce the final JSON and Markdown byte-for-byte. An
independent audit must recompute Level 0 before opening any downstream section of
the report.

## Claim boundary

A successful text deletion is causality at the rendered-token interface, not gene
or pathway causality. Orthogonal protein/functional gates are operational assay
labels, not an exhaustive biological ontology. A supervised decoder establishes
decodability, not knowledge. A controlled residual patch establishes local causal
influence only at the named model, layer, token position, direction, cohort, and
task. Even the complete local hierarchy does not prove human-like understanding,
clinical validity, a universal mathematical invariant, or a physical law.

## Bilingual summary / 한영 요약

**English.** Confirmatory execution is currently `NO-GO`: no public cohort meets
the frozen same-cell lineage-by-intracellular-state gate. The separately
preregistered, outcome-exposed Level-0 development run is authorized only to
engineer the open-weight, same-forward-pass two-token readout. Its `level0_only`
power result may select a development candidate in 12–20 but must leave final
`n_conf` unset. Confirmation requires prospective acquisition from at least four
development donors plus a held-out `n_conf` selected later by full-hierarchy power.
After Level 0 passes, text depletion/add-back
tests rendered lexical-cue use and a disjoint fact probe tests explicit access. An
adjudicated native shortfall plus natural alignment, true-label-aligned intact-cue rescue, selective
erasure, and paired mediation supports a local causal **integration/gain gap**.
Calling it relative underactivation additionally requires a frozen same-context
positive-control amplitude threshold and downstream amplitude restoration. Patching
only a depleted prompt is merely sufficiency.

**한국어.** 현재 confirmatory 실행은 `NO-GO`다. 동일 세포 수준의 계통 단백질과
세포내 상태 표지를 모두 갖춘 공개 코호트가 동결된 기준을 충족하지 못한다. 별도
사전등록된 outcome-exposed Level-0 개발 실행은 오픈웨이트 모델의 단일
forward-pass 두 토큰 readout 엔지니어링에 한해서만 허용된다. 현재
`level0_only` power 결과는 12–20명 중 개발 후보만 고를 수 있고 최종 `n_conf`는
반드시 미정으로 남겨야 한다. 확증에는 최소 4명의 개발 donor와, 이후 전체 claim
hierarchy power로 정한 독립 확증 donor `n_conf`를 새로 수집해야 한다. Level 0 통과
후 텍스트 제거/재추가는 렌더링된 어휘 단서
사용을, 분리된 사실 probe는 명시적 접근을 검사한다. 통계적으로 판정된 native
shortfall, 자연 activation 정렬, GNLY가 존재하는 입력에서의 정답 정렬 rescue, 선택적 erasure, 양방향
mediation은 국소적 인과 **integration/gain gap**을 지지한다. 이를 상대적
underactivation으로 부르려면 동일 맥락 positive-control의 동결된 activation amplitude
기준과 downstream amplitude 복원까지 추가로 통과해야 한다. GNLY 제거 입력만
복구하는 것은 sufficiency일 뿐이다.
