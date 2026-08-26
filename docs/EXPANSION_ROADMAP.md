# Expansion roadmap: mechanism, transfer, causal use, and response validation

*Updated 2026-08-02. These are independent evidence tracks. Passing one track
does not imply either of the others, and no new empirical result is reported here.*

## Shared biological priority

Use deep mutational scanning, or DMS, as the first biological primary. It provides
explicit wild-type-to-mutant constructs, measured functional outcomes, natural
counterfactual pairs, and protein-family holdouts. Use MPRA as the first independent
regulatory replication and Perturb-seq as the higher-complexity interventional
extension. `single-cell/*:anon` remains an engineering and interface control.

The preferred assay-specific questions are:

| Domain | Preferred biological endpoint | Binary compatibility question | Experimental/uncertainty unit | Generalization/split unit |
|---|---|---|---|---|
| DMS | signed assay-normalized mutant effect relative to a matched physical WT/control, with uncertainty | In separately registered increase/decrease or retained/loss tasks, is the complete interval outside the assay's indeterminate zone? | culture, library/run, tile/codon lineage | protein, then protein family |
| MPRA | signed, replicate-aware alternate-minus-reference log activity effect in cell context \(C\) | In a separately preregistered increase or decrease task, is the interval outside the assay-specific indeterminate zone? | construct, barcode, library, batch | locus/regulatory family, chromosome, study |
| Perturb-seq | donor-aware signed effect on a response program fixed independently of evaluation outcomes | Does perturbation \(P\) produce the preregistered signed increase or decrease relative to its matched control in cell type \(C\) and timepoint \(T\), outside an indeterminate zone? | guide/barcode and cell nested in donor/batch | perturbation, cell type, donor, study |

The threshold, units, orientation, replicate aggregation, missingness policy, and
experimental context are part of each target. Alternate prompt or representation
renderings of one endpoint are not independent biological questions.
For DMS and Perturb-seq, a binary endpoint can preserve compatibility with the current
answer-token estimand, but the signed continuous endpoint must be retained for
calibration, threshold-robustness, and regression analyses. For MPRA, the signed
continuous effect is primary; a binary rendering must preregister increase and decrease
as separate directions rather than collapse them into an unsigned "change."
Every empirical binary task freezes an indeterminate zone. Failure to cross one
directional boundary is not evidence for neutrality or the opposite direction.

## Best next biological expansion: genotype-by-cofactor response

The strongest immediate CBS target separates the biological estimand from the model
evaluation question.

**Biology:**

> For the same CBS variant `v`, what is the signed assay-level
> high-minus-composite-low vitamin-B6 contrast, with uncertainty at the authenticated
> culture/library dependency level?

**Model:**

> Can a completely frozen source-trained score predict that signed contrast with no
> CBS response labels exposed to model selection, fitting, calibration, or threshold
> choice, and does it beat the frozen raw-input and surface-feature controls?

This is stronger than another generic damaging-versus-neutral question because B6 is
an explicit experimental condition and the same variant supplies its own biological
comparison. The primary target should be the continuous high-minus-low fitness
difference, restricted to variants with measured values in both conditions. Missing
high- or low-condition values must not be imputed into the primary analysis; any
imputation is a separately labeled, nonconfirmatory sensitivity analysis.

Here “low B6” is a composite of the 0 and 1 ng/ml conditions. Its weights,
replicate aggregation, and covariance treatment must be authenticated and frozen
before any contrast is emitted; the word “increase” is reserved for a positive
adjudicated result rather than built into the estimand.

For [Sun et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6993387/), low B6
combines the 0 and 1 ng/ml conditions and high B6 is 400 ng/ml. The published
B6-remediability rule is evaluated among variants first classified as deleterious
under low B6 and uses a lower 95% confidence bound on the high-minus-low difference
above 0.22 at 5% FDR. Keep that rule separate from the low-B6
functional-deleterious rule, whose upper 95% confidence bound is below 0.60. Failure
to meet that deleterious rule is not automatically evidence of neutral/retained
function. The continuous difference is primary; the 0.22 rule is a frozen,
publication-defined binary rendering rather than a threshold tuned during evaluation.

This is a candidate T5 assay-level genotype-by-environment response only after the
treatment assignment and culture/library dependency provenance are authenticated;
until then it is a signed assay contrast. It can test
assay-outcome alignment and, with controlled model interventions, contribute to a
local causal integration/gain-gap test and, only with separate amplitude calibration,
an activation-gap test. It does **not** by itself show clinical pyridoxine
responsiveness, prove latent knowledge, demonstrate that the unperturbed model uses
that knowledge, or establish a physical law.

The candidate-stage pair registry, high-B6 native adapter, and structural pair
contract are now implemented for
[MaveDB CBS high-B6 `urn:mavedb:00000005-a-6`](https://www.mavedb.org/score-sets/urn%3Amavedb%3A00000005-a-6).
They freeze the exact low/high core-body digests, 1,656-nt target hash, primary
`hgvs_nt` join, annotation checks, exact codon-copy rule, no-imputation policy,
signed orientation, and the expected 10,098-row native overlap. This is a
variant-aligned two-condition contrast, not evidence that the underlying experimental
replicates are paired.

The persisted pair statuses intentionally do not claim that decoded source bodies are
present. On 2026-07-26, the intermittent monolithic mapped-variants endpoints
nevertheless completed under the unchanged fail-closed validator for both conditions.
The resulting low- and high-B6 source-lock summaries were atomically materialized,
reloaded, hash-pinned in the pair registry, and independently confirmed by second
mapped-body replays. They bind complete OpenAPI, metadata, score, count, mapping, and
target snapshots while retaining no decoded response bodies. The exact score replay
still matches the 10,098/1,380/704 native-row partition and the two separately defined
protein-set counts.

`eval/cbs_b6_public_native_evidence.py` and its persisted lock now freeze the official
Additional File 3 workbook identity and six-sheet structure, the paper-reported
replicate/run design, and publication-window candidates from both paper-linked code
repositories. The workbook exposes only ordinal replicate columns; neither it nor the
generic CBS repository test fixture authenticates the B6-specific mapping from those
columns to dose, biological culture, sequencing library/run, tile, or shared ancestry.
Column order therefore cannot create a pairing or independence claim.

`eval/cbs_b6_uncertainty_contract.py` now locally hash-locks joint post-count TileSeq
bootstrap as the production method, with exactly 10,000 draws and a seed derived from
the pair-registry digest and serialized as fixed-width lowercase hexadecimal. Direct
covariance remains a reference schema that requires
a versioned amendment; zero covariance and independent-condition fallback are
forbidden. This is not a complete or externally registered specification. The status
still contains no evidence body and cannot unlock readiness. Authenticated post-count
TileSeq score reconstruction,
the authenticated biological sample/dependency graph, condition-specific QC and codon
lineage, joint-bootstrap evidence, CI construction, and externally authenticated
registration remain required before any delta or binary outcome can be emitted. CBS
alone also cannot satisfy protein-family-disjoint transfer or
independent-biological-family replication; it is the variant-aligned
genotype-by-B6-condition primary, not the entire domain-coverage claim. The
paper-reported two cultures per condition fail the contract's conservative
eight-independent-block percentile-CI gate, so 10,000 draws cannot promote a
descriptive point estimate into a CI, label, or outcome. A CBS-only causal run is
pilot-only and `NOT_ADJUDICATED`; variants, codons, tiles, and ordinal channels cannot
serve as pseudo-family groups.

## Track 1: execution of specified mechanistic and mathematical constraints

### Falsifiable question

Given every parameter and assumption, does the model select the exact-solver result for
a specified biological equation or constraint under held-out parameter ranges,
equivalent units, and monotonic counterfactual changes?

Use Michaelis-Menten kinetics first, followed by Nernst potential, Hill or mass-action
occupancy, \(\Delta G\)-to-\(K_d\) consistency, and one-compartment pharmacokinetics.
Keep the first targets binary: an item supplies parameters and a proposed conclusion,
and the label states whether the conclusion is equation-consistent. Use one normalized
two-token logit readout, counterbalanced answer order, and opaque-label remapping;
do not reuse target-conditioned self-reported probabilities unless they first pass the
Level-0 coherence gate in
[`COHERENT_BINARY_READOUT_DESIGN.md`](COHERENT_BINARY_READOUT_DESIGN.md). The exact
solver remains the oracle.

These examples are not one scientific category. Michaelis-Menten, Hill occupancy, and
one-compartment pharmacokinetics are conditional mechanistic models. Nernst calculations
are conditional on the stated equilibrium model. Reserve **physical-constraint**
language for explicitly tested dimensional, conservation, or thermodynamic consistency
relations, such as unit invariance, mass or charge balance, and a declared
\(\Delta G\)-to-equilibrium relation.

Freeze the validity domain, not only the equation name:

- Michaelis-Menten uses an initial-rate, single-substrate, constant-enzyme,
  quasi-steady-state model with positive finite parameters and no substrate
  inhibition or allostery;
- Nernst freezes temperature, nonzero ion valence, positive activity convention,
  membrane-potential sign, and equilibrium;
- the thermodynamic task freezes the standard state and the sign convention in
  `Delta G degree = -R T ln K`;
- Hill/mass-action tasks freeze stoichiometry and whether the Hill coefficient is a
  fitted phenomenological parameter; and
- pharmacokinetics freezes dose route, absorption model, compartment count,
  observation time, and positive parameter range.

Exclude zero denominators, log-of-nonpositive values, singular limits, and boundary
cases whose label changes within the numerical tolerance. For each counterfactual,
register the parameter-specific derivative sign and its validity conditions; a generic
“monotonic” label is insufficient.

Minimum build:

- `signal/mechanistic/generate.py` and a versioned source manifest;
- mechanistic task registrations in `eval/benchmark_tasks.py`;
- paired unit-conversion, algebraic-reexpression, dimensional-error, and parameter
  counterfactual records;
- `eval/analyze_mechanistic.py`; and
- two independent oracle implementations—one symbolic/closed-form and one
  arbitrary-precision numerical path—plus invariance contract tests.

Success requires exact-solver agreement at preregistered accuracy and numerical
tolerances on held-out parameter regimes, oracle-to-oracle agreement before item
release, invariance to equivalent units and notation, rejection of dimensional
inconsistencies, and the registered derivative-sign response to parameter
interventions.

Allowed claim: **exact solver agreement for specified mechanistic or mathematical
constraints under the stated assumptions**. A physical-constraint claim is allowed only
for the explicitly declared dimensional, conservation, or thermodynamic subset.

Not allowed: the activation gap itself is a physical law, a conditional model equation
is a universal physical law, or solver agreement proves latent biological knowledge.

## Track 2: zero-target-label assay-outcome alignment

### Falsifiable question

Can a decoder selected and fitted entirely on one DMS source dataset predict measured
functional direction in an independent DMS target dataset when zero target-dataset
labels are available during layer selection, fitting, calibration, thresholding, or
intervention design?

Source labels are allowed. Target labels are attached only after immutable predictions
have been written. The smallest credible build is:

- one release-grade DMS task with upstream variant IDs, assay provenance, exact
  physical-baseline pool/link identities, protein-family `split_group_id`, and
  row-level `intervention_pair_id`;
- a DMS registration that names the now-supported row-level
  `intervention_pair_field` rather than leaving the current tasks' pair IDs null;
- `eval/latent_transfer.py`, reusing
  `probe_common.predict_with_frozen_decoder`; and
- raw-input, layer-0, surface-feature, and shuffled-label frozen-transfer controls.

Success requires protein-family-disjoint source and target datasets, a programmatic
target-label firewall, positive target transfer beyond the source-trained controls,
agreement between the predicted and measured signed mutant-minus-baseline direction,
and replication on another dataset or model family.

Current status: the aggregate schema-1 pilot and transfer runner remain mechanically
nonconfirmatory. Candidate-stage MaveDB source locking and the independent raw-assay
schema-2 outcome contract are implemented. Schema 2 replays physical baseline links,
transformations, QC, exclusions, and outcomes, but it never grants confirmatory
eligibility from self-contained hashes. No candidate has yet passed assay-specific
admission or a frozen sequence-derived family partition, and no transfer run is reported.

Allowed claim: **zero-target-label, assay-outcome-aligned transferable representation
evidence** for the named constructs, assays, transformations, model, and family
partition.

Not allowed: proof of human-like knowledge, natural use by the unperturbed model, or
spontaneous verbal accessibility. Natural use is a separate causal question.

## Orthogonal target/truth-alignment and causal-use axes

For empirical T5 targets, truth alignment is operationalized only as alignment with the
named assay outcome:

- the **alignment axis** asks whether a frozen source-derived representation predicts
  independently measured target-assay outcomes with zero target labels during
  selection, fitting, calibration, or thresholding;
- the **causal-use axis** asks whether controlled interventions show local causal
  availability, partial natural use, and routing in the unperturbed model's computation.

Passing either axis does not pass the other. For future reporting, distinguish:

1. **local causal integration/gain gap:** an independently adjudicated native
   shortfall plus an intact-input, true-label-aligned causal rescue passes for one
   locked task/model/cohort;
2. **relative activation-amplitude gap:** the local gain result additionally shows
   native relation-aligned activation below a frozen same-context positive-control
   amplitude and downstream restoration; and
3. **replicated causal tier:** the corresponding complete result repeats in an
   independent biological study/family and an independent model lineage.

The implemented status string `CAUSAL_ACTIVATION_GAP_SUPPORTED` is a legacy local
suite name, not permission to skip these semantic gates. Reserve **biological causal
activation gap** for a replicated relative-amplitude result plus a passed Track-2
assay-outcome alignment gate under compatible target semantics. Even that conjunction
does not prove human-like knowledge or a universal latent-knowledge store.

## Track 3: causal integration and activation-gap identification

### Falsifiable questions

1. On intact inputs with an adjudicated native shortfall, does fixed-direction
   steering improve the true-label-aligned answer-logit margin beyond matched controls
   and the item-wise random-direction null without answer-token bias?
2. Does target-direction erasure selectively damage naturally correct answers, and
   does a paired held-out counterfactual source rescue them?
3. Do bidirectional content and routing patches beat observed random and shuffled
   sources, pass the locked content-by-routing interaction, and survive a route-
   necessity/occlusion test?

A gap claim has mandatory prerequisites: a Level-0-valid readout; an independently
adjudicated material native shortfall rather than merely `A-O>0` under unequal
supervision; improvement on intact inputs that reaches the frozen adequacy target or
closes the preregistered fraction of the decoder-native gap; complete finite-logit and
full-vocabulary format adherence in every intervention arm; order/remapping
equivalence; and random, shuffled, surface-feature, and unrelated-direction controls.
Patching a depleted input shows sufficiency. Routing interaction alone shows routing
dependence, not a bottleneck. Activation wording additionally requires the amplitude
calibration in
[`COHERENT_BINARY_READOUT_DESIGN.md`](COHERENT_BINARY_READOUT_DESIGN.md).

The implementation foundation now exists in:

- `eval/model_hooks.py`;
- `eval/causal_intervention.py`;
- `eval/analyze_causal.py`;
- fold-frozen artifacts in `eval/probe_common.py`; and
- the matching causal-intervention contract tests, which are not in the public release.

The next build is a DMS-specific run adapter and immutable preregistration, producing
`steering.json`, `erasure_rescue.json`, `routing_patch.json`, and
`causal_suite_analysis.json`.

The claim ladder is:

- steering only: `CAUSAL_AVAILABILITY_SUPPORTED`;
- erasure, specificity, and paired non-trivial rescue:
  `PARTIAL_NATURAL_USE_SUPPORTED`;
- routing interaction alone: **routing dependence**; the legacy
  `ROUTING_BOTTLENECK_SUPPORTED` string requires a separate necessity/occlusion pass
  before bottleneck wording; and
- one compatible three-artifact suite plus native-shortfall and intact-rescue gates:
  **local causal integration/gain gap**. The legacy
  `CAUSAL_ACTIVATION_GAP_SUPPORTED` string does not upgrade this to underactivation
  without the positive-control amplitude gate.

Every status remains specific to the task, model revision, fold, direction, layer,
token position, biological groups, intervention pairs, and held-out items. A causal
result is not automatically assay-outcome-aligned; Track 2 supplies that separate
evidence. Replication in an independent biological study/family and model lineage is
required for the replicated tier. Stronger biological activation-gap wording requires
both the Track-2 gate and the relative-amplitude calibration.

The confirmatory artifact contract is stricter than a collection of matching
checksums. Every measurement trace must be identity-bound to its registered `item_id`.
The measured margin and trace must be emitted together in one exact identity-bound
measurement record; positional arrays are not admissible confirmatory evidence.
Receipt binding accepts only an exact, versioned, label-free schema; labels are
attached afterward by a separate exact identity join, with duplicate, missing, extra,
or non-bijective joins rejected. Scientific patch and rescue claims require both the
recipient trace and an identity-bound donor/source execution trace. Activation-capture
manifest top-level objects, provenance objects, and records likewise use exact,
versioned, label-free schemas. For routing, the content-manifest source and recipient
token-ID lists must exactly equal the corresponding executed donor/source and
recipient token lists under the declared tokenizer revision. These are requirements
for a future confirmatory run, not evidence that one has occurred.

## Track 4: local linear-response validation and a later regularity candidate

A later diagnostic is the local response relationship

\[
\Delta m(\alpha) \approx \alpha \nabla_{h_\ell}m^\top v.
\]

This is the first-order Taylor approximation of a differentiable intervention response.
Agreement near \(\alpha=0\) tests local numerical linearity; it is not by itself a
nontrivial empirical law, a biological mechanism, or an architecture-conditioned
theorem.

Fit no regularity on the same model-task cells used to propose it. A claim-bearing
preregistration must freeze:

- the symmetric alpha domain and the rule defining the local regime;
- dimensionless or explicitly normalized margin, direction, dose, and residual scales;
- numerical error metric, tolerance, and prediction interval;
- comparisons with a null response and a preregistered quadratic or other curvature
  comparator;
- coefficient-stability and held-out prediction thresholds;
- at least three independent biological task families spanning DMS, MPRA, and
  Perturb-seq, and at least two independent model architecture families, with no family
  used both to propose and confirm the relation;
- family-held-out and model-family-held-out extrapolation; and
- multiplicity control across doses, layers, tasks, models, and candidate normalizations.

A future Track 4 analyzer must return `NOT_SUPPORTED` when a valid, fully adjudicable
analysis misses its preregistered prediction or stability thresholds. It must return
`NOT_ADJUDICATED` when the alpha domain, normalization, independent-family minima,
comparison models, preregistration, or required artifacts are missing. These are
future analysis outcomes, not current implementation statuses. Only a prospectively
successful held-out analysis may support a **cross-family empirical response
regularity**. It still does not establish a physical law of biology.

## Recommended phase order

1. Complete assay-specific admission and external registration of one release-grade DMS
   dataset contract; the candidate-lock and raw schema-2 infrastructure is implemented,
   but no record is admitted.
2. Verify the grouped signal gate, clone-refit-bound train-only decoder, label-free
   activation/prediction artifact frozen before outcomes, label-attached evaluation
   artifact, intermediate-block hook identity, complete execution-context receipts,
   exact label-free receipt and capture-manifest schemas, identity-bound
   measurement-emitted tokenized-input traces, pair-within-bootstrap-group nesting,
   donor-item pair/group resolution, required donor/source traces, and recomputable
   source/recipient activation captures with decoder-width and executed-token-list
   checks against the content manifest and declared tokenizer.
3. Run zero-target-label DMS transfer before expensive causal intervention runs.
4. Run the locked three-artifact causal DMS suite.
5. Replicate the causal result in an independent biological dataset and model family;
   use MPRA as the first independent biological family and Perturb-seq as the next.
6. Only after the independent-family minima exist, preregister the held-out local
   linear-response and cross-family regularity analysis.

The mechanistic/mathematical execution track can run in parallel after Phase 1 because
it is scientifically independent of the DMS transfer and causal claims.
