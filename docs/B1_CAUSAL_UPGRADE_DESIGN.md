# B1: confirmatory causal tests of a decodability-native-output gap

*Design and preregistration target. Updated 2026-08-02. No confirmatory B1 model run
or causal result is reported here. Existing observational arm scores do not satisfy
this contract and must not be used as causal evidence.*

## Question and claim boundary

The observational benchmark can show that a supervised decoder predicts a biological
label from a hidden state better than a locked native-output prompt. That contrast does
not by itself show latent knowledge, natural use of the decoded feature, a routing
mechanism, or a law.

B1 separates three questions:

1. **Causal availability, or operational sufficiency:** does a fixed, train-derived
   intervention on the decoded direction cause the predeclared answer-logit margin to
   change more than matched control interventions?
2. **Partial natural use:** on items the unperturbed model answers correctly, does
   targeted erasure selectively damage the answer, and does an independent,
   non-trivial rescue restore it?
3. **Activation-to-output routing:** can bidirectional content and routing patches
   recover and suppress expression in the predicted directions, including the
   preregistered content-by-routing interaction?

A steering result alone can support causal availability in one model-task setting. It
cannot establish that the unperturbed model naturally uses the direction. Natural-use
language requires the erasure, specificity, and rescue gates. A causal activation-gap
code status additionally requires the bidirectional routing gates. Semantic
integration/gain-gap wording also requires a valid readout, adjudicated native
shortfall, and intact-input rescue; relative underactivation additionally requires a
matched amplitude calibration. None of these single-model results is a physical law
or an unconditional proof of biological knowledge.

## Primary biological target and expansion order

### Confirmatory primary: deep mutational scanning

Deep mutational scanning, or DMS, is the first confirmatory target because it offers a
measured functional outcome, explicit wild-type and mutant constructs, and natural
counterfactual pairing without requiring a cell-state aggregation model.

The frozen primary question should be one assay-specific binary contrast, for example:

> For this protein and assay, is the signed mutant-minus-physical-WT effect, including
> its registered uncertainty interval, outside the preregistered indeterminate zone in
> the separately declared retained or loss-of-function direction?

The threshold, direction of better function, assay units, replicate aggregation,
missing-value rules, and permissible sequence context must be fixed per assay. Splits
must hold out proteins or protein families, not merely mutation rows. Every row needs:

- `item_id`
- upstream assay and variant identifiers
- `biological_question_id`
- `task_family_id`
- `split_group_id`, preferably protein family
- `intervention_pair_id` for the wild-type/mutant or paired-rendering contract
- experimental label provenance

### Stronger biological expansions

1. **MPRA:** reference and alternate regulatory sequences with measured allele-specific
   or sequence-specific activity in a locked cell context. Hold out loci or regulatory
   elements and keep assay batch and donor dependencies out of opposing folds.
2. **Perturb-seq:** perturbation versus matched control in a locked cell type and
   timepoint, with a preregistered gene-program or response endpoint. Hold out biological
   perturbation groups and account for donor, batch, guide, and cell-level
   pseudoreplication.

MPRA and Perturb-seq provide stronger intervention-derived biological semantics than an
arbitrary interface task, but their dependency structures make them harder confirmatory
targets.

`single-cell-anon` is retained only as an interface and hook-control task. Its arbitrary
codebook can test whether the implementation can move a supervised direction through a
model. It is not a headline test of latent biological knowledge, because the mapping is
introduced by the benchmark and learned with labels. Synthetic spectra and other
representation transforms likewise remain engineering controls unless their biological
measurement provenance is independently established.

## Locked output estimand

For fixed positive and negative answer-token sets, define the answer-logit margin

\[
m_i(c) =
\operatorname{logsumexp}_{t \in T_+} z_{i,t}(c)
-
\operatorname{logsumexp}_{t \in T_-} z_{i,t}(c),
\]

where \(c\) is an intervention condition and the logits are taken at the locked answer
position. Prompts, answer aliases, tokenization rules, answer position, decoding
settings, and model revision are frozen before the outer test fold is evaluated.

The positive-class direction \(v\) is oriented once using only outer-training labels.
Its normalized raw-space vector is hashed once and the same
`applied_direction_sha256` is recorded in the contract and every target-direction
execution row. A held-out decoder prediction may not flip this sign per item.
For steering,

\[
h'_{\ell}=h_{\ell}+\alpha s_{\mathrm{RMS}}\frac{v}{\lVert v\rVert_2},
\qquad
s_i =
\frac{m_i(+a)-m_i(-a)}{2a},
\]

where \(s_{\mathrm{RMS}}\) is the train-fold residual-feature RMS saved in the
intervention contract. Normalization occurs inside the hook, so multiplying the
caller-supplied vector by an arbitrary constant cannot change the intervention.
Alternatively, the preregistered linear slope over a symmetric alpha grid is used.
Positive alpha must always mean movement toward the positive class, for both positive
and negative test items. Negative alpha must use the same direction with the opposite
fixed sign.

The primary steering estimand is the paired target-direction slope and its contrast
with matched control-direction slopes. It is **not** label-signed `P(correct)`, and the
held-out label may never choose the intervention sign, alpha, source activation,
layer, token position, or prompt. A per-item intervention of
\((2y_i-1)\alpha v\) is an oracle policy and is forbidden.

Every label-free execution trace must carry the registered `item_id` and be
identity-bound to that item before a receipt is frozen. The helper must emit the
measured margin and its trace together in one exact identity-bound measurement record;
separate positional margin and trace arrays are not confirmatory. The receipt-binding stage
accepts one exact, versioned, label-free schema: missing fields, unknown fields, label
fields, and free-form extensions are rejected. Only after all model executions and
receipt binding are complete may a separate stage attach labels through an exact
identity join on `item_id`; duplicate, missing, extra, or non-bijective joins are
rejected. Define

\[
m_i^{\mathrm{correct}}=(2y_i-1)m_i
\]

only for the natural-use and task-performance analyses. This post-execution
transformation must not feed back into intervention construction.

## Fold-frozen direction and selection contract

All selection happens within the outer-training partition. The implementation must use
the actual shared probe interface:

```python
result = probe_common.nested_layer_auroc(
    H,
    y,
    groups=split_group_ids,
    item_ids=item_ids,
    return_decoder_artifacts=True,
    decoder_artifact_dir=artifact_dir,
    artifact_provenance=provenance,
    hidden_states_include_embedding=True,
)
```

For each outer fold, this returns the nested-training-selected layer and a frozen
`StandardScaler` plus balanced logistic decoder fitted only on that fold's training
rows. The artifact records train/test entity IDs, train/test group IDs, the selected
layer, scaler parameters, positive-class decoder coefficients, and a training-data
checksum. Export also deterministically clones and refits the pipeline on the supplied
training activations and labels, then compares scaler state, decoder state, and
training probabilities. A classifier fitted on different arrays or labels is rejected.
This direction artifact remains train-only: its checksum never includes a held-out
label, activation, or prediction.

Held-out evidence is split into two more objects. First, a label-free prediction
artifact stores the selected-layer and pre-decoder activations, item IDs, biological
groups, frozen selected/reference probabilities, activation checksums, and immutable
extraction provenance. Its builder has no label argument. Only after this artifact is
frozen does the evaluation artifact attach labels and compute AUROC,
selected-minus-reference selectivity, and group-bootstrap confidence intervals. The
analyzer reapplies both immutable decoders to the stored activations and verifies the
prediction-artifact checksum before scoring. Thresholds have hard floors (AUROC at
least 0.65, AUROC CI lower-bound threshold at least chance, selectivity at least 0.05,
and selectivity CI lower-bound threshold at least zero) and may only be made stricter
in `design.decoder_signal_gate`. A zero target decoder, fabricated probability,
activation/provenance mismatch, or failed signal gate stops B1.

The final flag is load-bearing for Hugging Face causal-LM outputs: hidden-state index
zero is the pre-decoder embedding state, and each intermediate hidden-state index
\(1 \leq k < |H|-1\) maps to decoder block \(k-1\). The final returned hidden state is
commonly post-final-normalization and is **not** the tensor exposed by the final
decoder-block output hook. It is therefore excluded from confirmatory selection until
a separately identified final-normalization hook is implemented. The embedding is
retained only as a pre-decoder reference and is also not intervention-hookable. An
unspecified activation-site mapping is recorded as `unspecified_not_hookable` and
cannot drive a confirmatory hook.

The raw residual-stream intervention direction is derived from that fold's standardized
decoder without refitting on test data. If \(w\) is the logistic coefficient and
\(\sigma\) the saved per-feature scaler scale, use the normalized raw-space gradient
\(v \propto w/\sigma\), oriented toward frozen positive class 1. The saved train-fold
mean supplies any centering operation. Zero-scale features and numerical tolerances
must be handled by one locked conversion function and tested before a confirmatory run.

The following are also train-only selections:

- layer and one-standard-error band
- residual token position or pooling rule
- alpha normalization and symmetric dose grid
- successful versus native prompt pair used for routing tests
- content and routing subspaces
- erasure strength and patch strength
- rescue-source rule
- all nuisance and surface-control directions

A pilot may choose these values, but the pilot entities and biological groups must then
be excluded from the confirmatory test. Test-fold outcomes cannot choose a best layer,
best alpha, best prompt, or best intervention family.

## Intervention families and confirmatory gates

### B1a: fixed-direction steering, causal availability

Run the locked symmetric alpha sweep for each held-out item with the same direction
sign. Required conditions are:

- a separately executed unhooked baseline
- target direction
- shuffled-label decoder direction
- a representation/surface direction when one is measurable
- covariance-matched random directions
- alpha zero identity condition

Random directions are constructed from outer-training residual covariance only. The
generation method, regularization, seeds, number of controls, and matching tolerance
must be locked. Controls must match the target intervention in Euclidean norm,
projected train-activation variance, token position, layer, and alpha schedule. Merely
drawing a unit Gaussian vector is not a sufficient matched control.

Causal availability is supported only when:

1. the cluster-bootstrap 95% confidence interval for the mean target slope is above
   zero;
2. the paired target-minus-control slope interval is above zero separately for every
   declared control kind; averaging a failed shuffled or surface control with another
   control is forbidden;
3. the target slope exceeds each item's preregistered 95th-or-higher random-direction
   quantile under the biological-group bootstrap, without first averaging away
   item-by-direction heterogeneity;
4. alpha zero reproduces the separately executed unhooked baseline within the locked
   numerical tolerance;
5. the effect is not explained by the shuffled-label or surface direction;
6. record-derived collateral output and fluency diagnostics remain within
   preregistered tolerances.

This supports an intervention-sensitive direction, not natural use or per-item
sufficiency for correct classification.

### B1b: targeted erasure and non-trivial rescue, partial natural use

Use directional erasure centered and oriented from the outer-training artifact. The
confirmatory subset is defined as items with a correct unperturbed margin,
\(m_i^{\mathrm{correct}}>0\), under deterministic locked logits. The subset definition
and minimum sample size are fixed before intervention outcomes are inspected.

Required paired conditions are:

- unperturbed baseline
- target-direction erasure
- covariance-matched random-direction erasure
- target erasure followed by rescue
- nuisance-task and fluency checks under every condition

The scientific rescue must not simply cache the same target activation before erasure
and add it back. That is an engineering identity test. A non-trivial rescue uses a
predeclared independent source, preferably a paired rendering of the same biological
item whose source relationship is fixed by `intervention_pair_id`, without consulting
the target label or output. Donor selection and any frozen-decoder-based source rule
must be learned and locked on outer-training data. Every rescued row records the
source item, matching intervention-pair ID, locked strength, source condition, and
distinct source/recipient activation checksums. Reusing one donor across unrelated
items or omitting this linkage yields `NOT_ADJUDICATED`.

The checksums must resolve to an embedded, label-free activation-capture manifest, not
only to caller-supplied strings. Its top-level object, provenance object, and every
record use exact, versioned, label-free schemas; missing or unknown fields, labels, and
free-form extensions are rejected. Each manifest row binds the exact source and
recipient inputs, token IDs, attention masks, and their hashes; source item and
condition; recipient condition; direction; intervention pair; model revision; layer;
token position; hook kind; prompt and execution-code hashes; execution-context hash;
and the source and recipient tensors. A scientific patch or rescue additionally
requires an identity-bound donor/source execution trace, not merely a recipient trace
and a source tensor checksum. That trace must resolve to the declared `source_item_id`,
source condition, pair, biological group, model context, and executed source token
sequence.

The analyzer recomputes all input and tensor hashes, requires one hidden vector with
the frozen decoder's feature width, requires exact row coverage, resolves the source
item against the real held-out item-to-pair/group map, and requires source and
recipient to share the declared dependency group. For routing, the source and
recipient token-ID lists in the content-equivalence manifest must exactly equal the
corresponding executed token lists in the donor/source and recipient traces under the
declared tokenizer revision. Routing capture inputs must also be identical to the
locked content-equivalence manifest. This makes internal fabrication or post-hoc
substitution detectable; it is not remote attestation that the model execution
occurred.

The scaler-derived center checksum and erasure strength are locked in the intervention
contract. Every target, random-control, and rescue erasure row must reproduce that same
center and strength. Stronger target erasure than random-control erasure is forbidden.

Partial natural use is supported only when all three paired cluster-bootstrap intervals
are above zero:

1. baseline minus target-erased correct-answer margin;
2. target-erasure damage minus random-erasure damage;
3. rescued minus target-erased correct-answer margin.

The target-erasure damage must also exceed the preregistered item-wise
random-direction damage quantile. Averaging positive and negative random-direction
effects before this gate is forbidden.

The rescue must be recorded as `held_out_counterfactual_source`. A same-run identity
rescue cannot adjudicate natural use. The effect must also be selective: broad damage
to language quality, unrelated labels, or all answer logits invalidates the
representation-specific interpretation.

### B1c: bidirectional content and routing patching

Content and routing are defined on outer-training data, then frozen. The paired
renderings must preserve the biological item and target truth while changing only the
predeclared interface or elicitation condition. Source-target assignment is fixed by
entity and `intervention_pair_id`, never by test performance.

The tested cohort is locked from a behavioral run completed before interventions. For
the combined legacy suite status, every suite item must satisfy the preregistered
`native_incorrect_elicited_correct` rule: native correct-answer margin at or below zero
and elicited correct-answer margin above zero. The analyzer verifies the exact item-set
checksum and requires a positive group-bootstrap interval for
`elicited - native`. Routing without this observed expression gap can support no
combined suite status, and the expression gap alone does not identify underactivation.

The minimum routing artifact contains these paired conditions:

- `native`
- `elicited`
- `route_success_to_native`
- `route_native_to_success`
- `route_random_to_native`
- `route_random_to_success`
- `route_shuffled_to_native`
- `route_shuffled_to_success`
- `no_content_no_route`
- `route_only_no_content`
- `content_and_route`

The forward test asks whether a successful routing state increases the native
recipient's correct-answer margin. The reverse test asks whether a native routing state
damages the elicited recipient. The random and shuffled source conditions test both
directions against observed controls. Both directions are necessary because a one-way
patch can reflect a generic activation boost. Every patched row records the locked
patch strength, expected source condition, and source-activation checksum.
The same logical donor key must resolve to the same checksum across cells, while one
donor checksum cannot be reused across distinct biological items.
The same exact activation-capture manifest contract used for rescue applies to every
routing source/recipient row.

Content equivalence is quantitative, not a manually asserted `PASS`. Every patched row
records its source-recipient token-count difference and embedding cosine distance.
The artifact locks their thresholds, reported maxima, tokenizer revision, embedding
model revision, measurement scope, and comparison-manifest checksum. The analyzer
hashes the embedded comparison manifest; recomputes source and recipient hashes from
the embedded canonical strings, token-count differences from embedded token-ID lists,
and cosine distances from embedded non-zero embedding vectors; binds those values to
the exact routing row/direction; recomputes both maxima; and rejects missing,
mismatched, or out-of-threshold evidence.

For the factorial test, `native` is the content-present/native-route cell,
`no_content_no_route` is its covariance-matched content-ablation control,
`route_only_no_content` adds the elicited route without target content, and
`content_and_route` contains both components in the same fixed recipient. Content and
routing effects are not assumed additive. The interaction must be reported as

\[
\Delta_{C \times R}
=
\bigl(m_{\mathrm{content+route}}-m_{\mathrm{native}}\bigr)
-
\bigl(m_{\mathrm{route-only}}-m_{\mathrm{neither}}\bigr).
\]

The analyzer consumes `no_content_no_route` and calculates the full
difference-in-differences expression above. A conditional
`content_and_route - route_only_no_content` contrast alone is insufficient.
Full-state and directional component patches can alter interacting mediators, so
component effects must not be summed into a total causal decomposition.

A routing-dependence component is supported only if the forward routing effect,
reverse routing damage, both target-versus-random and target-versus-shuffled
specificity tests in both directions, and the content-by-routing contrast all have
cluster-bootstrap 95% intervals above zero. The forward target effect and reverse
target damage must also exceed their item-wise random-direction quantiles. A routing
artifact alone does not produce the final combined status. The legacy analyzer may
name this `ROUTING_BOTTLENECK_SUPPORTED`, but bottleneck wording additionally requires
a preregistered route-necessity/occlusion intervention; interaction and bidirectionality
alone establish dependence.
`CAUSAL_ACTIVATION_GAP_SUPPORTED` requires a compatible three-artifact suite in which
steering, erasure-rescue, and routing use the same task, model revision, fold, layer,
token position, direction checksum, held-out items, labels, biological groups,
intervention pairs, and causal-suite ID. It also requires the same globally oriented
applied-direction checksum and the locked native-failure/elicited-success expression
gap.

That identifier is a legacy code status, not a semantic proof of underactivation. A
reported **local causal integration/gain gap** additionally requires the Level-0-valid
readout, independently adjudicated native shortfall, intact-input true-label-aligned
rescue to the frozen adequacy rule, full-vocabulary adherence, and answer-order/opaque-
mapping equivalence specified in
[`COHERENT_BINARY_READOUT_DESIGN.md`](COHERENT_BINARY_READOUT_DESIGN.md). A **relative
activation-amplitude gap** further requires the frozen same-context positive-control
amplitude and downstream restoration. Without these additions, retain the lower B1
mechanism statuses even if the legacy combined string is emitted.

## Biological splitting, inference, and multiplicity

- The confirmatory unit is the biological dependency group, not a row. Use protein
  family for DMS, locus or regulatory element for MPRA, and a defensible perturbation,
  donor, or batch hierarchy for Perturb-seq.
- Train/test entity IDs and group IDs must be disjoint. Snapshot-row or entity-proxy
  grouping produces `NOT_ADJUDICATED`, even if a point estimate is positive.
- The group scope must be an approved biological scope, and the artifact must link its
  source field and dataset checksum to a recomputed item-to-group map checksum.
- All effects are paired by item and cluster-bootstrapped by `split_group_id`.
- Every non-null `intervention_pair_id` must be nested within exactly one
  `split_group_id`; otherwise pair-linked dependence would cross bootstrap clusters.
- The implementation refuses confirmatory status below 30 held-out items, 8
  biological groups, 20 observed random directions, or 1,000 bootstrap draws. The
  preregistered power analysis may set stricter values, which are stored in
  `design.analysis_lock` and cannot be lowered through the CLI or API.
- DMS is the single primary family. MPRA, Perturb-seq, additional layers, models, and
  tasks are confirmatory only under their own preregistration; otherwise they are
  exploratory and multiplicity-adjusted.
- Report every locked alpha and condition. Do not report only the best dose.
- A syntactically complete artifact with a failed firewall, unavailable hook, or
  insufficient naturally correct items yields `NOT_ADJUDICATED`, not a negative
  causal result. Missing condition-required fields or incomplete source-pair records
  are malformed artifacts and are rejected before adjudication.

## Infrastructure contract

The repository contains implementation scaffolding for this design:

- `eval/model_hooks.py` captures residual states and applies fixed steering,
  directional erasure, full-state patching, or directional-component patching.
- `eval/causal_intervention.py` defines the positive-versus-negative answer-logit
  margin, forbids evaluation-label-derived intervention signs, validates train/test
  firewalls, runs intervention measurements, and writes versioned artifacts. Its
  execution-context checksum binds the task, immutable model, fold, layer, token,
  hook kind, prompt protocol, decoding configuration, execution code, answer-token
  sets, direction artifacts, dose scale, and erasure strength. For a confirmatory
  trace, the helper must pass the recorded per-item input IDs and attention masks into
  the forward callback and emit the actual layer, token position, and answer-token sets
  used by that helper call. The trace must be identity-bound to its registered
  `item_id`, and the executed margin must remain in the same identity-bound measurement
  record; scientific patch and rescue rows also require an independently
  identity-bound donor/source trace. Receipt binding must accept only the exact
  versioned label-free trace schema and must bind its recomputed input hash, actual
  margin, biological group and pair, direction, dose, patch/source metadata,
  collateral diagnostics, control-matching diagnostics, and content-equivalence
  measurements. Labels may be attached only later by a separate exact identity join.
  A closure-only call that omits these model inputs remains an engineering measurement
  and cannot populate a confirmatory execution record.
- `eval/analyze_causal.py` keeps causal availability, partial natural use, and routing
  claims separate, recomputes alpha-zero and control diagnostics, performs paired
  group-aware analysis, and assigns the combined claim only through a compatible
  three-artifact suite.
- `eval/probe_common.py:nested_layer_auroc` produces the fold-specific frozen decoder
  artifacts used to select the intervention layer and direction.

Each confirmatory causal artifact embeds the immutable train-only decoder plus its
label-free prediction and label-attached evaluation artifacts. The train-only
deterministic JSON checksum, fold, mapped decoder block, exact test item/group map,
model/task provenance, dataset checksum, and firewall counts must match the causal
contract. The analyzer regenerates held-out selected/reference probabilities from the
stored activations before applying the locked AUROC, selectivity, and
confidence-interval gate. The positive-class raw direction must be non-zero; its
applied checksum and train-derived RMS dose scale must be stable for every target row.
Each control `direction_id` must map to one stable checksum, and distinct random IDs
must represent distinct vectors. Every scored row must still match its pre-label
execution receipt. Rescue and routing artifacts must also embed the exact
activation-capture manifest whose checksum is locked in the design. A summary claiming
`outer_train_only` without these machine-verifiable artifacts cannot receive a positive
status.

The writer and analyzer share one family/condition schema. Baselines must carry null
intervention fields. Steering cells require an alpha that agrees with the condition
name and permit steering fields only; random cells additionally require observed
matching diagnostics. Erasure and routing cells require only their preregistered
erasure, patch, source, and content fields. Stale target hashes, patch strengths, or
source metadata on a baseline are invalid, as are missing fields on an intervention
cell. Every scored row uses one exact key set; free-form fields cannot be used as a
second label channel. Such structural violations are rejected as malformed artifacts
before statistical adjudication.

These checks are tamper-evident reproducibility contracts, not remote attestation.
Anyone able to fabricate every artifact and recompute every checksum can still invent
a run. A confirmatory release therefore must execute the model in a controlled runner,
write and hash the label-free prediction artifact before the label file is made
available to the scoring process, preserve immutable model/prompt/dataset/code
identities, and publish the raw artifacts needed for independent replay.

The presence of these files does not mean a supported architecture has passed hook
identity tests, that a DMS dataset has been registered, or that a confirmatory artifact
exists. Before any model run, unit and integration tests must verify layer resolution,
hook removal, alpha-zero identity, token alignment, batch isolation, source-target
pairing, artifact checksums, and exact reproducibility of an unintervened forward pass.

## Claim ladder and stopping rules

1. **Engineering validity:** hook and artifact tests pass. Otherwise stop.
2. **Supervised decodability:** nested group-held-out decoder passes its independent
   signal gate. Otherwise no causal direction is tested.
3. **Causal availability:** B1a passes all target-versus-control gates.
4. **Partial natural use:** B1b passes erasure, specificity, non-trivial rescue, and
   collateral-damage gates.
5. **Legacy combined suite status:** one compatible suite jointly passes B1a, B1b,
   both B1c routing directions, the locked native-failure/elicited-success expression
   gap, observed routing controls, and the mediator-interaction gate.
6. **Local causal integration/gain gap:** Level 5 plus an independently adjudicated
   native shortfall and valid intact-input rescue under the coherent-readout contract.
7. **Relative activation-amplitude gap:** Level 6 plus matched positive-control
   calibration showing deficient native amplitude and downstream restoration.
8. **Truth-aligned latent biological representation:** requires semantic
   counterfactual alignment and zero-target-label transfer to an independent dataset,
   beyond the B1 intervention result.
9. **Law-like relationship:** requires a separate preregistration that predicts
   normalized intervention effects in held-out model families and biological task
   families. B1 cannot establish this.

At each rung, failure prevents the stronger language but does not erase valid evidence
from a lower rung.

## Candidate law track, explicitly outside B1

One later mechanistic hypothesis is the local relationship

\[
\Delta m(\alpha)
\approx
\alpha \nabla_{h_\ell}m^\top v.
\]

B1 can generate the intervention data needed to test this approximation, but fitting
the relationship and evaluating it on the same models or tasks would be exploratory.
A law-like claim requires preregistered numerical predictions, normalized variables,
multiple architecture families and sizes, independent DMS/MPRA/Perturb-seq families,
and held-out extrapolation. Even a successful result would be an empirical
computational regularity or an architecture-conditioned theorem, not a physical law.

## Deliverables

1. A registered DMS manifest with upstream identifiers, assay contract,
   `intervention_pair_id`, and biological group split.
2. Train-only fold-frozen direction artifacts, label-free activation-backed prediction
   artifacts, and label-attached evaluation artifacts produced through
   `probe_common.nested_layer_auroc`.
3. A locked causal preregistration containing model revision, prompts, answer tokens,
   token position, alpha grid, control construction, rescue rule, estimands, gates,
   decoder-signal thresholds, expression-gap cohort checksum, quantitative content
   equivalence, minimum sample sizes, and multiplicity plan.
4. Label-free source/recipient activation-capture manifests plus versioned steering,
   erasure-rescue, and routing artifacts.
5. Machine-generated analyses from `eval/analyze_causal.py`.
6. A result report that begins with the highest supported rung and lists every failed
   or non-adjudicated gate.

Until those artifacts exist, the correct project statement is:

> B1 is a specified and partially scaffolded experimental design for testing whether a
> supervised biological direction is causally available, partially naturally used,
> and gated by a separable routing state. No confirmatory causal result has yet been
> established.
