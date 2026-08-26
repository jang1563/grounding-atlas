# Coherent readout v3 cross-codebook anti-copy preregistration

Freeze date: 2026-08-02  
Mode: prospective, development-only, synthetic and non-biological  
Primary model: `Qwen/Qwen2.5-1.5B-Instruct`  
Primary site: final attended context token at a frozen `resid_post` block  
Candidate block grid: `[8, 12, 16, 20, 24]`

## 1. Status, purpose, and hard boundary

The authoritative v2 result is
`NONSPECIFIC_DECISION_STATE_TRANSFER_REPLICATED`: complete final-token residual
replacement transferred a late X/Y decision state, but paired, same-pair, and
unrelated correct-X sources were essentially indistinguishable. That result is
not revised here.

V3 asks a different, prospectively identifiable question:

> Can a rank-one queried-content coordinate, fitted on a complete factorial in
> disjoint symbolic worlds, be interchanged from a donor that has the same
> native answer as its recipient yet cause the recipient to emit the opposite
> answer required by the donor's content under the recipient's unchanged
> codebook?

The donor construction makes generic answer copying predict the native answer,
whereas content recomposition predicts the opposite answer. Full-state patches
are controls, not the primary intervention.

This study cannot authorize a biological forward pass or revise any stopped
biological status. It contains no biological facts and cannot establish latent
knowledge, a biological activation gap, underactivation, a physical law,
real-world semantic knowledge, a general variable-binding circuit, or
model-family generality.

No model forward is authorized until the pre-forward `design.json` has bound all
dynamic implementation and tokenization identities required in Section 9.

## 2. Frozen prospective bank and identity ledger

The synthetic bank was built with zero model calls. Its 56 symbolic worlds and
all cell IDs are disjoint from both v2 banks. World, not prompt row, is the
inferential and resampling unit.

| Object | Frozen path | SHA-256 |
|---|---|---|
| bank builder | `signal/syntax/build_coherent_readout_v3_content_routing_bank.py` | `74192ecda496667b094b6ad0420fd56efcc41644de41aae06cf42758751263f1` |
| bank fixture | `signal/syntax/coherent_readout_v3_content_routing_bank.json` | `a63dced290410ef6d463a0f2c04431dcea871ea564f2a6d0b2e0a05b4bb0d78f` |
| fixture canonical JSON | same fixture | `65a4f768fcebd648f51667dce84e594049804cb412248e7abe37351e3ac4e5b4` |
| bank manifest | `signal/syntax/coherent_readout_v3_content_routing_bank.manifest.json` | `bf8559d7922a01403d7d77aa914f459cc88ecec94311da815bd37c28a4340425` |
| shared hook helper | `eval/model_hooks.py` | `62495bd77adc40d7fd5e5643df334eb98aba363f5b81b4b7925314e877bad0c4` |

To avoid a mutual-hash cycle, dynamic runner, analyzer, tests, dependency lock,
and tokenizer-receipt hashes are not embedded in this static ledger. The
pre-forward `design.json` binds their exact paths and hashes plus the final
preregistration hash. The runner alone embeds and verifies that final
preregistration hash. Runner and analyzer must reject a path mismatch, hash
mismatch, unknown file, or self-consistent edit to only one member of the
pre-forward bundle.

The prospective roles are fixed by world ID before model execution:

| Role | Worlds | Complete factorial cells | Selected recipients | Permitted use |
|---|---:|---:|---:|---|
| direction fit | 16 | 512 | 128 | estimate axes only |
| localization | 8 | 256 | 64 | choose one grid layer only |
| confirmatory holdout | 32 | 1,024 | 256 | final adjudication only |
| total | 56 | 1,792 | 448 | no cross-role reuse |

No fit-world or localization-world result contributes to a holdout interval or
status. Holdout activations, logits, correctness, and intervention results are
inaccessible to direction fitting and layer selection.

## 3. Complete factorial and balanced recipient fraction

Each world contains the complete `2^5` design over:

- queried content `p`: property `P` (`-1`) or `Q` (`+1`);
- codebook `m`: identity, `P -> X; Q -> Y` (`-1`), or swapped,
  `P -> Y; Q -> X` (`+1`);
- distractor content `d`: property `P` (`-1`) or `Q` (`+1`);
- queried entity `e`: entity `a` (`-1`) or `b` (`+1`); and
- fact order `o`: queried fact first (`-1`) or second (`+1`).

Let answer sign `a=+1` denote X and `a=-1` denote Y. Under the fixture's sign
convention, the exact native-answer relation is

\[
a = p m.
\]

All 32 cells per world are used to fit and behaviorally validate the factorial.
Interventions use an eight-cell regular fraction fixed independently of model
outputs. Its base columns are `e`, `p`, and `m`, with

\[
d=e p,\qquad o=e m,\qquad a=p m.
\]

Thus the six columns `(e,p,m,d,o,a)` have Gram matrix `8 I_6`: every factor and
the native answer are each 4/4 balanced and mutually orthogonal in the selected
recipient set. The fraction is not a complete factorial and is not used to fit
axes or to estimate unrestricted interactions; its fixed alias structure limits
the claim to the preregistered paired interventions.

All 32 cells and all eight recipients in every world remain intention-to-treat.
Correctness or patch response may not select cells, donors, worlds, strata, or
directions.

## 4. Exact prompt, tokenization, and answer channel

The system message is exactly:

```text
Follow the user's codebook. Your entire response must be exactly one character: X or Y. Do not write any other text.
```

The user message is byte-for-byte the fixture cell's `prompt_text`, rendered by
this exact template:

```text
Codebook: P maps to {code_for_P}; Q maps to {code_for_Q}.
Fact: {first_entity} has property {first_property}.
Fact: {second_entity} has property {second_property}.
Question: Which code does the codebook assign to the property of {queried_entity}? Answer with X or Y.
```

The two fact lines are the fixture-registered queried and distractor facts in the
registered order. No wrapper, cue, prefill, paraphrase, or demonstration is
added. Chat-template flags are `add_generation_prompt=true`,
`continue_final_message=false`, and `enable_thinking=false`.

Tokenizer-only exhaustive preflight must establish all of the following before
the first model forward:

- all 1,792 complete rendered chat inputs contain exactly 103 attended tokens;
- literal content symbols P and Q and answer symbols X and Y are single
  contextual tokens at every registered occurrence;
- next-answer contextual token IDs are exactly X=`55` and Y=`56`;
- every fixture prompt digest and token-ID-list digest matches the frozen plan;
- every registered donor and recipient has the same length and attention-mask
  shape; and
- no truncation, padding-dependent site shift, or tokenization exception occurs.

Failure stops execution. The model is never asked to generate. Every call is one
teacher-forced prompt forward, and outcomes are measured directly from the full
next-token logit row.

## 5. Anti-copy identification and source graph

For recipient

\[
r=(p,m,d,e,o),\qquad a_r=pm,
\]

the frozen within-world sources are:

1. anti-copy donor
   \[
   s_{AC}=(-p,-m,d,e,o),\qquad a_{AC}=(-p)(-m)=a_r;
   \]
2. text counterfactual
   \[
   s_{TC}=(-p,m,d,e,o),\qquad a_{TC}=-a_r;
   \]
3. same-content, opposite-codebook donor
   \[
   s_{SC}=(p,-m,d,e,o),\qquad a_{SC}=-a_r;
   \]
4. self source `s_ID=r`.

These are the fixture fields `anti_copy_donor_cell_id`,
`text_counterfactual_cell_id`,
`same_content_opposite_codebook_donor_cell_id`, and `self_cell_id`.
Every transformation is an involution and all sources remain in the same world.

The anti-copy donor and recipient have the same native X/Y answer. A generic
source-answer copy therefore predicts no answer flip. If only donor content
`-p` is transplanted while the recipient codebook `m` is retained, the
counterfactual target is

\[
t_r=m(-p)=-a_r,
\]

which is opposite both native answers. Emission of `t_r` after the selective
intervention is the anti-copy criterion. The fixture's distractor-, query-, and
order-flip references are design diagnostics; they are not alternate primary
patch sources.

## 6. Model, activation site, and numerical execution lock

- model ID: `Qwen/Qwen2.5-1.5B-Instruct`;
- revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- cached `model.safetensors` SHA-256:
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`;
- cached `config.json` SHA-256:
  `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670`;
- cached `tokenizer_config.json` SHA-256:
  `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`;
- cached `tokenizer.json` SHA-256:
  `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`;
- effective chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`;
- architecture: 28 Qwen2 decoder blocks, residual width 1,536;
- candidate block grid: `[8,12,16,20,24]`, with no other block selectable;
- hook: output of `model.model.layers[layer]`, `resid_post`;
- token site: final attended context token, index `-1`, only;
- batch size 1, evaluation mode, `torch.inference_mode()`, `use_cache=false`,
  explicit SDPA attention, no gradient, no logits processor, and no generation;
- model parameters and persistent buffers: float32 on MPS; and
- patch strength: exact coordinate replacement, `1.0`, with no dose search.

The hook must fire exactly once and be removed after each call. The final HF
hidden state and final RMS-normalized state are not substituted for `resid_post`.
Passing one member of the coarse grid localizes only that tested site; it does
not establish an exact layer boundary, neuron, head, circuit, or bottleneck.

## 7. Fit-only factorial axes

Let `h_l(w,x)` be the unpatched final-context `resid_post` vector in fit world
`w`, layer `l`, and complete factorial cell `x`. For each nonempty subset `S` of
the five factors, define its Walsh sign `chi_S(x)` and pooled coefficient

\[
\beta_{S,l}=\frac{1}{16}\sum_w\frac{1}{32}\sum_x
\chi_S(x)h_l(w,x).
\]

There are 31 non-intercept coefficients. The raw queried-content coefficient is
`beta_p`. The five structural coefficients are:

- native answer: `beta_{p m}`;
- codebook: `beta_m`;
- distractor content: `beta_d`;
- queried entity: `beta_e`; and
- fact order: `beta_o`.

For target coefficient `beta_j`, form a nuisance matrix from every other
non-intercept Walsh coefficient. Using CPU float64 SVD and the fixed rank rule
`sigma_k > 1e-6 * sigma_max`, let `P_{-j}` project onto that nuisance span and
define

\[
r_{j,l}=(I-P_{-j})\beta_{j,l},\qquad
u_{j,l}=r_{j,l}/\lVert r_{j,l}\rVert_2.
\]

The queried-content direction is `u_C=u_p`. Each structural control is built by
the same leave-one-coefficient-out rule. The intercept

\[
\mu_l=\frac{1}{16\cdot32}\sum_{w,x}h_l(w,x)
\]

is used only as the erasure origin.

Four content-shaped null axes are built from fit worlds only. Let
`beta_{p,l,w}` be the within-world content coefficient. The sign matrix is the
first four nonconstant columns of the canonical Sylvester Hadamard matrix
`H_16`, with fit worlds in increasing frozen world-ID order. Each column has
eight `+1`, eight `-1`, and is orthogonal to the others and the all-ones column.
For `k=0,...,3`, define

\[
\widetilde\beta_{N_k,l}=\frac1{16}\sum_w H_{16}[w,k+1]\beta_{p,l,w}.
\]

Residualize each null coefficient against the span of all 31 genuine
non-intercept coefficients using the same SVD rule, then normalize it to obtain
`u_{N_k,l}`. Null signs, ranks, and directions cannot be redrawn after any
activation is observed.

Every required residual must be finite and satisfy
`||r|| / max(||beta||, 1e-12) > 1e-6`; all directions must have unit norm within
`1e-6`. Because Stage C freezes all five grid layers and exactly 4,480 rows,
failure at any grid layer makes the complete fit basis lock invalid and stops
before localization; no reduced grid is executed or reinterpreted. The fit artifact stores and
hashes fit IDs, raw activations, all coefficients, singular values, retained
ranks, residuals, the intercept, float64 calculation arrays, and the exact
little-endian float32 vectors applied by the hook. Holdout data cannot enter any
of these objects.

## 8. Selective interventions

For unit direction `u`, recipient activation `h_r`, and source activation `h_s`
at the same frozen layer and site, coordinate replacement is

\[
T_u(h_r,h_s)=h_r+uu^T(h_s-h_r).
\]

Erasure sets that coordinate to the fit-only intercept:

\[
E_u(h_r)=h_r-uu^T(h_r-\mu).
\]

After queried-content erasure, rescue from an independent source is

\[
R_C(h_r,h_s)=E_{u_C}(h_r)+u_Cu_C^T(h_s-\mu).
\]

This is not same-run activation addback. The rescue source is a different prompt
with the same content and opposite native answer. No vector is norm-scaled to a
recipient outcome, and no alpha, sign, or source is chosen per item from model
behavior. The projector, source graph, and strength are fixture- and
fit-determined.

Algebraically, `R_C(h_r,h_s)=T_{u_C}(h_r,h_s)`. Consequently, independently
executed `content_rescue_same` and `content_same` rows must agree within the
identity tolerance, as must `content_rescue_opposite` and `content_anticopy`.
The duplicate executions test hook determinism and support differently oriented
contrasts; they are not counted as independent evidence families. The
natural-use upgrade comes from selective erasure damage beyond null erasures and
restoration from a different, opposite-answer prompt, not from counting the
same projected endpoint twice.

Numerical gates for every selective call are:

- projector symmetry and idempotence error at most `1e-6`;
- `(I-uu^T)(h'-h_r)` norm at most `1e-6 * max(1,||h_r||)`;
- selective displacement no larger than the corresponding full
  donor-recipient displacement plus `1e-6` relative tolerance;
- finite activations and logits; and
- exactly one hook call with no surviving hook.

## 9. Staged execution, admission, and localization

### Stage A: immutable plan and engineering preflight

Before any forward, write `design.json`, `plan_manifest.json`, tokenization
receipt, dependency lock, and complete baseline/intervention row templates under
`results/benchmark/single_cell/coherent_readout_v3_cross_codebook/qwen2.5-1.5b-instruct/`.
They must report `model_calls=0`, bind every ledger identity, split, cell, source,
layer, condition, seed, threshold, and expected row count, and refuse overwrite
of non-identical files.

In particular, `design.json` must bind the exact final file hashes of this
preregistration, `eval/run_coherent_readout_v3_cross_codebook.py`,
`eval/analyze_coherent_readout_v3_cross_codebook.py`, every v3 test file, the
tokenization receipt, the dependency lock, and the static artifacts in Section
2. It must also bind the Python, PyTorch, Transformers, NumPy, and tokenizer
revisions, effective chat template, runtime platform class, all expected row
counts, and the deterministic `H_16` sign matrix. The runner's embedded
preregistration hash must equal the design value. This one-way binding is
mandatory and avoids a runner/preregistration mutual-hash cycle.

The runner must rebuild the fixture and plan in memory before every phase and
require byte- and semantic-equality with disk. Completed phase artifacts are
immutable inputs to later phases. Re-entry requires their file hashes, canonical
hashes, exact row sets, and activation-sidecar hashes; a partial or duplicate
phase cannot be silently resumed. Atomic temporary-to-final writes are required.

### Stage B: baseline behavioral admission

All 32 unpatched cells per world are executed before interventions in that role.
Direction-fit, localization, and holdout roles must each separately satisfy:

- global-next-token native-answer accuracy at least `0.95` overall;
- accuracy at least `0.90` in each of the four `p x m` cells;
- text-level `p -> -p` counterfactual accuracy at least `0.95` separately for
  both content-transition directions and both codebooks;
- no global-argmax ties;
- unique global argmax in `{X,Y}` for at least `0.95` of cells;
- mean `{X,Y}` probability mass at least `0.95`; and
- for the registered eight-cell recipient fraction, mean text-counterfactual
  gap `G` and its 95% world-bootstrap lower bound are above zero.

All leave-one-world-out aggregate `G` denominators must also be positive. Role
failure stops before that role's patch calls and is not evidence that the
internal content representation is absent. In particular, localization-role
failure emits `LOCALIZATION_BASELINE_STOP_NOT_ADMITTED` before any localization
patch; it is not relabeled as an engineering failure.

### Stage C: disjoint localization

At each grid layer, use the fit-only frozen axes and the 64 localization
recipients. Run 14 conditions: primary projected anti-copy, same-content
projected, five structural-axis projected, four null-axis projected, full
anti-copy, identity, and full text-counterfactual. This is exactly
`64 * 5 * 14 = 4,480` localization condition rows. All projected controls use
the exact same anti-copy donor displacement as the primary except the named
same-content control; only the projector changes. Natural-use erasure/rescue
does not participate in layer selection.

A layer passes localization only if all hold:

- aggregate `F/G >= 0.30`;
- every leave-one-world-out `F/G >= 0.20`;
- at least 6/8 world-mean `F` values are positive;
- mean `F` is positive in each content-transition by recipient-codebook cell;
- primary minus each of seven negative controls--same-content, five structural,
  and full anti-copy--is at least `0.20 G`;
- every leave-one-world-out primary-minus-structured-control contrast is
  positive; and
- primary exceeds the maximum of the four null mean effects, including in every
  leave-one-world-out analysis.

The selected layer `ell*` is the earliest passing member of
`[8,12,16,20,24]`. If none passes, execution stops before all holdout patches.
The holdout cannot revise `ell*`.

### Stage D: untouched holdout

After holdout behavioral admission, execute exactly 21 frozen conditions for
each of 256 recipients: 5,376 recipient-condition rows at `ell*`. No native
correctness or preliminary patch response may remove a row.

| # | Frozen condition | Source/projector or operation |
|---:|---|---|
| 1 | `content_anticopy` | `T_{u_C}(r,s_AC)`; primary |
| 2 | `content_same` | `T_{u_C}(r,s_SC)` |
| 3 | `answer_anticopy` | `T_{u_answer}(r,s_AC)` |
| 4 | `codebook_anticopy` | `T_{u_m}(r,s_AC)` |
| 5 | `distractor_anticopy` | `T_{u_d}(r,s_AC)` |
| 6 | `query_anticopy` | `T_{u_e}(r,s_AC)` |
| 7 | `order_anticopy` | `T_{u_o}(r,s_AC)` |
| 8-11 | `null_{0..3}_anticopy` | `T_{u_Nk}(r,s_AC)` |
| 12 | `full_anticopy` | full `h_r <- h_AC` |
| 13 | `identity` | full `h_r <- h_r` |
| 14 | `full_text_counterfactual` | full `h_r <- h_TC` |
| 15 | `content_erase` | `E_{u_C}(h_r)` |
| 16 | `content_rescue_same` | `R_C(r,s_SC)`; same content/opposite answer |
| 17 | `content_rescue_opposite` | `R_C(r,s_AC)`; opposite-content/same-answer sham |
| 18-21 | `null_{0..3}_erase` | `E_{u_Nk}(h_r)` |

Conditions 12 and 14 are full-state diagnostics. Condition 14 is expected to be
a strong source-answer positive control but cannot support content specificity.
Condition 12 tests whether full anti-copy retains/copies the shared native
answer. Condition 13 must be numerically identical to an unhooked duplicate.

## 10. Persisted measurements and artifact verification

For every baseline and intervention row persist:

- role, world, cell, recipient, source, layer, site, condition, and factor signs;
- native answer and counterfactual target fixed from the fixture;
- raw X and Y logits and their oriented margins;
- full-vocabulary global maximum token, maximum value, tie count, log-sum-exp,
  `{X,Y}` probability mass, and SHA-256 of the transient little-endian float32
  full-vocabulary row;
- hook count/removal state and numerical projector checks; and
- source, recipient, applied-direction, and patched-activation hashes.

Persist every unpatched source/recipient grid activation in finite
little-endian float32 sidecars with row, matrix, logical-ID-map, and file hashes.
Each source logical ID must resolve to the same activation hash everywhere. The
analyzer independently reloads sidecars, reconstructs each projected or erased
activation, verifies every source link and plan identity, and recomputes all
estimands. Unknown, missing, duplicate, non-finite, misordered, or extra rows are
fatal engineering failures.

Full-vocabulary rows need not be persisted, so global maxima, log-sum-exp, and
mass are replay-verifiable commitments rather than independently recomputable
from logits X/Y alone. Hook-call counts, hook removal, and non-target-token
equality are likewise runtime trace attestations unless the forward is replayed;
the persisted patched-state sidecars make the selected-token intervention
algebra independently reconstructible but do not reconstruct the entire hidden
sequence. No stronger artifact-verifiability claim is allowed.

## 11. Scores and primary estimands

For recipient `i`, let `A_i` be its frozen native answer and `T_i` the opposite
counterfactual target. Define the target-oriented margin

\[
M_i(c)=z_{T_i}(c)-z_{A_i}(c).
\]

Let `B` be the unpatched recipient, `TC` the unpatched text-counterfactual cell,
and `P` the primary projected anti-copy condition. Define

\[
G_i=M_i(TC)-M_i(B),\qquad F_i=M_i(P)-M_i(B).
\]

`G` is the behavioral text-counterfactual scale. `F` is selective causal movement
toward the recipient-codebook interpretation of donor content. Recovery is the
ratio of aggregate means, `mean(F)/mean(G)`, never the mean of item-wise ratios.

For each structured control `j` (same-content, answer, codebook, distractor,
query, order, and full anti-copy), define

\[
K_{ij}=M_i(j)-M_i(B),\qquad S_{ij}=F_i-K_{ij}.
\]

Define four null effects `N_{ik}` analogously. The primary flip indicator is one
only when the unique full-vocabulary global argmax is exactly `T_i`. The weaker
within-channel sign indicator `1[M_i(P)>0]` is reported descriptively and cannot
replace this gate.

For natural-use analysis, define the native-answer margin

\[
Q_i(c)=z_{A_i}(c)-z_{T_i}(c)=-M_i(c).
\]

With C erasure `EC`, same-content rescue `RS`, opposite-content sham rescue
`RH`, and null erasure `EN_k`, define

\[
L_i=Q_i(B)-Q_i(EC),
\]
\[
R_i=Q_i(RS)-Q_i(EC),\qquad H_i=Q_i(RH)-Q_i(EC),
\]
\[
L_{ik}^{null}=Q_i(B)-Q_i(EN_k).
\]

Positive `L` is damage from selective C erasure; positive `R` is restoration.
Reverse damage is not automatically necessity, and rescue after a full-state
replacement is not part of this natural-use estimand.

## 12. Dependence, bootstrap, ratios, and multiplicity

Average all eight recipients within symbolic world first. The 32-world holdout
uses 10,000 percentile cluster-bootstrap draws with seed `260804`. The same
resampled world-index matrix is reused for every primary, control, stratum,
flip-rate, channel, erasure, and rescue metric. Report point estimates, medians,
all world means, positive-world counts, 95% intervals, and leave-one-world-out
values.

For a bootstrap recovery ratio, compute the ratio of resampled aggregate means
on every draw without dropping unfavorable denominators. A zero or non-finite
draw makes the interval invalid and fails the gate. A nonpositive aggregate or
leave-one-world-out `G` makes the relevant ratio undefined and also fails.

The final causal status is an intersection-union decision: every named
structured control must pass separately, so failed controls cannot be averaged.
For the four nulls, first compute `F_i-max_k(N_ik)` within each recipient, then
aggregate by world and bootstrap that simultaneous contrast. Any additional
layer, rank, prompt, strength, axis, split, model, or task is a separate
exploratory family. Standalone claims outside the final intersection use Holm
correction.

The bootstrap quantifies stability across this finite bank of symbolic worlds;
it does not by itself identify a population of models, tasks, or natural facts.

## 13. Frozen holdout gates

### Engineering and channel gates

All scientific statuses require:

- every identity, re-entry, sidecar, row-count, hook, projector, and numerical
  gate in Sections 2, 4, 6, 8, 9, and 10;
- `identity` changes X/Y logits by at most `1e-4` and preserves the exact
  float32 global maximum-token set and tie count;
- each algebraically duplicate direct-patch/rescue pair differs by at most
  `1e-4` in X/Y logits;
- unique global argmax lies in `{X,Y}` for at least `0.95` of primary rows;
- mean `{X,Y}` probability mass under the primary decreases by no more than
  `0.02` from its paired baseline; and
- all other engineering gates pass.

### Content-specific causal recomposition

The causal recomposition tier requires all of:

1. aggregate `F/G >= 0.30`;
2. the 95% bootstrap lower bound for `F/G` is strictly above `0.20`;
3. the 95% bootstrap lower bound for mean `F` is strictly above zero;
4. at least 24/32 within-world stratum means are positive separately in both
   content-transition directions by both recipient codebooks, all four strata
   being mandatory;
5. at least 24/32 world-mean `F` values are positive;
6. the 95% world-bootstrap lower bound for the unique global-argmax
   counterfactual-target flip rate is strictly above `0.50`;
7. for each of seven structured controls separately--same-content, answer,
   codebook, distractor, query, order, and full anti-copy--mean
   `S_j >= 0.20 mean(G)`
   and its 95% bootstrap lower bound is strictly above zero;
8. the simultaneous primary-minus-max-null mean is at least `0.20 mean(G)` and
   its 95% bootstrap lower bound is strictly above zero; and
9. all engineering, behavioral-admission, and channel gates pass.

The four transition-by-codebook stratum requirements cannot be replaced by a
pooled interaction. Likewise, the same-content and full-state anti-copy controls
cannot be merged with the five structural controls, and the structured controls
cannot be replaced by the max-null test.

### Partial natural use

The partial-natural-use tier requires the complete recomposition tier plus all
of:

1. C-erasure damage `mean(L) >= 0.10 mean(G)` and the 95% lower bound for `L`
   is above zero;
2. every C-minus-null-erasure contrast `L-L_k^{null}` has a 95% lower bound
   above zero;
3. same-content/opposite-answer rescue satisfies
   `mean(R)/mean(L) >= 0.70`, `mean(R) >= 0.10 mean(G)`, and a 95% lower bound
   for `R` above zero;
4. `R-H` has a 95% lower bound above zero.

The rescue denominator and all leave-one-world-out `L` denominators must be
positive. Otherwise rescue recovery is undefined and the natural-use gate
fails. This supports only a selective contribution under the frozen linear
intervention. It does not establish that `u_C` is necessary, unique, exhaustive,
or a bottleneck.

## 14. Status hierarchy and claim ladder

Exactly one terminal analyzer status is emitted after the highest reachable
stage:

- `FIT_STOP_BASIS_LOCK_INVALID`;
- `LOCALIZATION_BASELINE_STOP_NOT_ADMITTED`;
- `LOCALIZATION_STOP_ENGINEERING_INVALID`;
- `LOCALIZATION_STOP_NO_PREREGISTERED_LAYER`;
- `HOLDOUT_BASELINE_STOP_NOT_ADMITTED`;
- `FINAL_STOP_ENGINEERING_INVALID`;
- `NO_REPLICATED_PROJECTED_CONTENT_RECOMPOSITION`;
- `NONSPECIFIC_PROJECTED_TRANSFER_REPLICATED`;
- `CONTENT_RECOMPOSITION_SUPPORTED_NATURAL_USE_NOT_ESTABLISHED`; or
- `CONTENT_RECOMPOSITION_AND_PARTIAL_NATURAL_USE_SUPPORTED`.

Engineering failure takes precedence over scientific statuses. A primary
transfer pass without all specificity controls emits
`NONSPECIFIC_PROJECTED_TRANSFER_REPLICATED`. A specificity pass with failed
natural-use gates emits
`CONTENT_RECOMPOSITION_SUPPORTED_NATURAL_USE_NOT_ESTABLISHED`; natural-use
failure does not erase the lower recomposition result. Any failure is scoped to
the frozen rank-one fit rule, grid, site, prompt, model, and bank.

The permitted claim ladder is:

1. **Behavioral composition:** admitted baseline behavior follows the supplied
   symbolic codebook across the complete factorial.
2. **Linear queried-content readout:** the fit worlds contain a nuisance-
   residualized rank-one content coefficient. This is correlational.
3. **Content-specific causal recomposition:** a selective anti-copy transplant
   changes output toward donor content interpreted by the recipient codebook,
   beyond every frozen structured and null control.
4. **Query-resolved specificity:** passing distractor, query, order, answer, and
   codebook controls supports queried content rather than generic surface or
   answer geometry in this setting.
5. **Partial natural use:** selective erasure damage and independent
   same-content/opposite-answer rescue support a local contribution to naturally
   correct computation.

No rung licenses the next one unless its separate gates pass.

## 15. Hard falsifiers and prohibited adaptations

The content-specific interpretation is falsified or bounded if any of the
following occurs:

- the effect does not reverse with the recipient codebook or one codebook alone
  drives it, consistent with direct X/Y steering;
- full-state text-counterfactual transfer works but projected anti-copy transfer
  fails, leaving only generic/full-state transfer;
- the same-content, answer, codebook, distractor, query, order, or max-null
  control reproduces the primary effect;
- either content-transition direction, codebook, or more than eight worlds
  lacks positive transfer;
- apparent flips arise from loss of `{X,Y}` mass or off-channel maxima;
- C erasure does not damage the native-answer margin beyond null erasures;
- rescue is no better than the opposite-content/same-answer sham;
- behavior fails to implement the explicit text counterfactual; or
- fit-only axes, selected layer, source links, or row identities cannot be
  reconstructed exactly.

After the first model forward, do not change the prompt, bank, split, recipient
fraction, source graph, sign coding, answer tokens, layer grid, token site,
projector algorithm, SVD tolerance, null signs, patch strength, conditions,
thresholds, bootstrap seed, status logic, or claim language. Do not:

- select only native-correct items or successful donors;
- tune axis rank, sign, source, or strength on localization or holdout outputs;
- search additional layers or token spans and report them as confirmatory;
- average a failed structured control into a passing composite;
- reinterpret full-state source-answer copying as content transfer;
- use a same-run erased coordinate as the rescue source; or
- reopen the biological fixture from this result.

Any nonlinear subspace, multi-rank representation, other site, prompt, relation,
vocabulary, model, or biology study requires a new preregistration and new
outcome-untouched worlds.

## 16. Maximum claim and nonclaims

The maximum claim after the recomposition status alone is:

> In Qwen2.5-1.5B-Instruct, under the exact synthetic prompt and at the earliest
> passing member of a frozen coarse `resid_post` grid, a fit-only rank-one
> queried-content coordinate causally moved held-out outputs toward donor
> content recomposed through the recipient's unchanged codebook, despite donor
> and recipient sharing the same native answer, and exceeded every
> preregistered structured and sign-flip null control.

With the stronger status, append:

> Selective erasure damaged the native-answer margin beyond null erasures, and
> an independent same-content donor carrying the opposite answer restored a
> preregistered fraction of that damage beyond the opposite-content sham.

Even the stronger result would not prove latent knowledge because every fact and
codebook is supplied in the prompt. It would not prove a biological or natural
activation gap, relative underactivation, causal sufficiency of the whole model,
global necessity, a unique mediator, a bottleneck, a neuron/head circuit,
nonlinear or distributed completeness, general variable binding, transfer to
real semantics, model-family generality, or a physical law. Failure would reject
only this frozen linear-coordinate intervention at the tested sites.

## 한국어 요약

V2의 확정 결과는 item-specific binding이 아니라 일반적인 late X/Y decision
state transfer였다. V3는 이 결과를 재해석하지 않고, 완전히 새로운 56개
symbolic world에서 content와 answer route를 분리한다.

각 recipient의 queried content와 codebook을 동시에 뒤집은 anti-copy donor를
사용한다. Recipient가 `(p,m)`이면 donor는 `(-p,-m)`이므로 두 prompt의 원래
정답은 `pm=(-p)(-m)`으로 같다. 그러나 donor의 content만 recipient에 이식하면
recipient의 고정 codebook이 요구하는 답은 `m(-p)=-pm`, 즉 두 원래 정답의
반대다. 따라서 단순 X/Y answer-state 복사는 성공을 설명할 수 없다.

16개 world의 완전한 `2^5` factorial만 사용해 각 layer의 queried-content
coefficient를 추정하고, 나머지 30개 non-intercept Walsh coefficient span을
제거한 rank-one 방향을 만든다. 별도의 8개 world에서 `[8,12,16,20,24]` 중
causal anti-copy와 control separation을 모두 만족하는 가장 이른 layer 하나를
고정한다. 마지막 32개 holdout world는 layer와 방향이 고정되기 전에는 열지
않는다.

Holdout은 world당 균형 잡힌 8개 recipient, 총 256개에 대해 21개 조건,
5,376개 intervention row를 실행한다. Primary content patch 외에 same-content,
answer, codebook, distractor, query, order, 네 개 sign-flip null, full-state,
identity, erasure와 독립 rescue/sham 조건을 모두 따로 판정한다.

Primary 성공에는 `F/G >= .30`, ratio bootstrap 하한 `>.20`, `F` 하한 `>0`,
두 content 방향과 두 codebook의 네 stratum 각각에서 32개 world 중 최소
24개 양수, 전체에서도 최소 24개 world 양수, global-target flip-rate 하한
`>.50`, 일곱 structured control 각각 대비 최소 `.20G` 우위, max-null 대비
최소 `.20G` 우위, X/Y channel 보존이 모두 필요하다. 자연적 사용에 대한
더 강한 status는 C erasure damage `>=.10G`, null-erasure 우위,
same-content/opposite-answer rescue `>=70%` 및 `>=.10G`, 그리고
opposite-content/same-answer sham 우위를 추가로 요구한다.

전체 성공 후에도 가능한 결론은 이 모델, prompt, coarse layer, synthetic
task에서 queried-content coordinate의 국소적 causal recomposition과 제한된
partial natural use까지다. 잠재지식, 생물학, activation gap, underactivation,
고유 circuit, 일반 variable binding, model-family 일반화 또는 물리 법칙은
증명되지 않는다.
