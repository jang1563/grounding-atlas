# Pre-registered design: layer-localization of the encode-vs-express gap (Qwen3-8B)

Status: PRE-REGISTERED. Hypotheses, thresholds, and decision rules below are fixed BEFORE
running the outer-fold scoring. Naive (selection-biased) numbers may be inspected during the
pipeline dry-run; the unbiased nested-CV outer-fold AUROCs and the verbalize curves are the
locked endpoints and are not to be looked at before the analysis in Section 5.

The protocol is nested CV + selectivity + cluster bootstrap + a positive control. The unbiased
machinery is a hard pre-submit precondition for EVERY task (it existed in code for hERG and the SFM
arm only); the PRIMARY task's surface confound is controlled by an explicit residualization gate;
the H2 gap is re-baselined against the positive control so "supervision helps" is subtracted out;
and the launch path is reconciled with the scripts that actually submit.

Repo: `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval`
Primary model: `Qwen/Qwen3-8B` (36 transformer blocks, hidden 4096, so 37 hidden-state tensors
including the layer-0 embedding output). Open-weight, already cached on Cayuga.

---

## 1. Question and why now

We have repeatedly measured an "encode much greater than express" gap: an open LLM (or an SFM)
linearly encodes a binary biological property in its hidden states (held-out probe AUROC well
above chance) while verbalizing it near chance. The forward goal is a calibrated LLM-SFM bridge:
a trained read-out head that reads the encoded signal, attached at the right place, with a
calibration that tells a router when to trust it. Before training that head we need to know
WHERE in the network the property is linearly available and WHERE verbalization breaks, so we
know which layer to attach the read-out and the calibration to. This experiment is the cheap
GPU warm-up (one forward pass over each task, then CPU/sklearn probing) that localizes both
halves. It also fixes a concrete methodological bug from our own prior runs: the 0.5B SFM probe
reported best-layer 0.71 but held-out-layer 0.599 (a +0.11 selection-bias inflation from picking
the best layer on the same data used to score it; `results/sfm_embedding_activation_Qwen2.5-0.5B-Instruct.json`).
The fix is nested cross-validation plus selectivity controls, so a high probe AUROC means the
layer encodes the PROPERTY, not a surface artifact or probe memorization.

A high probe AUROC is only interpretable as "encoding" if three confounds are ruled out, and the
review made clear that two of them were asserted in prose but not enforced in code: (a) selection
bias over the 37 layers, which the nested-CV fix removes but which existed in code for hERG/SFM
only; (b) a SURFACE confound (the probe reading lexical composition rather than the biological
property), which a shuffled-label control alone cannot catch because permuting labels destroys the
confound-to-label correlation along with everything else; and (c) the supervised-vs-zero-shot
asymmetry in the encode-vs-express gap, where a trained linear read-out beats a greedy decode on
almost any signal even with no genuine express deficit. This design treats all three as gating
preconditions, not as caveats.

---

## 2. Pre-registered hypotheses (each falsifiable, with a stated threshold)

All AUROC thresholds are on the UNBIASED nested-CV outer-fold estimate (Section 4.3), with a
95 percent CLUSTER bootstrap CI on the grouped tasks (resample groups, take all their rows;
Section 4.6) and an iid bootstrap CI on the genuinely ungrouped single-cell task. "Chance" = 0.50.
Every threshold below is additionally gated on a pre-registered power floor (Section 2.1): a task
may only CONFIRM a hypothesis if its minimum detectable effect at the realized n is at or below the
threshold it is being asked to clear.

- H1 (encoding peak is at a mid-to-late band, not the embedding floor and not only the final
  layer). For each expression-gap task, the per-layer peak-SELECTIVITY layer (Section 4.4, not the
  raw-AUROC peak) falls in the depth band fraction 0.45-0.80 of Qwen3-8B, i.e. layers ~16-29 of 36
  (literature band 0.55-0.76, anchors 2505.14352 L32/42, 2510.01070 single-layer 0.55-0.76,
  2303.08112 mid-depth, 2312.01037 "especially middle layers"). FALSIFIED for a task if the
  peak-selectivity layer sits at layer 0-3 (lexical/surface; depth < 0.10) OR only at the final
  layer 35-36 (depth > 0.95) across all 5 outer folds. Because the selected layer is itself a
  high-variance pick over 37 correlated candidates, H1 is evaluated on a layer BAND, never a point:
  per task we report the across-fold selected-layer set AND the within-1-SE band of the inner
  argmax (Section 4.3). A jumpy argmax with no consistent band (selected layers spanning more than
  half the network, e.g. range > 18 layers, across folds) is reported as "broad/no clean peak" and
  also falsifies the point-localization form of H1. H1 is adjudicated on the two HEADLINE tasks
  (DNA, hERG) only; the single-cell pair contributes a depth-band DESCRIPTION, not a layer-shift
  conclusion (Section 2.1).

- H2 (a real encode-vs-express gap that PERSISTS under nested CV, survives selectivity, and is not
  merely "supervision helps"). For the expression-gap tasks the raw gap = (unbiased nested-CV
  held-out probe AUROC) - (verbalize-arm output AUROC on the SAME items, same orientation). The
  LOCKED H2 endpoint is the EXCESS gap = raw_gap(task) - raw_gap(`msa/conservation`), where the
  positive control msa is the case the model DOES verbalize, so its raw gap is the floor that a
  supervised probe beating a zero-shot decode produces with no express deficit. H2 CONFIRMED for a
  task requires EXCESS gap >= +0.15 AND the excess-gap cluster-bootstrap 95 percent CI excludes 0
  AND probe selectivity at the selected layer >= +0.15 (Section 4.4) AND, for DNA specifically, the
  GC-residualized probe-minus-surface-floor margin >= +0.10 (Section 4.4 control 2). For hERG the
  raw gap is also reported on the LOW-FAMILIARITY (non-approved-drug) subset (Section 4.4 control 3)
  so the gap is not carried by memorized famous blockers. Predicted raw magnitude +0.30 to +0.55
  (probe 0.85-0.92 vs verbalize ~0.50-0.55); predicted msa floor ~0.10-0.20. FALSIFIED for a task
  if, after the nested-CV deflation, the EXCESS gap drops below +0.05 (CI includes 0) OR
  selectivity < +0.15 OR (DNA) the GC-residualized margin < +0.10 OR (hERG) the gap does not
  survive on the low-familiarity subset. The headline methodological number is the
  naive-minus-nested optimism gap, reported with a PAIRED bootstrap CI on the difference (Section
  4.6); we predict +0.05 to +0.12 (internally anchored to our own +0.11, FLAGGED as the
  least-externally-anchored prediction) and treat a deflation whose paired CI includes 0 as "no
  measurable optimism," not as a real shrink.

- H3 (the read-out's per-item confidence is calibrated and adds routing signal over a FAIRLY-TUNED
  output baseline). On the outer-fold out-of-fold probe predictions at the selected layer ℓ*
  (returned by the nested-CV function, Section 4.3), the read-out is better calibrated and more
  selectively-predictive than a TEMPERATURE-SCALED verbalize baseline (the bare zero-shot output is
  a supervision-free comparator and is rigged toward the fitted probe; Section 4.5). Concretely:
  probe ECE <= 0.15 with its bootstrap CI reported, AND the probe-minus-baseline AURC difference is
  negative with a bootstrap 95 percent CI that EXCLUDES 0 (lower AURC = a confidence ranking that,
  when you abstain on the least-confident items, recovers accuracy faster), AND probe
  risk-coverage is monotone (accuracy at 50 percent coverage > accuracy at 100 percent coverage by
  at least +0.05). The PRIMARY H3 claim is the conditional/stacking form: does ℓ*-probe confidence
  add routing signal OVER the verbalize prediction (incremental AURC of probe-given-output)? FALSIFIED
  if the probe-minus-baseline AURC-difference CI includes 0 on every expression-gap task (the
  encoded read-out gives no routing signal beyond a fairly-tuned output) OR probe ECE > baseline ECE
  everywhere. ECE is reported at 10 bins AND 5 bins with a bootstrap CI on each, because at the
  per-class OOF counts here (~190-310 items) a single 10-bin ECE has a ~0.03-0.05 half-width.

### 2.1 Power floor (pre-registered minimum detectable effect)

The locked thresholds (gap CI excludes 0, selectivity >= +0.15, H1 across-fold range) are only
admissible on a task whose realized n can resolve them. Per task we pre-register the minimum
detectable gap whose CI excludes 0 at 80 percent power, computed from the realized balanced n and
fold structure (5 outer folds):

- DNA (n=1500, ~300 test/fold): minimum detectable gap ~0.06-0.08; can adjudicate H1 and H2.
- hERG (n<=1250, capped by 625 positives, see Section 3): minimum detectable gap ~0.07-0.09; can
  adjudicate H1 and H2 as a SINGLE-SHOT estimate (no held-out positive block for replication,
  because the balanced sample consumes the entire minority class; stated, not worked around).
- single-cell (n<=470 realized in the committed file, ~77-94 test/fold): minimum detectable gap
  ~0.12-0.15. A two-argmax peak-SHIFT between name and anon, each localized on ~300 train rows,
  has NO power to pin a layer to +/-9. Therefore single-cell is DEMOTED to descriptive-only for any
  layer-localization or peak-shift claim: it contributes an OUTPUT contrast (web-rich name vs
  web-zero anon verbalization) and a depth-band description with wide CIs, and CANNOT confirm or
  kill H1's web-exposure peak-shift. The Section 8.4 attach-depth-by-web-exposure idea is therefore
  EXPLORATORY (Known limitations), not a locked deliverable.
- msa positive control (n per signal file): used as the H2 floor; its pass-band is numeric
  (Section 6), not "small."
- ESM `protein/esm2_emb` (n=214): methodological replication only; read off the CI, not the point.

---

## 3. Models and tasks

Primary model: `Qwen/Qwen3-8B` (open-weight hidden states; the frontier Claude models have no
exposed hidden states, so layer-localization is necessarily done on the open 8B and is NOT a
claim about whether scale recovers the OUTPUT). Optional cross-architecture replication on
`meta-llama/Llama-3.1-8B-Instruct` and `allenai/OLMo-2-1124-7B-Instruct` for the H1 layer-band
generalization only (same sbatch, `ACT_MODEL=` override), reported as secondary.

Tasks (ranked from Grounding 2; n is the balanced sample available). Each is run through the
existing per-modality activation arm so the probe, the verbalize arm, and the selectivity control
are all on the SAME items under ONE split. Every arm is brought to parity on the unbiased machinery
before submission (Section 4.3 / Section 7); the table below states the ranking AFTER the
data-validity audit.

| Rank | Task id (registry / arm) | n | Why | web | Specialist ceiling |
|---|---|---|---|---|---|
| 1 ANCHOR/HEADLINE | `admet/herg` (`eval/activation_arm.py`, `data/herg.csv`) | <=1250 balanced (625 pos cap; Murcko-scaffold GroupKFold) | Canonical gap (held-out act 0.760 vs out 0.453); scale/architecture-invariant; notation-invariance already tested; scaffold-GROUPED (leakage-controlled); 4 prior controls | rich | 0.892 (Morgan) |
| 1 HEADLINE | `dna/promoter` (`eval/activation_arm_dna.py`) | 1500/class, balanced (sequence-CLUSTER GroupKFold; see control 2) | Largest robust gap in prior runs (act 0.880 vs out 0.396); output anti-correlated. Data identity DISCHARGED (3000 rows pure ACGT, balanced; Section 3). Stays headline ONLY while the GC-residualized margin clears +0.10 (the GC confound is empirically confirmed, class-GC 0.482 vs 0.625); until then it is a possibly-surface-confounded upper bound | mixed | 0.88 (probe / 6-mer) |
| CONTROLLED PAIR (DESCRIPTIVE) | `single_cell/cd8t_nk` name (`cell_sentence`) and `anon` (`eval/activation_arm_sc.py`) | <=470 realized (StratifiedKFold) | SAME cells, global-consistent anon relabel, bag-of-tokens ceiling ~0.99 both forms; provides the OUTPUT web-rich-vs-web-zero contrast. Demoted from any layer peak-shift claim by the power floor (Section 2.1) | rich / zero | 0.989-0.992 |
| POSITIVE CONTROL + H2 FLOOR | `msa/conservation` (`eval/activation_arm_msa.py`) | per signal file | Web-RICH mapping the model verbalizes (output 0.795, activation ~1.0). Doubles as (a) the pipeline parser sanity gate and (b) the QUANTITATIVE subtraction baseline for the H2 excess gap (Section 2 H2) | rich | 0.999 |
| SELECTIVITY CONTROL (built into every task) | shuffled-label / control-task refit + GC-residualization (DNA) + token-shuffle (hERG/DNA) + re-notation (hERG) | same n | Hewitt-Liang control plus the confound-residualization and chemistry-destroying controls the review required (Section 4.4) | n/a | n/a |
| OPTIONAL SECONDARY | `histo/pcam_tumor` (`eval/activation_arm_histo.py`, Qwen2.5-VL-7B) | 400 | Largest vision gap (act 0.827 vs out 0.463) but small n, frontier-specific closure, no frontier-VLM hidden states; run only if the 7B VLM hosts locally | rich | 0.90 (CONCH) |

Note (Grounding 2 caveat): the web-zero numeric-vector tasks (`single_cell/*:anon`, `methyl/age`)
encode essentially to ceiling because a linear probe on tokenized input recovers input variance.
They are RETAINED only as the web-zero arm of the controlled pair (the OUTPUT contrast is the
result), NOT as standalone H2 targets. H2's clean targets are tasks where encoding is BELOW ceiling
(hERG 0.892 ceiling vs ~0.76 held-out act; DNA after GC residualization), so content is cleanly
separated from surface. The positive control (`msa/conservation`) anchors the gap scale and is the
H2 subtraction floor.

Data-identity precondition (PRIMARY): DISCHARGED by direct audit 2026-06-26. `signal/dna_promoter.csv`
contains 3000 rows, ALL pure A/C/G/T, length 251, balanced 1500 positive / 1500 negative; it IS genuine
promoter SEQUENCE data, not SMILES. The column header reads `smiles` (a cosmetic mislabel from a reused
build template; the arm reads it as the sequence string, so the misnomer is harmless to scoring). Two
RESIDUAL preconditions remain and are gating: (a) the rows are CLASS-BLOCK ordered (all 1500 positives
then all 1500 negatives), so the loader/splitter MUST shuffle and stratify; a non-shuffled split would be
single-class. Confirm shuffle is on in the arm before locking. (b) The GC-content confound is EMPIRICALLY
CONFIRMED, not hypothetical: measured class-0 mean GC 0.482 vs class-1 mean GC 0.625 (delta ~0.14, sd
~0.11-0.13), so promoter vs non-promoter is strongly separable on GC/composition ALONE. The Section 4.4
control-2 GC-residualization is therefore MANDATORY and its +0.10 kill criterion is the load-bearing
safeguard for DNA (a layer-0 or surface probe will score high on this task BY CONSTRUCTION). DNA remains
a possibly-surface-confounded upper bound until the GC-residualized margin clears +0.10; the data
identity itself is no longer in question.

The SFM-embedding probe (`protein/esm2_emb`, `eval/sfm_embedding_activation.py`, n=214 tercile-extreme)
is run as the DIRECT methodological replication of the prior inconclusive result: we re-measure
naive best-layer vs nested-CV held-out at 8B and report whether the +0.12 selection bias collapses.
It is a partial-encoding case (ceiling 0.81-0.85, prior held-out 0.655), NOT a clean expression
gap; flagged as such per the Section 2 / Grounding 2 partial-encoding note.

---

## 4. Method

### 4.1 Per-layer hidden-state extraction
Reuse the established arms (`eval/activation_arm.py` and the per-modality `activation_arm_*.py`).
Each forwards every item once with `output_hidden_states=True`, yielding 37 tensors for Qwen3-8B
(index 0 = post-embedding floor; indices 1..36 = post-block residual states). KEEP layer 0 in the
sweep as the surface floor (a property readable at layer 0 is lexical, not computed; load-bearing
for the H1 selectivity argument). Token position: primary readout = LAST non-pad token (`h[0,-1]`
in the single-example arms; `h[arange, attention_mask.sum(1)-1]` with `padding_side="left"` in the
batched SFM arm) because in a causal decoder only the last position has attended over the full
input and is the state that conditions generation (the verbalize half). ROBUSTNESS arm: add
mean-pool over content tokens (exclude pad/BOS/template) and report BOTH; do not average them. The
last-vs-mean dissociation is itself a result (last >> mean = assembled at the decision position;
mean >> last = distributed but not routed to the output position, the mechanism of an express gap).
This mean-pool path exists in NO current arm and is added in the shared helper (Section 4.3), not
left as prose. Cache activations to disk once (`.npy` per task; ~1.2 GB for n=2000 at fp16) so all
probing is CPU/sklearn.

### 4.2 Probe (encoding question) -- linear, NOT the tuned lens
Per (layer, position): `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
class_weight="balanced"))`, scored by AUROC. The linear, regularized probe is the defensible
choice for "is the property linearly decodable here": low capacity keeps the control-task AUROC
near chance (Hewitt-Liang). The tuned lens is deliberately NOT used for encoding -- it decodes
into the 150k-token vocabulary (the EXPRESS question) and is far higher-capacity, which would
overstate encoding and reintroduce the memorization risk we are controlling.

Regularization is PRE-REGISTERED as fixed L2 with C = 1.0 (the value already in the committed
arms), NOT a tuned grid. The earlier draft described tuning C on {0.01, 0.1, 1, 10} jointly with
the layer INSIDE the inner CV; the committed `heldout_layer_auroc` does not implement that, and a
37-layer x 4-C = 148-candidate inner argmax both reinvites optimism into the SELECTED layer (the
H1 endpoint) and is not what the code does. We therefore lock fixed C = 1.0 so the inner candidate
count is 37, not 148, and `class_weight="balanced"` is used identically in BOTH the probe pipeline
and the nested-CV `clf_factory` so the operating point that ECE/sel_acc read is consistent with the
AUROC (the committed arms used unweighted LogisticRegression in the factory; the helper is updated
to pass the balanced factory). If a C sweep is ever revisited it must be pre-registered separately
with ℓ* reported as a band and the multiplicity-corrected SE of the inner argmax; it is out of
scope here.

### 4.3 The EXACT nested-CV protocol (removes the +0.11 best-layer optimism), now in EVERY arm
This is the locked fix. It existed in code for hERG (`eval/activation_arm.py` lines 134-160) and
the SFM arm (`eval/sfm_embedding_activation.py` lines 43-60) ONLY; the DNA, single-cell, and MSA
arms emitted just `best = max(roc_auc_score ... for L in range(layers))`, which is the
selection-BIASED number this whole design exists to remove. Porting the protocol into those arms is
a BLOCKING pre-submit precondition (Section 7): until the PRIMARY (DNA), the controlled pair
(single-cell), and the sanity-gate (MSA) emit a nested-CV number, their hypotheses cannot be scored
as written, and "do not look at the biased number first" is violated because the biased `max` is
the only number they produce.

Implementation (single source of truth): lift `heldout_layer_auroc` into a shared module
(`eval/probe_common.py`) and call it from `_dna`, `_sc`, `_msa`, and the SFM arm, parameterized by
a `splitter_factory` so the same logic serves GroupKFold (hERG/DNA/SFM) and StratifiedKFold
(single-cell, which has no groups) without forking. Two signature changes (one-line each) so the
downstream metrics are computable: the function RETURNS the outer-fold OOF probability vector and
the per-fold ℓ* list, not just `(auroc, picked)` (the OOF vector is currently discarded at line
160). Protocol, stratified/grouped throughout:

1. OUTER loop: `GroupKFold(5)` (group = Murcko scaffold for hERG; sequence CLUSTER for DNA, see
   control 2; sequence/structure cluster for SFM; `StratifiedKFold(5)` for single-cell, which has
   no group structure). Each outer-test fold is touched ONCE for final scoring; nothing -- layer,
   scaler -- is chosen on it.
2. INNER loop: on the outer-TRAIN rows only, `GroupKFold(min(5, n_groups_train))` (or
   `StratifiedKFold` for single-cell), select the best layer ℓ* (argmax mean inner-validation AUROC
   over the 37 layers) at fixed C. Because ℓ* is a max over 37 correlated inner AUROCs, the SELECTED
   layer is reported as a band: alongside each fold's argmax we record the within-1-SE set of layers
   and report the across-fold union as the H1 localization (never a single point).
3. Refit scaler+probe at ℓ* on the full outer-train, score AUROC on the untouched outer-test, and
   write the per-fold OOF probabilities into the returned vector. Standardizer is refit INSIDE each
   fold (fitting it on all data is a subtle leak).
4. Headline = mean of the 5 outer-test AUROCs (this already accounts for selection, so the +0.11
   gap collapses). Also report the INNER-OOF AUROC of ℓ* next to the outer; if inner >> outer for
   the selected layer, that gap is the residual selection inflation and belongs in the optimism
   headline.

Report ALL THREE for transparency: (i) naive max-over-37-layers AUROC (the biased number, printed
as "MAX over layers, selection-biased"), (ii) nested-CV unbiased AUROC, (iii) their difference =
the quantified optimism, with a PAIRED bootstrap CI on the difference (Section 4.6) so a +0.05
deflation is not reported as real when its CI includes 0. Showing (iii) shrink with a CI excluding 0
is the methodological contribution.

### 4.4 Selectivity controls (so high AUROC means encoding, not probe power or SURFACE)
Four orthogonal controls. The existing arms implement (1) for hERG/DNA only; (2)-(4) are added
because the review showed a shuffled-label control alone is blind to a composition/orthography
confound (permuting y destroys the confound-to-label correlation, so the probe can be 100 percent
surface and still show large selectivity).

1. Control-task / shuffled-label baseline (Hewitt-Liang, EMNLP 2019). Refit the IDENTICAL probe
   pipeline on permuted labels (`np.random.RandomState(123).permutation(y)`), same nested splits,
   in EVERY arm (single-cell and MSA currently run no selectivity control; this is added).
   Selectivity = real-task AUROC - control-task AUROC, reported PER LAYER. The honest localization
   of where the property is COMPUTED is the layer of peak SELECTIVITY, not peak raw AUROC. Keep the
   probe linear+regularized so control stays near 0.50. Localize ℓ* on a SEPARATE inner resampling
   (nested bootstrap of the inner argmax) from the one used to score, so the "picked-layer" stability
   list is an honest localization and not a double-use of the scoring partition.

2. SURFACE-CONFOUND residualization (the DNA-blocking control). For DNA the linear probe can read
   GC-content / k-mer composition, which is the most lexical possible cue and is observationally
   identical to "the residual stream carries the input's GC content." The shuffled-label control
   cannot see this. Therefore, as a PRE-REGISTERED kill criterion: (a) fit logistic regressions
   predicting y from GC% alone and from the full 6-mer vector and report their AUROC as the
   "surface floor"; (b) report, for the selected layer, the GC-RESIDUALIZED probe AUROC -- either
   the probe refit on hidden states with GC% (and length) partialled out, OR the AUROC computed
   WITHIN GC-decile strata and averaged. H2-DNA is KILLED if probe-minus-surface-floor < +0.10
   after GC residualization (the layer encodes composition, not promoter semantics). The same
   logged GC separation (per-class GC%, sequence-length sd) is the provenance check of Section 3.

3. Chemistry-destroying token control + familiarity covariate (the hERG strengthening). (a)
   Re-notation: `ACT_RANDOMIZE=1` feeds deterministic non-canonical SMILES while keeping the
   scaffold split and Morgan probe canonical (prior: held-out 0.787->0.739, still above char-n-gram
   0.812). Because BPE keeps most chemical substring tokens under atom renumbering, ADD a
   token-IDENTITY-preserving but chemistry-DESTROYING control: probe on SHUFFLED SMILES BPE tokens
   (same token multiset, broken connectivity). Chemistry read-out should fall toward chance; a
   bag-of-tokens read-out should not. Report both. (b) Pretraining-familiarity covariate: a scaffold
   split prevents structural leak between folds but not pretraining recall of famous blockers
   (astemizole, cisapride, dofetilide, terfenadine) with their cardiotox label. Stratify the hERG
   probe AUROC by approved/named drug (high web-exposure) vs screening compound, and report the gap
   and probe AUROC on the LOW-familiarity subset. If encoding only survives on famous drugs the
   "encode" claim is partly memorization (and this directly serves the web-exposure thesis).

4. Layer-0 floor + the probing-classifier caveat (Belinkov 2102.12452). If layer-0 AUROC is near
   the peak, the property is lexically given, not computed (report it; it reframes the claim). All
   probe results are phrased "linearly decodable at layer ℓ," NEVER "the model knows at layer ℓ."
   OPTIONAL stronger causal check (stretch, not gating any hypothesis): amnesic removal of the
   probe direction (INLP/nullspace) from the layer and re-measure the verbalize arm.

### 4.5 Verbalize arm (the express half), measured on the SAME items
Two complementary readouts, both on the identical items/orientation as the probe:

a) Zero-shot output AUROC. Use each arm's OWN generate step (the `OUTPUT` line:
   `activation_arm_dna.py:130`, `_sc:101`, `_msa:99`, `activation_arm.py:235`) as the verbalize
   number for H2/H3. The earlier draft routed this through `eval/run_grounding_eval.py --model
   oai:Qwen/Qwen3-8B`, but (i) that path needs a separately scheduled local vLLM server the compute
   plan never budgeted, and (ii) the registry (`eval/benchmark_tasks.py`) has NO `dna/promoter` key,
   so `--tasks dna/promoter` hits the unknown-tid skip branch and silently scores nothing for the
   PRIMARY task. We therefore import the harness metric FUNCTIONS (`ece`, `aurc`, `sel_acc`, `ci`
   from `run_grounding_eval.py`, lines 145-185) and call them DIRECTLY on each arm's `outp` and the
   probe's returned OOF vector, rather than running the harness CLI. Respect each task's orientation
   (`orient="oppose"` flips y for ames/solubility/permeability; align elsewhere) so encode and
   express are compared in the same direction, and PRINT the parser fallback rate per task
   (`activation_arm.py` already does; `_dna`/`_sc`/`_msa` are updated to print it). For the H2 and
   H3 comparisons, items pinned at the 0.5 parser fallback are reported as a rate and a calibrated/
   few-shot output AUROC is reported alongside argmax for the two HEADLINE tasks so the gap is not
   carried by fallbacks. The H2 endpoint is the EXCESS over the msa floor (Section 2), and the H3
   comparator is the TEMPERATURE-SCALED output (below), so neither headline rests on a bare
   supervised-vs-zero-shot contrast.

b) "Where verbalization breaks" -- TRIANGULATED, not lens-only. The express-break layer is NOT
   drawn from the tuned lens alone, because a tuned lens is trained to match the final-layer
   distribution by KL and therefore makes intermediate distributions look more like the output the
   deeper they are; a "rise then fall" of the answer-token probability can be a translator-capacity
   artifact (the same identifiability problem the lens is meant to solve for the encoding half). The
   break-layer claim REQUIRES agreement of two methods that do not decode to vocabulary, with the
   lens as corroboration only: (i) the linear-probe AUROC curve already computed -- if probe AUROC
   is high at L_mid while the answer-token logit is not top-1 at the final layer, that dissociation
   localizes the break without a lens; (ii) a CAUSAL patch -- patch the L_mid hidden state (or its
   probe-direction component) into a forward pass and check whether the OUTPUT token flips; if
   patching at L_mid recovers the answer but the unpatched run does not, the break is downstream of
   L_mid. The tuned lens (Belrose 2303.08112; per-block affine `h'_ℓ = A_ℓ h_ℓ + b_ℓ`, init `A_ℓ=I`,
   `b_ℓ=0`, frozen final LayerNorm + unembedding, trained by KL to the model's FINAL-layer
   distribution on a DISJOINT text split, never the eval entities) is reported as a third,
   corroborating curve; if its training is not worth the first-pass cost, report the logit lens and
   FLAG it as a low-confidence estimate. This is NEW code; gate the entire break-layer arm behind a
   confirmed H1/H2, and label the break-layer the most EXPLORATORY deliverable.

### 4.6 Bootstrap and CIs (cluster-resampled on grouped tasks)
The earlier code did iid example resampling (`bootstrap_ci` used `rng.choice(idx, len(idx))`; the
harness `ci()` used `rng.integers(0, len(y), len(y))`), which UNDERSTATES the CI when entities
recur (hERG scaffolds with multiple analogs, DNA homolog families) -- the precise small-n
over-estimate failure mode in our history. Implement a CLUSTER (block) bootstrap (resample GROUPS
with replacement, take all their rows) and use it for every HEADLINE CI on the grouped tasks (hERG,
DNA) and for the H2/H3 CIs on those tasks. Keep the iid bootstrap only for the genuinely ungrouped
single-cell task. Seed each bootstrap from the per-task `rng` (not a hard-coded `RandomState(0)`) so
CIs across tasks do not share one stream. The naive-minus-nested optimism, the H2 excess gap, and
the H3 AURC difference are each evaluated with a PAIRED bootstrap (resample once, recompute both
quantities on the same resample, CI on the difference).

---

## 5. Metrics and analysis

Per task, report:

1. Per-layer probe AUROC curve (37 points), last-token and mean-pool, each with a 95 percent
   bootstrap CI band (cluster-resampled on hERG/DNA); overlay the per-layer SELECTIVITY (real minus
   control) curve.
2. Unbiased best-layer: nested-CV outer-fold mean AUROC with cluster/iid-bootstrap CI, the
   across-fold selected-layer set AND the within-1-SE band (H1 localization as a band), the
   inner-OOF AUROC of ℓ* next to the outer, and the depth fraction ℓ*/36.
3. The optimism gap: naive max-over-layers AUROC minus nested-CV AUROC with a PAIRED bootstrap CI
   (predicted +0.05..+0.12; reported as "no measurable optimism" if the CI includes 0).
4. The encode-vs-express gap (H2): the RAW gap (nested-CV held-out probe AUROC minus same-item
   verbalize AUROC) AND the LOCKED EXCESS gap (raw gap minus the msa raw gap) with the excess-gap
   CI; plus probe selectivity at ℓ*; plus, for DNA, the GC-residualized probe-minus-surface-floor
   margin; plus, for hERG, the gap on the low-familiarity subset.
5. Calibration / selective prediction (H3), reusing `ece`, `aurc`, `sel_acc` from
   `run_grounding_eval.py` applied to the probe's OOF vector at ℓ* (returned by the nested-CV
   function) with the SAME orientation as the output: probe ECE (10-bin and 5-bin, each with CI) and
   AURC vs the TEMPERATURE-SCALED verbalize baseline, the probe-minus-baseline AURC difference with
   its CI, the INCREMENTAL routing value of probe-given-output, and the risk-coverage point
   (accuracy at 50 percent vs 100 percent coverage).
6. The triangulated express-break layer: probe-curve dissociation + causal-patch flip, with the
   tuned-lens property-token curve (emergence layer, any suppression layer) as corroboration.

Cross-task synthesis: do the two HEADLINE expression-gap tasks (DNA, hERG) share a layer band (H1)?
Does peak-layer selectivity rank-track where a read-out should attach (feeds Section 8)? The
single-cell pair contributes the OUTPUT web-rich-vs-web-zero verbalization contrast and a depth-band
DESCRIPTION only; any web-exposure peak-SHIFT statement is explicitly exploratory (Section 2.1).

---

## 6. Pre-registered decision rules and falsification

Confirm / kill, per hypothesis (all on the UNBIASED nested-CV numbers, with cluster/iid CIs, and
only on tasks that clear their Section 2.1 power floor):

- H1 CONFIRMED if for the HEADLINE tasks (DNA, hERG) the peak-SELECTIVITY layer band lands in depth
  0.45-0.80 (layers ~16-29) AND the across-fold selected-layer range is <= 18 layers (a band, not
  noise). KILLED if the peak-selectivity layer is at layer 0-3 (surface) or only 35-36 (final) for
  both headline tasks, OR the selected layer is effectively uniform-random across folds (range > 18)
  for both -> conclude "no clean encoding peak; localization is broad," report a band. Single-cell
  is NOT used to confirm or kill H1 (power floor). (Pre-registered split per Grounding 4 caveat D:
  the decoder-LLM verbalization-localization band is 0.55-0.76; the SFM-side encoding probe
  `protein/esm2_emb` may legitimately peak LATER, even final, for scalar regression -- judged
  separately, not against the 0.45-0.80 window.)

- H2 CONFIRMED for a task if the EXCESS gap (raw gap minus msa raw gap) >= +0.15 with its CI
  excluding 0 AND selectivity >= +0.15 AND (DNA) the GC-residualized probe-minus-surface-floor
  margin >= +0.10 AND (hERG) the gap survives on the low-familiarity subset AND the naive-minus-
  nested optimism is reported with its paired CI. KILLED (gap is an ARTIFACT) if after nested-CV the
  EXCESS gap < +0.05 / CI includes 0, OR selectivity < +0.15, OR the chemistry-destroying token
  control / GC residualization does NOT drop the probe (the probe read format/composition, not
  content), OR (DNA) the data-identity precondition (Section 3) is unmet. The positive control
  `msa/conservation` must clear a NUMERIC pass-band: raw gap < 0.10 AND output AUROC within the
  bootstrap CI of the known 0.795. If msa fails this band the verbalize arm is mis-instrumented
  (parser/orientation bug) and ALL gaps are suspect -> halt and debug (the pipeline sanity gate; a
  gate with no numeric band and +/-0.07 noise could pass a broken parser, so the band is mandatory).
  A specific watch on DNA: the output is anti-correlated, so verify whether the layer-level signal
  is also reversed (encoding present, sign-correct) or the reversal is an output-stage phenomenon;
  report which.

- H3 CONFIRMED if on at least one expression-gap task the probe-minus-(temperature-scaled-output)
  AURC difference is negative with its bootstrap CI excluding 0 AND probe ECE <= 0.15 (with CI) AND
  risk-coverage is monotone (acc@0.5cov - acc@1.0cov >= +0.05) AND the incremental routing value of
  probe-given-output is positive. KILLED if the probe-minus-baseline AURC-difference CI includes 0
  on every expression-gap task (no routing signal beyond a fairly-tuned output) OR probe ECE >
  baseline ECE everywhere.

Global "do not over-claim" rules (from our own history): (a) trust only robust n -- DNA(1500),
hERG(<=1250) are headline; single-cell(<=470)/histo(400) report wider CIs and are corroborating
or descriptive, not decisive; anything n<200 (e.g. ESM n=214) is a methodological replication, read
off the CI not the point. hERG's balanced sample consumes the entire 625-positive minority class,
so it is a SINGLE-SHOT estimate with no held-out positive block for replication; this is stated, not
worked around. (b) Every reported AUROC carries a bootstrap CI (cluster-resampled where entities
recur); cross-sample replication where the n permits it. (c) Layer-peak location is not mechanism --
interpret it only via the controlled comparisons (re-notation, chemistry-destroying token shuffle,
GC residualization, layer-0 floor), never from the bare peak.

---

## 7. Compute plan (Cayuga)

IMPLEMENTATION STATUS (2026-06-26): the BLOCKING port is DONE. The nested-CV machinery now lives in a
single source, `eval/probe_common.py` (`nested_layer_auroc` returns the OOF vector + per-fold layer +
fold AUROCs, fixing the discarded-OOF bug; plus `layer_curve`, `selectivity_at`, `cluster_boot`,
`ece`/`aurc`/`sel_acc`, `dump_layerloc`), and is wired into all five arms (`activation_arm.py`,
`activation_arm_dna.py`, `activation_arm_sc.py`, `activation_arm_msa.py`, `sfm_embedding_activation.py`);
each now emits a task-tagged `results/layer_loc_<task>_<model>.json` with the per-layer curve, the
naive-minus-nested optimism, the nested headline + OOF, selectivity, and the verbalize vector for H2/H3.
The four launchers now export `PYTHONNOUSERSITE=1`. An end-to-end smoke (DNA arm, Qwen2.5-0.5B, n=48,
local MPS) ran clean: nested-CV picked mid-layers [5,18,5,17,5], naive 0.924 vs unbiased 0.882 (the
+0.042 optimism the design targets), expression gap 0.455, full JSON schema written. STILL PENDING (not
blocking the first run, tracked in Known limitations): the DNA sequence-cluster GroupKFold + GC
residualization (control 2), mean-pool readout, and the tuned-lens break-layer arm. Staging must copy
`eval/probe_common.py` FLAT to `~/bge/` alongside the arms (the launchers run flat). The SFM launcher is
`cayuga_sfm_activation.sbatch` (not `run_activation_sfm_cayuga.sh`).

Infra: open-weight on Cayuga `scu-gpu` via the per-arm `run_activation_*_cayuga.sh` launchers.
GPU type: request `gpu:a40:1` for the DNA/single-cell/MSA/SFM runs (an 8B in fp16 is ~16 GB weights,
fits the a40's 48 GB; the A100 partition has historically sat PENDING for days, so a40 is the
correct ask). The hERG arm's existing launcher requests a100; either accept that or switch it to a40
for parity -- pick one and make it match. Do NOT route the DNA/sc/msa runs through the a100
`run_activation_cayuga.sh`.

Pre-submit ENGINEERING pass (BLOCKING -- the launch path in the prior draft did not match the
scripts; do all of this, then a `--n 8` smoke submit, BEFORE any locked outer-fold scoring):

1. Env truth. The launchers do `source venv/bin/activate`, NOT the bioguard conda env, and none
   export `PYTHONNOUSERSITE`, so the `~/.local` numpy2-vs-sklearn break is not actually guarded.
   `eval/activation_arm.py` imports `from rdkit import Chem` at the TOP (it is NOT lazy, despite the
   old comment), so the hERG and SFM arms crash on import if `~/bge/venv` lacks rdkit. Choose ONE
   path: (i) add `export PYTHONNOUSERSITE=1` to each launcher and `pip install rdkit-pypi` into
   `~/bge/venv` (DNA/sc/msa do not import rdkit; hERG does), OR (ii) rewrite the launchers to call
   `PYTHONNOUSERSITE=1 ~/.conda/envs/bioguard/bin/python` and confirm rdkit is in bioguard (it is
   NOT, per our ops notes, so (i) is the real path). Verify with a 1-item smoke submit of
   `activation_arm.py` that `import rdkit` succeeds in the venv.
2. Port the unbiased machinery. Factor the nested-CV + selectivity + per-layer curve + mean-pool +
   task-tagged JSON-dump block out of `activation_arm.py` into `eval/probe_common.py` and call it
   from `_dna`, `_sc`, `_msa` (the SFM arm has its own copy; align it). Without this the PRIMARY
   single-cell contrast and the MSA sanity gate cannot be scored and the biased `max` is the only
   number sc/msa emit. The shared function returns the OOF vector and per-fold ℓ* (Section 4.3) and
   uses the balanced `clf_factory` (Section 4.2).
3. Paths. Pick ONE root. The launchers `cd "$HOME/bge"` and set `HF_HOME=$HOME/bge/...`, but the
   prior draft staged to `~/pge/bge`; files rsynced to `~/pge/bge` are not found by scripts that
   `cd ~/bge`. Stage to `~/bge` (matching the launchers). For MSA, DROP the launcher's
   `ACT_CSV=$HOME/bge/msa_conservation.csv` override and let the arm use its `signal/msa/...`
   default (which the rsync populates), or stage the CSV to the override path -- not both.
4. Verbalize number. Use each arm's own OUTPUT line and import the harness metric functions
   directly (Section 4.5); do NOT depend on `run_grounding_eval.py --tasks dna/promoter` (no such
   registry key) or stand up a vLLM server.
5. hERG n. `data/herg.csv` has 625 positives / 3338 negatives (3963 rows), and `load()` caps at
   `min(n//2, len(pos), len(neg))`, so `ACT_N=1250` realizes <=625/class. Pre-register hERG n as
   "<=1250, whatever balances" and read decisions off the realized-n CI. Print `len(pos)/len(neg)`
   before locking.

Env line (CRITICAL, to be ADDED to each launcher per step 1):

```bash
export PYTHONNOUSERSITE=1     # the ~/.local numpy2-vs-sklearn break; without it sklearn import fails
export HF_HOME=$HOME/bge/hf_cache
# venv path: source ~/bge/venv/bin/activate; pip install rdkit-pypi into it (activation_arm.py
# imports rdkit at top-level, NOT lazily) so the hERG/SFM arms do not crash on import.
```

Submissions (each ~1 a40, 1 h walltime is ample; n=2000 Qwen3-8B forward + CPU probing ~20-40 min).
Use the per-arm launchers so the GPU type and `cd`/env are correct; smoke-test each with `ACT_N=8`
first:

```bash
# 1 ANCHOR hERG (canonical) + re-notation + chemistry-destroying token-shuffle controls
ACT_MODEL=Qwen/Qwen3-8B ACT_N=1250 ACT_CSV=herg.csv sbatch run_activation_cayuga.sh
ACT_MODEL=Qwen/Qwen3-8B ACT_N=1250 ACT_CSV=herg.csv ACT_RANDOMIZE=1 sbatch run_activation_cayuga.sh
# 1 HEADLINE DNA promoter (after data-identity + GC-floor preconditions)
ACT_MODEL=Qwen/Qwen3-8B ACT_N=1500 sbatch run_activation_dna_cayuga.sh
# CONTROLLED PAIR single-cell (runs name AND anon in one script; DESCRIPTIVE)
ACT_MODEL=Qwen/Qwen3-8B ACT_N=384 sbatch run_activation_sc_cayuga.sh
# POSITIVE CONTROL + H2 FLOOR  MSA conservation
ACT_MODEL=Qwen/Qwen3-8B sbatch run_activation_msa_cayuga.sh
# METHOD REPLICATION  SFM ESM-2 embedding (naive-vs-nested at 8B)
ACT_MODEL=Qwen/Qwen3-8B sbatch run_activation_sfm_cayuga.sh
# OPTIONAL cross-arch H1 generalization (DNA + hERG only)
ACT_MODEL=meta-llama/Llama-3.1-8B-Instruct ... ; ACT_MODEL=allenai/OLMo-2-1124-7B-Instruct ...
# OPTIONAL secondary VLM histopath (only if Qwen2.5-VL-7B hosts)
ACT_MODEL=Qwen/Qwen2.5-VL-7B-Instruct ACT_N=400 ACT_SCRIPT=activation_arm_histo.py sbatch run_activation_cayuga.sh
```

Data staging: repo is NOT checked out on Cayuga; rsync a structure-preserving subset to `~/bge`
(`rsync -aR eval/ data/herg.csv signal/{dna_promoter.csv,single_cell,msa,sfm_embedding}/ cayuga-login1:~/bge/`),
matching the launchers' `cd "$HOME/bge"`. Run with `python -u` (the sbatch already does) and watch
via Monitor / an until-loop on the `activation_%j.log` tail (foreground `sleep` is blocked). Total
compute: ~6-9 GPU-hours including the optional cross-arch and VLM runs; the headline (hERG + DNA +
single-cell + MSA + SFM) is ~5 a40-hours. The tuned-lens corroboration (express half, NEW code) is
one extra ~1 h GPU step per model on a disjoint text split; gate it behind a confirmed H1/H2.

Outputs land in `results/` as TASK-TAGGED JSON per arm (e.g. `layer_loc_herg_Qwen3-8B.json`),
including the per-layer curves, the per-layer selectivity, the nested-CV headline, the OOF vector
at ℓ*, and the per-fold ℓ* list; the per-layer curves + selectivity + nested-CV headline are pulled
back and analyzed per Section 5. `eval/layer_profiles.py` is a LOG SCRAPER (it reads
`/tmp/layer_profiles.txt`, hand-assembled from cluster logs) and is NOT an arm output; the analysis
reads the task-tagged JSON, not a scraped file. Do not overwrite the prior `results/*.json`.

---

## 8. What it feeds (the calibrated LLM-SFM bridge)

The located layer + its calibration plug directly into the next experiment:

1. ATTACH POINT. The nested-CV peak-selectivity layer band ℓ* is where the read-out head for the
   LLM-SFM bridge attaches (read hidden state at ℓ*, train the head there). Anchor 2601.18468
   (Cox HR 2.6, biomedical task) predicts the high-probe layer is where a trained read-out acquires
   FASTEST, so ℓ* is both the most-decodable and the cheapest-to-train attach point; the per-layer
   probe-strength ranking becomes the prior over attach depth. Reported as a band, consistent with
   the H1 localization.
2. CALIBRATION. The probe-readout ECE/AURC at ℓ* (H3), measured against a temperature-scaled output
   baseline, is the calibration the router uses to decide when to trust the bridge vs defer (feeds
   the calibration-routing line: continuous CONF threshold, web-exposure law as the routing prior).
   A well-calibrated, low-AURC read-out at ℓ* that adds INCREMENTAL routing value over the model's
   own output means the bridge can be routed like an oracle on its confident items.
3. WHERE-TO-FIX. The TRIANGULATED verbalize-break layer (probe-curve dissociation + causal-patch
   flip, lens corroborating) localizes the express failure: if the answer token is promoted
   mid-network then SUPPRESSED before output, the fix is a read-out at the promotion layer (bypass
   the suppression), not more scale -- a concrete, testable claim for the bridge design and
   consistent with our "route, do not train" verdict (retrieve/orchestrate cover the space; the
   read-out is the orchestrate lever attached at ℓ*).
4. The web-rich vs web-zero single-cell pair informs, as an EXPLORATORY signal (power-limited,
   Section 2.1), whether the attach depth should DIFFER by representation web-exposure (web-zero
   shallower/surface, web-rich deeper/semantic). It is carried as a hypothesis for a properly-powered
   follow-up, not as a locked conclusion of this experiment.

---

## Known limitations (deliberately deferred)

These are scoped OUT of this pass by choice, with the reasoning recorded so a reader does not mistake
them for oversights:

1. Single-cell layer-localization is under-powered. At n<=470 with 5 folds the minimum detectable
   gap is ~0.12-0.15 and a two-argmax peak-shift between name and anon cannot pin a layer to +/-9
   (Section 2.1). Single-cell is therefore DESCRIPTIVE-only: it supplies the OUTPUT web-rich-vs-web-
   zero verbalization contrast and a depth-band description, and the web-exposure peak-SHIFT
   (Section 8.4) is explicitly exploratory. Resolving it needs a larger single-cell sample, deferred.

2. The committed `cd8t_nk_obscure.csv` uses a DIFFERENT global anon map than `cd8t_nk.csv` (e.g.
   SF3B5 -> feat_513 vs feat_496), so it is stale relative to the current global-consistent build
   (`build_cd8t_nk.py`, which now yields 0 within-file map collisions). Before the single-cell run,
   pin ONE file built with the global map and verify the relabel is clean: position-shuffled
   bag-of-tokens AUROC for name and anon must match within CI, AND a probe trained on name tokens
   mapped through the global permutation must transfer to anon at ~equal AUROC. The order within
   each sentence is expression-rank-sorted in BOTH forms (symmetric), so rank is carried by position
   in both, not smuggled into the anon token identity; that symmetry is what makes the OUTPUT
   contrast admissible even while the peak-shift is deferred.

3. The DNA data-identity precondition is DISCHARGED (audit 2026-06-26): `signal/dna_promoter.csv` is
   3000 rows of pure-ACGT promoter sequence, balanced 1500/1500, under a cosmetically mislabeled
   `smiles` header (read as the sequence string, harmless). What REMAINS is the surface confound, now
   empirically confirmed rather than hypothetical: class-0 GC 0.482 vs class-1 GC 0.625 (delta ~0.14),
   so the task is strongly GC-separable and a surface/layer-0 probe scores high BY CONSTRUCTION. DNA
   stays a possibly-surface-confounded upper bound until the Section 4.4 control-2 GC-residualized
   margin clears +0.10; that residualization (not data identity) is the live safeguard, and hERG
   remains the clean content-vs-surface anchor. Note also the rows are class-block ordered, so the
   splitter must shuffle.

4. hERG is a single-shot estimate. The balanced sample consumes the entire 625-positive minority
   class, so there is no held-out positive block for an internal replication split. We accept hERG
   as single-shot and read decisions off the realized-n cluster-bootstrap CI rather than a held-out
   replication; a fresh hERG positive set would be needed for replication and is deferred.

5. The nested-CV deflation magnitude (predicted +0.05..+0.12) is the only quantitative prediction
   anchored to our own prior (+0.11), not to external literature. It is reported with a paired
   bootstrap CI and treated as "no measurable optimism" if that CI includes 0, but it remains the
   least-externally-anchored number in the design.

6. The express-break (tuned-lens) arm is NEW code and is the most exploratory deliverable. It is
   gated behind a confirmed H1/H2 and its conclusion requires probe-curve + causal-patch agreement
   (lens corroborating only), precisely because a tuned lens can manufacture a rise-then-fall by its
   per-layer fit quality. If the triangulation does not converge, the break layer is reported as
   indeterminate rather than read off the lens alone.

7. The C-regularization sweep is deliberately not run. Fixed L2 C = 1.0 is locked to keep the inner
   candidate count at 37 and avoid reinflating the SELECTED-layer variance; a pre-registered C sweep
   with band-reported ℓ* and a multiplicity-corrected inner SE is left to a separate study.
