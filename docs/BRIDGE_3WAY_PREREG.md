# Pre-registered design: the 3-way calibrated LLM<->SFM bridge experiment

Status: PRE-REGISTERED (hypotheses + decision rules fixed BEFORE results). Date 2026-06-27.
Substrate, transfer split, n, thresholds, parity plumbing, and falsification rules below are committed.
No em dashes (repo style).

This design folds three adversarial reviews (3-arm fairness/parity, transfer-eval validity,
executability) into a single committable plan. Where the reviews converged on a smaller first cut
being wiser, the staged plan (v1 minimal -> v2 full 3-way) is stated explicitly in Section 4 and is
the order of execution.

Companion docs: `docs/LAYER_LOCALIZATION_PREREG.md` (the warm-up that fixes the per-model attach
layer + the H3 routing edge), `decision_map/DECISION_MAP.md` (the "route, don't train" prior this
experiment must fairly test), `internal/PRELIMINARY_DATA.md` (the killed lit over-claims).

---

## 1. Question and why now

### The deep-research white space
Many papers build ONE LLM<->SFM bridge (ProteinGPT 2408.11363, Cell2Text 2509.24840, BioVERSE
2510.01428, STELLA 2506.03800, MutaPLM 2410.22949). Nobody has put, on ONE task with ONE shared
frozen SFM embedding and ONE scoring head, all three placements side by side:

1. a learned LLM<->SFM **BRIDGE** (projection that lets the LLM verbalize the embedding),
2. **EXTERNAL ORCHESTRATION** of the frozen SFM (a trained read-out head, LLM untouched),
3. **IN-WEIGHT LoRA** of the LLM (no SFM),

and measured them under (a) a held-out-PROPERTY transfer eval (does the read generalize the SKILL or
memorize one property) and (b) calibrated permissioning (when to trust the read, vs a FAIR output
baseline). That triple-with-transfer-and-calibration is the white space. This is the experiment.

### Why now (what makes it answerable today)
- The layer-localization warm-up fixed the one novel design choice for hERG: the read-out depth is
  PER-MODEL (Qwen3-8B band [2,24], Llama-3.1-8B band [1,15]). The bridge attaches at the model's
  PEAK-SELECTIVITY layer (real minus shuffled-control per layer), not an assumed mid-band and not the
  raw-AUROC argmax (`results/benchmark/layerloc/SUMMARY.md`, `results/layer_profiles.json`). The
  attach layer is currently localized for hERG ONLY; extending it to the other endpoints is a
  pre-run gating step (Section 2, the attach-layer table; Section 4 v1 sidesteps this by running hERG
  as the headline test endpoint).
- The same warm-up fixed the calibration signal: the encoded read-out is a better router than the
  model's own output (probe AURC beats output AURC by dAURC -0.18 to -0.77 across both models/all
  tasks). H3 operationalizes this against a FAIR (temperature-scaled, per-arm) baseline.
- The transfer-eval data is buildable at ~0 cost on a substrate that gives a GENUINE held-out
  property (Section 5): molecular FM (ChemBERTa/MoLFormer) x 7 distinct ADMET endpoints, each
  ceiling-validated.

### The prior this must be able to refute
The decision-map verdict is "route, don't train": retrieve + orchestrate cover the space; the weights
lever is approximately empty for property-prediction; a thin bridge/head wins; in-weight FM
fine-tuning loses to external steering in low data (external steering of a frozen FM beats DPO,
2505.15093). The 3-way is the FAIR test of that verdict. To be refutable, all three arms run on
identical data/split/scoring/calibration (Section 5, the parity contract), and a clean
**bridge > orchestrate on held-out property** (skill transfer) is pre-committed as the result that
WOULD overturn the verdict (H1b / H2b). We design FOR that outcome, not against it.

### Killed lit over-claims to NOT re-import (honest stance)
- Bigger LLM is NOT reliably a better bridge substrate (we run BOTH Qwen3-8B and Llama-3.1-8B; H1 is
  over the ordering of ARMS, not models).
- Probes are NOT truer-than-output (Belinkov 2102.12452); the orchestrate number is an upper bound on
  a trained read, not "what the model really knows."
- The encode>>express gap is real (survives GC / single-cell-shift controls), but the bridge claim is
  about a TRAINED projection, not prompt-pasting (prompt-pasting an ESM-2 embedding is chance 0.47
  zero-shot / 0.56 ICL; that strawman is excluded by construction).
- Closed-weight asymmetry: SFMs are OPEN (post-trainable); frontier LLMs are not. The orchestrate arm
  is the closed-weight-friendly placement; the LoRA arm needs open weights; the bridge needs open
  weights AND an open SFM. This asymmetry is part of the contribution (Section 8).

---

## 2. The three arms, precisely defined

All three consume the **same frozen SFM embedding matrix** (ChemBERTa-77M-MTR, 384-dim, mean-pooled;
MoLFormer-XL 768-dim as a v2 robustness check) and are scored on the **same held-out item list** with
the **same scoring-and-calibration pipeline** (Section 5). The ONLY difference is the read-out locus.
Param counts are reported for all three (capacity-fairness).

### Arm A: LEARNED BRIDGE (new build, `eval/bridge_arm.py`)
SFM embedding -> trained projection -> soft-prompt token(s) that enter the LLM, which then ANSWERS
yes/no in language. The injection mechanism is specified as the actually-buildable pair below; the
review correctly flagged that "prepend k NEW positions at layer 21 via a forward hook" is incoherent
(new sequence positions must exist from layer 0 for attention/RoPE/the KV cache to be consistent; a
mid-stack hook can only add to or overwrite hidden states of positions that already flow through).

- **Projection**: `e in R^384 -> W2 . GELU(W1 . e) -> R^{k . d_model}`, reshaped to k soft-prompt
  token vectors (d_model = 3584 Qwen3-8B / 4096 Llama-3.1-8B). Default 2-layer FF, bottleneck 512.
  Trainable params approx 384*512 + 512*(k*d_model); at k=4, Qwen approx 7.5M (~ LoRA-r32 scale).
- **Injection (PRIMARY = layer-0 soft-prompt, standard prompt-tuning)**: the k projected vectors are
  prepended as the FIRST k positions of the sequence at the embedding layer, via `inputs_embeds`
  (or a `register_forward_pre_hook` on `model.model.embed_tokens`). These positions exist from layer
  0, so attention/RoPE/KV-cache are consistent. This is the buildable, conventional bridge and is the
  PRIMARY condition.
- **Injection (ABLATION = additive-at-peak hook)**: a `register_forward_hook` on
  `model.model.layers[L]` that ADDS the broadcast projection output to the hidden state of the last
  prompt token at the peak-selectivity layer L (it does NOT create new positions). This is the
  "read enters where the model encodes the property" variant. The PRIMARY-vs-additive-at-peak
  contrast is itself a reported result; the design's earlier "prepend at L21" headline is REPLACED by
  "add at L (peak-sel)" as the ablation. (v1 runs PRIMARY only; the additive-at-peak ablation is v2.)
- **Training**: LLM frozen, train ONLY the projection. Stage B = next-token CE on " yes"/" no",
  target construction IDENTICAL to `ws3_lora.py:build_example` (lines 89-96). Stage A
  (contrastive/MSE alignment) is an optional ablation, skipped for v1.
- **Bridge tuning budget (pre-committed, TRAIN endpoints only, never the held-out test)**: to
  separate placement from under-training, the bridge gets its own small pre-registered sweep on the
  train-fold CV: k in {4, 8, 16}, lr in {1e-4, 5e-4}, epochs in {5, 10}. Pick by train-fold CV AUROC,
  FREEZE, then run the held-out eval once. A learning curve (train loss / train-fold val AUROC vs
  epoch) is reported so "converged" is auditable, not assumed. The LoRA arm gets a symmetric
  pre-registered sweep (Arm C). Without this, H1 cannot separate placement from optimization.
- **LLM-bypass control (pre-committed, the load-bearing parity check)**: train the IDENTICAL FF
  projection (same param budget, same k, same recipe) to predict yes/no directly from the embedding
  through a fixed linear read-out, with NO transformer in the loop (transformer attention/MLP zeroed
  or the projection mapped straight to the two-token logit gap). If the full bridge does NOT beat this
  bypass control by the H1 threshold, the bridge is NOT reading THROUGH the LLM and any H1b ("bridge
  beats orchestrate") is discounted as a capacity artifact, not an in-language read. This makes
  "bridge vs orchestrate" a real distinction rather than head-size.
- **Reuses**: ChemBERTa loader pattern (`sfm_embedding_activation.py` L45-50), the READ-path hook
  pattern (`activation_arm.py` L135-188, adapted from read to write/add), the shared scorer
  (Section 5, NOT `ws3_lora.eval_output_auroc` directly; see parity contract).
- **Builds**: the projection module + layer-0 prepend + additive-at-peak hook + Stage-B loop +
  bypass control. Approx 250 lines.

### Arm B: EXTERNAL ORCHESTRATION (mostly have, `eval/probe_common.py`)
The frozen SFM embedding -> `balanced_lr` trained head (StandardScaler + L2 C=1.0 + class_weight
balanced) -> probability surfaced/routed. LLM weights untouched. The 2505.15093 "external steering
beats DPO" placement; the closed-weight-friendly arm.

- **Reuses**: `probe_common.balanced_lr`, `control_curve` (shuffled-label selectivity, Hewitt-Liang
  1909.03368), `ece`, `sel_acc`, `cluster_boot`. The `aurc` and `nested_layer_auroc` are used but
  with the two parity fixes below.
- **Builds**: (i) the shared held-out item list + scorer/calibration wrapper (Section 5, used by ALL
  arms); (ii) a held-out-PROPERTY wrapper (fit the head on the pooled train-endpoint embeddings, score
  on the shared held-out item list).
- **Selectivity guard**: selectivity (real minus shuffled-label) must stay >= 0.10 or the cell is
  dropped.

### Arm C: IN-WEIGHT LoRA (have + a ~5-line patch, `eval/ws3_lora.py`)
LoRA-finetune the LLM on the property; measure VERBALIZED output AUROC before vs after, on the SHARED
held-out item list. NO SFM. The "train the LLM's own weights" placement.

- **Config (locked, from the PoC)**: r=16 (primary) and r=32 (the PoC's best), lora_alpha=2r,
  dropout=0.05, target q/k/v/o_proj, manual AdamW loop lr=1e-4, grad-accum every 8 steps, 3-5 epochs.
  Symmetric pre-registered sweep allowed (epochs in {3, 5}, r in {16, 32}), picked by train-fold CV,
  frozen before the held-out eval.
- **Two required patches (the review showed "reuse as-is / builds nothing" is FALSE for the transfer
  arm)**:
  1. `{property}` slot. `PROMPT` currently has only `{rep}` (line 35), so pooling 5 endpoints under
     one fixed prompt makes every row ask the SAME question regardless of which endpoint the label
     came from (the LoRA then learns label noise). Add a `{property}` field and have
     `load()`/`build_example()` read `r["property"]` per row (every ADMET record already carries
     `"property": "<endpoint>"`, confirmed). Template e.g.
     `"SMILES: {rep}\nDoes this molecule show {property}? Answer yes or no.\nAnswer:"`. The same
     templated prompt is used by Arm A and at held-out eval time (ask the held-out endpoint's
     question). This is a ~5-line change.
  2. External `test_ids`. `load()` calls its OWN `GroupShuffleSplit(test_size=0.3, random_state=42)`
     (line ~66), so as-is Arm C would score on a LoRA-PRIVATE 30% fold, not the shared held-out list.
     Refactor `load()` to accept an external `test_ids` list (the shared fold file) and NOT call
     `GroupShuffleSplit` for this experiment. See Section 5 parity contract.
- **Negative control already wired**: LORA_SHUFFLE=1 (train on shuffled labels, must stay at chance;
  PoC 0.484).

### The attach layer = per-model PEAK-SELECTIVITY layer (corrected, and currently hERG-only)
The review caught a real contradiction: the prior draft cited "Qwen approx L34-35 (depth 0.97)" but
told the code to read `argmax` from `results/layer_profiles.json`, whose `stats.herg.
peak_layer_canonical = 2`. The 0.97 is a selectivity-DEPTH scalar, the picked BAND is [2,24], and a
raw-AUROC argmax returns the SHALLOW layer 2 (a near-surface layer the warm-up itself flagged can be
a GC/surface artifact). Resolution:

- `BRIDGE_PEAK_LAYER` is defined as the **peak-SELECTIVITY** layer = argmax over layers of
  (canonical_AUROC[L] - randomized_AUROC[L]) from `layer_profiles.json`, NOT the raw-AUROC argmax.
  For hERG this is computed and FROZEN as a literal integer in the table below (it is NOT "default =
  argmax" and the launcher does NOT silently resolve to layer 2).
- `layer_profiles.json` currently contains profiles for `herg` and `withdrawn` ONLY. There is NO
  profile for clearance/cyp3a4/cyp2d6/solubility/permeability/ames, so "the per-model argmax" is
  undefined for 6 of 7 endpoints AND for any pooled-train union. Therefore:
  - **v1** uses hERG as the headline TEST endpoint, where the peak-selectivity layer IS defined and
    frozen. The bridge's distinguishing move is fully pinned for v1.
  - **v2** (any non-hERG test endpoint or the additive-at-peak ablation on a pooled union) is GATED
    on first running `activation_arm.py` per-layer on the needed endpoints to populate
    `layer_profiles.json`, then freezing the resulting `peak_sel_layer` integers in this table as a
    literal per-endpoint x per-model grid BEFORE any bridge run. For a pooled-train union the
    pre-committed rule is: re-localize the peak-selectivity layer on the POOLED train embedding (one
    profiling pass on the union), and use that single integer; do NOT post-hoc pick per endpoint.

Frozen attach-layer table (the only values committed now):

| model | endpoint | peak-selectivity layer (canonical - randomized argmax) | source |
|---|---|---|---|
| Qwen3-8B | herg | recompute from `herg_canonical - herg_randomized`, freeze the integer; raw-AUROC argmax L2 is REJECTED as the attach layer | `results/layer_profiles.json` |
| Llama-3.1-8B | herg | band [1,15], freeze the peak-selectivity argmax integer | `results/benchmark/layerloc/SUMMARY.md` |
| both | all other 6 endpoints | UNDEFINED until `activation_arm.py` is run on them; freeze here before any v2 bridge run | (to populate) |

The additive-at-peak ABLATION uses this layer. The layer-0 soft-prompt PRIMARY does not depend on it,
so v1's primary bridge result is robust to the layer question.

---

## 3. Pre-registered hypotheses (each falsifiable, threshold stated)

All AUROC differences are read against **PAIRED cluster-bootstrap 95% CIs on the SAME held-out item
list** (Section 5; this requires ADDING a paired-difference bootstrap to `probe_common`, which
currently has only the unpaired `cluster_boot`). "Wins" = point estimate higher AND the
paired-bootstrap CI of the difference excludes 0, computed on identical molecules. Threshold delta =
0.03 AUROC (below this, within the n=hundreds-per-endpoint noise band documented in the AMES
n=206->n=2000 lesson) is a TIE.

### H1 - ORDERING (within-property, the placement question)
Prediction from the prior (route, don't train): **orchestrate >= bridge >> in-weight LoRA**, and ALL
THREE are dominated by (or tie) a cheap non-LLM specialist (Morgan-fingerprint probe / k-NN) on
fingerprint-local endpoints.

- **H1a (the prior holds)**: on the headline endpoint, orchestrate AUROC >= LoRA-output AUROC by
  >= 0.03 (paired CI of diff excludes 0). The trained-head read of the frozen SFM beats in-weight LoRA.
- **H1b (REFUTATION of the prior)**: bridge AUROC > orchestrate AUROC by >= 0.03 (paired CI excludes
  0) on the within-property eval, AND the bridge ALSO beats the LLM-bypass control by >= 0.03 (else
  H1b is a capacity artifact, not an in-language read; Section 2 Arm A). Both conditions are required
  for H1b to fire.
- **Specialist-dominance control (the firmest decision-map result)**: every LLM arm (A/B/C) is
  reported against the SAME-SPLIT Morgan-probe and k-NN ceiling (`signal/admet/verifiability_report.
  json` cold_auroc: herg 0.895, permeability 0.878, ames 0.847, cyp3a4 0.83, cyp2d6 0.828, solubility
  0.791, clearance 0.746, MEASURED AT FULL n; the at-capped-n clearance ceiling is re-measured per
  P5 below before it is used as a gate). PRE-COMMITTED reading: on fingerprint-local endpoints
  (herg/permeability) the specialist is expected to dominate all three LLM placements; the place to
  look for an LLM win is the LOWEST-ceiling, lowest-selectivity endpoint (clearance: cold 0.746 /
  selectivity 0.242) where the cheap specialist is weakest. An LLM arm "wins" only if it strictly
  exceeds cheap AND retrieve AND specialist on the held-out split (the ws3 train-placement rule).

FALSIFIED IF: orchestrate does NOT beat LoRA on the headline endpoint AND bridge does not beat
orchestrate anywhere (a flat 3-way tie on every endpoint refutes the ordering and would say
"placement does not matter here").

### H2 - TRANSFER (the white-space point: skill vs memorization)
Train the read on the train endpoints, TEST on held-out endpoint(s) (the read answers a property it
never trained on). The metric is FLOOR-ANCHORED, not a bare ratio (the review showed a bare ratio R =
held-out/in-property is uninterpretable because the cross-property floor is approximately chance:
a head trained on 5 endpoints applied raw to a held-out endpoint scores approximately 0.52, so an arm
that learns NOTHING transferable but has a weak in-property denominator can land R approximately 0.95
in the "skill" band).

- **Pre-registered transfer FLOOR**: BEFORE any arm runs, compute and FREEZE the cross-property
  raw-transfer floor = the P2 head-transfer matrix value into the held-out endpoint (head trained on
  the train union, applied raw to the held-out endpoint embedding). The empirical stand-in measured
  on Morgan fingerprints is approximately 0.52 (pooled-5 -> herg approximately 0.525, -> clearance
  approximately 0.515; best single-endpoint raw transfer maxes at approximately 0.62). This number is
  recomputed on the actual frozen SFM embedding and committed as `AUROC_floor` per held-out endpoint
  BEFORE the arms run, NOT discounted post hoc.
- **Normalized transfer** NT_arm = (AUROC_holdout - AUROC_floor) / (AUROC_inprop - AUROC_floor), where
  AUROC_inprop is computed on the SAME held-out items / SAME scaffold groups / SAME bootstrap as the
  numerator, the ONLY difference being train = union-of-train-endpoints vs train = held-out endpoint
  (no estimator mismatch; Section 5). Bands on NT: NT >= 0.90 = SKILL transfers; 0.60 <= NT < 0.90 =
  PARTIAL; NT < 0.60 = MEMORIZES one property.
- **Primary number** = the absolute Delta-over-floor (AUROC_holdout - AUROC_floor) with a paired
  cluster-bootstrap CI. A transfer claim REQUIRES the held-out-AUROC CI to exclude the floor, not just
  NT >= 0.90. NT is the descriptive normalization; the threshold rides on the paired difference.
- **H2a (prediction)**: orchestrate transfers best (NT_orch highest), because reading a fixed strong
  encoder's task-relevant axis is the most generic skill; LoRA transfers WORST (NT_lora lowest),
  echoing the cross-task LoRA result (variant_seq 0.652 vs AlphaMissense 0.96 = distant second).
- **H2b (REFUTATION)**: (NT_bridge - NT_orch) paired CI excludes 0 with magnitude >= 0.05 on the SAME
  held-out items, AND the bridge beats the LLM-bypass control on transfer too. If the in-language
  bridge generalizes the read-skill to a held-out chemical property BETTER than the vector-space head
  (and it is not just a bigger head), the "train a thin head is the safe win" verdict is overturned.

FALSIFIED IF: no arm's held-out-AUROC CI excludes the floor on any held-out endpoint (then the
transfer premise fails: the read is property-specific for all placements, and "generalize the skill
of reading the SFM" is not a thing any arm does). This is a real possible negative and is reported as
such.

### H3 - CALIBRATION / PERMISSIONING (when to trust the read)
The read-out's per-item confidence is calibrated and routes better than a FAIR output baseline, with
PER-ARM temperature scaling (the review showed the prior draft temperature-scaled only the output
baseline, leaving the bridge/LoRA two-token softmax over-confident vs the head's `predict_proba`,
which rigs ECE toward the head before any modeling).

- **Per-arm temperature scaling (do NOT skip for any arm)**: fit ONE temperature scalar T per arm
  (bridge, LoRA, orchestrate, raw output) on a held-out CALIBRATION split (minimize NLL) BEFORE
  computing ECE / sel_acc / reliability. ECE and reliability are reported on the temperature-scaled
  probabilities of EACH arm. AURC is rank-based and INVARIANT to a single monotone T, so AURC is the
  calibration-free ranking metric and is reported on the raw scores; ECE/reliability get the per-arm
  T. This is stated explicitly so the AURC comparison is not confounded by probability shape.
- **Per-arm operating threshold (NOT hard 0.5)**: `aurc`/`sel_acc` as written use confidence =
  |p - 0.5|, which presumes a 0.5 decision threshold that is WRONG under `class_weight="balanced"` and
  under skewed positive fractions (permeability 0.915, cyp2d6 0.111). Pre-commit a per-arm operating
  threshold tau via Youden-J on the calibration split; define confidence = |p - tau| (or the
  calibrated-probability margin) for the risk-coverage and selective-accuracy computations. `aurc`
  and `sel_acc` are extended to take tau (a small patch).
- **H3a (routing edge)**: the orchestrate head AURC (area under risk-coverage, lower = better) beats
  temperature-scaled LLM-output AURC by >= 0.05 absolute. This is the layer-loc dAURC -0.18..-0.77
  result, re-tested against a FAIR baseline.
- **H3b (calibration)**: orchestrate ECE (10-bin AND 5-bin) <= temperature-scaled-output ECE, both
  computed on per-arm temperature-scaled probabilities. The read-out is at least as calibrated as the
  rescaled model.
- **Permissioning rule (reported, not hypothesis-gated)**: route to the arm with the highest per-item
  calibrated confidence at operating coverage 0.5, else abstain; report routed-AUROC vs the per-item
  ORACLE (routed approximately oracle for a well-calibrated router).

FALSIFIED IF: orchestrate AURC does NOT beat temperature-scaled output by >= 0.05 on the headline
endpoint (then the "probe is a better router than output" edge is a raw-output artifact that a single
temperature scalar erases, and the calibrated-permissioning layer loses its motivation).

### Pre-committed outcome table (so no post-hoc story)
| outcome | H1 | H2 | H3 | verdict |
|---|---|---|---|---|
| prior holds | orch >= LoRA, all <= specialist | NT_orch highest, NT_lora lowest | orch AURC beats temp-scaled | "route, don't train" CONFIRMED, fairly |
| bridge surprise | bridge > orch AND > bypass (H1b) | (NT_bridge - NT_orch) CI excludes 0 AND > bypass (H2b) | (either) | verdict OVERTURNED: in-language read transfers |
| flat | 3-way tie everywhere | no arm's held-out CI excludes floor | no AURC edge vs temp-scaled | placement-agnostic NEGATIVE: report honestly |

---

## 4. Staged plan (v1 minimal -> v2 full), and the data substrate

The three reviews converged: each independently recommended a smaller first cut, and each ended at the
same v1 (Qwen3-8B / one fold / ChemBERTa / primary-only). The staged plan is therefore the order of
execution, not a fallback.

### v1 (minimal, runs first, answers the headline)
- **One model**: Qwen3-8B (the model whose hERG peak-selectivity layer is localized and frozen).
- **One substrate**: ChemBERTa-77M-MTR (384-dim) embedding, built locally.
- **One fold**: the headline LOPO fold below, with the CYP-cluster fix applied.
- **One test endpoint for the bridge's pinned-layer claim**: hERG (peak-selectivity layer defined).
  The transfer eval still uses both held-out endpoints {hERG, clearance} for orchestrate and LoRA
  (which do not need a per-endpoint attach layer); the bridge's additive-at-peak variant is restricted
  to hERG in v1.
- **Bridge**: layer-0 soft-prompt PRIMARY only (no additive-at-peak, no Stage-A, no projection+LoRA),
  PLUS the LLM-bypass control (cheap, same recipe) so H1b/H2b are interpretable even in v1.
- **All three arms** scored through the ONE shared item list + scorer + per-arm temperature scaling
  (Section 5). This is the load-bearing parity plumbing and is built once, in v1.
- **Compute**: Qwen / 1 fold / ChemBERTa / primary-bridge + bypass + LoRA(r16,r32) + orchestrate.
  Bridge primary approximately 2 runs (bridge + bypass) at approximately 25 min; LoRA approximately
  30-45 min; orchestrate CPU-minutes. Fits in < 3 a40 GPU-hours. v1 answers H1/H2/H3 for the headline.

### v2 (full matrix, GATED on v1 showing arm separation)
Promote to a second sbatch ONLY if v1 shows separation worth the spend:
- add Llama-3.1-8B (second model);
- add the additive-at-peak injection ablation (requires the frozen non-hERG peak-selectivity layers,
  Section 2; run `activation_arm.py` to populate `layer_profiles.json` FIRST, freeze the integers);
- add MoLFormer-XL (768-dim) robustness pass;
- add folds 2-3 (with the CYP-cluster constraint, below);
- add the pretraining-naive SFM control (the dissociation control, below).
- **Compute**: the full pre-registered matrix is approximately 10-20 a40 GPU-hours (the bridge alone,
  across models x ablations x folds x in-property baselines, is the bulk; this is the number for v2,
  not v1).

### Substrate decision: molecular FM x ADMET (committed)
**Use ChemBERTa-77M-MTR (384-dim; MoLFormer-XL 768-dim v2 robustness) x 7 ADMET endpoints.**
Pre-committed justification:
- It is the ONLY substrate that gives a GENUINE held-out-PROPERTY split: 7 physically different
  properties (mutagenicity / clearance / 2 CYP isoforms / hERG block / permeability / solubility) on a
  SHARED diverse molecular space. "Train on train endpoints, test on held-out endpoints" is real
  property transfer.
- FLIP is REJECTED for the transfer eval: each landscape is a single-protein deep-mutational scan of
  ONE fitness, so a held-out-property split is perfectly confounded with held-out-protein (domain
  shift, not skill transfer).
- Meltome (Tm-only, n=320, ceiling 0.633) CANNOT support a property-transfer claim (one property) and
  its ceiling is too weak (P1). Kept ONLY as a held-out-ENTITY control and a protein-side robustness
  check.
- Build cost approximately 0: both molecular FMs are cached locally; embedding all approximately 22k
  SMILES is approximately 1.5 min on this Mac's MPS (no Cayuga, no download). Embeddings approximately
  34 MB (ChemBERTa) / 68 MB (MoLFormer).

### Pretraining-naive SFM dissociation control (v2, pre-registered second substrate)
The review raised a real confound: ChemBERTa-77M-MTR is pretrained by MULTI-TASK REGRESSION over
approximately 200 computed molecular descriptors (the MoleculeNet/RDKit-style set), so its embedding
is already organized along physchem-property axes that correlate with solubility/permeability/
clearance. "The head reads a held-out property it never trained on" could then be "the SFM baked that
property axis in during pretraining," i.e. pretraining coverage, not read-out skill transfer. (And
MoLFormer-XL is ALSO property-adjacent, so it does not dissociate this.) Pre-registered control:
re-run the FULL transfer matrix on a pretraining-naive embedding = ChemBERTa-MLM (the masked-LM
variant, NOT MTR) and/or a Morgan-fingerprint baseline. If transfer SURVIVES on the MLM/fingerprint
embedding, it is structural; if it collapses to the floor, the MTR "transfer" is pretraining leakage
and is reported as such. This is a v2 gate on the transfer claim's interpretation.

### Endpoints and their validated ceilings (the precondition each property is genuinely encoded)
From `signal/admet/verifiability_report.json` (Morgan-probe, scaffold GroupKFold):

| endpoint | cold AUROC | selectivity | n (cap 4400) | pos frac | leakage_drop |
|---|---|---|---|---|---|
| herg | 0.895 | 0.365 | 3963 | 0.158 | 0.025 |
| permeability | 0.878 | 0.377 | 2057 | 0.915 | - |
| ames | 0.847 | 0.331 | 517 | 0.801 | 0.078 |
| cyp3a4 | 0.830 | 0.312 | 4000 | 0.147 | 0.044 |
| cyp2d6 | 0.828 | 0.338 | 4000 | 0.111 | 0.051 |
| solubility | 0.791 | 0.289 | 866 | 0.212 | 0.059 |
| clearance | 0.746 | 0.242 | 4000 (from 46958) | 0.186 | - |

All 7 pass the VIABLE gate (ceiling >= 0.65, selectivity >= 0.10) at full n. clearance's ceiling is
RE-MEASURED at the capped n before it is used as the P1 gate (P5 below). Skewed positive fractions
(permeability 0.915, ames 0.801, cyp2d6 0.111) are the reason H3 uses a per-arm operating threshold,
not hard 0.5.

### The exact transfer split (Leave-2-Properties-Out, CYP-cluster-safe)
Properties ranked by Morgan-ceiling so a fingerprint-local endpoint is always in the test fold. The
review found the one genuinely correlated pair (cyp3a4 <-> cyp2d6 share 1261 molecules, Jaccard 0.187,
label agreement 0.921 -- by far the strongest coupling) and showed it must be treated as ONE cluster,
never split across the train/test boundary, or fold-2 becomes textbook leaky transfer (train cyp3a4 ->
test cyp2d6 on approximately 1261 near-duplicate, 92%-agreeing molecules).

- **CYP cluster rule (pre-committed)**: {cyp3a4, cyp2d6} are a single property cluster. No fold may put
  one in train and the other in test. They are also exact-SMILES-deduplicated AGAINST EACH OTHER
  (P3), not only across the train/test seam.
- **Headline LOPO fold (committed)**: TRAIN on {ames, cyp3a4, cyp2d6, solubility, permeability}, TEST
  on {herg (fingerprint-local, high ceiling 0.895), clearance (lowest ceiling 0.746, the LLM-win
  candidate)}. Both CYP endpoints are on the SAME (train) side, so the cluster is intact. As a
  robustness read on the headline fold, ALSO report the pooled transfer with cyp3a4 AND cyp2d6 ablated
  from train, to show the held-out transfer does not depend on the one correlated block.
- **Rotation (v2, reported with CIs)**: pre-register 3 folds total; fold-1 above; **fold-2 holds out
  the WHOLE CYP cluster {cyp3a4, cyp2d6}** (replacing the prior leaky "{cyp2d6, solubility}" fold);
  fold-3 holds out {permeability, ames}. Report all 3, headline = fold-1.
- For each arm: train ONCE on the union of the train endpoints, test on EACH held-out endpoint on the
  shared item list. The in-property baseline (H2 denominator) trains/tests ON the held-out endpoint
  using the SAME folds/groups/bootstrap as the numerator (Section 5).

### Controls against property leakage (pre-registered)
- **P1 weak-ceiling guard**: every test endpoint must have Morgan-ceiling >= 0.65 AND selectivity
  >= 0.10. For the protein robustness check ONLY, re-extract ESM-2 650M (esm2_t33_650M) before use.
- **P2 transfer floor (now load-bearing, see H2)**: compute and FREEZE the cross-property head-transfer
  matrix (head trained on a train set, applied raw to a held-out endpoint) BEFORE the arms run; the
  held-out endpoint's `AUROC_floor` is its pooled-train raw transfer (empirically approximately 0.52
  for herg/clearance). A held-out endpoint whose best single-endpoint raw cross-transfer already
  >= 0.70 is flagged LEAKY and its transfer number is discounted (none of the headline-fold held-out
  endpoints hit this; cyp pair is the only >0.70-risk pair and is cluster-isolated).
- **P3 cross-property entity leakage**: GLOBAL Murcko-scaffold dedup across ALL 7 endpoints BEFORE any
  fold is cut. Each scaffold is assigned to exactly one endpoint's pool (or excluded). Additionally,
  exact-SMILES dedup across the CYP pair specifically. The headline number is reported on the
  scaffold-disjoint subset; the full-set number is secondary. (The raw seam overlap is approximately
  8.4% herg / 7.6% clearance test molecules also present in the train union; the global scaffold
  firewall removes this free-ride.)
- **P4 within-split entity leakage**: always use the existing Murcko-scaffold groups as the GroupKFold
  groups; never random-split (the ws3_retrieve_random_split FLIPPED result is the cautionary tale).
- **P5 capped-n ceiling for clearance (new)**: clearance's 0.746 ceiling was measured at n=46958/24219
  groups; the arms cap at 4000-4400. Re-run the verifiability ceiling for clearance at the CAPPED n and
  commit THAT number as the P1 gate. If the capped-n clearance ceiling drops below 0.65, clearance is
  demoted from the "LLM-win candidate" and the headline LLM-win search falls back to the next-weakest
  viable specialist endpoint.

### Held-out-ENTITY control axis (the weaker fallback, reported alongside)
Within ONE endpoint (and within meltome for the protein side), cluster/scaffold GroupKFold = can the
read predict held-out scaffolds/protein-families. This is entity not property generalization; the
LOWER bar, reported to show the within-property read is not memorizing specific scaffolds.

---

## 5. Method per arm and the parity contract (one shared fold, one scorer, one calibration)

The reviews showed the prior draft's parity was prose-only: "one held-out item list shared by all
three arms" was contradicted by `ws3_lora.py` re-splitting internally, the "same scorer" was false
(orchestrate `predict_proba` vs bridge/LoRA two-token softmax), and the CIs were computed over
non-identical supports so "X - Y CI excludes 0" was not a paired test. The parity plumbing below is
the structural change; it is built ONCE in v1 and used by all arms.

### Shared fold + item list (built first, persisted)
- One embedding matrix per endpoint per FM: `signal/admet/<endpoint>/<fm>.npz` with schema
  `{emb, y, groups, ids, model}` (built by `eval/sfm_embed_admet.py`).
- One held-out item list per fold, persisted as `signal/admet/folds/<fold>.json` =
  `{test_ids, train_ids, scaffold_groups}` (test = held-out endpoints' rows AFTER the global scaffold
  firewall P3; within-property folds = a frozen scaffold-GroupKFold index). EVERY arm reads this file:
  - Arm B fits on the train_ids embedding, scores `predict_proba` on the test_ids embedding.
  - Arm C's patched `load()` takes `test_ids` and does NOT call `GroupShuffleSplit`.
  - Arm A trains the projection on train_ids, scores on test_ids.
- The H2 in-property baseline reads the SAME `folds/<fold>.json` scaffold groups (same items, same
  bootstrap), differing only in train = union vs train = held-out endpoint. No native-splitter
  denominators.

### One scorer + one calibration pipeline (shared module, all arms)
- A single `score_arm(p_hat, y, ids, groups)` used by ALL arms produces: AUROC, AURC (rank-based, raw
  scores), and -- AFTER fitting a per-arm temperature T on the calibration split -- ECE(10-bin,
  5-bin), sel_acc@cov0.5, reliability. The per-arm operating threshold tau (Youden-J on the
  calibration split) is fitted here and passed to `aurc`/`sel_acc`.
- For bridge/LoRA, p_hat = P(" yes") via the two-token renormalization (the `ws3_lora.eval_output_
  auroc` formula), but the AUROC/AURC/ECE come from the SHARED `score_arm`, NOT from
  `eval_output_auroc` directly, so every arm is scored by the same function on the same rows.
- Paired comparisons use a NEW `paired_cluster_boot(y, pA, pB, groups)` added to `probe_common`
  (resample scaffold groups once, recompute (metric_A - metric_B) per resample, CI on the difference).
  This is the missing piece that makes "X - Y CI excludes 0" a real paired test.

### Arm A (bridge) - the build
1. Load `<endpoint>/chemberta.npz` (frozen 384-dim). Concatenate train endpoints (train_ids only).
2. Projection `e -> W2 GELU(W1 e) -> reshape (k, d_model)`; init small (std 0.02).
3. PRIMARY: prepend the k vectors as the first k positions via `inputs_embeds` at layer 0 with the
   `{property}`-templated prompt `"SMILES: <soft>\nDoes this molecule show <property>? Answer yes or
   no.\nAnswer:"`. ABLATION (v2, hERG-only in v1): additive forward hook at the frozen peak-selectivity
   layer adding the broadcast projection to the last prompt token.
4. Bridge-only sweep on train-fold CV (k, lr, epochs per Section 2), freeze, train projection ONLY
   (LLM frozen), Stage-B next-token CE on " yes"/" no" on train_ids. Report the learning curve.
5. Score test_ids through `score_arm` (shared). ALSO train+score the LLM-bypass control identically.
   Write `results/bridge_arm.json` (tag, base/ft AUROC, AURC, ECE, n_train/n_test, param_count,
   attach_layer, bypass_auroc, transfer Delta-over-floor, NT).

### Arm B (orchestrate) - mostly reuse
1. Fit `balanced_lr` on the train_ids pooled embedding; `predict_proba` on test_ids. `control_curve`
   for the shuffled-label selectivity guard (>= 0.10).
2. H2 in-property baseline = same `balanced_lr` trained on the held-out endpoint's train_ids, scored on
   the SAME test_ids/groups (no nested-CV denominator mismatch).
3. Score through `score_arm` (shared). Report param_count (384), AUROC, AURC, ECE, sel_acc, NT.

### Arm C (LoRA) - reuse + the two patches
1. Pooled-train invocation: LORA_PAIRS = concatenated train-endpoint jsonl, the `{property}`-templated
   LORA_PROMPT, external test_ids from `folds/<fold>.json`, LORA_N up to 4400, r=16 (primary) and
   r=32, sweep epochs in {3,5} on train-fold CV then freeze.
2. Eval verbalized output on the test_ids (base vs finetuned). In-property baseline = LoRA trained AND
   tested on the held-out endpoint via the SAME test_ids.
3. LORA_SHUFFLE=1 negative control. Score through `score_arm` (shared). Writes `results/ws3_lora.json`.

### Parity guardrails (pre-registered)
- Same train_ids/test_ids per arm (cap 4400 balanced), same scaffold groups, same calibration split,
  same `score_arm`, same `paired_cluster_boot`.
- Param counts reported: bridge approximately 7.5M (k=4) up to its swept k, LoRA-r16 approximately
  6.8M / r32 approximately 13.6M, head approximately 384, bypass = bridge param budget. "Bridge wins"
  must be a placement effect, not size: H1b requires beating BOTH orchestrate AND the equal-budget
  bypass control, and the r32 LoRA (larger) is the fair LoRA comparator.
- Same fixed `{property}`-templated prompt skeleton across bridge and LoRA (only the soft-prompt
  injection differs).

---

## 6. Metrics and pre-registered decision rules

### Primary metrics
- **Ordering (H1)**: per-endpoint within-property AUROC, all three arms + bypass control + Morgan-probe
  + k-NN, with PAIRED cluster-bootstrap 95% CIs on the shared items. Decision: arm X "wins" iff point
  estimate higher by >= 0.03 AND `paired_cluster_boot` CI of (X - Y) excludes 0.
- **Transfer (H2)**: per-arm Delta-over-floor (AUROC_holdout - AUROC_floor) with paired CI (primary)
  and NT (descriptive). A transfer claim requires the held-out CI to exclude the frozen floor.
- **Calibration (H3)**: per-arm AURC (raw scores), ECE (10-bin and 5-bin on per-arm temperature-scaled
  probabilities), sel_acc@cov0.5 with per-arm tau, EACH vs temperature-scaled LLM-output. Reliability
  diagram per arm.

### Pre-registered decision rules (the verdict map)
1. H1a fires AND H1b does NOT AND H2a holds (NT_orch highest, NT_lora lowest) AND H3a holds:
   -> "route, don't train" CONFIRMED on a fair triple-with-transfer-and-calibration. Headline.
2. H1b fires (bridge > orch by >= 0.03 paired AND bridge > bypass by >= 0.03) OR H2b fires
   ((NT_bridge - NT_orch) CI excludes 0, magnitude >= 0.05, AND bridge > bypass on transfer):
   -> verdict OVERTURNED for this substrate; the in-language bridge read is the win. Reported as the
   refutation (the pre-committed "could refute" path; do not suppress).
3. Flat outcome (3-way tie everywhere, no held-out CI excludes floor, no AURC edge):
   -> placement-agnostic NEGATIVE. The triple does not separate here; reported honestly (the
   data-war / spectra_ms precedent: a pre-registered negative is a result).
4. Specialist-dominance (always reported): if every LLM arm <= the cheap Morgan/k-NN specialist on an
   endpoint, that endpoint is declared "do not use an LLM here" regardless of H1 ordering. The LLM-win
   search is concentrated on clearance (weakest specialist, pending the P5 capped-n re-check).

### Falsification summary (one line each)
- H1 falsified: flat 3-way tie on the headline endpoint (no >= 0.03 separation, no paired CI excludes 0).
- H2 falsified: no arm's held-out-AUROC CI excludes the frozen floor on any held-out endpoint.
- H3 falsified: no arm's AURC beats temperature-scaled output by >= 0.05 on the headline endpoint.

### Anti-confirmation discipline
- All thresholds, the headline LOPO fold, the transfer floor procedure, and the outcome table are
  fixed HERE, before any run. The transfer floor and the capped-n clearance ceiling are computed and
  frozen BEFORE the arms run.
- The bridge's "could overturn" paths (H1b/H2b) carry the SAME 0.03/0.05 thresholds as the prior's
  paths and ADD the bypass-control requirement (which guards against a bridge over-claim, not a
  bridge under-claim); the bridge tuning budget guards symmetrically against a bridge under-claim.
- Small-n discipline: read effect sizes off the largest viable split; per-endpoint n in the hundreds
  gets paired cluster-bootstrap CIs; the AMES n=206 (0.376) -> n=2000 (0.145) over-estimate is the
  named cautionary case. ames (n=517) is a TRAIN endpoint in the headline fold and contributes < 3% of
  pooled train rows, so its noise is diluted; this is noted, not upsampled.

---

## 7. Compute plan (Cayuga a40)

### What runs where
- **Embedding build (Arm B/A inputs)**: LOCAL Mac MPS, `eval/sfm_embed_admet.py`, approximately 1.5
  min, no download (ChemBERTa/MoLFormer cached). Produces `signal/admet/<endpoint>/{chemberta,
  molformer}.npz`. SMOKE-TEST on 8 SMILES locally before the full approximately 22k run.
- **Orchestrate arm (B)**: LOCAL or Cayuga CPU, `probe_common`, minutes. No GPU.
- **Bridge arm (A)** and **LoRA arm (C)**: Cayuga a40 GPU.

### Embed fork is NOT 1:1 (submit-time crash fix, pre-registered)
`eval/sfm_embed_admet.py` is a fork of `sfm_embed_meltome.py` but MUST add `trust_remote_code=True`.
`sfm_embed_meltome.py` calls `AutoModel.from_pretrained(MODEL)` and `AutoTokenizer.from_pretrained
(MODEL)` with NO `trust_remote_code` (lines 57-58). The molecular FM configs require it:
- MoLFormer-XL is `MolformerForMaskedLM` with an `auto_map` -> `AutoModel.from_pretrained(...,
  trust_remote_code=True, deterministic_eval=True)`, and `AutoTokenizer(..., trust_remote_code=True)`.
- ChemBERTa-77M-MTR is `RobertaForRegression` (a DeepChem custom class); load the base encoder, also
  with `trust_remote_code=True` for safety, and mean-pool `last_hidden_state` (the regression head is
  not used).
- SMILES BPE tokenization differs from ESM (no `padding`/special-token assumptions carried over); use
  each model's own tokenizer. Budget the MoLFormer remote-code path as the risk; ChemBERTa-only is the
  de-risked v1 first cut.

### Cayuga launcher (template = `eval/cayuga_ws3_lora_cells.sbatch`)
Mirror it for `eval/cayuga_bridge_arm.sbatch`:
- `#SBATCH --partition=scu-gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=01:00:00
  --exclude=g0001`, `--output=logs/bridge_arm_%j.log` (walltime 1:00 backfills the busy partition;
  the 1:30->0:40 lesson).
- `REPO=${REPO:-$HOME/bge}`; `cd "$REPO"`. `PY=${PY:-python}` pointing at the env with
  torch(+cu124)/transformers/peft/sklearn; rdkit is lazy.
- Env per invocation: `BRIDGE_LLM=Qwen/Qwen3-8B` (v2 adds Llama-3.1-8B), `BRIDGE_PEAK_LAYER` = the
  FROZEN peak-selectivity integer for the test endpoint (NOT "default argmax"; the launcher must NOT
  silently resolve to layer 2), `BRIDGE_K=4`, `BRIDGE_PAIRS` = pooled train jsonl,
  `BRIDGE_HOLDOUT` = test endpoint list, `BRIDGE_FOLD` = `signal/admet/folds/<fold>.json`,
  `BRIDGE_BYPASS=1` for the control run.

### Gotchas (pre-registered, from prior Cayuga ops)
- FLAT staging: `rsync -aR` a structure-preserving subset to `~/bge`; the repo is NOT checked out on
  Cayuga. `probe_common.py` and the new `score_arm`/`paired_cluster_boot` staged flat too. Arms write
  to `~/results/` then pull back.
- `export PYTHONNOUSERSITE=1` BEFORE python (a shared conda `~/.local` numpy2-vs-sklearn break
  otherwise; with the guard, numpy2.4/sklearn1.8/torch2.6+cu124/transformers/peft all import).
- lmod-init before `module load cuda`; `HF_TOKEN` for gated Llama-3.1-8B (Qwen3-8B already cached).
- Use `python -u` + a Monitor until-loop on the log (foreground sleep is blocked; unwatchable without
  -u).

### Runtime estimate (split v1 vs v2)
- **v1** (Qwen / 1 fold / ChemBERTa / primary bridge + bypass + LoRA r16/r32 + orchestrate): bridge
  approximately 2 GPU-runs at approximately 25 min, LoRA approximately 30-45 min, orchestrate
  CPU-minutes. **< 3 a40 GPU-hours total.**
- **v2** (the full matrix: + Llama + additive-at-peak ablation + MoLFormer + folds 2-3 + pretraining-
  naive control + in-property baselines per arm): each bridge training step is a full forward+backward
  through a frozen 8B in bf16 (the projection grad backprops through layers above the inject point),
  so the bridge ALONE is roughly folds x models x variants x (transfer + in-property) which is tens of
  runs. **Approximately 10-20 a40 GPU-hours**, promoted only if v1 separates.

---

## 8. What it feeds / the contribution

### The measurement -> routing -> calibration layer
- **Measurement** (done): encode>>express is real and per-model-localized (layer-loc warm-up).
- **Routing** (this 3-way): on ONE task with ONE frozen SFM and ONE scorer, WHICH placement reads the
  encoded signal best, and does the read TRANSFER as a skill (floor-anchored, bypass-controlled). The
  pre-committed prediction (orchestrate >= bridge >> LoRA; orchestrate transfers best) makes "route,
  don't train" a falsifiable claim with a stated, equally-weighted refutation path (bridge >
  orchestrate AND > bypass on held-out property).
- **Calibration** (H3): the read-out's per-item, per-arm-temperature-scaled confidence as the
  PERMISSIONING signal, validated against the fair temperature-scaled-output baseline and a per-arm
  operating threshold. This is what the calibrated-permissioning layer routes on.

### Closed-weight asymmetry (the policy-relevant point)
- Orchestrate (Arm B) needs ONLY an open SFM + a tiny head; the LLM can be closed-weight frontier. If
  H1a holds, the closed-weight-friendly placement is the win -> frontier LLMs are NOT disadvantaged for
  biological grounding (consistent with the methylation ICL and AlphaMissense orchestrate results).
- LoRA (Arm C) needs open LLM weights; bridge (Arm A) needs open LLM AND open SFM. SFMs are open;
  frontier LLMs are not. The 3-way maps directly onto deployability: the winning arm tells you whether
  biological grounding REQUIRES open weights or can be done by orchestration around a closed model.
- This is the contribution beyond "another bridge": a FAIR, pre-registered, refutable test of WHERE to
  place the read (around vs inside the model), with floor-anchored transfer and per-arm calibration,
  that informs the open-vs-closed weight question for biological AI.

### Honest stance preserved
The verdict could be confirmed OR overturned; both paths carry identical thresholds (and the bridge
gets both a tuning budget and a bypass control so neither over- nor under-claiming is baked in). If the
bridge surprises us (H1b/H2b), that is the headline and we report it. If the flat negative fires, that
is reported too. We do not import the killed over-claims (bigger-LLM-better-bridge; probes-truer-than-
output), and the bridge is a TRAINED projection, not the prompt-paste strawman already shown to be
chance.

---

## Known limitations (deliberately deferred or accepted)

1. **MTR pretraining coverage is not dissociated in v1.** ChemBERTa-77M-MTR is pretrained by
   multi-task regression on approximately 200 computed descriptors, so a v1 transfer result could
   reflect the SFM having seen the property rather than read-out skill transfer. The
   pretraining-naive control (ChemBERTa-MLM / Morgan) is deferred to v2. v1 transfer claims are
   therefore stated as "transfer on a property-pretrained substrate," and the interpretation
   (structural vs pretraining-leakage) is settled only after the v2 control. This is the single
   biggest deferred interpretive caveat.

2. **Llama-3.1-8B and the additive-at-peak injection are v2.** v1 runs Qwen only with the layer-0
   soft-prompt primary, because the non-hERG peak-selectivity layers are not yet localized (only hERG
   and withdrawn profiles exist in `layer_profiles.json`). The cross-model generality of any v1
   finding and the "read enters at the encoding depth" claim are explicitly NOT established by v1.

3. **The additive-at-peak hook adds, it does not prepend new positions.** The conceptually cleanest
   "inject k new tokens at the encoding layer" is not buildable (positions must exist from layer 0).
   The realized peak-layer variant adds to the last prompt token's hidden state. This is a deliberate
   mechanism substitution; the layer-0 soft-prompt is the primary so the headline does not hinge on it.

4. **clearance at capped n is the LLM-win candidate but its capped-n ceiling is unconfirmed at write
   time.** The 0.746 ceiling is a full-n number; P5 re-measures it at n approximately 4000 before it
   gates anything. If it falls below 0.65 the LLM-win search loses its cleanest target and falls back
   to the next-weakest viable specialist. The headline H1/H2 on hERG do not depend on this.

5. **Small-n train endpoints (ames n=517, solubility n=866) remain in the pooled train union.** They
   are not upsampled; their contribution is < 3% (ames) of pooled rows and is accepted as diluted
   noise rather than corrected, to keep the train set a faithful union.

6. **The global scaffold firewall (P3) may shrink the held-out test set.** Assigning each Murcko
   scaffold to exactly one endpoint and reporting the headline on the scaffold-disjoint subset reduces
   effective n on the held-out endpoints (raw seam overlap is approximately 8% per endpoint, but
   scaffold-level removal is larger). The full-set number is reported as secondary; if the
   scaffold-disjoint subset is too small for a stable paired CI, that is flagged and the claim is
   downgraded to the entity-control axis for that endpoint.

7. **Three folds, not the full C(7,2)=21 rotation.** Transfer generality across all property pairs is
   not measured; the 3 pre-registered folds (CYP-cluster-safe) are a sample, not the population, and
   fold-to-fold variance beyond these three is unknown.

8. **Meltome remains a weak protein-side control only (ceiling 0.633, Tm-only).** It cannot carry a
   property-transfer claim and is reported only as the held-out-entity / protein robustness check; the
   protein side of the cross-domain story is acknowledged as under-powered.

---

## Key file paths (all absolute)
- LoRA arm (Arm C; needs the `{property}` + external-`test_ids` patches, see Section 2/5):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/eval/ws3_lora.py`
- Orchestrate head + calibration metrics (Arm B; ADD `paired_cluster_boot` + tau to `aurc`/`sel_acc`):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/eval/probe_common.py`
  (`balanced_lr`, `control_curve`, `ece`, `aurc`, `sel_acc`, `cluster_boot`)
- Embed pipeline to fork (ADD `trust_remote_code=True`):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/eval/sfm_embed_meltome.py`
  -> new `eval/sfm_embed_admet.py`
- Per-endpoint ceilings/selectivity (the H1 specialist bar + P1 gate; clearance re-measured at capped n):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/signal/admet/verifiability_report.json`
- ADMET SMILES (the 7 endpoints; each record carries `"property"`):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/signal/admet/<endpoint>/pairs.jsonl`
- Per-model attach layer (peak-selectivity, hERG-only today):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/results/layer_profiles.json`
  and `results/benchmark/layerloc/SUMMARY.md`
- Cayuga launcher template:
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/eval/cayuga_ws3_lora_cells.sbatch`
- Meltome (Tm-only, ceiling 0.633; protein robustness + entity control only):
  `/Users/jak4013/Dropbox/Bioinformatics/Claude/Bio_Grounding_Eval/signal/sfm_embedding/meltome_esm2.npz`
- NEW to write: `eval/bridge_arm.py` (projection + layer-0 prepend + additive-at-peak hook + bypass
  control + Stage-B), `eval/sfm_embed_admet.py`, `eval/cayuga_bridge_arm.sbatch`, the shared
  `score_arm` + `paired_cluster_boot` (in `probe_common.py`), `signal/admet/folds/<fold>.json`,
  `results/bridge_arm.json`

## Sources
- Bridge recipes: ProteinGPT 2408.11363, Cell2Text 2509.24840, BioVERSE 2510.01428, STELLA 2506.03800,
  MutaPLM 2410.22949.
- External steering beats DPO in low data: 2505.15093; reward-guided steering 2507.00445.
- Transfer benchmark context: FLIP (Dallago et al., bioRxiv 2021.11.09.467890), FLIP PLM eval
  2501.18223; PEER (Xu et al. 2022).
- Calibration baseline: Guo et al. 2017 (temperature scaling). Selectivity control: Hewitt-Liang
  1909.03368. Probes-not-truer: Belinkov 2102.12452.
