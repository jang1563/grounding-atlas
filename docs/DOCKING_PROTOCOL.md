# Bias-arbiter protocol for the Experiment-3 high-budget RL edge

Proportionate, locally-feasible arbiter for the hERG generative arm (route-don't-train).
Status of the effect under test: REAL but MODEST, seed-variable, sub-threshold, INDETERMINATE.
This protocol is designed to REMOVE ONE explanation (shared reward-oracle ML bias). It cannot,
by construction, produce an overturn (see Section 8). Plain-ASCII. No cluster required for the
primary verdict.

---

## 1. The question and why it matters

Anchored to [results/benchmark/rl_env/budget_sweep.md](../results/benchmark/rl_env/budget_sweep.md).

The Experiment-3 budget sweep (hERG, M=1000, reward-query budget Q swept) was expected to confirm
the "route-don't-train" tie across budgets. At high budget it did not. Pooled across 3 seeds at
Q=10000:

- arm A (internalized PPO RL): 280 / 3000 draws = 0.0933 oracle-pass
- arm B (external guidance):    29 / 1000 draws = 0.0290 oracle-pass
- (A - B) success-rate-over-draws = +0.064, scaffold-clustered 95% CI [+0.026, +0.104]

The CI excludes 0 (A genuinely beats B at high budget) but its lower bound 0.026 sits JUST UNDER
the pre-registered 0.03 overturn margin, so the cell is INDETERMINATE. Seed 0 alone (170/1000 =
0.170, (A-B)=+0.141) is a HIGH OUTLIER; seeds 1/2 give 0.063 / 0.047. The number under test is the
POOLED +0.064, never the seed-0 0.141.

### The confound this protocol exists to break

Arm A optimized the REWARD (ChemBERTa logistic-regression). The held-out ORACLE that scores
oracle-pass is a Morgan-fingerprint random forest on block-O. These are a DIFFERENT model and a
DIFFERENT featurization, but they were trained on the SAME internal hERG label lineage
(chemberta_herg.npz, ChEMBL/TDC-derived) and BOTH read a 2D-substructure feature surface. If the
reward and oracle share a bias -- both over-calling a drug-like chemotype that is not truly a hERG
blocker -- then at high budget arm A pushes into that shared-bias region and the +0.064 is
reward-hacking a correlated in-silico pair, not producing genuine hERG-blocker-like actives.

The prior property check already argues AGAINST crude gaming: arm A's passers match the known
hERG-blocker chemotype (large, lipophilic, polycyclic; QED 0.31 / MW 508 / 5.1 rings) and match
guidance's passers on the same axes. So the surviving loophole is NARROW and specific: a shared
ML bias that over-calls a genuinely-drug-like-looking-but-inactive chemotype. This protocol targets
exactly that loophole.

---

## 2. Primary arbiter and its two-part self-validation gate

### 2.1 What the primary arbiter is

An independent-featurization + external-label hERG classifier, run entirely on local CPU, built so
its error is INDEPENDENT of the reward+oracle shared failure mode on BOTH axes it must break:

- DIFFERENT feature surface: RDKit physicochemical descriptors + MACCS keys. NOT Morgan
  fingerprints, NOT ChemBERTa embeddings.
- DIFFERENT label lineage: an EXTERNAL hERG assay whose PRIMARY PROVENANCE is disjoint from the
  internal ChEMBL/TDC lineage (see 2.2). This is the load-bearing leg.

Two variants:

- 2b (REQUIRED leg, load-bearing): trained/validated on the external lineage-disjoint hERG set.
- 2a (OFFLINE FALLBACK ONLY): trained on block-R (different featurization, but SAME internal label
  lineage). 2a breaks only the feature surface, not the label lineage. It is necessary-not-sufficient
  and can NEVER on its own license a "genuine" verdict (see decision rule, Section 7).

The frozen reward (herg_reward_ensemble.pkl) and oracle (herg_oracle_rf.pkl) are reused for
reference scores ONLY and are never refit.

### 2.2 Label-lineage independence requirement (BLOCKER fix from the independence critic)

ChEMBL/hERG_Karim is ALSO a ChEMBL aggregation and therefore re-imports the exact blind-spot the
arbiter must be independent of. It is NOT an acceptable 2b source. The 2b external set MUST have an
independent PRIMARY provenance, not merely a different ChEMBL re-packaging. Acceptable sources, in
preference order:

1. A named single-lab / consortium automated-patch-clamp hERG panel (e.g. an NCATS / Kramer-style
   patch-clamp release) with primary electrophysiology data.
2. PubChem BioAssay hERG AIDs that do NOT feed TDC hERG_central / Karim (verify by AID provenance).

Hard independence checks, all pre-registered:

- InChIKey de-duplicate the 2b training molecules against BOTH the internal 3963-molecule pool
  (chemberta_herg.npz) AND the TDC hERG_central pool. Report the overlap fraction.
- If overlap > 5%, the "external" label is contaminated: the 2b independence claim FAILS and 2b is
  not usable as the load-bearing leg.
- The report MUST carry a lineage-provenance statement: assay type, source lab/consortium, and the
  measured de-dup overlap number. The "confirms_genuine" verdict is CONDITIONAL on overlap being
  near zero.

Pin an exact, versioned URL/accession/DOI for the 2b source and CACHE it into the repo now while
network is confirmed up (RCSB/PyPI/GitHub return HTTP 200 at write time), so 2b is offline-reproducible
and a run-time network hiccup cannot silently downgrade the verdict to the 2a fallback.

### 2.3 The self-validation gate: THREE co-requirements, all blocking

The arbiter is NOT permitted to score arm A until it passes ALL THREE gates below on the block-E
controls (208 known actives / 1113 known inactives, held-out and scaffold-disjoint from blocks R and
O -- see Section 4). Passing merely "better than useless" is not enough to adjudicate a 6-point
fraction difference.

GATE A -- COMPETENCE (discrimination floor).
  - block-E AUROC >= 0.70 (pre-registered floor), AND
  - a NUMERIC early-enrichment floor, because the verdict rests on the top-scoring passer tail under
    a ~16% base rate where global AUROC can hide near-zero enrichment: EF1% >= 3x OR BEDROC(alpha=20)
    >= 0.30.
  Empirical note: the 2a offline fallback (physchem+MACCS RF on block-R, scored on block-E) was
  measured at AUROC = 0.741 -- it clears 0.70 by only ~0.04. This is a barely-discriminating
  instrument, which motivates gates B and C.

GATE B -- EFFECT-SCALED POWER (must be able to see +0.064).
  Via a label-permutation / down-sampling power simulation on block-E at the arbiter's MEASURED
  AUROC and at arm B's realized N (~29 delivered passers, or the full-draw N=1000 for the primary
  estimand in Section 6), require >= 80% power to detect a +0.064 fraction-active difference with the
  pre-registered two-sample bootstrap CI machinery. If the minimum detectable effect (MDE) at the
  arbiter's AUROC and these Ns EXCEEDS 0.064, the arbiter is under-powered for this question: route
  to INDETERMINATE BY CONSTRUCTION rather than reporting a false "survives" or "collapses".

GATE C -- INDEPENDENCE (must part company from the reward+oracle pair).
  A high block-E AUROC is also achievable by a NON-independent arbiter that shares the blind-spot, so
  competence alone does not certify independence. Construct a disagreement probe: assemble molecules
  where reward AND oracle both score high but an assay-independent flag (a known-inactive from the 2b
  external source, or a hard cross-assay negative) says inactive. Require the arbiter to side with the
  external source on these divergence cases at a pre-registered rate materially above chance. An
  arbiter that passes A and B but fails C is reported as "competent but independence-unverified" and
  routes to INDETERMINATE.

If any gate fails: HALT, declare the arbiter uninformative, report "inconclusive; need a different
orthogonal oracle (independent-assay ML, interaction-fingerprint/pharmacophore, or MM-GBSA)."

### 2.4 Why this arbiter (and why not docking as primary)

This is the only immediately-runnable arbiter whose error is independent of the reward+oracle shared
(2D-substructure-feature x internal-label-lineage) failure mode -- the exact logical requirement that
separates "genuine binder" from "reward-hacked a correlated ML pair." Physics docking cannot be the
primary: hERG docking discrimination is expected WEAK (large promiscuous central cavity; control-gate
AUROC likely 0.5-0.6), so a null docking result is ambiguous, not confirmatory, and cannot arbitrate
a sub-threshold effect. Best cost/decisiveness ratio of any available check; no docking engine
required to reach a verdict.

---

## 3. Arbiter-weakness and OOD guards (fixes that keep a weak arbiter honest)

Because the primary arbiter (2a measured 0.741) is WEAKER than the incumbent oracle
(block_e_auroc = 0.882 stored in herg_oracle_rf.pkl), two additional guards are mandatory so a
"collapse" is not mis-read as gaming and an OOD score is not mis-read as signal.

### 3.1 Discrimination-loss confound guard (guards the confirms_gaming branch)

A weaker arbiter has more ranking error, which regresses ANY small between-arm gap toward zero. So a
shrunk (A-B) edge under the primary arbiter is confounded with the arbiter simply being noisier.

Calibration: score the block-E known-actives with BOTH the incumbent oracle (0.882) and the primary
arbiter, and measure how much the weaker arbiter compresses a synthetic +0.064 separation (inject a
known +0.064 fraction-active gap between two block-E-derived pseudo-arms and measure the shrinkage the
arbiter alone induces). A "collapse" counts as shared-bias GAMING only if the edge shrinks by MORE
than this discrimination-loss shrinkage. Also run the SAME full-draw (A-B) contrast under the strong
incumbent oracle as a reference point.

### 3.2 Arbiter-native applicability-domain (guards against OOD extrapolation)

The AD falsifier that already fired (arm-A passers max-Tanimoto 0.716 ~ known-active self-reference
0.723, referenced to the FULL 625-actives library -- see Section 5) rules out OOD gaming relative to
the known-active chemistry manifold. It does NOT establish that arm-A passers are in-domain to the
ARBITER'S OWN training manifold. Arm-A passers are near-novel generated molecules (168/170 not in the
internal pool), so one can sit inside the known-active Tanimoto ball yet fall in a low-density corner
of the descriptor+MACCS space where arbiter predictions are unreliable extrapolations.

Required before scoring:
- Compute each test molecule's distance to the arbiter's OWN training set in the arbiter's feature
  space (k-NN distance in standardized RDKit-descriptor+MACCS space, or leverage/hat-value for the
  descriptor block). Define an AD threshold from the training-set distance distribution.
- Report the fraction of arm-A / arm-B / control molecules OUTSIDE the arbiter's AD.
- Restrict the load-bearing (A-B) comparison to IN-DOMAIN molecules; report the out-of-domain
  fraction separately as an uncertainty band. If a large share of arm-A passers are OOD for the
  arbiter, the arbiter is uninformative for exactly the surplus in question -> INDETERMINATE.

---

## 4. Molecule sets and controls

All canonicalized with RDKit; molecules that fail to parse are dropped with a logged count.

### 4.1 The primary estimand fix (BLOCKER from the logic critic)

The +0.064 is a SUCCESS-RATE-OVER-DRAWS (fraction of DRAWS the pipeline delivers as oracle-pass), an
UNCONDITIONAL quantity. Scoring only the old-oracle passers measures P(new-active | old-passer), a
CONDITIONAL quantity on a different scale -- comparing the two is a category error. TEST_A_EXTRA (the
"+0.064 surplus") is a ~193-molecule PHANTOM, not an identifiable set, and arm B has only 29 identified
passers, so no clause may "score the extra passers."

Therefore the primary metric is computed on ALL DRAWS:

- FULL_A: ALL 3000 arm-A draws (pooled 3 seeds, herg_armA_ppo_s{0,1,2}_Q10000.json).
- FULL_B: ALL 1000 arm-B draws (herg_armB_guidance.json, Q=10000 delivered slice, M=1000).

The new arbiter scores every draw at a pre-committed threshold; the success-rate-over-draws is the
fraction of DRAWS called active. The (A-B) of THESE full-draw success rates is compared to +0.064 on
the SAME scale. The verdict is a two-DISTRIBUTION comparison, never a scored "extra" subset.

### 4.2 Passer sets (secondary conditional diagnostic only, explicitly a different estimand)

- TEST_A: arm-A oracle-passers. PRIMARY diagnostic set = the POOLED 3-seed passers (272 unique of
  280), NOT the seed-0 170 outlier. Seed 0 (170), seed 1 (63), seed 2 (47) reported per-seed as
  diagnostics only.
- TEST_B: arm-B oracle-passers = the Q=10000 DELIVERED-SLICE passers = 29 (top-1000 by guidance score
  from the 10000-candidate pool). NOT the whole-pool count 37; 29 keeps the denominators matched
  (1000 vs 1000). Low N -> always report CIs, never a point estimate.

### 4.3 Controls

- POSITIVE_CONTROL: 208 block-E known hERG actives (y==1 AND scaffold_to_block==E). Held-out,
  scaffold-disjoint from blocks R (reward) and O (oracle) by construction in build_holdout_oracle.py.
  No training contamination.
- NEGATIVE_CONTROL (unmatched): 1113 block-E known hERG inactives (y==0 AND scaffold_to_block==E),
  same disjointness guarantee.
- NEGATIVE_CONTROL (matched, MANDATORY baseline -- controls-critic fix): property/scaffold-matched
  decoys. For each arm-A passer, nearest-neighbor-match a block-E inactive on (MW, LogP, QED, ring
  count, HBD, HBA). This prevents the arbiter from separating arm-A passers on gross physicochemistry
  instead of hERG-blocker-likeness. The load-bearing (A-B) statistic is reported against BOTH the
  matched and the unmatched negative sets; SURVIVAL AGAINST THE MATCHED SET is required for
  "confirms_genuine".

### 4.4 Novelty / memorization dedup (controls-critic fix)

Before scoring, drop (or separately flag) any arm-A or arm-B passer whose canonical SMILES appears
anywhere in chemberta_herg.npz['smiles'] (2 of 170 seed-0 passers are exact matches = trivial
memorization, neither a block-E active). Report passer counts before and after dedup so the edge is
stated on genuinely novel molecules.

---

## 5. Standing free falsifiers (computed in STEP 4, not pre-existing)

Both are FAST local CPU checks (the full primary featurization of ~4280 molecules plus the AD
Tanimoto falsifier reproduced in ~7 s on this Mac). They are re-derived in STEP 4, NOT carried as
prior outputs. They only NARROW the loophole the primary arbiter must close; they are never
standalone genuine-binding evidence.

- APPLICABILITY-DOMAIN / OOD falsifier: max-Tanimoto (Morgan/Tanimoto) of each molecule to the
  nearest known active. REFERENCE SET = the FULL 625-actives library (this is a chemistry-library
  question, not a train/test-leakage question; block-E disjointness applies to ML-arbiter TRAINING
  only). Against the 625-library the carried values reproduce (arm-A 0.716 ~ known-active self-ref
  0.723); against block-E-only they would mislead (0.546 vs 0.641). PIN the reference to the 625
  library. Already points away from crude OOD gaming.
- SCAFFOLD-CONCENTRATION mechanism check: Murcko-scaffold distribution of arm-A passers (89 distinct
  scaffolds among the 170; top-1 ~21%, top-2 ~31%, masked by the 0.822 diversity metric). Flag
  scaffolds over-represented in arm-A passers but rare/absent among the 625 known actives. This
  partial-concentration subset is the suspect subset that gates any secondary docking (Section 6).

---

## 6. Executable steps (local CPU)

STEP 0 -- Extract sets. From herg_armA_ppo_s{0,1,2}_Q10000.json take ALL designs (FULL_A, 3000) and
the oracle_pass_vec==1 subset (TEST_A passers, 170/63/47). From herg_armB_guidance.json take ALL
delivered-slice designs (FULL_B, 1000) and oracle_pass==1 (TEST_B, 29). From chemberta_herg.npz +
herg_partition.json['scaffold_to_block'] take y==1 & block==E (208 pos) and y==0 & block==E (1113
neg). Canonicalize, drop parse failures (log count), run the novelty dedup (4.4).

STEP 1 -- Build the primary arbiter WITHOUT touching block-O or the reward's Morgan/ChemBERTa surface.
Featurize with RDKit physchem descriptors + MACCS. Labels: (2b, required) fetch/load the cached
lineage-disjoint external hERG set (2.2), InChIKey-dedup vs internal pool and TDC hERG_central, report
overlap; (2a, offline fallback only) train on block-R. Serialize the arbiter. Do NOT refit the frozen
reward/oracle.

STEP 2 -- SELF-VALIDATION GATE (blocking, all three co-requirements of 2.3): score block-E POS (208)
vs NEG (1113). Compute AUROC + EF1% + BEDROC(alpha=20) (Gate A); run the effect-scaled power/MDE
simulation (Gate B); run the reward-oracle disagreement probe (Gate C). If any gate fails -> HALT,
report uninformative-arbiter INDETERMINATE. Only an arbiter passing A+B+C may proceed.

STEP 3 -- PRIMARY full-draw scoring. Score FULL_A (3000) and FULL_B (1000) at the pre-committed
threshold; compute success-rate-over-draws per arm and (A-B) with two-sample bootstrap 95% CI
(scaffold-clustered where applicable; paired_cluster_boot is inapplicable to these disjoint new-molecule
sets). Compare to +0.064 [+0.026,+0.104]. Stratify arm A by seed and pool. Restrict to arbiter-in-domain
molecules per 3.2; report OOD fraction as an uncertainty band. Also score the passer sets (TEST_A pooled +
per-seed, TEST_B) and the controls (POS, matched NEG, unmatched NEG) as the CONDITIONAL diagnostic,
explicitly labeled a different estimand. Apply the discrimination-loss guard (3.1) before interpreting any
shrinkage.

STEP 4 -- Compute the two free falsifiers (Section 5): AD max-Tanimoto to the 625-library; Murcko-scaffold
over-representation of arm-A vs the 625 known actives. Report as loophole-narrowing evidence only.

STEP 5 -- Decision (Section 7). If genuine-or-ambiguous AND the effect matters, PROCEED to the gated
secondary docking arm (Section 6b) on the scaffold-concentration suspect subset ONLY. If the primary
collapses the edge (past the discrimination-loss guard), report shared-ML-bias gaming and stand the
effect down to a tie WITHOUT spending docking compute.

STEP 6 -- Report every arbiter with its own discrimination ceiling attached (block-E AUROC + EF1%/BEDROC
for the primary; Stage-1 control AUROC for docking). Keep the prereg verdict INDETERMINATE /
route-don't-train-holds until the primary reports; no single weak arbiter flips the sub-threshold call.

### 6b. Secondary/confirmatory arbiters (all gated, none a sole overturn gate)

DOCKING (pre-registered physics co-primary, DEMOTED to secondary/confirmatory, GATED, and NOT on the
verdict's critical path -- feasibility-critic BLOCKER fix). The verdict is COMPLETE and reportable on the
primary arbiter + the two free falsifiers ALONE. Docking is optional bonus.

  - Stage 0 (install; unverified on arm64): FIRST run `conda search -c conda-forge smina` and confirm an
    osx-arm64 (or working Rosetta osx-64) build. smina/obabel/vina/qvina2/meeko are ALL absent locally and
    the conda shell function was not even invokable at review time, so the conda-forge install is NOT
    assumed to work. If no arm64-native build resolves, substitute an arm64-native engine (AutoDock Vina
    1.2 pip wheel, or QuickVina2 via homebrew) or DROP docking. Pre-committed branch:
    docking-unavailable -> INDETERMINATE (never a stall, never a verdict-mover).
  - Stage 1 VALIDATION GATE (must pass first): prep receptor from PDB 5VA1 (Kv11.1 central cavity below
    the selectivity filter; box ~22 A cube centered on the four Y652/F656; waters/K+ removed; polar H;
    Gasteiger; PDBQT via obabel). Dock a balanced block-E known-active vs known-inactive set. Report AUROC
    + EF1%/BEDROC of best score. Pre-registered go/no-go: control AUROC >= 0.65 AND EF1% >= 3x (or
    BEDROC(alpha=20) >= 0.30). LIKELY to FAIL for hERG.
  - Stage 2 (ONLY if Stage 1 clears the bar): dock ONLY the scaffold-concentration suspect subset (top-1/
    top-2 scaffolds ~31% of passers) plus matched known actives and guidance passers. Compare docking-
    affinity distributions RELATIVELY (arm-A-extra vs known-active vs guidance) with CIs, stratified by
    seed. Report the Stage-1 control AUROC alongside.
  - If Stage 1 fails (the likely hERG outcome): STOP, report docking uninformative. Docking NEVER alone
    flips the call.

CHEAP NON-DOCKING PHYSICS LEG (independence-critic fix, so the protocol is not one failed docking gate
away from having zero bias-independent arbiters): a pharmacophore / interaction-fingerprint match to a
Y652/F656 aromatic-cage model (aromatic + basic-nitrogen + hydrophobe geometry), engine-free. Reported as a
weak orthogonal physics-flavored signal.

SCAFFOLD-CONCENTRATION MECHANISM CHECK (Section 5): supporting evidence.

APPLICABILITY-DOMAIN / OOD FALSIFIER (Section 5): standing falsifier; already fires negative-for-gaming.

PROPERTY-MATCHED-DECOY ENRICHMENT (now MANDATORY, 4.3): tightens the negative baseline the primary scores
against.

---

## 7. Metrics and the sharp decision rule

### 7.1 Metrics

- Self-validation gate (primary): block-E AUROC (floor >= 0.70) + EF1% (>= 3x) / BEDROC(alpha=20) (>= 0.30)
  + effect-scaled power (>= 80% for +0.064 at realized N) + independence-probe agreement rate (above a
  pre-registered chance floor). Docking Stage-1: control AUROC (>= 0.65) + EF1%/BEDROC.
- PRIMARY (load-bearing) statistic: the FULL-DRAW success-rate-over-draws (A-B) under the validated
  primary arbiter, with two-sample bootstrap 95% CI, compared to +0.064 [+0.026,+0.104] on the SAME
  scale, restricted to arbiter-in-domain molecules, against BOTH matched and unmatched negatives, AFTER
  the discrimination-loss guard.
- Conditional diagnostic: per-set fraction-called-active and mean/median arbiter score with CIs for
  TEST_A (pooled + per-seed), TEST_B, POS, matched NEG, unmatched NEG. Labeled a different estimand.
- Falsifiers: AD mean/median max-Tanimoto to the 625-library and % below 0.3; Murcko scaffold count,
  top-1/top-2 share, over-represented-vs-known-actives scaffolds.
- Docking (only if gate passes): relative docking-affinity distributions (arm-A-extra vs known-active vs
  guidance) with CIs, stratified by seed, Stage-1 control AUROC alongside.

### 7.2 Decision rule (sharp)

CONFIRMS_GENUINE (shared-bias explanation ruled out; edge stands as real-but-modest). ALL required:
  - The 2b external-lineage-disjoint leg was USED and cleared all three self-validation gates (A+B+C) with
    near-zero InChIKey overlap. A 2a-only run can NEVER reach this verdict.
  - The full-draw (A-B) success-rate SURVIVES the featurization AND external-label swap: it does not
    collapse toward zero when neither the 2D-substructure surface nor the internal label lineage is
    available, as a CI STATEMENT (not a point estimate), restricted to in-domain molecules, and it survives
    against the MATCHED negative baseline.
  - Shrinkage (if any) is within the discrimination-loss guard, i.e. not attributable to the weaker
    arbiter.
  - Reinforced by: AD falsifier already in-domain (0.716~0.723 vs the 625-library); arm-A scaffolds
    overlapping the known-active distribution. If the gated docking secondary also runs and its Stage-1
    control gate passes, arm-A-extra docking >= known-active/guidance (reproducing Y652/F656 aromatic-cage
    contacts) is confirmatory but NOT required.
  - Verdict wording: "shared-bias explanation ruled out, edge stands as real-but-modest" -- NOT
    "demonstrated genuine binding" (in-silico surrogate of a surrogate; see Section 8).

CONFIRMS_GAMING (shared reward-oracle bias; stand the sub-threshold edge down to a tie). Required:
  - The full-draw (A-B) success-rate SHRINKS toward zero under the swap by MORE than the discrimination-
    loss guard allows (i.e. not explained by arbiter weakness), OR collapses even under the weaker 2a swap
    (a strong signal).
  - Signature strengthened if arm-A passers concentrate into 1-3 high-reward+high-oracle scaffolds rare/
    absent among the 625 known actives (scaffold falsifier) and, where the docking Stage-1 gate passes,
    those same scaffolds dock significantly WORSE than known actives (two ML models agree, physics
    disagrees).

INCONCLUSIVE / INDETERMINATE (moderate-budget tie remains; high-budget effect stays modest and
sub-threshold).
Any of:
  - The primary arbiter fails ANY self-validation gate: AUROC < 0.70, OR EF1%/BEDROC below floor, OR
    effect-scaled MDE > 0.064 (under-powered by construction), OR the independence probe fails
    ("competent but independence-unverified").
  - Only the 2a fallback was available (external 2b unreachable): a SURVIVING edge caps at INDETERMINATE
    because block-R shares the internal label lineage and can preserve the edge via the same blind-spot.
    (A collapse under 2a may still support CONFIRMS_GAMING.)
  - The (A-B) CI straddles zero at the realized N, leaving the surplus neither clearly preserved nor
    collapsed.
  - A large arm-A OOD fraction relative to the arbiter's own manifold.
  - Docking (if run) fails its Stage-1 control gate (AUROC < 0.65 -- the likely hERG outcome), so the
    physics tie-breaker is uninformative.
  In all these cases the prereg verdict stays INDETERMINATE (CI lower bound 0.026 < 0.03 overturn margin),
  and the burden passes to a heavier orthogonal oracle (independent-assay ML, interaction-fingerprint/
  pharmacophore, or MM-GBSA) once clusters return.

---

## 8. This validates a SUB-THRESHOLD effect (honest scope note)

- This arbiter is ONE-DIRECTIONAL and CANNOT produce an overturn. The prereg admits OVERTURN only if
  CI-lower > 0.03 AND docking co-primary confirmation; the pooled effect is already CI-lower 0.026 < 0.03
  = confirmed-tie, and docking is expected to fail its own hERG gate. So the CEILING outcome of this whole
  protocol is "route-don't-train tie stands, shared-bias artifact ruled out." Its only actionable outcome
  is CONFIRMS_GAMING (stand the edge down). It is cheap one-directional insurance, NOT a verdict-mover.
- The MODAL expected outcome is "primary ML arbiter informative-but-independence-contested + docking
  uninformative." In that case the verdict stays INDETERMINATE / route-don't-train-holds regardless of the
  primary point estimate. There is a real chance NO arbiter here is both independent AND informative; the
  cheap non-docking pharmacophore leg (6b) exists so the protocol is not one failed docking gate away from
  zero bias-independent signals.
- A passing primary + favorable docking removes only the shared-ML-bias explanation. It does NOT prove
  wet-lab binding: the whole chain is an in-silico surrogate of a surrogate. The prereg
  "surrogate-of-a-surrogate" caveat stays binding; a positive result is framed as "shared-bias explanation
  ruled out, edge stands as real-but-modest," never as demonstrated genuine binding.
- Proportionality is the governing principle: cheap CPU ML primary first; docking only if the primary is
  genuine-or-ambiguous AND restricted to the suspect subset; no multi-day physics campaign to move an
  already-indeterminate sub-threshold effect to another indeterminate.

---

## 9. Feasibility, effort, and what needs the cluster

FULLY FEASIBLE on the local Mac (8 cores, 24 GB, arm64, rdkit 2025.09.x, sklearn 1.6.1 verified) with
clusters down.

- PRIMARY arbiter: CPU-only, cheap. RDKit physchem+MACCS featurization of ~1640 test+control molecules is
  seconds-to-minutes; the full primary featurization of ~4280 molecules plus the AD Tanimoto falsifier was
  reproduced in ~7 s locally. Training a descriptor/MACCS classifier is minutes. The falsifiers (AD
  Tanimoto, scaffold concentration) are fast local CPU and are computed in STEP 4 (NOT pre-existing).
- The one network dependency is the 2b external-hERG fetch. Network is confirmed up at write time (RCSB
  5VA1 HTTP 200). PIN + CACHE the exact versioned 2b source into the repo NOW so it is offline-reproducible;
  the block-R-trained 2a variant is the fully-offline FALLBACK (fallback-only, caps at INDETERMINATE on a
  surviving edge).
- SECONDARY docking: feasible-but-blocked-on-install and NOT on the verdict's critical path. No engine
  present. Requires a verified arm64-native install (run `conda search` first; substitute Vina 1.2 pip
  wheel / QuickVina2 if conda-forge has no arm64 build). Wall-clock is ESTIMATED, UNVERIFIED (no engine
  installed to measure): report a real prep+one-dock timing only AFTER a successful install; do not cite
  "0.13 s/mol" or "15-45 s/ligand" as measured. The proportionate suspect-subset run + Stage-1 control gate
  is intended as an overnight-safe local run, gated on Stage 1 passing, and NOT required for a verdict.
- Provenance honesty: the AD (0.716/0.723), scaffold (0.822 / 89-scaffold), and docking-timing numbers were
  NOT found in any repo artifact. They are treated as to-be-computed in STEP 4 (falsifiers, fast) or
  to-be-measured after install (docking timing), not as pre-existing outputs.

NEEDS THE CLUSTER (OUT while Cayuga/Expanse unreachable, NOT required for this verdict): GPU arbiters
(gnina, graph-neural oracle), PPO re-runs, and any heavier orthogonal oracle (independent-assay ML head,
MM-GBSA) that the INDETERMINATE branch defers to once clusters return. This is exactly why the primary is a
CPU ML oracle, not physics.
