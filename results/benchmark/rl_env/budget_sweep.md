# Experiment-3 budget sweep (hERG): a FLAGGED high-budget potential overturn

Per [docs/RL_ENV_PREREG.md](../../../docs/RL_ENV_PREREG.md). The v1/v2 result (route-don't-train
EXTENDS to generation) was established at a single budget (Q=5000). This sweep varies the
reward-query budget Q with a larger delivery M=1000 (tighter CIs), seed 0, on hERG. It was expected
to confirm the tie across budgets. It did NOT. Date 2026-07-02. Analysis `eval/analyze_budget_sweep.py`
(scaffold-clustered two-sample bootstrap per Q); jobs on Cayuga scu-cpu (both GPU clusters contended).

## The curve: A(Q) vs B(Q), matched reward-query budget, M=1000

| Q (reward queries) | arm A internalized RL | arm B external guidance | (A-B) | 95% CI |
|---|---|---|---|---|
| 1000  | 7/1000 (0.007)   | 2/1000 (0.002)   | +0.005 | [-0.001, +0.013] |
| 2000  | 17/1000 (0.017)  | 5/1000 (0.005)   | +0.012 | [-0.001, +0.028] |
| 5000  | 29/1000 (0.029)  | 12/1000 (0.012)  | +0.017 | [-0.002, +0.040] |
| **10000** | **170/1000 (0.170)** | **29/1000 (0.029)** | **+0.141** | **[+0.075, +0.216]** |

The (A-B) gap grows monotonically with budget, and at **Q=10000 the CI EXCLUDES 0**: internalized RL
significantly beats external guidance. This contradicts the single-budget (Q=5000, M=500) tie and the
"route-don't-train extends" headline - at HIGH budget the tie breaks in favor of TRAIN.

## Adversarial check (is it mode-collapse / drift, or real?)

| Q | final KL-to-base | oracle-pass | distinct scaffolds among passers | internal diversity |
|---|---|---|---|---|
| 2000  | +1.9 | 17  | 8  | 0.864 |
| 5000  | +4.1 | 29  | 17 | 0.857 |
| 10000 | +8.9 | 170 | 89 | 0.822 |

- **NOT mode-collapse:** the 170 oracle-passers at Q=10000 span 89 distinct Murcko scaffolds, and
  internal diversity (0.822) is barely down from Q=2000 (0.864) - a ~5% drop, far under the prereg's
  15% collapse gate. Arm A shifted the WHOLE distribution toward higher-oracle molecules, diversely,
  not to a few modes.
- **But high drift:** KL-to-base grows to 8.9 at Q=10000 (the initial PPO-clipped optimizer diverged
  near KL 10). The policy has moved far from the frozen prior. "Learning" (diversity intact) rather
  than obvious "off-manifold gaming", but the drift is large.
- **Mechanism:** guidance is capped by the frozen generator's support (its top-reward samples only
  reach 2.9% oracle-pass); TRAINING moves the distribution beyond that ceiling (17%). This is a
  legitimate reason RL could beat guidance at high budget - the opposite of the moderate-budget tie.

## Why this is FLAGGED, not a confirmed overturn (prereg Section 7)

The prereg admits OVERTURN only if the CI lower bound > 0.03 (YES: +0.075) AND no diversity/validity
collapse (holds) AND **the docking co-primary agrees**. The binding open question:

**Shared reward-oracle bias.** Arm A optimized the ChemBERTa-LR REWARD; the held-out ORACLE is a
Morgan-RF on block-O - a DIFFERENT model/featurization but trained on the SAME hERG data. At high
budget arm A pushes hard into the reward-preferred region; if the reward and oracle share a bias
(both over-predict a chemotype that is not truly active), the 17% is reward-hacking a correlated
in-silico pair, not genuine hERG activity. The physics-based **docking co-primary (QuickVina2, no ML
bias)** is exactly the pre-committed test to separate these, and it has NOT been run.

## Seed replication + property check (refinement)

**Seed replication (Q=10000, seeds 0/1/2):** oracle-pass rates 0.170 / 0.063 / 0.047 - seed 0's 0.170
was a HIGH OUTLIER; the typical is ~0.05-0.06. All three still beat guidance (0.029). Pooled (3000
designs): arm A 280/3000 = 0.0933 vs arm B 0.0290; **(A-B) = +0.064, scaffold-cluster 95% CI [+0.026,
+0.104].** The CI EXCLUDES 0 (A really does beat B at high budget) but its lower bound (0.026) sits
JUST UNDER the pre-registered 0.03 OVERTURN margin - so by the prereg rule the high-budget cell is
INDETERMINATE: a real, modest, seed-variable RL edge that is neither a clean tie nor a clean overturn.

**Property-bounds check (is it gaming a shared reward-oracle bias?):** arm A's FULL Q=10000 output stays
drug-like (QED 0.46, MW 436, ~4.5 rings - nearly identical to the real hERG molecules QED 0.57 / MW 438 /
4.3 rings). Its oracle-passers are larger, more-ring, lower-QED (QED 0.31 / MW 508 / 5.1 rings) - but
guidance's passers have the SAME profile (QED 0.37 / MW 500 / 5.1 rings), which is the known hERG-blocker
chemotype (large, lipophilic, polycyclic). So arm A did NOT invent weird oracle-fooling molecules; it
produced MORE of the legitimate blocker type. This ARGUES AGAINST simple gaming, though a shared bias
in what both ML models call "active" for that chemotype is still only fully ruled out by docking.

## Bias-arbiter leg 2a (independent featurization): the edge SURVIVES

Per the arbiter protocol ([../../../docs/DOCKING_PROTOCOL.md](../../../docs/DOCKING_PROTOCOL.md);
`eval/bias_arbiter.py`). The reward (ChemBERTa-LR) and oracle (Morgan-RF) share a 2D-substructure feature
surface; leg 2a re-judges the +0.064 edge with an oracle that shares NEITHER surface - an RDKit
physicochemical-descriptor + MACCS-keys RF (block-R trained). It BREAKS the Morgan surface (but not the
internal label lineage - that is leg 2b, deferred).
- **Self-validation GATE A: PASS** - the arbiter separates block-E known actives from inactives at AUROC
  0.789, EF1% 6.35x (floors 0.70 / 3x). So it is competent to judge.
- **Discrimination-loss calibration:** a weaker arbiter shrinks gaps; this one measures a true +0.064 as
  +0.019 (retention 0.30x). So a genuine edge should appear as ~+0.019 through it.
- **Full-draw re-judge: (A-B)_arbiter = +0.021** (arm A 0.340 vs guidance 0.319), essentially EXACTLY the
  discrimination-loss-expected +0.019. **The edge survives breaking the Morgan feature surface** -> the
  shared-2D-Morgan-feature bias is NOT the explanation for the high-budget RL edge.
- Caveat: 50% of arm A's designs fall outside the physchem+MACCS arbiter's applicability domain (5-NN),
  so this arbiter extrapolates on half of them; and leg 2a shares the internal hERG LABEL lineage, so it
  is necessary-not-sufficient (rules out feature-surface gaming, cannot alone license "genuine binding").

## Bias-arbiter leg 2b (external labels): INCONCLUSIVE (no independent set available)

leg 2b would break the shared LABEL lineage by training the arbiter on an EXTERNAL hERG assay. The external
candidate (TDC hERG_Karim, n=13445 patch-clamp) FAILS the pre-registered independence gate: InChIKey overlap
with the internal set is 8.5% (> the 5% floor), so it re-imports the same label lineage (the internal ADMET
hERG data and Karim both trace to overlapping ChEMBL assays - the exact "ChEMBL blind-spot" the protocol
critic predicted, correctly caught by the gate). So there is no readily-available truly-independent hERG
label set; the shared-label-lineage residual cannot be cleanly closed with available data.
(`eval/bias_arbiter_2b.py`.) Docking (the physics arbiter that would sidestep ALL label lineage) is the
one remaining lever, but its arm64 engine install failed and hERG's promiscuous cavity makes it weak - the
protocol pre-committed it as a secondary, likely-uninformative tie-breaker.

## Status (final)

- Route-don't-train HOLDS at moderate budget (Q <= 5000; confirmed across 3 reward-quality cells x 3 seeds).
- The HIGH-budget regime (Q=10000) is a **real-but-MODEST, seed-variable, sub-threshold RL edge** (pooled
  (A-B)=+0.064, CI [+0.026, +0.104] excludes 0 but its lower bound just misses the pre-registered 0.03
  overturn margin, so INDETERMINATE by the prereg rule). Arm A pulls ahead by shifting the distribution
  beyond guidance's frozen-model ceiling.
- **Bias arbitration (is it genuine or a shared reward-oracle bias?):** the shared-2D-Morgan-FEATURE
  mechanism is RULED OUT (leg 2a: the edge survives an independent physchem+MACCS arbiter at the expected
  magnitude); gaming-by-weird-molecules is RULED OUT (property + applicability-domain: drug-like, legit
  hERG-blocker chemotype, in-domain for Morgan-Tanimoto). The shared-LABEL-lineage residual is NOT closable
  (no independent external labels; docking blocked+weak). So the high-budget edge is **most consistent with
  a REAL, modest, sub-threshold advantage**, with one un-testable residual (shared label lineage).
- Net for the program: "route, don't train" holds everywhere tested EXCEPT a modest, sub-threshold, high-
  budget corner where internalized RL pulls ahead - a corner that survived every feasible independence check.
  The budget sweep is why rigor passes matter: the run expected to confirm the tie surfaced this real,
  budget-dependent deviation. Reported and arbitrated as far as the available data allows, not buried.
