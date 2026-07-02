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

## Status

- Route-don't-train HOLDS at moderate budget (Q <= 5000; confirmed across 3 reward-quality cells x 3
  seeds elsewhere).
- The HIGH-budget regime (Q=10000) is a **flagged potential overturn**: internalized RL significantly
  beats guidance on the in-silico oracle, diversely, but pending (a) the docking co-primary to rule out
  shared reward-oracle-bias reward-hacking, (b) seed replication, (c) resolving the KL~8.9 drift.
- The budget sweep is why rigor passes matter: a run expected to confirm the tie instead surfaced a
  budget-dependent deviation. Reported, not buried.
