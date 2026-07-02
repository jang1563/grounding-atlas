# Experiment-3 v1 result (hERG): route-vs-train in the GENERATIVE regime

Per [docs/RL_ENV_PREREG.md](../../../docs/RL_ENV_PREREG.md). v1 scope: hERG, one matched budget
Q=5000, REINVENT arm A, one seed. The build/de-risking gates (reward, feasibility, generator,
oracle, reward-oracle agreement) are in [build_gates.md](build_gates.md). GPU on Cayuga; light
analysis local. Date 2026-06-28.

## The 3-arm head-to-head (+ controls), all judged on the held-out block-O oracle

| arm | what | oracle-pass / 500 | rate |
|---|---|---|---|
| **A internalized RL** | REINVENT (sigma 20) tunes the generator to the block-R reward, base anchor | **21** | 0.042 |
| **B external guidance** | Best-of-N: top-500-by-reward of the FROZEN generator at Q=5000 | **20** | 0.040 |
| A shuffle (drift guard) | the same RL on PERMUTED rewards | 0 | 0.000 |
| C base (no steering) | frozen generator, unselected | -- | 0.0046 |

The reward budget Q = 5000 reward-queries for BOTH A (100 steps x 50 batch) and B (draw 5000, keep
top 500). The oracle is the block-O Morgan-RF (held-out block-E AUROC 0.882), bar = 90th pct of
block-E (0.627), scaffold-disjoint from the reward.

## H1: route-don't-train EXTENDS to generation

**(A - B) = +0.0020; scaffold-clustered two-sample bootstrap 95% CI [-0.047, +0.046]** (479 vs 476
Murcko-scaffold clusters, n_boot 4000; `eval/compare_rl_orchestrate.py`). The CI includes 0 and the
point estimate sits far inside the 0.03 tie band, so internalized RL does NOT beat external guidance
at matched budget. CONFIRM.

**Drift guards (all pass):**
- The reward DRIVES the gain: arm A on the real reward gets 21 oracle-passes; on SHUFFLED rewards it
  gets 0 (meanReward stays at base ~0.09). The 21 is reward-driven, not optimization noise.
- Genuinely generative + RL: the headline is generative-design oracle-success, NOT a discriminator
  AUROC; arm A's loss contains the reward (REINVENT augmented likelihood) and survives the shuffle test.
- Stable, on-manifold: REINVENT meanReward rose 0.06 -> 0.618, KL-to-base bounded at 4.8 (the initial
  PPO-clipped optimizer diverged to KL 10 / reward 0 and was replaced; an optimization failure, not a
  verdict).

**Reading:** building a post-training RL environment for a generative bio FM is feasible and the
reward produces real (oracle-confirmed) gains, but INTERNALIZING the reward into the weights buys
nothing over EXTERNALLY selecting top-reward samples from the frozen model at the same budget. The
"route, don't train" verdict, established on the discriminative read-out
([REPORT.md](../../../docs/REPORT.md)), EXTENDS to the generative/RL lever.

## v1 caveats / v2
- One cell (hERG), one budget (Q=5000), one seed. The CI is wide (small pass counts, ~20/500). v2
  (below) adds seeds + a degraded-reward cell.

## v2: seed robustness + degraded-reward cell (done)

The tie is robust. arm A across three seeds: 0.038 / 0.024 / 0.036 (KL 4.6 / 3.8 / 5.7), pooled
49/1500; **(A-B) = -0.007, scaffold-clustered 95% CI [-0.054, +0.031] -> CONFIRM.** A degraded
LOW-DATA reward (block-R subsampled to 150 positives): arm A 24/500 (0.048) vs guidance 14/500
(0.028); **(A-B) = +0.020, CI [-0.020, +0.064] -> CONFIRM.** When the reward weakens, external
guidance degrades MORE than RL (guidance 20 -> 14 passes, RL holds), so the point estimate tips
toward RL in the low-data regime (the literature's predicted crossover) but stays inside the 0.03
tie band. No cell separates train from route. (`eval/compare_rl_orchestrate.py`, result
`signal/reward/herg_H1_compare.json`.)

## Clearance: the genuinely weak endpoint (run on SDSC Expanse H100)

The reward-quality axis, completed on a real weak endpoint (clearance-specific generator excluding
clearance block-O/E; Cayuga scu-gpu was contended so this ran on Expanse `nairr-gpu-shared`).
Clearance's reward barely enriches oracle-actives on generated molecules (top-5% enrichment **1.1x**
vs hERG's 14.8x), so guidance has little to select on. Result (matched budget Q=5000, M=500;
`signal/reward/clearance_H1_compare.json`):
- **arm A RL 57/500 (0.114) vs arm B guidance 44/500 (0.088); (A-B) = +0.026, 95% CI [-0.015, +0.067]
  -> CONFIRM.** Drift guard: shuffle drops to 0.044 (< arm A, so the reward drives the 0.114).
- RL extracts a bit more from the weak reward than top-selection does, so the A-B point estimate tips
  toward RL and grows as the reward weakens across the three cells (-0.007 strong -> +0.020 low-data
  -> +0.026 weak-endpoint) - the literature's predicted low-data crossover, directionally consistent -
  but every CI includes 0. On clearance the binding constraint is reward reliability, not the lever.

## Remaining caveats
- The CIs are still wide (pass counts ~20-57 / 500); a larger M or more budgets would tighten further.
- The clearance cell used its own generator + reward + oracle (all endpoint-parameterized), confirming
  route-don't-train on a genuinely weak endpoint; the low-data cell is the within-hERG degraded stand-in.
- OVERTURN was the only outcome needing the docking co-primary; since it CONFIRMED, docking is moot.
- sigma=20 (REINVENT reward scale) was not swept; the matched-budget contrast vs guidance is the
  controlled comparison regardless.
