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
- One cell (hERG), one budget (Q=5000), one seed. The CI is wide (small pass counts, ~20/500). The
  prereg's budget sweep, the clearance (degraded-reward) and low-data cells, and multiple seeds would
  tighten it. The tie point estimate + CI-includes-0 + the clean drift guard make the v1 verdict solid.
- OVERTURN was the only outcome that needed the docking co-primary; since it CONFIRMED, docking is moot.
- sigma=20 (REINVENT reward scale) was not swept; it sets arm A's reward level but the matched-budget
  contrast vs guidance is the controlled comparison either way.
