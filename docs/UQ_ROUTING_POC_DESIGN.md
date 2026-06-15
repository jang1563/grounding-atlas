# UQ routing PoC: inject specialist self-uncertainty into the per-item router

*Design doc, 2026-06-14. The first-lever experiment from `docs/POSITION_SUPERHUMAN_SCIENTIST.md`.
Tests whether a specialist's own per-input uncertainty closes the measured per-item routing
ceiling. Builds directly on `calibration_discovery/eval/per_item_router.py`, which already
established the gap. No em dashes.*

## The question

The per-item router (`calibration_discovery/`) found that routing on the model's own
continuous confidence to a real per-item specialist reduces to almost-always-call-the-
specialist (accuracy 0.81 vs solo model 0.65) and does NOT reach the per-item oracle
(0.81 vs 0.91), because the model cannot flag the roughly 10% of inputs where it BEATS the
specialist. That residual is specialist-side. So: does feeding the SPECIALIST'S OWN per-input
uncertainty into the router recover part of the 0.81 to 0.91 gap?

## The construct

Data already exist: `calibration_discovery/results/per_item*.csv` give the model's per-item
`pred / conf / route / label / err`; `per_item_router.py` computes a per-item specialist
probability for every rung (cheap out-of-fold CV classifier; AlphaMissense `am` for the
variant rung). The only new ingredient is a per-input specialist uncertainty U(x):

- **cheap CV specialists** (chem Morgan, MSA column-stats, single-cell bag-of-genes, NMR
  binned m/z): ensemble / bootstrap variance. Refit the out-of-fold classifier with K seeds
  or bootstrap resamples and take the per-item standard deviation of the predicted
  probability. This is epistemic uncertainty (MC-dropout's cheap cousin).
- **AlphaMissense** (variant): ambiguity from the score itself, U = 1 - 2 |am - 0.5| (or the
  binary entropy of `am`). AlphaMissense is deterministic, so its self-uncertainty is its
  calibrated distance from the decision boundary.

## Steps

1. **Signal validity.** Per rung, AUROC of U(x) against specialist-incorrect. Does the
   specialist's uncertainty predict the specialist's error? This is the load-bearing check.
2. **Router comparison** (per-item routed accuracy and AURC / risk-coverage):
   - R0 always-specialist (~0.81 baseline)
   - R1 route on model CONF alone (the measured ~0.81 ceiling)
   - R2 route on [model CONF, specialist U] (a small cross-validated meta-rule: take the
     model when it is confident AND the specialist is uncertain, else the specialist)
   - oracle (~0.91)
   - report the closed fraction (R2 - R1) / (oracle - R1).

## Prediction (falsifiable) and pre-registered risk

- Prediction: U predicts specialist error (step 1 AUROC > 0.6), and R2 > R1, closing a
  measurable part of the gap, because the inputs where the model beats the specialist
  overlap the inputs where the specialist is uncertain.
- **Pre-registered risk (the honest pre-mortem):** if the specialist tends to be CONFIDENTLY
  wrong on the model-beats-specialist inputs (low U where it errs), U will not fire and R2
  will not beat R1. That is itself an informative result: it localizes the failure to
  confident specialist errors, which no self-uncertainty can catch, and points the next step
  at a disagreement signal (model-specialist divergence) or a second independent specialist.

## Controls and honest residual

- Random-U shuffle (should give zero lift); cross-validation so the 2-feature meta-rule does
  not overfit the small n; a selectivity check.
- Even if R2 > R1, it will not reach the oracle: the residual is the out-of-distribution
  inputs where U is itself unreliable, the same novel regime that couples to the knowledge
  wall. Report R2 < oracle honestly.

## Why it matters

This is the cheapest decisive test of the position's first lever. It uses existing per-item
data and existing specialists, adds one feature, and either closes part of the routing
ceiling (the orchestrator gets safer and more accurate at no training cost, closed-weight
friendly) or pins the residual to confident specialist errors. Both outcomes are publishable
and on-thesis (route on signal quality, not on the model's say-so).

## First executable step

Extend `per_item_router.py`: have `specialist_proba` also return U (ensemble std for the CV
specialists via K-seed refits; `1 - 2|am - 0.5|` for the variant), run step 1 (AUROC of U vs
specialist error) per rung. No new API calls; the model side is read from `per_item.csv`.

## Implementation entry points

`calibration_discovery/eval/per_item_router.py` (add U + the 2-feature router),
`calibration_discovery/results/per_item*.csv` (model side), `plot_router.py` (R0/R1/R2/oracle
bars). Boltz-2 / AlphaGenome MC-dropout ensembles are the heavier specialist-UQ sources for a
follow-up; this PoC stays on the existing cheap specialists plus AlphaMissense.
