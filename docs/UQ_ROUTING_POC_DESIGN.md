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

- **Primary: proba ambiguity** U = 1 - 2 |p - 0.5| (binary entropy of the specialist's
  predicted probability). Works for every specialist including AlphaMissense `am`, and is
  comparable across rungs because it is in [0, 1], which lets us pool for power.
- **Secondary (heavy, follow-up): ensemble / bootstrap variance.** Bootstrap-resample the
  training data and take the per-item std of the predicted probability. Note: for the cheap
  deterministic specialists (logistic regression on fixed features) seed variance is near
  zero, so this is only meaningful for stochastic or deep specialists (Boltz-2, AlphaGenome
  MC-dropout); on the cheap specialists, ambiguity is the load-bearing U.

## Steps

0. **Alignment check (do first).** The specialist proba is aligned to `per_item.csv` rows by
   load order; verify it by reproducing the reported per-item specialist accuracy (~0.81 for
   the variant) before trusting anything downstream. A misalignment silently corrupts every
   metric below.

1. **The decisive, non-tautological check.** Do NOT use AUROC(U, specialist-error): a
   calibrated specialist's errors concentrate at its decision boundary (proba ~0.5), so
   ambiguity predicts error almost by construction, a false green light. Instead test whether
   U flags the RECOVERABLE inputs:
   - recoverable set R = {model correct AND specialist wrong} (this is the 0.91 - 0.81 oracle
     gap, the only items routing can recover).
   - is U elevated on R versus the specialist-correct items (AUROC / Mann-Whitney, pooled
     across rungs for power, per-rung secondary)?
   - and the conditional that actually drives routing: P(model correct | U high) versus
     P(specialist correct | U high). U helps only if the model is genuinely better where the
     specialist is uncertain.
2. **Router comparison** (per-item routed accuracy and AURC / risk-coverage):
   - R0 always-specialist (~0.81 baseline)
   - R1 route on model CONF alone (the measured ~0.81 ceiling)
   - R1b route on specialist confidence alone (1 - U), the naive baseline U must beat
   - R2 route on [model CONF, specialist U] (a small cross-validated meta-rule)
   - oracle (~0.91)
   - report the closed fraction (R2 - max(R1, R1b)) / (oracle - max(R1, R1b)).

## Prediction (falsifiable) and pre-registered risk

- Prediction: U is elevated on the recoverable set and the model is better where U is high
  (step 1), so R2 beats max(R1, R1b) and closes a measurable part of the gap. Measured on the
  recoverable-set conditional, NOT on the near-tautological AUROC(U, error).
- **Pre-registered risk, and the likely default:** if the specialist is CONFIDENTLY wrong on
  the recoverable inputs (low U where the model beats it), U will not fire and R2 will not
  beat R1b. This is the EXPECTED outcome for the variant rung specifically: the inputs where
  Claude beats AlphaMissense are web-rich variants it recalls from ClinVar while AlphaMissense
  scores them confidently (off-boundary) from sequence, so U is low exactly where it would
  need to be high. The informative question is therefore WHICH rungs U helps in (the cheap
  specialists whose errors are boundary-concentrated may behave better than the variant), and
  the failure on variant localizes the unrecoverable residual to confident specialist errors,
  pointing the next step at a disagreement signal (model-specialist divergence) or a second
  independent specialist rather than self-uncertainty.

## Controls and honest residual

- Random-U shuffle (should give zero lift); cross-validation so the 2-feature meta-rule does
  not overfit the small n; a selectivity check.
- Even if R2 > R1, it will not reach the oracle: the residual is the out-of-distribution
  inputs where U is itself unreliable, the same novel regime that couples to the knowledge
  wall. Report R2 < oracle honestly.

## Result (2026-06-14): the cheap version is RED

Ran Step 0 to 2 (`calibration_discovery/eval/uq_competence.py`, opus, 640 pooled items).
Step 0 reproduces the gap exactly (always-spec 0.811, always-model 0.689, oracle 0.909;
recoverable 9.8%), confirming alignment. The decisive checks are negative:

- Step 1a passes superficially: U flags the recoverable items (AUROC 0.75 separating
  recoverable from specialist-correct), and even the variant rung shows AlphaMissense
  uncertain (not confident) on its recoverable errors, refuting that part of the pre-mortem.
- Step 1b fails: in the U-high tercile the specialist is still better than the model (0.640
  vs 0.579), so routing the model in on high U loses.
- Step 2 confirms: cross-validated, no router beats always-call-the-specialist. R1 conf
  0.747, R1b U 0.686, R2 conf+U 0.702, all below always-spec 0.811 (the earlier 0.81 CONF
  number was in-sample best-threshold optimism; honest CV is 0.747).

So specialist self-uncertainty does not close the per-item gap. The recoverable 10% is real
but not extractable from {model confidence, specialist self-uncertainty}: the
specialist-uncertain inputs stay specialist-favorable, and the model's confidence does not
flag its own wins. This extends the calibration_discovery result: not only can the model not
flag where it beats the specialist, neither can the specialist's self-uncertainty. The
missing signal is one that identifies model-superiority specifically (the model's unique
recall beating the specialist), not uncertainty. Likely next signals: model-specialist
disagreement magnitude, or a per-item recall / web-exposure flag.

Follow-up (`uq_signals.py`): those next signals also fail, and the reason becomes clear. The
specialist is better than or tied with the model on every rung (the model "wins" only methyl,
where both are at chance ~0.50, so it is noise). Per-context (per-rung) reliability routing
therefore does nothing: in-sample upper bound 0.812, CV 0.792, against always-spec 0.811. And
CV feature routers on disagreement, web-exposure flag, and rung identity all land at 0.70 to
0.75, below always-spec. So the override is closed: across five signal families (model
confidence, specialist self-uncertainty, disagreement, web-exposure, per-context reliability)
nothing beats always-call-the-specialist. Much of the oracle gap is the model being right by
chance on hard items the near-chance specialist also misses (methyl carries the largest
recoverable fraction with both arms at chance), which carries no signal. The orchestrator's
optimal policy reduces to call the right specialist; Claude's value is domain routing, faithful
grounding of the specialist output, and calibration for safe deferral, not per-item override.

## Why it matters

This is the cheapest decisive test of the position's first lever. It uses existing per-item
data and existing specialists, adds one feature, and either closes part of the routing
ceiling (the orchestrator gets safer and more accurate at no training cost, closed-weight
friendly) or pins the residual to confident specialist errors. Both outcomes are publishable
and on-thesis (route on signal quality, not on the model's say-so).

## First executable step

Extend `per_item_router.py`: have `specialist_proba` also return U = 1 - 2 |p - 0.5|. Run
step 0 (reproduce the ~0.81 specialist accuracy to confirm alignment), then step 1 on the
recoverable set (is U elevated on {model correct AND specialist wrong}, pooled across rungs,
plus P(model correct | U high))? No new API calls; the model side is read from
`per_item.csv`. Treat the variant rung as the expected red case and report the per-rung
pattern.

## Implementation entry points

`calibration_discovery/eval/per_item_router.py` (add U + the 2-feature router),
`calibration_discovery/results/per_item*.csv` (model side), `plot_router.py` (R0/R1/R2/oracle
bars). Boltz-2 / AlphaGenome MC-dropout ensembles are the heavier specialist-UQ sources for a
follow-up; this PoC stays on the existing cheap specialists plus AlphaMissense.
