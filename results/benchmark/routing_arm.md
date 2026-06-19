# Calibration / routing arm

*Phase 3 of grounding-atlas-eval. Given the verbalization gap, what do you DO per compound?
Route it to the model when the model is competent, to a cheap specialist when it is not. Can
the model tell you which is which? Code: [`eval/routing_arm.py`](../../eval/routing_arm.py),
[`eval/elicit_confidence.py`](../../eval/elicit_confidence.py); data: [`routing.json`](routing.json).*

## Setup

Pooled over the 6 corrected-ADMET rungs, n ≈ 600 per model. The specialist is a per-item
out-of-fold Morgan + logistic-regression model on the oriented label. Two confidence signals
are compared as the routing key:

- **implicit** — `|P − 0.5|`, the decisiveness of the model's own probability (already in
  `raw.jsonl`, no extra calls).
- **explicit** — a fresh per-item query asking the model how reliable its own prediction is,
  0 to 1 (`elicit_confidence.py`).

| model | always-model | always-specialist | oracle | implicit routed (AURC) | explicit routed (AURC) | stated-conf ↔ correct |
|---|---|---|---|---|---|---|
| claude-opus-4-8 | 0.605 | 0.753 | 0.910 | 0.765 (0.380) | 0.755 (0.403) | **−0.068** |
| claude-sonnet-4-6 | 0.545 | 0.753 | 0.925 | 0.763 (0.417) | 0.763 (0.445) | **+0.032** |
| gpt-4o | 0.523 | 0.753 | 0.955 | 0.777 (0.405) | 0.767 (0.453) | **+0.050** |

`routed` = best accuracy of confidence-thresholded routing (keep the model on its most-confident
items, send the rest to the specialist); AURC = area under the model-alone risk-coverage curve
(lower is better).

## What it says

1. **The specialist dominates.** 0.753 vs the model's 0.52–0.61: the strong baseline is "just
   call the specialist." Provider-invariant.
2. **But the oracle is far above (0.91–0.96).** The model is uniquely right on 16–20% of items,
   so real complementary signal exists — *if* you could find those items.
3. **Implicit confidence recovers only a sliver.** Routing on `|P−0.5|` reaches 0.765–0.777,
   i.e. +0.01–0.02 over always-specialist, by keeping the 4–14% the model is surest of — about
   1–2 of the ~16–20 oracle points.
4. **Explicit self-confidence recovers nothing.** The correlation between stated confidence and
   actually being right is ≈ 0 for all three (−0.07 / +0.03 / +0.05), and explicit routing is
   equal-or-worse than implicit on every model. Asking "how sure are you?" adds no per-item
   signal — and this is provider-invariant.
5. **Reconciliation with the per-rung "calibrated router."** The frontier model's confidence
   tracks its grounding at the *task* level (across rungs, opus corr +0.90 — it knows which
   tasks it can do), but not at the *item* level (which molecules it will get right). Two
   distinct layers; the item-level competence lever is closed.

## Prescription

Route by the **a-priori-known web-exposure / rung prior** and call the specialist; do not rely
on the model's per-item self-assessment. The deferral decision should be made *before* the model
answers (from the representation's web-documentation), not by asking the model afterward.

## Caveats

Pilot scale (n = 100/rung pooled), a cheap Morgan specialist (not a tuned one), ADMET only, and
a single confidence-prompt phrasing. The null on explicit confidence is for *this* elicitation;
a different prompt could differ, but the implicit signal sets a low bar that explicit did not
clear on any of the three models.
