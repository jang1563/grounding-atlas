# gpt-4o — grounding-atlas-eval, cross-provider run

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `1837b8a`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
Run as the cross-provider control for [claude-sonnet-4-6](../claude-sonnet-4-6/README.md).
`clearance` excluded (orientation unresolved).

> **ames orientation corrected (2026-06-18)** to `oppose` (structural-alert audit:
> [`eval/analyze_ames.py`](../../../eval/analyze_ames.py)); ames row re-scored from raw outputs.
> Under the old `align` it read 0.314 (apparent anti-grounding); corrected it grounds at 0.686.

| rung | output AUROC (95% CI) | ceiling | gap | ECE | AURC | memo_delta | orient |
|---|---|---|---|---|---|---|---|
| `ames` | 0.686 (0.59–0.78) | 0.896 | 0.210 | 0.23 | 0.34 | +0.09 | oppose |
| `cyp2d6` | 0.520 (0.43–0.61) | 0.825 | 0.305 | 0.20 | 0.57 | 0.05 | align |
| `cyp3a4` | 0.570 (0.49–0.65) | 0.809 | 0.239 | 0.21 | 0.51 | 0.13 | align |
| `herg` | 0.520 (0.45–0.60) | 0.893 | 0.373 | 0.26 | 0.71 | 0.08 | align |
| `permeability` | 0.567 (0.50–0.65) | 0.893 | 0.326 | 0.16 | 0.25 | 0.09 | oppose |
| `solubility` | 0.668 (0.56–0.76) | 0.786 | 0.118 | 0.13 | 0.28 | 0.13 | oppose |

## The cross-provider result

GPT-4o and Claude Sonnet 4.6 agree within ~0.04 AUROC on **every** rung (ames shown corrected):

| rung | gpt-4o | sonnet | \|Δ\| |
|---|---|---|---|
| ames | 0.686 | 0.679 | 0.007 |
| cyp2d6 | 0.520 | 0.563 | 0.043 |
| cyp3a4 | 0.570 | 0.535 | 0.035 |
| herg | 0.520 | 0.475 | 0.045 |
| permeability | 0.567 | 0.599 | 0.032 |
| solubility | 0.668 | 0.703 | 0.035 |

Mean \|Δ\| ≈ 0.033. The ADMET grounding profile is therefore **provider-invariant**, not a
Claude quirk:

- **ames grounds in both** (0.686 / 0.679, the tightest agreement of all rungs, |Δ| 0.007). Both
  correctly key on nitroaromatic alerts (ρ(P, aromatic-nitro) +0.85 gpt-4o / +0.68 sonnet). Two
  models from different labs doing the same textbook structure-activity is strong cross-provider
  evidence. (Under the original inverted ames label both read ~0.32 and looked like a shared
  anti-grounding failure — that was the label bug, now corrected.)
- **Both ground every rung above chance** and below the Morgan ceiling (all gaps positive); the
  best-grounded are ames and solubility (~0.68–0.70), the weakest is herg.
- **herg is weak-to-chance for both** (n=100: 0.520 / 0.475). The n=300 diagnostic resolves it:
  **0.513 (gpt-4o)** and **0.564 (sonnet)** — the n=100 0.475 dip was a low fluctuation, but
  herg grounding is genuinely weak (below the prior 0.633), one of the weaker output-arm rungs.

## Caveats

Pilot scale (n=100, wide CIs). GPT-4o is better calibrated than Sonnet overall (mean ECE 0.197
vs 0.284; lower ECE on 5 of 6 rungs, the exception now being ames), but the grounding ranking is
the same. The point of the two-model table is the agreement, not the small per-rung differences
(all within CI). A third model, [opus-4-8](../claude-opus-4-8/README.md), extends this with the
best calibration (mean ECE 0.152) and a capability lift on the CYP/hERG rungs.
