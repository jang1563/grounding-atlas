# gpt-4o — grounding-atlas-eval, cross-provider run

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `1837b8a`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
Run as the cross-provider control for [claude-sonnet-4-6](../claude-sonnet-4-6/README.md).
`clearance` excluded (orientation unresolved).

| rung | output AUROC (95% CI) | ceiling | gap | ECE | AURC | memo_delta | orient |
|---|---|---|---|---|---|---|---|
| `ames` | 0.314 (0.22–0.41) | 0.896 | **0.582** | 0.25 | 0.66 | −0.09 | align |
| `cyp2d6` | 0.520 (0.43–0.61) | 0.825 | 0.305 | 0.20 | 0.57 | 0.05 | align |
| `cyp3a4` | 0.570 (0.49–0.65) | 0.809 | 0.239 | 0.21 | 0.51 | 0.13 | align |
| `herg` | 0.520 (0.45–0.60) | 0.893 | 0.373 | 0.26 | 0.71 | 0.08 | align |
| `permeability` | 0.567 (0.50–0.65) | 0.893 | 0.326 | 0.16 | 0.25 | 0.09 | oppose |
| `solubility` | 0.668 (0.56–0.76) | 0.786 | 0.118 | 0.13 | 0.28 | 0.13 | oppose |

## The cross-provider result

GPT-4o and Claude Sonnet 4.6 agree within ~0.04 AUROC on **every** rung, with identical
rank order (solubility best, ames worst/anti-grounded):

| rung | gpt-4o | sonnet | \|Δ\| |
|---|---|---|---|
| ames | 0.314 | 0.321 | 0.007 |
| cyp2d6 | 0.520 | 0.563 | 0.043 |
| cyp3a4 | 0.570 | 0.535 | 0.035 |
| herg | 0.520 | 0.475 | 0.045 |
| permeability | 0.567 | 0.599 | 0.032 |
| solubility | 0.668 | 0.703 | 0.035 |

Mean \|Δ\| ≈ 0.033. The verbalization gap on ADMET is therefore **provider-invariant**, not a
Claude quirk:

- **ames anti-grounds in both** (0.314 / 0.321, both below chance). Two models from different
  labs ranking mutagenicity backwards by the same amount is strong evidence this is a shared
  frontier-LLM failure on a subtle, poorly-web-documented structure→property mapping.
- **Both ground solubility best** and sit weak-positive on CYP/permeability; every rung is
  below its Morgan ceiling (all gaps positive) for both models.
- **herg is weak-to-chance for both** (n=100: 0.520 / 0.475). The n=300 diagnostic resolves it:
  **0.513 (gpt-4o)** and **0.564 (sonnet)** — the n=100 0.475 dip was a low fluctuation, but
  herg grounding is genuinely weak (below the prior 0.633), one of the weaker output-arm rungs.

## Caveats

Pilot scale (n=100, wide CIs). GPT-4o is somewhat better calibrated than Sonnet on these
endpoints (lower ECE on 5 of 6 rungs), but the grounding ranking is the same. The point of the
two-model table is the agreement, not the small per-rung differences (all within CI).
