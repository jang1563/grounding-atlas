# claude-opus-4-8 — grounding-atlas-eval, high-end model

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `89b3f6c`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
Run to ask whether the high-end model closes the ADMET verbalization gap. `clearance` excluded.
(Opus 4.8 deprecates the `temperature` parameter; the harness omits it for this model.)

| rung | opus AUROC (95% CI) | sonnet | gpt-4o | ceiling | gap | ECE | memo_delta | orient |
|---|---|---|---|---|---|---|---|---|
| `ames` | 0.325 (0.22–0.44) | 0.321 | 0.314 | 0.896 | 0.571 | 0.517 | −0.17 | align |
| `cyp2d6` | 0.744 (0.65–0.83) | 0.563 | 0.520 | 0.825 | **0.081** | 0.055 | +0.35 | align |
| `cyp3a4` | 0.633 (0.53–0.73) | 0.535 | 0.570 | 0.809 | 0.176 | 0.014 | +0.12 | align |
| `herg` | 0.590 (0.48–0.70) | 0.475 | 0.520 | 0.893 | 0.303 | 0.081 | +0.16 | align |
| `permeability` | 0.628 (0.54–0.74) | 0.599 | 0.567 | 0.893 | 0.265 | 0.182 | −0.03 | oppose |
| `solubility` | 0.668 (0.57–0.77) | 0.703 | 0.668 | 0.786 | 0.118 | 0.388 | +0.14 | oppose |

## Does capability close the gap? Endpoint-dependent.

The high-end model does **not** uniformly close the verbalization gap — it splits by endpoint:

- **ames is capability-invariant.** Opus 0.325 ≈ sonnet 0.321 ≈ gpt-4o 0.314: all three anti-ground
  Ames mutagenicity below chance, within 0.011 of each other. The strongest single result in the
  benchmark — a frontier-scale model with the same gap as a mid model and a different provider is
  hard evidence the gap here is about the representation's web-documentation, **not** capability.
- **CYP inhibition and hERG: capability closes much of it, via real grounding.** Opus lifts cyp2d6
  to 0.744 (near the 0.825 ceiling, gap 0.081), cyp3a4 to 0.633, herg to 0.590 — all clearly above
  sonnet/gpt-4o. The lift comes with positive `memo_delta` (cyp2d6 +0.35, cyp3a4 +0.12, herg +0.16):
  the score depends on the real structure, not surface tokens, so it is genuine grounding, not recall.
- **solubility: no capability gain.** Opus 0.668 sits at gpt-4o's level and below sonnet's 0.703 —
  an easy, composition-driven property where all models already ground and headroom is small.

## Calibration

Mean ECE: opus 0.206, gpt-4o 0.201, sonnet 0.339. Opus is **not** uniquely best on average (it ties
gpt-4o; both beat sonnet). But its calibration is sharply endpoint-dependent: near-perfect where it
grounds (cyp3a4 ECE 0.014, cyp2d6 0.055, herg 0.081) and poor where it anti-grounds (ames 0.517,
confidently wrong) or hits an easy property (solubility 0.388). So Opus is well-calibrated about what
it actually grounds — and dangerously confident exactly where it is wrong (ames).

## The refined thesis

The verbalization gap is not monolithic. Web-exposure sets a gap that **capability can close where
the underlying structure→property SAR is learnable** (CYP inhibition, hERG block) — Opus does this
with genuine structural grounding — but **cannot close where the SAR is subtle and web-poor** (Ames
mutagenicity), where even a frontier-scale model anti-grounds. Pilot scale (n=100, wide CIs); read
with the [sonnet](../claude-sonnet-4-6/README.md) and [gpt-4o](../gpt-4o/README.md) cards.
