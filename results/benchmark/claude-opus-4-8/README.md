# claude-opus-4-8 — grounding-atlas-eval, high-end model

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `89b3f6c`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
Run to ask whether the high-end model closes the ADMET verbalization gap. `clearance` excluded.
(Opus 4.8 deprecates the `temperature` parameter; the harness omits it for this model.)

> **ames orientation corrected (2026-06-18).** A structural-alert audit
> ([`eval/analyze_ames.py`](../../../eval/analyze_ames.py)) found the ames label direction was
> inverted: label-0 is the nitroaromatic-rich (mutagenic) class. ames is therefore oriented
> `oppose`, and the ames row below was re-scored from the committed raw outputs
> ([`eval/fix_ames_orientation.py`](../../../eval/fix_ames_orientation.py)). Under the old
> `align`, ames read as anti-grounding (~0.32); corrected, all models ground it (~0.68).

| rung | opus AUROC (95% CI) | sonnet | gpt-4o | ceiling | gap | ECE | memo_delta | orient |
|---|---|---|---|---|---|---|---|---|
| `ames` | 0.675 (0.56–0.77) | 0.679 | 0.686 | 0.896 | 0.221 | 0.194 | +0.17 | oppose |
| `cyp2d6` | 0.744 (0.65–0.83) | 0.563 | 0.520 | 0.825 | **0.081** | 0.055 | +0.35 | align |
| `cyp3a4` | 0.633 (0.53–0.73) | 0.535 | 0.570 | 0.809 | 0.176 | 0.014 | +0.12 | align |
| `herg` | 0.590 (0.48–0.70) | 0.475 | 0.520 | 0.893 | 0.303 | 0.081 | +0.16 | align |
| `permeability` | 0.628 (0.54–0.74) | 0.599 | 0.567 | 0.893 | 0.265 | 0.182 | −0.03 | oppose |
| `solubility` | 0.668 (0.57–0.77) | 0.703 | 0.668 | 0.786 | 0.118 | 0.388 | +0.14 | oppose |

## Does capability close the gap? Endpoint-dependent.

Every rung grounds above chance (0.475–0.744) and every rung sits below its cheap Morgan
specialist (gaps 0.08–0.30): the verbalization gap holds, but **nothing anti-grounds**. What
capability buys splits by endpoint:

- **CYP inhibition / hERG / permeability: capability narrows the gap, via real grounding.**
  Opus lifts cyp2d6 to 0.744 (near the 0.825 ceiling, gap 0.081), cyp3a4 to 0.633, herg to
  0.590, permeability to 0.628 — all above sonnet/gpt-4o. The lift comes with positive
  `memo_delta` (cyp2d6 +0.35, cyp3a4 +0.12, herg +0.16): the score depends on the real
  structure (it collapses on a character-scrambled SMILES), so it is genuine grounding.
- **ames: already grounded by all three, no capability lift.** opus 0.675 ≈ sonnet 0.679 ≈
  gpt-4o 0.686 — all key correctly on nitroaromatic alerts (Spearman ρ(P, aromatic-nitro)
  +0.71 / +0.68 / +0.85). A well-web-documented SAR that mid models already read; capability
  adds nothing because there is little headroom in *reading it out*, only in the gap to ceiling.
- **solubility: no capability gain** (easy composition-driven property, opus ≈ gpt-4o < sonnet).

## Calibration

Mean ECE: **opus 0.152, gpt-4o 0.197, sonnet 0.284** — Opus is the best-calibrated of the three.
It is near-perfect where it grounds strongly (cyp3a4 ECE 0.014, cyp2d6 0.055, herg 0.081) and
worst on the easy solubility rung (0.388). This is consistent with the project's prior finding
that the frontier model's self-confidence tracks its actual grounding — the property that makes
it a usable router.

## The thesis

The verbalization gap is real but not monolithic, and (corrected) nothing anti-grounds: all
models read ADMET endpoints below the cheap specialist, and **capability closes part of that gap
where the structure→property SAR is learnable** (CYP, hERG, permeability) with genuine structural
grounding, while adding little on an already-grounded well-documented SAR (ames) or an easy
property (solubility). The original "ames anti-grounding" was a label-direction artifact, now
corrected — a reminder that a benchmark's largest risk is label provenance, not the model. Pilot
scale (n=100, wide CIs); read with the [sonnet](../claude-sonnet-4-6/README.md) and
[gpt-4o](../gpt-4o/README.md) cards.
