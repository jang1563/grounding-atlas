# claude-sonnet-4-6 — grounding-atlas-eval, first run

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `1837b8a`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
`clearance` is excluded (heterogeneous units, orientation unresolved).

> **ames orientation corrected (2026-06-18)** to `oppose` after a structural-alert audit
> ([`eval/analyze_ames.py`](../../../eval/analyze_ames.py)) found label-0 is the
> nitroaromatic-rich (mutagenic) class; the row was re-scored from raw outputs
> ([`eval/fix_ames_orientation.py`](../../../eval/fix_ames_orientation.py)). Under the old
> `align` it read 0.321 (apparent anti-grounding); corrected it grounds at 0.679.

| rung | output AUROC (95% CI) | ceiling | gap | ECE | AURC | memo_delta | orient |
|---|---|---|---|---|---|---|---|
| `ames` | 0.679 (0.58–0.79) | 0.896 | 0.217 | 0.16 | 0.31 | +0.20 | oppose |
| `cyp2d6` | 0.563 (0.48–0.65) | 0.825 | 0.262 | 0.27 | 0.55 | 0.02 | align |
| `cyp3a4` | 0.535 (0.43–0.63) | 0.809 | 0.274 | 0.29 | 0.56 | 0.10 | align |
| `herg` | 0.475 (0.37–0.59) | 0.893 | 0.418 | 0.31 | 0.48 | −0.03 | align |
| `permeability` | 0.599 (0.50–0.71) | 0.893 | 0.294 | 0.26 | 0.39 | 0.13 | oppose |
| `solubility` | 0.703 (0.60–0.79) | 0.786 | 0.083 | 0.41 | 0.32 | 0.18 | oppose |

## What it says

- **The verbalization gap reproduces on ADMET.** Every rung sits below its cheap Morgan
  specialist (all gaps positive): the model verbalizes the property well under what a
  fingerprint + logistic regression decodes from the same SMILES.
- **ames grounds via nitroaromatic alerts (corrected).** AUROC 0.679; the model's P(mutagenic)
  correlates +0.68 with aromatic-nitro presence ([`eval/analyze_ames.py`](../../../eval/analyze_ames.py))
  — textbook structure-activity. Under the original inverted `align` label it read 0.321 and
  looked like anti-grounding; that was a label-direction bug, now fixed.
- **Weak-to-moderate elsewhere.** CYP inhibition (0.54–0.56), permeability 0.60, solubility 0.70
  ground above chance; herg sits at chance (0.475, see below). The oppose-orientation endpoints
  read correctly only because the clause + a priori orientation are applied.
- **Calibration is mixed.** mean ECE 0.284; worst on solubility (0.41). Sonnet is the least
  calibrated of the three models (opus 0.152, gpt-4o 0.197).
- **No memorization flags.** `memo_delta` in [−0.03, +0.20], small-to-moderate positive — the
  scores depend on the real structure (they drop on a character-scrambled SMILES), with no
  surface-recall signature.

## Validation and caveats

- **Cross-checked against the prior bespoke `eval/output_arm_admet.py`.** The qualitative pattern
  matches — but both shared an *inverted ames orientation*, corrected here via the structural-alert
  audit (so "agreeing with the prior run" reproduced its ames bug; the audit, not the agreement,
  is what fixed the direction). The standardized gap / calibration / `memo_delta` / bootstrap-CI
  scorecard is what the general harness adds.
- **Pilot scale (n=100).** CIs are wide (±~0.1).
- **herg, resolved at n=300.** The n=100 0.475 was a low fluctuation: a herg-only n=300 re-run
  gives **0.564 (sonnet)** and **0.513 (gpt-4o)**. So herg is not "closed", but its grounding is
  genuinely weak-to-chance (~0.51–0.56), below the prior 0.633 reference and one of the weaker
  output-arm rungs; the gap to the 0.893 ceiling stays large (~0.33–0.42). The leaderboard is
  kept uniform at n=100; this is a diagnostic note (bootstrap CIs are why the single n=100
  number should not be over-read).
