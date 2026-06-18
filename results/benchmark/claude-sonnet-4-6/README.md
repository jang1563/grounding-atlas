# claude-sonnet-4-6 — grounding-atlas-eval, first run

Output arm, 6 ADMET rungs, n=100/rung balanced, prompt v3, data commit `1837b8a`
(2026-06-18). Scored by [`eval/run_grounding_eval.py`](../../../eval/run_grounding_eval.py);
see [`scorecard.json`](scorecard.json), [`raw.jsonl`](raw.jsonl), [`manifest.json`](manifest.json).
`clearance` is excluded (heterogeneous units, orientation unresolved).

| rung | output AUROC (95% CI) | ceiling | gap | ECE | AURC | memo_delta | orient |
|---|---|---|---|---|---|---|---|
| `ames` | 0.321 (0.21–0.43) | 0.896 | **0.575** | 0.48 | 0.69 | −0.20 | align |
| `cyp2d6` | 0.563 (0.48–0.65) | 0.825 | 0.262 | 0.27 | 0.55 | 0.02 | align |
| `cyp3a4` | 0.535 (0.43–0.63) | 0.809 | 0.274 | 0.29 | 0.56 | 0.10 | align |
| `herg` | 0.475 (0.37–0.59) | 0.893 | 0.418 | 0.31 | 0.48 | −0.03 | align |
| `permeability` | 0.599 (0.50–0.71) | 0.893 | 0.294 | 0.26 | 0.39 | 0.13 | oppose |
| `solubility` | 0.703 (0.60–0.79) | 0.786 | 0.083 | 0.41 | 0.32 | 0.18 | oppose |

## What it says

- **The verbalization gap reproduces on ADMET.** Every rung sits below its cheap Morgan
  specialist (all gaps positive): the model verbalizes the property well under what a
  fingerprint + logistic regression decodes from the same SMILES.
- **Ames mutagenicity anti-grounds.** AUROC 0.321 is below chance with a tight CI; the
  verbalized P(mutagenic) anti-correlates with the assay label (verified `label-1 = mutagenic`).
  Widest gap (0.575). This is a genuine result, not an orientation bug — the prior bespoke
  script measured 0.379 on the same data with the same clause.
- **Weak-positive elsewhere.** CYP inhibition (0.54–0.56) and the membrane endpoints
  (solubility 0.70, permeability 0.60) ground above chance; the oppose-orientation endpoints
  read correctly only because the clause + a priori orientation are applied.
- **Poorly calibrated.** ECE 0.26–0.48 and high AURC (ames 0.69) mean self-confidence does not
  track correctness on ADMET output — abstention recovers little.
- **No memorization flags.** `memo_delta` stays in [−0.20, +0.18]; none is the large-positive
  signature of recall over grounding, consistent with ADMET being structure-judged, not
  name-recalled.

## Validation and caveats

- **Validates the instrument.** Orientations match the prior `eval/output_arm_admet.py` run
  exactly, and the qualitative pattern (ames anti-grounds; CYP/membrane weak-positive; all gaps
  positive) reproduces — the general harness agrees with the established measurement while
  adding the standardized gap / calibration / `memo_delta` / bootstrap-CI scorecard.
- **Pilot scale (n=100).** CIs are wide (±~0.1). Absolute AUROCs land ~0.07 below the n=200
  bespoke reference on average, but not uniformly (solubility is higher), so this is sampling
  variance, not a fixed prompt offset.
- **herg, resolved at n=300.** The n=100 0.475 was a low fluctuation: a herg-only n=300 re-run
  gives **0.564 (sonnet)** and **0.513 (gpt-4o)**. So herg is not "closed", but its grounding is
  genuinely weak-to-chance (~0.51–0.56), below the prior 0.633 reference and one of the weaker
  output-arm rungs; the gap to the 0.893 ceiling stays large (~0.33–0.42). The leaderboard is
  kept uniform at n=100; this is a diagnostic note (bootstrap CIs are why the single n=100
  number should not be over-read).
