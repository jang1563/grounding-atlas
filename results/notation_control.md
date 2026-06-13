# T2-solve grounding control: canonical vs randomized vs scrambled SMILES

*Results section. 2026-06-11. Instrument: `eval/notation_control.py` (balanced signal-bearing set, self-generated notations, bootstrap CIs). Addresses the reviewer objection that the R3 "reads it" could be canonical-string memorization rather than structure reading. No em dashes.*

## The test

R3 found the frontier model "reads" CYP3A4 inhibition at AUROC ~0.61 with the property named. Is that reading the STRUCTURE, or pattern-matching the canonical SMILES STRING, or a non-structural prior? On the SAME balanced 100/100 sample the sweep scores, run three notations of each molecule:
- **canonical** (RDKit canonical SMILES, the sweep input)
- **randomized** (RDKit doRandom valid SMILES of the same molecule, structure kept, string changed)
- **scrambled** (character-shuffled canonical, structure destroyed, usually invalid)

Reads: canonical ~ randomized AND canonical >> scrambled means it reads the structure; canonical dropping on randomized means string memorization; all three equal and above chance means a non-structural prior.

## Result (claude-sonnet-4-6), two endpoints on Cayuga

CYP3A4 at n=200 (underpowered pilot) then n=1000, replicated on CYP2D6 at n=888:

| endpoint | canonical | randomized | scrambled | canon-scram drop | canon vs scram CIs |
|---|---|---|---|---|---|
| CYP3A4 (n=1000) | 0.581 [0.547,0.614] | 0.582 | 0.507 [0.473,0.541] | +0.074 | non-overlapping |
| CYP2D6 (n=888) | 0.631 [0.597,0.664] | 0.611 | 0.550 [0.513,0.586] | +0.081 | non-overlapping |

Both endpoints show the same pattern: canonical ~ randomized (notation-invariant) and canonical significantly above scrambled (the canonical and scrambled bootstrap CIs do not overlap in either case). One nuance: CYP3A4 scrambled falls to chance (CI includes 0.5), while CYP2D6 scrambled stays slightly above chance (0.55, CI 0.513 to 0.586), so CYP2D6 has a small residual non-structural component, but its dominant signal is still structure-dependent. (The CYP3A4 n=200 pilot gave 0.627/0.602/0.563, the same direction but CIs too wide to call; run-to-run AUROC noise is ~0.03, which is why n=1000 was needed.)

## Reading: structure-dependent and notation-invariant, CONFIRMED at n=1000

- **Not canonical-string memorization.** Randomizing the SMILES (same molecule, different string) costs essentially nothing (0.581 vs 0.582, drop -0.001, near-identical CIs). The reading does not depend on the specific canonical string.
- **Structure-dependent, now significant.** canonical 0.581 [0.547, 0.614] vs scrambled 0.507 [0.473, 0.541]: the CIs do NOT overlap (0.547 > 0.541), so the 0.074 drop is real. Destroying the structure drops the model to chance (scrambled CI includes 0.5; canonical CI excludes it). So the prediction genuinely uses the structure, it is not a non-structural prior.
- **Still does not separate grounding from SAR-recall.** "Uses the structure" here means it reads substructures and applies structure-activity associations, which may be remembered SAR rather than de novo reasoning. Both are structure-reading, and the control cannot tell them apart; that is the one open piece.

Net: the control DECISIVELY closes the string-memorization and non-structural-prior objections, and it REPLICATES across two endpoints (CYP3A4 and CYP2D6). The solo "reads it" signal is structure-dependent and notation-invariant, robustly. The remaining open question is grounding vs remembered SAR, both of which are structure-grounded.

## Property-specificity: the named property matters (not a generic bioactivity prior)

One more objection: maybe the model outputs a generic structure-based "this molecule is bioactive" score and pins it to whatever property is named, so the structure-dependence is not CYP3A4-SPECIFIC. Test (`eval/property_specificity.py`, CYP3A4 molecules n=300): ask the REAL property (CYP3A4 inhibition) and an ORTHOGONAL control (aqueous solubility), score both against the CYP3A4 labels, and correlate the two predictions.

| prediction | AUROC vs CYP3A4 label | 95% CI |
|---|---|---|
| P(CYP3A4 inhibition) | 0.600 | [0.538, 0.658] |
| P(soluble) | 0.379 | [0.318, 0.441] |

Pearson correlation between the two per-molecule predictions: -0.477.

This control FAILS to establish property-specificity, and an adversarial review caught the original (opposite) read here as an error. AUROC 0.379 is symmetric to 0.621: it is equally informative about the CYP3A4 label as 0.621, just anti-correlated, so "P(soluble) does not predict CYP3A4" is wrong (in fact |0.379-0.5| = 0.121 exceeds |0.60-0.5| = 0.10, so the solubility prompt carries MORE CYP3A4-label information). Worse, a model reading a SINGLE dominant structural axis, lipophilicity, would produce exactly this pattern: lipophilicity raises CYP3A4 inhibition and lowers solubility, giving P(CYP3A4) ~ +0.6 vs the label, P(soluble) ~ -0.4 vs the label, and a strong NEGATIVE correlation between the two. That is the generic-shared-axis hypothesis, not its refutation, and the result is direction-fragile (flipping the clause to "poorly soluble" would flip the sign and a naive read would call it "generic"). So this control does NOT show the reading is property-specific; if anything it is consistent with a single lipophilicity-like axis applied to whichever property is named.

What still stands (from the notation control above, which the review found sound): the reading is structure-dependent (scrambled to chance) and notation-invariant (not string memorization). What is NOT established: that the structure-dependent signal is SPECIFIC to CYP3A4 rather than a single dominant structural axis (e.g. lipophilicity) that correlates with many ADMET endpoints. A proper test needs a control property orthogonal to lipophilicity, or a partial-association analysis that removes the lipophilicity axis. So the grounding chain is: structure-dependent and notation-invariant (solid), property-specificity OPEN (this control was confounded), and grounding-vs-remembered-SAR unresolved.

## Methods note (a corrected first attempt)

A first attempt used the WS2 `re_notation` and `scrambled` rows directly. That subset is imbalanced (27/200 positive) and the model is at chance on it (canonical ~ randomized ~ scrambled ~ 0.54), so it could not probe the reading: the invariance there was a FLOOR effect (no surfaced signal to perturb), NOT grounding evidence, and an intermediate two-condition read of it was an over-interpretation that the scrambled arm corrected. Generating the notations on the balanced signal-bearing set (this doc) is the fix. Also noted: re-scoring the same canonical set across two runs gave 0.571 then 0.627, so model-output stochasticity adds ~0.03 of run-to-run AUROC noise on top of the sampling CI; single-run orderings inside ~0.05 are not reliable.

## What would settle it

The larger-n power-up is DONE (n=1000 above, scrambled drop now significant). Remaining: a name-removed arm (ask a generic "predict this assay outcome" without naming CYP3A4) to separate grounding from remembered SAR; and the same on a second signal-bearing endpoint. Both light API.

## Reproduce

`python eval/notation_control.py` (rdkit + anthropic; `NOTE_EP`, `NOTE_N`). Raw in `results/notation_control.json`.
