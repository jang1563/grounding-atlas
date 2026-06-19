# T2 (apply): does T1 grounding transfer downstream, solo vs orchestrate

*Results section. 2026-06-10. Instrument: `eval/t2_apply.py` (reproducible, no GPU/API) over `signal/admet/*/verifiability.json` (WS2 ceilings) and the measured 3-arm anchors (`head_to_head.md` R2, `../PROJECT_DESIGN.md` 7.2). No em dashes.*

## What this is

WS1's capability ladder is T0 recognize, T1 comprehend, T2 apply (`eval/README.md`). T1 returned its first numbers (the 3-arm head-to-head across SMILES, variant, protein), which is the precondition the spec set for starting T2 ("full T2 scoring is deferred until the T1 head-to-head returns its first numbers"). T2 was the one unstarted WS1 tier (`../docs/WS1_BACKLOG.md`); this closes its SOLVE mode.

T2 has three modes (`eval/README.md` Bridge to T2): **solve** (predict the property), **propose** (generate a candidate, scored against ground-truth negatives), **evaluate** (judge a presented claim's reliability). Only SOLVE is scored here, because its arm (LLM-output vs ground truth) is already measured; PROPOSE and EVALUATE need fresh runs and are specified below, not scored.

The capability question the project rests on: **does higher T1 grounding predict higher T2 performance?** The answer splits by path.

## R1. Solo T1 does not transfer to T2-solve; the orchestrate path does

SOLVE is the LLM-output arm scored against ground truth. Across all five measured anchors the solo LLM sits near chance while the specialist ceiling an orchestrator would call sits at 0.70 to 0.96:

| modality / task | ceiling | activation | solo output | orchestrate headroom | expression gap | encoding gap | T2 route |
|---|---|---|---|---|---|---|---|
| smiles / hERG | 0.825 | 0.787 | 0.453 | +0.372 | +0.334 | +0.038 | WS3-weights: train read-out |
| smiles / CYP3A4 | 0.745 | 0.684 | 0.502 | +0.243 | +0.182 | +0.061 | WS3-weights: train read-out |
| variant / ClinVar text | 0.962 | 0.795 | 0.599 | +0.363 | +0.196 | +0.167 | weights + orchestrate (mixed) |
| variant / ClinVar seq | 0.962 | 0.740 | 0.494 | +0.468 | +0.246 | +0.222 | weights + orchestrate (mixed) |
| protein / meltome Tm | 0.699 | 0.609 | 0.486 | +0.213 | +0.123 | +0.090 | orchestrate (thin signal) |

(`eval/t2_apply.py`; solo output is the Qwen3-8B anchor, consistent with 7.2. The clean frontier hERG output is 0.633 (R3, sonnet-4-6 with the fixed system-message protocol); the older bare-prompt frontier run gave 0.581 and is superseded. n=200 balanced, so each AUROC 95% CI is roughly +/-0.07; within-band orderings below should be read as ties unless the gap exceeds that.)

The solo curve is flat near chance for these anchors no matter how much the model encodes internally (activation 0.61 to 0.80): T1 grounding fidelity does NOT predict T2-solve through the solo path here. This is the expression gap (`head_to_head.md` R2) carried one tier up. The orchestrate path, calling the specialist whose ceiling is in the table, recovers the property by construction, so the transfer is path-dependent, not capability-absent. (These anchors are the 8B 3-arm on hERG, variant, and protein; the frontier ADMET sweep in R3 shows solo reading is also endpoint-dependent, near chance on hERG but 0.61 to 0.72 on web-exposed physicochemical endpoints, so "near chance" is specific to model and endpoint, not universal.)

## R2. The headroom decomposition is the T2 routing rule (the WS3 seed)

The solo-to-orchestrate headroom (ceiling minus solo output) is not one undifferentiated gap. It splits into two parts that route differently:

- **expression gap** (activation minus output): the property is present in the model's hidden state but not surfaced. Recoverable by training the read-out, in weights (`head_to_head.md` R4; 2602.07812). This is the WS3-weights lever.
- **encoding gap** (ceiling minus activation): the property is not in the model at all. No read-out training can recover it; the specialist must be orchestrated.

The two SMILES endpoints are expression-dominant (encoding gap 0.04 to 0.06, expression gap 0.18 to 0.33), so T2-apply there is a read-out-training target. Variant is mixed (both gaps above 0.15), so it needs both a read-out and the specialist. Protein Tm is a thin-signal case (ceiling 0.70, little to surface), so orchestration dominates. This per-task route is the first content of the WS3 decision map (`../PROJECT_DESIGN.md` WS3): T2-apply is where the solo-vs-orchestrate-vs-weights placement question becomes empirical rather than asserted.

So the honest T2 headline is not "the LLM cannot apply." It is: **on these representation-grounded properties the solo LLM cannot apply, the signal needed to apply is measurably present (expression-limited) or absent (encoding-limited) per task, and that split prescribes the placement.** Negative for solo, constructive for the architecture.

## R3. The frontier solo sweep: T2-solo transfer is endpoint-dependent (not uniformly chance)

The R1 anchors are the open-weight 8B 3-arm. Running the solo arm of all SEVEN WS2 endpoints on a FRONTIER model (claude-sonnet-4-6, property named in the prompt, balanced n=200, clean system-message protocol, `eval/output_arm_admet.py`, raw numbers in `results/output_arm_admet.json`) shows solo T2-solve is NOT uniformly near chance. It is endpoint-dependent and mostly above chance, with the gradient tracking how structure-legible and web-exposed the property is:

| endpoint | n | orchestrate ceiling (cold) | solo raw AUROC | solo oriented AUROC | verdict |
|---|---|---|---|---|---|
| permeability | 200 | 0.878 | 0.281 | **0.719** | solo reads it |
| solubility | 200 | 0.791 | 0.347 | **0.653** | solo reads it |
| hERG | 200 | 0.895 | 0.633 | **0.633** | solo reads it |
| CYP3A4 inhibition | 200 | 0.830 | 0.611 | **0.611** | solo reads it |
| CYP2D6 inhibition | 200 | 0.828 | 0.608 | **0.608** | solo reads it |
| clearance | 200 | 0.746 | 0.485 | 0.485 | chance |
| AMES mutagenicity | 200 | 0.847 | 0.379 | 0.379 | below chance (anti) |

(oriented AUROC orients each score so that above 0.5 means the solo model discriminates the TRUE property. Solubility and permeability are flipped from raw because NegResultDB label-1 is the FAIL outcome = insoluble / impermeable, confirmed against the assay values; see the methods note. AUROC orientation is exact, 1 minus raw, so no re-run.)

Five of the seven physicochemical properties (permeability, solubility, hERG block, CYP3A4 and CYP2D6 inhibition) score 0.61 to 0.72. Three caveats bound what this means, and they matter:
- **Confirmed structure grounding (three controls), only grounding-vs-SAR-recall left.** The prompt names the property, so a priori the 0.61 could be named-property recall or a generic prior. Two controls (`notation_control.md`) hold: (1) NOTATION-INVARIANT (canonical 0.581 vs randomized 0.582, not canonical-string memorization), and (2) STRUCTURE-DEPENDENT (scrambled drops to chance, canonical CI excludes 0.5, non-overlapping; replicated on CYP2D6 n=888). So the solo model reads the STRUCTURE, not a string or a name. A third control (property-specificity, P(CYP3A4) vs P(solubility)) did NOT establish that the signal is CYP3A4-specific: the observed pattern (self 0.60, solubility-control 0.38 anti, correlation -0.48) is exactly what a single dominant structural axis (lipophilicity, which raises CYP3A4 inhibition and lowers solubility) would produce, so it is consistent with a generic structural axis rather than ruling it out. Open: whether the structure-dependence is property-specific (needs an orthogonal control), and grounding vs remembered SAR.
- **Two of the five depend on a post-hoc sign decision.** Permeability and solubility are 0.281 and 0.347 raw, lifted to 0.719 and 0.653 only by the label-1=fail direction flip (justified by assay medians, but a reader who disputes the call sees them as strongly anti). The orientation-independent statement is "3/7 read above chance in raw form (hERG/CYP3A4/CYP2D6 ~0.61); 2 more if the physicochemical sign flip is accepted."
- **CIs overlap within the band.** At n=200 each AUROC CI is ~+/-0.07, so the 0.61-to-0.72 ordering is not resolved; the defensible claim is the BINNED one (5/7 clear chance, 2/7 do not), not a gradient. AMES 0.379 is below chance (p~0.002, this one survives), but "under-flags real structural alerts" is a specific error-mode a single AUROC cannot establish, so it is dropped. The pattern is CONSISTENT WITH the web-exposure hypothesis (text-saturated property surfaced, specific-SAR property not), not a test of it.

Reconciliation with R1, and a scale finding: the R1 hERG anchor output is near chance (8B 0.453, the expression-gap mechanism with activation 0.787), but the clean FRONTIER reads the same hERG task at 0.633. So the near-chance hERG output is a model-SCALE expression gap plus, in the earlier bare-prompt frontier run (0.581 sonnet-4-6; 0.566 was opus-4-8), a parsing artifact. The lift is therefore confounded: the 8B-to-frontier comparison also changed the prompt protocol (the system-message fix), so "scale closes the gap" is not cleanly separable from "the better prompt closes it" here. The honest claim is that BOTH a frontier model AND the clean protocol are needed to reach 0.633, and the direction is consistent with the variant seq floor rising sonnet to opus (`../PROJECT_DESIGN.md` 7.2). The honest T2 claim is therefore path-dependent, endpoint-dependent, AND scale-dependent. Orchestration still wins for maximum accuracy everywhere (the cold ceiling 0.79 to 0.90 exceeds even the best solo 0.72), but a frontier solo model is already a usable partial reader for the web-exposed physicochemical endpoints, and fails only where the property needs specialist SAR (AMES) or the label is heterogeneous (clearance). The R2 routing rule therefore varies by endpoint and by model scale, which is exactly what the decision map is for.

### Methods note (this run, reviewer-relevant)
- **Parsing.** A bare "reply with only the number" prompt made sonnet-4-6 emit a long reasoning preamble that never reached a number within the token budget, giving a fallback (unparsed to 0.5) rate up to 96% on permeability, which silently floors AUROC at 0.5 and reads as false "chance". A system message forcing a bare number fixed it (all endpoints now 199 to 200 of 200 parsed). Lesson: a high fallback rate masquerades as chance, so parse counts are reported alongside every AUROC (`results/output_arm_admet.json`).
- **Direction.** `generate_signal.py` sets label-1 = the NegResultDB FAIL outcome, whose sign differs by endpoint: a hERG/CYP fail is an active blocker/inhibitor (clause aligned), but a solubility/permeability fail is the low-value compound (insoluble/impermeable), so the "is soluble/permeable" clause is inverted there. Confirmed against assay medians (solubility fail 1.1 vs pass 313 ug/mL; permeability fail 71 vs pass 579), which is why those two oriented AUROCs flip. Clearance pools heterogeneous units (fail 26.6% vs pass 22 mL/min/kg) and is left at chance.

## PROPOSE (scored) and EVALUATE (blocked on D)

- **Propose: scored** (`t2_propose.md`). The model generates valid, de novo, qualitatively pharmacophore-flavored molecules (93-100% valid SMILES) for CYP3A4/CYP2D6 inhibition, but the WS2 specialist probe judges them weakly active (mean P 0.14-0.18). The load-bearing caveat is that the probe is OOD on de novo scaffolds (and its references are resubstitution), so grounded activity is UNDETERMINED, not low; settling it needs an off-distribution verifier (docking/QSAR), which is itself a WS3 orchestrate step. K=15 per endpoint = feasibility probe, not a measurement.
- **Evaluate.** Judge a presented claim's reliability. This reuses the axis-D over-trust setup, which is still exploratory (kappa 0.36, no human-rater pass; `../docs/WS1_BACKLOG.md` B.D). So T2-evaluate is blocked on the same D human-rater pass: the two remaining WS1 gaps are coupled, and the D pass unblocks T2-evaluate as a side effect.

## Caveats

Zero-shot solo arm; n=5 modality anchors with ceilings spanning 0.70 to 0.96, so the cross-task numbers are descriptive, not a fitted relation (same ceiling-confound caveat as P1, `p1_webexposure.md`). The R1 anchor solo output is the 8B 3-arm. SOLVE solo is now measured on seven of seven ADMET endpoints (hERG/CYP3A4 from R1; the frontier sweep adds CYP2D6, AMES, solubility, permeability, clearance, R3). Frontier-sweep caveats: single frontier model (sonnet-4-6), the property is NAMED in the prompt (so this is named-property recall, an upper bound on solo grounding, not blind structure reading), balanced n=200 per endpoint (AUROC 95% CI roughly +/-0.08, so AMES 0.379 is below chance but the magnitude is noisy), and the oriented AUROC depends on the NegResultDB label-1=fail direction resolved in the R3 methods note. The route labels are a decision rule over the measured decomposition, not an outcome of a placement experiment; testing them is WS3.

## Reproduce

`python3 eval/t2_apply.py` reads the WS2 ceilings and measured anchors and writes `results/t2_apply.json` (the anchor decomposition plus per-task routes). Deterministic, no GPU or API. The R3 frontier sweep is `python3 eval/output_arm_admet.py` (needs ANTHROPIC_API_KEY; kill-safe incremental save per endpoint; ADMET_DRY=1 validates data and prompts with no API call); raw and oriented numbers in `results/output_arm_admet.json`.

## References

T1 arms and mechanism: `head_to_head.md` (R2 the 3-arm gap, R4 the read-out lever). Read-out training recovers an expression gap: 2602.07812. Spec: `eval/README.md` (Bridge to T2). Placement prior art to cite for WS3: Ovadia 2312.05934, In-Tool Learning 2508.20755.
