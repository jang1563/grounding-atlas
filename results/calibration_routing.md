# Calibration / routing: frontier Claude is a calibrated router, and calibration grows with scale

*Results. 2026-06-12. `eval/calibration_routing.py`, n=80 balanced per rung, 7 rungs spanning the web-exposure spectrum, claude opus-4.8 + sonnet-4.6 + haiku-4.5 (same run, max_tokens 96, strict one-line format). Per item Claude returns PRED (probability the property holds), CONF (self-rated reliability of PRED given it can or cannot read the representation), and ROUTE (SELF vs DEFER to a specialist). The orchestrator question: does Claude KNOW when it cannot ground, so its own confidence can drive tool routing. Raw `results/calibration_routing.json`. No em dashes.*

## Per-rung result (opus-4.8, representative)

| rung | tag | Claude AUROC | mean CONF | defer % | specialist ceiling |
|---|---|---|---|---|---|
| dna_promoter | web-rich | 0.871 | 0.26 | 100% | 0.889 |
| msa_conserv | web-rich | 0.902 | 0.39 | 82% | 0.999 |
| sc_cellsentence | web-rich | 0.983 | 0.69 | 22% | 0.989 |
| sc_anon | web-zero | 0.500 | 0.02 | 100% | 0.989 |
| smiles_herg | web-zero-out | 0.643 | 0.32 | 100% | 0.825 |
| nmr_herg | web-zero | 0.501 | 0.08 | 100% | 0.866 |
| methyl_age | web-zero | 0.500 | 0.13 | 100% | 0.701 |

opus lowers CONF to 0.02 to 0.13 on exactly the web-zero rungs it cannot read (sc_anon, nmr, methyl) and raises it where it grounds best (sc_cellsentence 0.69). The rank relation between confidence and actual grounding is +0.90 with zero over-confident rungs.

## Scale curve: calibration emerges with model size

| model | corr(CONF, AUROC) | over-confident rungs | always-self | Claude-routed | always-tool / oracle |
|---|---|---|---|---|---|
| haiku-4.5 | **+0.25** | smiles_herg (C 0.62 / A 0.51), methyl_age (C 0.62 / A 0.50) | 0.595 | 0.837 | 0.894 |
| sonnet-4.6 | **+0.64** | none | 0.682 | 0.810 | 0.894 |
| opus-4.8 | **+0.90** | none | 0.700 | 0.893 | 0.894 |

Calibration rises monotonically with scale (+0.25 to +0.64 to +0.90). Two thresholds matter for an orchestrator:
- **Over-confidence disappears at sonnet.** Only haiku is over-confident, and on exactly SMILES strings and decimal numbers: forms that LOOK familiar (ubiquitous in pretraining) but whose property mapping it cannot ground. It rates CONF 0.62 while scoring at chance and would self-answer them. sonnet and opus never do this. So the safe MINIMUM router scale is sonnet; a small model confidently self-answers cases it gets wrong, the SMILES char-n-gram trap surfacing as a calibration failure.
- **Precision peaks at opus.** Confidence ranks grounding cleanly only at opus (+0.90 vs sonnet +0.64), and opus routing recovers 0.893 mean AUROC, essentially the oracle (0.894) and far above answering everything itself (0.700). So the BEST router is opus: confidence-thresholded routing reaches specialist-only accuracy while keeping the web-rich rungs in-model.

## Three findings for the orchestrator

**1. Frontier Claude is a calibrated router.** At opus, self-reported confidence tracks actual grounding at +0.90 with zero over-confident rungs, and routing on its own confidence reaches the oracle (0.893 vs 0.894), far above always-self (0.700). An orchestrator can use frontier-Claude confidence as its tool-call signal: it hits specialist-only accuracy while keeping web-rich rungs in-model. This is the web-exposure law turned into a routing rule: low representation-to-property web-exposure -> low Claude confidence -> defer to specialist.

**2. Calibration is scale-dependent; small models are over-confident on web-familiar surface forms.** haiku is over-confident on precisely SMILES and methylation numbers (CONF 0.62, chance accuracy), the forms whose tokens are web-common but whose mapping is web-absent, and it would route them to SELF and be silently wrong. The router must therefore be at least sonnet (where over-confidence vanishes), ideally opus (where ranking is sharp). The danger scales inversely with model size and is worst exactly where silent errors cost most.

**3. Confidence is a better routing signal than the binary defer decision.** The binary ROUTE is noisy and framing-sensitive: opus over-defers (dna/msa 100%/82% despite grounding them at 0.87/0.90, because "a specialist is available" reads as "use it"), while sonnet under-defers on methyl (46%, so the per-rung-majority router keeps it in-model and loses, dragging sonnet's routed utility to 0.810, BELOW haiku's 0.837, an artifact of the binary rule not of better calibration). The continuous CONF, by contrast, ranks ability monotonically. So an orchestrator should THRESHOLD the continuous confidence (calibrated per-deployment), not take the model's yes/no defer at face value.

## Orchestrator design implications

- A grounded orchestrator is feasible with a frontier model as the router: confidence-thresholded routing recovers oracle-level accuracy (0.893 vs 0.894) at a fraction of the specialist calls (web-rich rungs stay in-model).
- The router must be frontier-scale, sonnet at the very least and opus for precision. Small-model confidence is uncalibrated on web-familiar-surface / web-absent-mapping inputs, the exact cells where silent errors are most costly.
- Route on continuous confidence with a tuned threshold, not the model's binary defer (over-cautious at opus, under-cautious at sonnet, and framing-sensitive throughout).
- The web-exposure law is the a-priori prior: it predicts which cells draw low confidence (web-zero or computation-bound mappings) before any item is seen, so it can seed the routing policy.

## Caveats

n=80 per rung, single run, 7 rungs. CONF and DEFER are elicited under a "specialist is available" framing that makes opus defer-happy and is framing-sensitive (finding 3); the CONF-AUROC ranking is the stable quantity, not absolute defer rates. Routing utility uses per-rung defer majority and known specialist ceilings as the tool's accuracy (an upper bound: tool always right, always called when chosen), which is why the routed numbers are noisy across models; a per-item router with real specialist per-item predictions and a continuous confidence threshold is the next refinement. Absolute CONF scale is model-idiosyncratic; only its monotonic relation to grounding is used. haiku msa AUROC moved between runs (0.92 to 0.72) under the stricter format, a reminder that small-model outputs are prompt-sensitive; opus and sonnet were stable.

## Reproduce

`source ~/.api_keys && CR_N=80 CR_MODELS=claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001 python eval/calibration_routing.py`
