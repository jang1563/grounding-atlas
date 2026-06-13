# Calibration inside discovery: per-item selective prediction (Move 2 result)

*2026-06-13. `calibration_discovery/eval/selective_eval.py`, n=80 balanced per rung, 8 rungs (3 in / 4 out / 1 novel) spanning the web-exposure spectrum, pooled n=640 per model. opus-4.8 + sonnet-4.6 + haiku-4.5, one run, same PRED/CONF/ROUTE elicitation as `results/calibration_routing.md` so the two are comparable. Raw `selective_eval.json` + `per_item.csv`. The per-item promotion of the rung-level +0.90 router. No em dashes.*

**Figures** (all data-driven from the saved runs, reproducible; each also `_hires.png` 300dpi + `.svg`):
- `results/risk_coverage.png` (`eval/plot_risk_coverage.py`): the opus risk-coverage curve + oracle/random baselines + AURC area + the two framing operating points (v3) on one curve.
- `results/scale_frontier.png` (`eval/plot_scale_frontier.py`): the three models' confidence curves, the frontier sharpening with scale (P1, finding 1).
- `results/router_placement.png` (`eval/plot_router.py`): per-item placement, cheap specialists dominate and CONF-routing nears them but not the oracle (the v4 section).

## Scale axis (the headline)

| model | AURC_conf | E-AURC | sel-acc@0.5 | sel-acc@0.8 | AURC_webexp | AURC_random | sigAUC CONF | sigAUC ROUTE | ECE |
|---|---|---|---|---|---|---|---|---|---|
| haiku-4.5 | 0.290 | 0.201 | 0.669 | 0.643 | 0.309 | 0.388 | 0.612 | 0.578 | 0.151 |
| sonnet-4.6 | 0.191 | 0.135 | 0.762 | 0.742 | 0.201 | 0.314 | 0.662 | 0.636 | 0.106 |
| opus-4.8 | **0.155** | **0.100** | **0.847** | 0.736 | 0.198 | 0.310 | **0.725** | 0.599 | **0.038** |

(AURC lower = better; oracle lower bound 0.055-0.089. sigAUC = AUROC of ranking per-item correctness by that signal.)

## Three findings

**1. The confidence frontier is real and improves monotonically with scale (P1 confirmed).** AURC_conf falls haiku 0.290 -> sonnet 0.191 -> opus 0.155, E-AURC 0.201 -> 0.135 -> 0.100, selective accuracy at 50% coverage rises 0.669 -> 0.762 -> 0.847, and PRED calibration (ECE) tightens 0.151 -> 0.106 -> 0.038. So IF the model abstained by its own graded confidence, selective prediction would be strong and would sharpen with scale, the per-item form of the +0.90 router curve. At opus, answering the most-confident half of items yields 0.847 accuracy against a pooled base of ~0.66.

**2. The abstain ACTION underuses the graded confidence, but the SIZE of that gap is mostly a framing effect (P2, revised by v3 below).** Under the orchestrator framing (a specialist is available), the binary ROUTE (SELF vs DEFER) ranks correctness worse than the graded CONF at every scale: opus 0.599 vs 0.725, sonnet 0.636 vs 0.662, haiku 0.578 vs 0.612, the gap largest at opus (0.126). That large gap is NOT mainly a calibration fact: it is mostly an artifact of the defer-nudging prompt, which pushes opus to defer 86% and so compresses the binary signal's variance. The v3 neutral-framing control (below) halves the deferral and collapses the opus gap to 0.044, leaving only a small framing-robust residual (~0.04 to 0.07 across models and framings, the information lost by binarizing a graded signal). The behavioral operating point under this framing is itself framing-dominated and inconsistent across scale, never landing on the model's own frontier:

| model | behavioral coverage (SELF) | behavioral risk | coverage-gap | reads as |
|---|---|---|---|---|
| opus | 0.136 | 0.000 | +0.005 | OVER-defers (defers 100% even on dna_promoter it grounds at 0.883, msa at 0.918); insists on zero risk and throws away coverage |
| sonnet | 0.661 | 0.227 | -0.225 | OVER-answers (answers smiles_herg at 0.693, methyl at chance); covers 0.225 beyond what its confidence justifies |
| haiku | 0.263 | 0.250 | +0.070 | erratic (mild over-defer, high risk) |

This is the per-item quantification of `calibration_routing.md` finding 3 (the binary DEFER is framing-sensitive, the continuous CONF is the stable quantity). The orchestrator lesson: route on a tuned threshold over continuous CONF, not the model's yes/no defer, both because binarizing a graded signal costs a small consistent margin (~0.04 to 0.07, v3) and because the binary is extremely framing-sensitive (the same model defers 86% or 47% of items depending only on the system prompt, v3).

**3. Web-exposure is a competitive model-free deferral prior.** Using only the item's web-exposure tag as the selective score (no model confidence, decided a-priori from the notation) gives AURC_webexp 0.198 / 0.201 / 0.309, far above random (0.310 / 0.314 / 0.388) at every scale, within 0.04 of opus's own confidence and EQUAL to sonnet's (0.201 vs 0.191). So you can decide where to defer before running the model, nearly as well as the model decides for itself, the web-exposure law as the a-priori routing prior this project predicted, now scored on the selective-prediction axis.

## The novel stratum: an honest reversal (P3 not confirmed, and why)

P3 predicted web-exposure would FAIL on the novel-but-named corner (post-cutoff variant: gene name web-rich, so webexp says "trust", but the specific variant is novel so the model should be wrong). It did NOT fail: opus scores variant_novel at AUROC 0.937, acc 0.85, and webexp correctly says "trust." The reason is informative: a post-2026-01 ClinVar variant is novel at the VARIANT level but its GENE-level pathogenicity prior persists past the cutoff (a new variant in BRCA1 is still in BRCA1), so the gene+HGVS text form remains answerable by the gene prior, exactly the variant-grounding finding that the text form's 0.79 is largely gene-prior recall. So the post-cutoff TEXT variant is not a clean discovery-hard case. The clean should-defer-on-novel case requires the web-POOR notation: the same novel variant as raw SEQUENCE, where the model falls toward chance (variant-grounding seq form 0.58) while the specialist (ESM-1v / AlphaMissense) reads it (0.92-0.96). That is the v2 stratum (`variant_novel_seq`), and the prediction is that THERE webexp and the model both correctly defer while only the specialist answers.

## v2: the within-entity notation contrast on the calibration axis (variant text vs seq)

The clean discovery-hard case the v1 reversal called for: the SAME post-cutoff variants in two notations, web-rich text (gene + HGVS) and web-poor sequence (a WT protein window + the substitution, no gene name), with a REAL per-item specialist (AlphaMissense `am` score on the same variants, not a cited ceiling). `SE_RUNGS=variant_novel,variant_novel_seq`, n=80 each, raw `selective_eval_variant.json`.

| model | AUROC text/seq | CONF text/seq | defer text/seq | refuse seq |
|---|---|---|---|---|
| opus-4.8 | 0.933 / 0.839 | 0.42 / **0.28** | 90% / 100% | 0 |
| sonnet-4.6 | 0.901 / 0.764 | 0.59 / **0.46** | 5% / **0%** | 0 |
| haiku-4.5 | 0.822 / 0.739 | 0.55 / **0.44** | 90% / 85% | 0 |

AlphaMissense specialist (real per-item, same variants): AUROC **0.955** (94% coverage), above every model's seq arm.

**1. The web-exposure law reaches the calibration axis, within a single entity.** Changing only the notation (content held fixed), grounding falls text -> seq for all three models (AUROC drop 0.08 to 0.14) AND the model's confidence falls in step (CONF drop 0.11 to 0.14, same direction every model). So the same web-exposure knob that governs whether the model grounds an entity also governs how confident it is, at the within-entity level, the cleanest possible test (no ceiling confound). This is the calibration-axis form of the variant text-vs-seq result.

**2. Confidence is graded by web-exposure, not binary.** The seq form is web-POOR not web-zero (opus reads it to 0.839, near the specialist), and the CONF drop is correspondingly PARTIAL (0.42 -> 0.28), not the collapse to ~0.02 seen on the truly web-zero rungs (sc_anon, NMR, methyl in v1). So CONF tracks a web-exposure gradient: web-rich text ~0.42 to 0.59, web-poor seq ~0.28 to 0.46, web-zero ~0.02. The model registers degree of groundability, not just its presence.

**3. Over-trust on the web-poor notation is scale-dependent (the risk case).** sonnet ANSWERS the seq form (0% defer, CONF 0.46) despite grounding it at only 0.764, the over-answer behavior surfacing on exactly the notation where it should be most cautious. opus drops its CONF further (0.28) and does not lean on seq. So the smaller frontier model is the one that over-trusts the web-poor form, the calibration-shadow of the v1 finding that the abstain action lags the confidence signal.

**4. The specialist wins seq, and the lowered CONF is the routing signal.** AlphaMissense (0.955) beats every model's seq arm decisively, while the models' own CONF drops precisely on that notation, so the within-entity confidence drop is exactly the signal that should trigger the specialist call. This is a genuine per-item orchestrate arm (real AlphaMissense predictions, not a ceiling upper bound), closing part of the `SYNTHESIS.md` open refinement.

Honest notes: opus over-defers both notations (the framing artifact), so CONF, not the binary defer, is the clean within-entity signal here too. The text form is itself post-cutoff yet scores 0.93 (gene prior persists past the variant-level cutoff, the v1 finding reconfirmed), and opus's CONF on it is only 0.42, mild under-confidence on a memorization-driven answer, which is the safe direction. No seq refusals at any scale (the careful "judge only from the sequence" framing avoids the Claude seq-refusal gotcha).

## v3: framing robustness of P2 (the gap is mostly a defer-nudge artifact; framing is a deferral knob)

A self-review flagged that the v1 P2 gap tracked the defer RATE: opus deferred 86% under the "specialist available" framing, which compresses the binary action's variance and inflates its gap from the graded confidence. v3 is a single-variable control: an identical run with a NEUTRAL system prompt that drops the specialist-defer nudge (answer when you can, decline only when you genuinely cannot), same 8 rungs, same items, same PRED/CONF/ROUTE format. `SE_FRAMING=neutral`, raw `selective_eval_neutral.json` / `per_item_neutral.csv`.

| model | defer% spec/neut | sigAUC CONF spec/neut | sigAUC ROUTE spec/neut | gap spec/neut |
|---|---|---|---|---|
| opus-4.8 | 86% / 47% | 0.725 / 0.720 | 0.599 / 0.676 | +0.126 / **+0.044** |
| sonnet-4.6 | 34% / 13% | 0.662 / 0.630 | 0.636 / 0.561 | +0.027 / +0.069 |
| haiku-4.5 | 74% / 31% | 0.612 / 0.633 | 0.578 / 0.562 | +0.034 / +0.070 |

- Neutral framing roughly HALVES deferral at every scale (opus 86 -> 47, sonnet 34 -> 13, haiku 74 -> 31): the "specialist available" nudge was inflating deferral, most at the frontier.
- The large opus gap was mostly that artifact: it collapses 0.126 -> 0.044 once the nudge is removed (ROUTE-AUROC rises 0.599 -> 0.676 while CONF holds at 0.72). The v1 "worst at opus" ordering REVERSES under neutral (opus now the SMALLEST gap, the small models largest), exactly what better calibration predicts once framing no longer dominates. The framing-robust residual is small (~0.04 to 0.07): binarizing a graded confidence loses a little, no more.
- CONF quality and P1 are framing-robust: sigAUC_CONF barely moves (opus 0.725/0.720) and AURC_conf stays monotone with scale under neutral (0.254 -> 0.209 -> 0.162, E-AURC 0.172 -> 0.151 -> 0.104). The continuous confidence is the stable, framing-independent signal; the binary action is not.

The new robust finding: orchestrator prompt framing is a deferral-QUALITY knob. Under the specialist nudge opus defers even the ANSWERABLE rungs (in-stratum defer 68%, wasting coverage it grounds at 0.85); under neutral it answers them (in-defer 13%) while still deferring the unanswerable (out-defer 75%), so the in-vs-out defer spread widens from 32 to 62 points. The binary action CAN be competence-aligned, but only when the prompt does not nudge it. One honest consequence: the AbstentionBench "scaling degrades abstention" pattern does NOT cleanly hold here once framing is controlled, under neutral framing the frontier model's binary action is the best-aligned, not the worst, so we do not claim it.

Net: the durable P2 claim is the conservative one. Route on continuous CONF (a small, consistent, framing-robust edge over the binary action), and set the orchestrator's deferral behavior by the prompt framing deliberately (it is a ~2x deferral-rate lever), rather than trusting the model's raw yes/no defer.

## Real per-item specialists: the per-item router (closes the SYNTHESIS open refinement)

calibration_routing routed on a per-RUNG ceiling (the specialist assumed perfect when called) and per-rung defer majority, a per-rung upper bound. Here every rung gets a REAL per-item specialist prediction (a cheap cross-validated out-of-fold classifier on the FULL raw representation: Morgan FP for SMILES, an LR on the methylation betas, k-mers for DNA, column-stats for MSA, a bag-of-genes for single-cell, binned m/z for NMR, AlphaMissense `am` for the variant), and the model's per-item PRED/CONF/ROUTE are read from the saved v1 `per_item.csv` (no new API calls). `eval/per_item_router.py`.

Per-item specialist AUROC (real, cross-validated): sc_cellsentence 0.994, sc_anon 0.994, msa 1.000, variant 0.929, smiles_herg 0.858, dna_promoter 0.834, nmr_herg 0.718, methyl_age ~0.50 (the cheap LR on this file's generic CpG panel is at chance; the project's 0.70 to 0.95 clock uses age-specific CpGs, so this rung's specialist is conservatively weak, which only lowers always-specialist below).

| model | always-model | always-specialist | own-DEFER | CONF-route | oracle | spec-call% |
|---|---|---|---|---|---|---|
| opus-4.8 | 0.689 | 0.811 | 0.812 | **0.817** | 0.909 | 84% |
| sonnet-4.6 | 0.684 | 0.811 | 0.762 | 0.811 | 0.908 | 90% |
| haiku-4.5 | 0.609 | 0.811 | 0.777 | 0.811 | 0.911 | 100% |

(accuracy, pooled over the 8 rungs; always-specialist is identical across models because the specialist does not see the LLM; oracle = per-item best of {model, specialist}, the upper bound any router could reach.)

**1. Cheap per-item specialists dominate the solo LLM (0.811 vs 0.61 to 0.69).** Confirms orchestrate >> solo at the per-item level: a cheap classifier on the raw representation beats the frontier model answering everything in-model by 0.12 to 0.20 accuracy.

**2. With real specialists, CONF-routing reduces to "almost always call the specialist."** CONF-route (0.811 to 0.817) barely exceeds always-specialist, and the best threshold routes 84 to 100% of items to the tool, because the specialists are strong enough that calling them nearly always is near-optimal and the LLM rarely beats a cheap specialist in this panel. So the routing GAIN over always-orchestrate is small here; the routing value is panel-dependent and would grow only where some rungs lack a cheap specialist (so the LLM is the better source).

**3. CONF-routing still beats the model's own binary DEFER (sonnet 0.811 vs 0.762).** Routing on the continuous confidence is better than trusting the model's yes/no, reconfirmed against real specialists: sonnet under-defers (keeps items in-model where the specialist is better), so its own-DEFER policy loses 0.05 that a tuned CONF threshold recovers.

**4. The idealized "routed ~ oracle" does NOT hold with real per-item specialists (the honest correction).** Routed accuracy 0.81 sits well below the per-item oracle 0.91: the model genuinely beats the specialist on ~10% of items, but its self-confidence cannot identify WHICH ones, so CONF-routing captures almost none of that complementary value (opus +0.006 over always-specialist). calibration_routing's "routed reaches oracle (0.893 vs 0.894)" was a per-rung-ceiling artifact; per-item with real specialists, confidence is good at "know when you cannot" (defer the web-zero rungs) but not at "know when you uniquely can" (flag the items where the LLM outperforms a specialist). That residual is the real ceiling on confidence-routing.

## Orchestrator implications (per-item, scored)

- Selective prediction works and scales: a frontier router thresholding its own continuous CONF reaches 0.85 selective accuracy at 50% coverage, and the frontier sharpens monotonically with scale.
- But the model's native binary abstain DECISION must not be trusted as-is: it is extremely framing-sensitive (the same model defers 86% or 47% of items on prompt wording alone, v3) and even at its best loses a small consistent margin to the graded confidence (~0.04 to 0.07 AUROC). The orchestrator must impose a tuned CONF threshold, not read the model's SELF/DEFER, and must set the prompt framing deliberately (it is a ~2x deferral-rate lever).
- Web-exposure seeds that threshold for free: it is a competitive a-priori deferral prior (within a model it nearly matches the model's own confidence: sonnet 0.201 vs 0.191, opus 0.198 vs 0.155), so a routing policy can be initialized from notation before any model call.

## Honest scoping

Selective prediction, risk-coverage, and E-AURC are standard (Geifman-El-Yaniv; AbstentionBench NeurIPS 2025); the contribution is the per-item promotion of our +0.90 router onto cross-representation biology grounding, with the web-exposure model-free prior and the (corrected) novel stratum. One run, n=80 per rung, binary tasks; the AURC ordering and the CONF-vs-ROUTE gap are the stable quantities, absolute defer rates are framing-sensitive by construction (the "specialist available" system prompt). CONF scale is model-idiosyncratic, only its ranking is used. The P2 gap was re-checked against a neutral-framing control (v3) after a self-review found it entangled with the defer-nudging prompt; the conservative residual is what we claim. This is the calibration cell of the measurement layer, an extension of calibration-routing, not a standalone headline. Done since v1: the `variant_novel_seq` web-poor stratum (v2), the framing robustness check (v3), and the real per-item specialist router (v4, `eval/per_item_router.py`, all 8 rungs). Next: multi-seed CIs on the AURCs, and a panel that includes a rung with NO cheap specialist (where the LLM is the better source) to show where confidence-routing's value actually lives, since this panel's cheap specialists dominate.

## Reproduce

`source ~/.api_keys && SE_N=80 SE_WORKERS=12 SE_MODELS=claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001 python calibration_discovery/eval/selective_eval.py`
