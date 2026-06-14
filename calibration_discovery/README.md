# calibration_discovery - Calibration inside discovery (Move 2)

*Branch started 2026-06-13. Promotes the rung-level calibrated-router result (`results/calibration_routing.md`, corr +0.90) to a PER-ITEM selective-prediction / risk-coverage benchmark for discovery. The roadmap's Move 2: the correct move on an out-of-competence or unfalsifiable task is to abstain or hand off, the axis AbstentionBench shows scaling degrades, which no discovery eval scores. Capability-first, calibration lane. No em dashes.*

**Status (2026-06-13): v1 run complete (3-model scale axis, n=640/model). See `results/RESULTS.md`.** P1 confirmed (frontier improves with scale: AURC_conf 0.290 -> 0.191 -> 0.155). P2 REVISED by v3: the binary abstain ACTION underuses the graded CONF, but the large v1 gap (opus 0.126) was mostly a defer-nudge artifact, it collapses to 0.044 under a neutral framing; the framing-robust residual is small (~0.04-0.07). Web-exposure is a competitive model-free deferral prior. P3 (novel stratum) gave an honest reversal: post-cutoff variant in TEXT form stays answerable via gene prior, so the clean discovery-hard case needs the web-poor SEQ notation (v2). **v2 done (same-variant text vs seq, real per-item AlphaMissense 0.955):** the web-exposure law reaches the CALIBRATION axis within a single entity, CONF falls text->seq in step with AUROC for every model (opus 0.42->0.28), graded (web-poor seq CONF ~0.3, not web-zero ~0.02); over-trust on the web-poor form is scale-dependent (sonnet answers seq at 0% defer); the lowered CONF is the signal that routes to the winning specialist. See `results/RESULTS.md` v2 section. **v3 done (neutral-framing control after self-review):** the v1 P2 gap was mostly a defer-nudge artifact (opus gap 0.126->0.044, deferral 86%->47%, ordering reverses so opus is now best-aligned); CONF quality + P1 are framing-robust; new robust finding = orchestrator prompt framing is a ~2x deferral-rate / deferral-QUALITY knob (specialist nudge makes opus defer answerable rungs 68%, neutral 13%). See `results/RESULTS.md` v3 section. **v4 done (real per-item specialist router, `eval/per_item_router.py`, no new API):** cheap CV specialists on every rung (+ AlphaMissense for variant); cheap specialists DOMINATE solo LLM (0.81 vs 0.65 acc); CONF-routing reduces to almost-always-call-the-specialist (spec-call 84-100%) and does NOT reach the per-item oracle (0.81 vs 0.91) because CONF cannot flag the ~10% where the LLM beats the specialist; the per-rung "routed~oracle" was a ceiling upper bound. Closes the SYNTHESIS §6 open refinement. See `results/RESULTS.md` final section.

## Why this branch (and what it is NOT)

The existing calibration result is COARSE: it correlates rung-level mean CONF with rung-level AUROC (7 points, +0.90). That says "Claude knows which MODALITY it cannot ground." It does NOT score the per-item decision an autonomous scientist actually makes: given THIS item, answer or abstain. This branch builds that.

- NOT a re-measurement of `calibration_routing.md` (rung-level corr stays cited, not redone).
- NOT the binary ROUTE decision at face value (that result already showed it is framing-sensitive and noisy; finding 3 there).
- This IS the per-item selective-prediction object: risk-coverage curves, AURC, and the gap between the model's calibration signal and its abstention BEHAVIOR.

## The bottleneck it instruments

The AI-scientist roadmap names the field's #1 bottleneck: trustworthy autonomous verification (the system reliably knowing when its own intermediate result is wrong). AbstentionBench (NeurIPS 2025): abstention does NOT improve with scale, and reasoning-tuning makes it ~24% worse. No discovery eval scores abstention as an outcome. This branch scores it on biology grounding, with the web-exposure law as the a-priori risk prior.

## Hypothesis

A frontier model HAS the calibration signal (its own CONF ranks grounding at +0.90) but does not USE it: when given an explicit abstain/defer option, it over-answers, so its abstention BEHAVIOR sits well inside (worse than) the risk-coverage frontier its own confidence could achieve. That action gap does not close, and may widen, with scale, even as the confidence frontier itself improves. And web-exposure is a FREE selective score: you can predict where to abstain a-priori from the item's notation, without running the model, except on the novel-but-web-rich-looking corner (post-cutoff variant), where notation over-predicts competence and only the model's own CONF (or a specialist) catches it.

## The three selective-prediction objects (the deliverable)

1. **Risk-coverage frontier (confidence-optimal).** Per item: PRED (property prob), CONF (selective score). Sort by CONF descending, sweep coverage, plot risk (error on answered) vs coverage. AURC (lower = better), E-AURC (AURC minus the oracle's), selective accuracy at coverage 0.5 / 0.8. This is the achievable frontier if the model abstained optimally by its own confidence.
2. **Behavioral abstention operating point.** The same items, model given an explicit SELF-vs-DEFER choice. Its operating point = (coverage = SELF rate, risk = error on the SELF subset). The CALIBRATION-ACTION GAP = risk_behavioral minus the confidence-frontier risk at the same coverage. Positive gap = the AbstentionBench effect (it could have done better by trusting its own CONF). Reported per model scale.
3. **Web-exposure as a free a-priori score.** Risk-coverage using the item's web-exposure tag (web-rich = answer, web-zero = abstain) as the selective score, no model confidence used. If AURC_webexp ~ AURC_conf, web-exposure alone is as good a deferral policy as the model's confidence (the web-exposure law as a routing prior, this project's signature). Broken down by competence stratum to expose where it fails (novel-but-named).

Baselines (selective scores compared on the same risk-coverage axis): model CONF, prediction margin |PRED - 0.5| (no separate CONF needed), web-exposure tag, random (chance), oracle (sort by correctness, the lower bound).

## Item set (the discovery substrate)

Reuses the proven 7-rung set from `calibration_routing.py` (spans web-rich -> web-zero = in-competence -> out-of-competence) plus a NOVEL stratum, with a per-item competence label:

| stratum | rungs | the right move |
|---|---|---|
| `in` (web-rich, answerable) | sc_cellsentence, msa_conserv, dna_promoter | answer |
| `out` (web-zero / computation-bound) | sc_anon, nmr_herg, methyl_age, smiles_herg | abstain / defer |
| `novel` (post-cutoff, web-rich-LOOKING) | variant_novel (ClinVar post-2026-01, gene+HGVS) | defer (absent from train-time web) |

The `novel` stratum is what makes this DISCOVERY, not just web-exposure: a variant whose gene name is web-rich but whose specific effect is post-cutoff. If the model answers it confidently from the gene prior, that is the over-confidence-on-novel failure the bottleneck is about, and the web-exposure-of-notation free score over-predicts competence there.

## Falsifiable predictions

- P1: AURC_conf improves with scale (haiku -> sonnet -> opus), consistent with the +0.90 calibration curve.
- P2 (AbstentionBench): the calibration-action gap (behavioral minus confidence-frontier) does NOT shrink with scale and may widen; explicit abstention stays inside the frontier its own confidence allows.
- P3 (web-exposure prior): AURC_webexp ~ AURC_conf on the in/out strata, but web-exposure FAILS on the novel stratum (named so it scores "trust", but the model is wrong), where only CONF or the specialist defers correctly.
- If P2 is false (behavioral abstention reaches the frontier), the model already self-abstains well and the orchestrator needs no confidence-thresholding layer (also a clean result).

## Honest scoping

- Selective prediction and risk-coverage are standard (Geifman-El-Yaniv, AbstentionBench); the contribution is running them on cross-representation biology grounding with the web-exposure free-score and the novel/post-cutoff stratum, as the per-item promotion of our +0.90 router. Cite the prior art, claim the bio-grounding instantiation.
- Same elicitation as `calibration_routing.py` (PRED/CONF/ROUTE), so results are directly comparable; CONF scale is model-idiosyncratic, only its ranking is used.
- Binary per-rung tasks, n per rung modest (single run); the AURC ordering is the stable quantity. This is the calibration cell of the measurement layer, an extension of calibration-routing, not a standalone headline.

## Layout / reproduce

```
calibration_discovery/
  README.md                  this spec
  eval/selective_eval.py     per-item harness: risk-coverage + AURC + behavioral gap + web-exposure score + framing toggle (SE_FRAMING)
  eval/per_item_router.py    v4: real per-item specialists + the model-vs-specialist router (no API; reuses per_item.csv)
  eval/plot_risk_coverage.py the risk-coverage figure (fig 1), data-driven from the saved runs
  eval/plot_scale_frontier.py  the P1 scale figure (fig 2): 3 models' confidence curves
  eval/plot_router.py        the v4 placement figure (fig 3): reads router_results.json
  results/RESULTS.md         v1-v4 writeup (read this first)
  results/{risk_coverage,scale_frontier,router_placement}.{png,svg}   figures (+ _hires.png)
  results/router_results.json                aggregate router accuracies (for fig 3)
  results/per_item{,_neutral,_variant}.csv   raw per-item rows (specialist / neutral / variant runs)
  results/selective_eval{,_neutral,_variant}.json   aggregate metrics per run
```

Reproduce: `python calibration_discovery/eval/selective_eval.py` (the eval, needs API); then with `PYTHONPATH=calibration_discovery/eval` and no API: `per_item_router.py` (router + router_results.json), `plot_risk_coverage.py`, `plot_scale_frontier.py`, `plot_router.py` (the three figures).

`SE_N=80 SE_MODELS=claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001 python calibration_discovery/eval/selective_eval.py`
