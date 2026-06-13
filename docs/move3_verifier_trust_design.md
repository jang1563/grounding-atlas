# Move 3: verifier-trust x web-exposure (3-condition design)

*Design doc, 2026-06-12. The verifier-trust corner of the placement program, redesigned around our anonymization weapon. Developed slowly; this is the living anchor, revise as we check each piece. Prior-art verdict in `move3_verifier_trust.md`/`move3_priorart.md`. Skeptical posture: this MEASURES whether the orchestrator's trust depends on web-exposure; it does not settle anything. No em dashes.*

## Hypothesis

A frontier LLM orchestrator's per-instance decision to TRUST or ESCALATE a specialist-FM verifier depends on the task's WEB-SOLVABILITY. When the task is web-poor (the LLM cannot solve it alone), the LLM depends on the verifier and can calibrate trust in it. When the task is web-rich (the LLM can self-recall), the LLM ignores the verifier and over-trusts its own answer (the "Tool Ignored" failure, ATTC arXiv:2604.08281). So web-exposure, the variable governing our whole placement map, is also a hidden variable in orchestration / verifier-trust, which no prior (Trust-or-Escalate, PROBE, ATTC) examined.

## The 3 conditions (the anonymization weapon applied to verifier-trust)

| condition | task / context shown to the LLM | predicted LLM behavior | predicted trust-calibration |
|---|---|---|---|
| (1) web-poor | binding or stability (LLM cannot solve alone) | depends on the verifier | high (escalates when verifier is wrong) |
| (2) web-rich-named | variant + gene NAME | self-recalls, ignores the verifier | LOW (over-trusts its own answer = Tool-Ignored) |
| (3) web-rich-anon | the SAME variant, gene name stripped | cannot self-recall, must depend on the verifier | high again |

The decisive contrast is (2) vs (3): the SAME variant task, only the gene name removed, and if verifier-trust recovers, web-exposure is isolated as the cause. This is the verifier-trust version of our PPI 0.95 -> 0.50 anonymization result. (1) is the cross-task robustness check (a genuinely web-poor task, different verifier) so the effect is not a variant-only artifact.

## Verifiers (candidates, to finalize)

- web-poor (1): ESM-based stability / ddG predictor (ground truth = Tsuboyama 2023 mega-scale ddG ~776k; runnable on Cayuga, ESM already in use) OR Boltz-2 binding affinity (documented miscalibration is ideal, but heavy to run). Lean: start with ESM-stability for tractability; Boltz as an optional second web-poor verifier.
- web-rich (2)(3): a variant-effect FM scoring ClinVar variants we already hold. Candidates: AlphaMissense (public genome-wide scores, easy match) or Evo2 (JK asset, 215k variants already scored, documented benign-error on TERT/non-coding). Lean: AlphaMissense for genome-wide coverage; Evo2 as the JK-asset variant with a known failure region.

Open requirement for ALL verifiers: a documented or measured BENIGN-ERROR structure (a predictable subset where the verifier is wrong), because the escalate-correct cases live there. We will measure each verifier's error map ourselves (run it, see where it misses) rather than assume.

## Measurement (score the DECISION, not the task)

Per instance the LLM receives `verifier output + context` and emits a trust/escalate decision plus its own PRED. Against known ground truth:
- trust-calibration: does trust/escalate match where the verifier is actually right/wrong (the direct extension of our +0.90 self-confidence router to tool-confidence)
- Tool-Ignored rate: verifier is right but the LLM overrides
- override-accuracy: when it distrusts, did it actually fix a verifier error
- report per condition; the headline is the (2)-vs-(3) swing in trust-calibration

## Baselines (the prior-art verdict demands these or it collapses)

- the FM's own self-reported confidence as the trust signal
- a trained calibrated probe on FM embeddings (PROBE, arXiv:2605.00640)
- the classical two-threshold accept/escalate/reject policy (When to Trust the Cheap Check, arXiv:2602.17633)

The LLM router must match/beat the FM's own confidence and approach the classical probe; if a trivial threshold dominates, the contribution collapses to "web-exposure modulates it," which is still the narrow defensible claim.

## Prediction (falsifiable)

trust-calibration high in (1), LOW in (2), recovered in (3). The (2)->(3) recovery under anonymization isolates web-exposure as the orchestration variable. If (2) == (3) (no recovery), the cause is not web-anchored self-recall but something else (e.g. the LLM never trusts a numeric verifier regardless), which is itself a clean result.

## Honest limits + positioning

- Narrow claim only: the orchestrator-judges-a-black-box-scientific-FM-verifier instantiation + the web-exposure modulation. Do NOT claim the trust-vs-escalate mechanism (Trust-or-Escalate ICLR 2025), calibrated escalation (2602.17633), or scientific surrogate trust (PROBE).
- This is one cell of the measurement layer, the natural extension of calibration-routing, not a standalone headline. Same question as CausalAtlas P2b on a different substrate, do not double-count.
- web-rich variant carries a memorization confound by design; that is the POINT here (it drives the (2) effect), but it means the variant arm measures recall-vs-verifier, not pure grounding.

## LIGHT MEASUREMENT RESULT (2026-06-12, `eval/verifier_trust_variant.py`)

ClinVar variants, simulated verifier (correct on 184, deliberately wrong on 216), opus, named vs anon (gene/accession stripped, c./p. change kept). LLM given the verifier verdict + variant, asked its own pathogenic probability.

| rendering | verifier | LLM-correct | follows-verifier |
|---|---|---|---|
| named | correct | 0.891 | 0.891 |
| named | WRONG | 0.713 | 0.287 |
| anon | correct | 0.918 | 0.918 |
| anon | WRONG | 0.574 | 0.426 |

Verdict: hypothesis DIRECTION confirmed but effect MODEST. On the verifier-wrong subset, named LLM-correct 0.713 > anon 0.574 and named follows-verifier 0.287 < anon 0.426, i.e. with the gene name visible the LLM relies LESS on the verifier (self-recalls, overrides the wrong verifier), and anonymized it relies MORE (follows the wrong verifier). So web-exposure does modulate verifier-reliance, but the swing is ~0.14, not the dramatic PPI-style 0.95->0.50 collapse. Two honest caveats: (a) effect is modest; (b) anon is NOT fully web-zero, the variant nomenclature (frameshift / stop-gain) leaks a general pathogenicity rule, so the LLM still solves 57% anonymized, blunting the contrast. Combined with the prior-art finding (SMART / ATTC already establish competence-modulated tool reliance and the Tool-Ignored phenomenon), Move 3 is a NARROW measurement cell (web-exposure modestly lowers verifier-reliance), an extension of calibration-routing, NOT a standalone headline. Recommendation: record it as a cell; move the energy to Move 1 (causal grounding, cleaner prior-art, CausalAtlas infrastructure).

## Data + next steps (slow, check each)

1. Decide verifiers (web-poor: ESM-stability vs Boltz; web-rich: AlphaMissense vs Evo2).
2. Build / fetch verifier outputs + ground truth: Tsuboyama ddG + an ESM ddG predictor; ClinVar variants + AlphaMissense/Evo2 scores (some already held).
3. Measure each verifier's benign-error map (where it misses) = the escalate-correct substrate.
4. Narrow research (only if needed): is the verifier-trust x web-exposure hypothesis itself anticipated; does ATTC connect Tool-Ignored to task-solvability.
5. Then the 3-condition run, reusing the calibration_routing harness.
