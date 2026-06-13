# Move 3 prior-art verdict: is "calibrated orchestrator verifier-trust" white space?

*2026-06-12. Targeted prior-art check (workflow `move3-priorart-verifier-trust`, full dossier in workflow output). Honest verdict, the defensible narrow claim, building blocks, and the design + make-or-break baselines. No em dashes.*

## Verdict: QUALIFIED white space, NOT a clean greenfield

The exact 5-way combination (frontier LLM orchestrator + per-instance verifier-trust + known-answer + calibration-metric + escalate-decision) does not exist, but three direct-priors own pieces and MUST be cited and differentiated. Name them before a reviewer does.

- **Trust or Escalate (Jung, Brahman, Choi; ICLR 2025, arXiv:2407.18370)** - the most dangerous neighbor; appears in two search angles. Already scores the trust-vs-escalate DECISION itself with per-instance calibrated LLM-judge confidence (Simulated Annotators) and a coverage guarantee. The abstraction is theirs. We cannot claim the mechanism; we can claim the FM-verifier instantiation. Differentiators: trusted object is a NON-LLM specialist FM (alien competence), not a peer LLM judge; escalation target is a higher-fidelity experiment with asymmetric cost, not a stronger model/human; scientific known-ground-truth and in-silico-vs-experiment asymmetry.
- **When to Trust the Cheap Check (Kiyani et al.; arXiv:2602.17633, 2026)** - owns the escalation-decision THEORY (calibrated two-threshold accept/escalate/reject with cost in the objective). Differentiator: a classical thresholding rule over a reward-model score, NOT an LLM orchestrator reasoning per-instance; math/Sudoku not a scientific FM. Cite as the NORMATIVE policy we benchmark the LLM router against. Do not claim "calibrated escalation."
- **PROBE (Knowing when to trust MLIPs; arXiv:2605.00640, 2026)** - closest scientific prior: per-instance trust-MLIP-vs-escalate-to-DFT via a calibrated classifier on frozen FM embeddings (93.2% acc @ 23.9% coverage). Differentiator: NO LLM/agent, classical supervised classification on embeddings. Our decider is a frontier LLM with NO access to FM internals, judging from returned output + context (the realistic agentic, harder problem).

White-space corners (confirmed): ATTC (arXiv:2604.08281) establishes the PHENOMENON ("Tool Ignored," 15-60% of errors) but is a method, math-only, single tool, no trust-decision metric. AstaBench (arXiv:2510.21652) has the cost-Pareto science harness but explicitly excludes grading verification decisions. The fusion (LLM-orchestrator over a scientific-FM-verifier, calibration as the scored object) is unoccupied.

## The defensible novel claim (narrow, reviewer-proof)

The first benchmark scoring a frontier LLM orchestrator's per-instance calibrated decision to TRUST or ESCALATE a specialist-FM in-silico verifier (Boltz-2-class), where (i) the decider is a general LLM with NO access to the verifier's internals or self-confidence (judges from returned output + context); (ii) the verifier is non-adversarial and benignly wrong on a structured subset (e.g. Boltz-2 OOD molecular glues, documented negative correlation); (iii) ground truth is KNOWN so the trust decision is scored as a calibration object (trust precision/recall, override-rate/accuracy, selective-reliance risk-coverage), not final-task or FM accuracy; (iv) escalation target is an asymmetric-cost oracle. Thesis no prior states: extending "does the model know ITSELF" (our +0.90 self-confidence router) to "does the model know its TOOL" whose error structure it never sees. Connect the web-exposure law as the verifier-trust prior (kinase-trustworthy vs glue-escalate = a coverage prior the router should exploit).

Do NOT claim: first to notice tool-distrust (ATTC), first "trust vs escalate" (2407.18370), calibrated escalation / two-threshold (2602.17633).

## Building blocks (build on, do not reinvent)

- ATTC "Tool Ignored" audit + token-probability trust signal C = (prod max p)^(1/n) as the baseline-to-beat (2604.08281).
- TRUST-Bench matched-pair design + asymmetric metric penalizing over-trust AND under-trust, swapping malicious triggers for benign FM errors (2605.17453).
- Quantile-regression PRM calibration (Know What You Don't Know, NeurIPS 2025) as a trust-gate signal; Calibrated Reasoning (2509.19681) guardrail: verifier reliability != verifier's own confidence.
- Two-threshold objective Pr[escalate] + lambda1 TypeI + lambda2 TypeII (2602.17633) as the normative escalation rule; RouteLLM P(strong beats weak) recipe repurposed to P(experiment overturns FM) (2406.18665).
- PROBE accuracy-at-coverage reporting as the standard operating-point format (2605.00640).
- Boltz-2 documented miscalibration (poses do not predict affinity; OOD glues negative correlation, bioRxiv 2025.06.14.659707) = the benign-error structure and ground-truth generator.
- "Trust or Escalate" Cascaded Selective Evaluation + Simulated Annotators as the LLM-decider template (swap human-agreement for FEP/experiment-agreement); RouterBench oracle-gap metric; AstaBench harness for a credible leaderboard.

## Sharpened design + make-or-break baselines

Substrate: Boltz-2 binding affinity on known-ground-truth instances (FEP+ sets, ChEMBL/BindingDB measured, plus the molecular-glue OOD set where Boltz-2 fails). Replicate on a SECOND FM-verifier with a different error structure (MLIP-vs-DFT mirroring PROBE, or an AlphaMissense variant-effect verifier from our variant line) so it is not a one-FM anecdote.

Primary scored object = the DECISION, not the task: trust precision/recall, override-rate + override-accuracy, selective-reliance risk-coverage + AURC, accuracy-at-coverage (PROBE format), and ECE / confidence-vs-correctness of the TRUST decision (the direct +0.90 extension; the headline is whether +0.90 DEGRADES when the object is a tool the model never sees inside). Cost-aware Pareto (RouterBench oracle-gap style).

Make-or-break baselines a reviewer will demand: (1) the FM's own self-reported confidence as the trust signal; (2) a trained calibrated probe on FM embeddings (PROBE); (3) the classical two-threshold reward-score policy (2602.17633). The LLM orchestrator MUST match/beat the FM's own confidence and approach the classical probe to justify a black-box LLM router. If a trivial threshold on the FM's score dominates, the contribution collapses.

## Honest implication for the program

Move 3 standalone is narrow and risky (Trust-or-Escalate owns the abstraction; the classical baselines may dominate). It is best positioned as ONE cell of the larger measurement layer (grounding + calibration +0.90 + placement + Move 1 causal), the natural extension of our self-knowledge-calibration to tool-knowledge, not a standalone headline. Re-evaluation signal: because Move 3 is this crowded, Move 1 (causal grounding eval: beats-additive-baseline + anon-invariant + held-out-intervention) is likely the cleaner white space; check its prior art next before committing effort.
