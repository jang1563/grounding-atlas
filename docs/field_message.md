# Field message: what the placement results say to agentic AI, AI scientists, and AI for science

*2026-06-12 (updated 2026-06-13 with the negative-class expression gap and the multimodal verifiability gate). The framing layer of Bio_Grounding_Eval, distilled. Anchored to the measured results (`results/decision_map_placement.md`, `results/SYNTHESIS.md`, `results/calibration_routing.md`). The empirical claims are measured; the philosophical framing is the interpretation. No em dashes.*

## One-line message

A frontier model's job in science is not to KNOW biology but to GROUND and ROUTE it: read a specialist's output faithfully, know when it cannot, and call the right tool. Capability lives in callable specialists and in-context retrieval; calibration is what makes the orchestration safe. Training the reasoner into a biologist is the wrong target; building a calibrated grounded orchestrator is the right one.

## "Understanding" is not one thing (the ladder our instrument separated)

What people compress into "the model understands biology" is really a ladder, and the model is on different rungs for different tasks:
- recall / recognition (knows the entity): variant 0.98, gene-name lookup 0.99 (largely memorization)
- surface decode (the property is linearly present in the hidden state): the probe reads it
- verbalize (says it in words): web-exposure gated, the expression gap. Now shown directly on the NEGATIVE class with the activation arm: an open 8B model encodes a confirmed-inactive compound near the specialist ceiling (activation held-out 0.69-0.88, selectivity-controlled) yet verbalizes it at or near chance (0-shot 0.45-0.59), replicated across two model families (Qwen3-8B and OLMo-2-7B); the largest robust gap is hERG (0.33 at n=1250). So "encodes it but cannot say it" is how a general LM handles the negative too, and web-exposure governs the saying, not the knowing. (Caveat learned here: the first AMES point looked sharpest at 0.376 but shrank to 0.145 when re-run at n=2000, so read sizes off robust n.)
- compute (calculates the property from the representation): binding from 3D, the model cannot
- mechanism / causal (why, and what an intervention does): the model is weak; this is the sibling causal project

The model does the lower rungs and not the upper ones. Most of the impressive numbers are the lower rungs.

## Does the LLM need to understand biology to accelerate science? No.

Measured: retrieve + orchestrate cover the capability space, and "understanding" is not in that loop. The model need not understand binding (3D computation) if it can call Boltz-2. The analogy is exact: a strong PI does not personally understand or run every technique; they know who can do what (calibration), attach the right tool or expert (orchestrate), and read the result faithfully (ground). More fundamentally, AlphaFold itself does not understand folding (it has no physics); it predicts structure by pattern, and it transformed biology. Understanding-free acceleration is real, present, and powerful.

## The difference between understanding and not (our data draws the line)

| | not understanding (recall / pattern / orchestrate) | understanding (mechanism / principle) |
|---|---|---|
| range | interpolation (inside the distribution) | extrapolation (outside it, novel) |
| nature | correlation, recall | causation, reason-from-principle |
| novel / OOD | collapses (PPI anon 0.95 -> 0.50, web-poor) | holds |
| intervention / design | cannot, must call a specialist | can (why -> what to change) |
| self-knowledge | over-confidence risk (knows-not that it knows-not) | naturally calibrated (principle bounds itself) |

The model that does not understand is fast at combining and applying the known, but cannot discover the new. The frontier where the LLM is weak in our map (causal = train-wins-sibling, novel design, structure-computation) is exactly the frontier where understanding/principle is required. That is not a coincidence; it is the same boundary seen from two sides.

## Does understanding the order of nature make a better scientist? Yes, decisively.

Acceleration and discovery are different. Acceleration (doing existing science faster) works without understanding (retrieve + orchestrate, deployable now). Discovery (finding a new order of nature) needs principle. Understanding a law lets you (a) extrapolate to conditions never seen (laws generalize), (b) intervene (causal design: why, therefore what to change), (c) compress many observations into few principles (Kepler's tables to Newton's law), (d) generate new hypotheses by deduction. Interpolation cannot do any of these; it only fills between seen points, never reaches past them. So a retrieval-scientist does the known faster; an understanding-scientist shifts the paradigm. Today's LLM + specialist stack is the former, which is enormous value, but the Kepler-to-Newton leap is still the latter.

## What this says to each field

- **Agentic AI:** the orchestrator architecture is right, and we have the quantitative evidence: retrieve + orchestrate replace train across 17+ cells. Calibration (corr +0.90 at frontier, and a per-item selective-prediction frontier that sharpens with scale) is the load-bearing part that makes routing safe; without it (small-model over-confidence on SMILES and numbers) the agent is confidently wrong. But the model's native binary abstain decision should not be trusted as-is: it loses only a small framing-robust margin to the graded confidence (~0.04 to 0.07 AUROC) yet is extremely framing-sensitive (the same model defers 86% or 47% of items on system-prompt wording alone, a single-variable control), so the orchestrator must route on a tuned threshold over the continuous confidence and set the deferral framing deliberately, not trust the model's yes/no defer (`calibration_discovery/`). Principle: do not train the reasoner, ground it, and route it on its confidence (not its say-so).
- **AI scientist:** "the model says X about a sequence" is not "the model grounds X from the sequence." Grounding must be measured, not assumed. The AI scientist is an orchestrator, not an oracle: descriptive property to retrieve/tool, novel discovery (causal, design, structure) to specialist plus experiment. Knowing its own limits (calibration) is the safety mechanism for autonomous science.
- **AI for science:** the bottleneck is not capability but grounding plus calibration. Specialist FMs (Boltz-2, ESM, AlphaFold) are already strong; the LLM's strength is reasoning and orchestration; the answer is to combine them with the LLM as a calibrated router. Closed-weight models are fully viable here (no training needed). The web-exposure law is an a-priori risk map: it predicts, before seeing an item, where to trust the model and where to call a tool.

## The honest boundary (the hype check, which may be the most important part)

Our results are a calibration of the "LLMs understand biology" hype. Much of the high numbers (variant 0.98, gene-name 0.99) is retrieval/memorization of web-documented entities, not novel grounding; anonymize the entity (PPI 0.95 -> 0.50) and it collapses. So: the AI scientist's impressive "discoveries" are often web-documented recall, and on the genuinely novel (web-poor, causal, structure) it is weak without a specialist. The frontier of real discovery is still specialist FM plus experiment; the LLM's contribution is to ground and orchestrate it. The measured truth sits between the hype (the model understands everything) and the dismissal (the model is a parrot). This memorization-versus-grounding split is now readable from the SIGNAL side before the model even answers: a content-feature verifiability gate scores PPI-by-name at chance (the 0.95 was name-lookup, there is no interaction signal in the text itself) while it passes the same task's in-data-pattern modalities even anonymized (single-cell 0.92). So a high number can be flagged as recall rather than grounding from the data alone, not only from the output.

## Where our work sits

We built the ruler that separates the two regimes: how far you get without understanding (retrieve/orchestrate) and where understanding (causal, novel, principle) becomes necessary. The job is to deploy honestly what works without understanding, and to point precisely at where understanding is needed. That is the contribution: not a model that grounds, but a measurement of grounding and of the train/retrieve/orchestrate boundary, with calibration as the bridge.

## Key framings

The sharpest framings: the one-line message and the closed-weight conclusion (`results/decision_map_placement.md`); the understanding-vs-orchestration line and the honest boundary. The "encode but cannot say" point now has a direct activation-arm proof on the negative class (`results/negative_expression_gap.md`) and a signal-side memorization detector (`signal/verifiability_multimodal.md`); use hERG (n=1250, activation 0.78 vs output 0.45, robust gap 0.33, cross-family) as the headline example, not AMES (its dramatic 0.376 was a small-n over-estimate that fell to 0.145 at n=2000, which is itself a good honesty-about-robustness anecdote).
