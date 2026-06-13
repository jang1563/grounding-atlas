# Thinking notes: can today's AI become a discoverer? Causality as a data / architecture / signal problem

*Living note, started 2026-06-12. An ongoing dialogue (JK + Claude) developing the discoverer question beyond the placement results. Keep updating. Companion to `docs/field_message.md` and `docs/ai_scientist_roadmap.md`. No em dashes.*

## The framing that organizes everything (JK's insight)

The limit on going from ACCELERATOR (interpolation, recall) to DISCOVERER (extrapolation, causation) is not one problem but three entangled ones, and they map cleanly onto the three questions we were chasing:
- **a. why observation alone cannot give causation = DATA problem**
- **b. world model vs LLM = ARCHITECTURE problem**
- **c. does real discovery only come from experiment = SIGNAL problem**

## a. DATA: observation under-determines causation

The ice-cream analogy, deepened. "Ice-cream sales up AND drownings up" is produced equally well by several causal structures: ice-cream -> drowning, drowning -> ice-cream, summer -> both (the truth), or coincidence. Observation cannot tell them apart because they all generate the same correlation. Formally, many causal graphs share one observational distribution (a Markov equivalence class), so observation under-determines the structure. This is not a smartness problem; the information is absent from the data.

The only cut is INTERVENTION: actually do(block ice-cream), see drownings unchanged, conclude not-a-cause. So causation needs intervention data (Perturb-seq, RCT).

The alarming measured fact: even WITH intervention data (Perturb-seq), perturbation FMs lose to a dumb additive baseline (Ahlmann-Eltze, Nat Methods 2025). So data alone is not the whole story; if it were, the intervention data would be enough. It is not, which forces b and c.

## b. ARCHITECTURE: does the model treat ACTION as first-class

The difference in one word: action (intervention).
- LLM: P(next token | context). No notion of intervening on the world in its structure; it recalls written causal claims.
- world model (LeCun): P(next state | state, ACTION). The action (= intervention, do(X)) is a first-class part of the structure, so it expresses do() natively. This is why LeCun says causation/planning need a world model.

Cooking analogy: the recipe-memorizer (LLM) knows "salt makes it salty" from reading; the person who has cooked (world model) knows "this action changes that" from doing, and is stronger in novel situations.

The science twist: AlphaFold/Boltz/GEARS are already input->output world models, BUT most are OBSERVATIONAL mappings (sequence->structure), not action->effect. GEARS attempting "knock out gene -> what changes" and losing to additive is the evidence: an architecture that handles action does not guarantee it learned the causal structure. So: an action-aware architecture (world model) is NECESSARY but not SUFFICIENT.

## c. SIGNAL: what does the model learn from

Training signal = what the model sees to learn.
- LLM signal = next-token = "imitate human-written text" = imitating CORRELATION. Not a causal signal.
- To learn causation the signal must be "the result of an intervention": change X, observe Y move. That comes only from experiment (wet-lab) or simulation (a world model as grader).

The decisive limit: "cannot exceed its verifier." Training an LLM against a specialist grader (rBio) teaches only what the grader already encodes; the genuinely new is not in that signal. So the signal for novel discovery is the result of a never-run experiment. The AI's role then narrows: not knowing causation directly, but choosing WHERE to experiment to learn the most (active learning) and updating itself on the result (closed-loop).

One escape hatch: if simulation is accurate enough (Boltz-2 approximates experiment ~1000x cheaper), in-silico intervention can substitute for the signal, partly replacing experiment. But you must know when the simulation lies (sim-to-real gap), which is itself a calibration problem.

## Synthesis: the three are entangled

| axis | deficit | what it needs |
|---|---|---|
| data | observation under-determines cause (confounding) | intervention data (experiment / Perturb-seq) |
| architecture | LLM has no action concept | a world model that treats action as first-class (necessary, not sufficient) |
| signal | next-token imitates correlation | intervention feedback (experiment / simulation, closed-loop) |

Entanglement: data (intervention) creates the signal, architecture (world model) represents it, signal (experiment) updates the data. Fixing only one fails. The perturbation FM had the data (intervention) and still lost, because architecture and signal did not follow. So "just scale the LLM" (same signal, same architecture) does NOT reach discoverer, and that is the core message of our measurements.

## The three paths (and our honest read)

- A. Scale the LLM bigger: skeptical (it memorizes more, does not discover; scale even degrades calibration / knowing-it-is-wrong).
- B. LeCun general world model (JEPA): right that architecture must handle action, but a general scientific world model is still infant; domain world models (specialist FMs) are ahead in practice.
- C. Hybrid (LLM orchestrator + specialist world models + intervention + closed-loop): the path our placement map points to. Limit: orchestration combines what the specialists know; it cannot exceed them, so genuinely novel cause may come only from new experiment.

Working view: discoverer is the hybrid, but novel causal extrapolation (Pearl rung 2 beyond additive) is not yet demonstrated by ANY path. That non-result is exactly what our causal grounding eval (Move 1) would measure, settling the LLM-vs-world-model debate empirically rather than rhetorically.

## Open threads to keep developing

- Causality's MEASURABLE definition: is "beats additive/no-change baseline + anon-invariant + held-out intervention" enough to claim rung 2? how to even test counterfactual (rung 3)?
- Boundary of the world model: if specialist FMs are the domain world models, where does the LLM stop being orchestrator and need its own world model (e.g. mental simulation for multi-step experiment design)?
- Escaping "cannot exceed its verifier": if orchestration only recombines the known, does novelty come only from experiment? Then is the AI's real job active learning (choosing the most informative experiment) + closed-loop update, with calibration as the safety layer?
- Does an LLM secretly hold an implicit world model (Othello-GPT style), and is the science bottleneck that the PHYSICAL world is under-represented in text (vs games)?

## The three concrete moves this feeds (kept for execution)

1. Causal grounding eval (name-vs-content on non-additive perturbations + mandatory additive baseline; pairs with sibling CausalAtlas).
2. Calibration-inside-discovery (defer/escalate as a scored risk-coverage outcome).
3. Closed-loop placement probe (Boltz-2 behind MCP; measure the orchestration decision = the verifier-reliability signal nobody has).
