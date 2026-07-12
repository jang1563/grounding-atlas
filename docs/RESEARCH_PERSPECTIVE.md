# Research perspective: calibrated grounding and specialist orchestration

This note summarizes the engineering interpretation of the grounding-atlas evidence. It is
model-agnostic and separates measured results from proposed system design.

## Evidence scope

- GroundBench currently contains 24 tracked tasks across 9 modalities and reports output results for
  three frontier models plus a cheap-head baseline.
- Hidden-state encoding probes are run on open-weight models. Frontier models contribute output and
  routing measurements because their hidden states are unavailable.
- The repository therefore does not establish a same-model encode-plus-verbalize result for the
  frontier models on the leaderboard.
- All reported studies are pilot scale and should be interpreted with their task, model, split, and
  confidence interval.

## What the measurements support

1. **Grounding depends on the representation.** Model output changes when the same or related content
   is expressed through names, anonymized identifiers, sequences, embeddings, images, or numeric
   vectors.
2. **No single web-exposure law explains the panel.** Token familiarity, reasoning capacity, mapping
   documentation, representation parsing, and medical-image behavior contribute differently by model
   and task. The `web` tag is a competitive prior, not an item-level decision rule.
3. **Specialist read-outs are strong baselines.** In the measured discriminative cells, retrieval or a
   specialist head matched or exceeded in-weight adaptation. These results do not imply that training
   never wins outside the tested cells.
4. **Confidence routing remains below the per-item oracle.** With real per-item specialists, routed
   accuracy is about 0.81 versus 0.91 for an oracle that always selects the correct source. The router
   is useful for deferral, but it does not identify most items where the language model uniquely beats
   the specialist.
5. **The generative result is budget-dependent.** Internalized RL and external guidance were not
   separated at moderate reward-query budgets. At high budget, pooled RL showed a modest,
   seed-variable edge whose lower confidence bound missed the pre-registered overturn margin.

## System-design implication

A practical research assistant can combine:

- a reasoning model for task decomposition and synthesis;
- retrieval for documented knowledge;
- callable scientific models and deterministic tools for specialized computation;
- calibrated confidence and input-derived priors for deferral; and
- explicit provenance so each conclusion can be traced to the model, tool, and data used.

The placement decision should be measured per capability. A specialist call is appropriate when it
improves verified performance or supplies information unavailable to the general model. In-weight
adaptation remains appropriate when it wins a matched comparison on the intended distribution.

## Open research questions

- Can a router combine model confidence, specialist uncertainty, disagreement, and input features to
  close part of the 0.81-to-0.91 per-item gap?
- Which results transfer across model families, prompts, and data shifts?
- Where does in-weight adaptation outperform retrieval or specialist orchestration under matched
  compute and data budgets?
- Can same-model encoding and verbalization be measured on stronger open models with controlled
  surface-feature baselines?

The durable contribution is the measurement framework: compare representations, preserve honest
baselines, separate evidence streams, and make placement decisions from verified performance.
