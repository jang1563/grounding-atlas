# Position: the path to a superhuman scientist is calibration and orchestration, not knowledge

*JangKeun Kim, 2026. A one-page synthesis of the grounding-atlas evidence into a
direction for Claude. The empirical claims are measured in this repository
(`results/SYNTHESIS.md`, `results/decision_map_placement.md`,
`results/calibration_routing.md`); the framing is the interpretation. No em dashes.*

## Thesis

A frontier model becomes a superhuman scientist not by knowing more biology but by
knowing where it can and cannot ground biological content, and by routing accordingly.
The lever is grounding, calibration, and orchestration, not training knowledge into the
weights. This is measured, not asserted, and it is closed-weight friendly.

## Split the question: acceleration is reachable, discovery is walled

"Superhuman scientist" is two different targets. **Acceleration** (doing known science
faster and broader) is reachable now with a calibrated orchestrator. **Discovery**
(finding a new order of nature) hits a structural wall: novel means web-poor, and the
ground truth for a novel causal claim is a not-yet-run experiment, so no static model
or verifier holds the signal. Most of what reads as superhuman in practice is
acceleration, and it does not require the model to understand biology, just as AlphaFold
transformed structure without physics.

## What we measured (the map)

One three-arm instrument across 17 representations decomposes grounding into encoding (a
probe on hidden states), expression (verbalized output), and a specialist ceiling.

- **LLMs encode far more biology than they verbalize.** The encoding gap is under 0.10
  for 13 of 17 representations; the verbalization gap runs 0.12 to 0.49 and is set by how
  web-documented the representation-to-property mapping is, not by the modality. The
  controlled proof: methylation and MSA are both encoded to ceiling, but MSA verbalizes
  (0.795) and methylation does not (0.487), differing only in web documentation.
- **High numbers are recognition, not grounding.** Variant 0.98 and gene-name 0.99 are
  memorization; anonymize the entity and it collapses (PPI 0.95 to 0.50). In this work,
  cell-identity marker recall saturates at the frontier even for rare and disease cell
  states, so it is memorized text, uniform and content-free.
- **Capability lives in tools, not weights.** Across 17+ placement cells, retrieve and
  orchestrate cover the space and train wins nowhere for descriptive prediction; the one
  cell where training weakly wins is causal, and it is bounded by its verifier.
- **The frontier is calibrated about all this.** Opus self-confidence tracks actual
  grounding at corr +0.90; routing on it reaches the oracle (0.893 vs 0.894) against
  0.700 answering everything itself. The web-exposure law is a free a-priori risk map.

## The first lever: reading per-input competence (a UQ problem, not a wall)

The single highest-value improvement is not knowledge but per-input competence: knowing,
for THIS input, whether the model or the specialist is right. That is an
uncertainty-quantification problem, tractable engineering, not a wall. The evidence: the
per-item confidence frontier sharpens with scale (AURC 0.290 to 0.155; selective accuracy
at 50% coverage 0.67 to 0.85, haiku to opus), and the model uses a good continuous signal
near-optimally (rung-level routing reaches the oracle, 0.893 vs 0.894). So the bottleneck
is signal quality, not the model's use of it: a better per-input competence signal in,
better routing out.

It has a measured ceiling, and the ceiling names the missing piece. With real per-item
specialists, routing on the model's own confidence reduces to almost-always-call-the-
specialist (0.81) and does not reach the per-item oracle (0.91), because the model cannot
flag the roughly 10% of inputs where it BEATS the specialist. That residual is
specialist-side. A first candidate, the specialist's own per-input uncertainty, was tested
and is insufficient on its own (`docs/UQ_ROUTING_POC_DESIGN.md`): measured on 640 items, no
router on model confidence and specialist self-uncertainty beats always-call-the-specialist,
because the inputs where the specialist is uncertain stay specialist-favorable. So the
competence lever cleanly delivers one half, reaching specialist-level safely by deferring
when the model is unreliable, the load-bearing safety value, achievable now. The other half,
beating the specialist by flagging where the model's unique recall wins, is not extractable
from uncertainty and needs a model-superiority signal (model-specialist disagreement, or a
per-item recall flag), which is open. This stays closed-weight friendly: Claude exposes no
hidden states, so the competence reader is external and injected, not trained into the
weights. The honest residual: UQ degrades out of distribution, so the un-readable part is the
novel regime, which couples back to the knowledge wall. Competence raises the routing floor
to specialist-level safely, but beating the specialist per item is an open signal-design
problem, not a free win.

## Current limits, and which are improvable

| limit | improvable? |
|---|---|
| encodes but cannot verbalize | yes, by a trained read-out, but retrieve and orchestrate are cheaper and usually win |
| recognition mistaken for grounding; collapses on novel | not by more knowledge; fix by calibration and by canonicalizing the input form before the model reads it |
| cannot compute from structure (3D binding) | not in-weights; orchestrate a specialist (Boltz-2) |
| weak on causal and novel discovery | structural wall; needs new perturbation data and experiments, not a better model (cannot exceed its verifier) |
| over-confident small models; framing-sensitive abstention | yes, by routing on tuned continuous confidence, not the binary defer |

The leverage is the ground-route-calibrate layer. The "knows more biology" layer is the
wrong target: it is memorization that breaks exactly in the discovery regime.

## Does improving it matter for real tasks

- **Acceleration tasks** (annotation, triage, hypothesis generation from known biology):
  yes, decisively. A calibrated orchestrator gives superhuman breadth and speed now, and
  closed-weight models are fully viable because no training is required.
- **Autonomous-science safety**: calibration is load-bearing. Knowing when to trust
  itself versus call a tool is the safety mechanism for an autonomous agent; the
  web-exposure law predicts where it will quietly mis-ground before any item is seen.
- **Discovery tasks**: improving the model yields diminishing returns. The bottleneck is
  data production in disease cell states and the experiments themselves (the post-
  AlphaGenome data war). The model-side leverage is to make Claude a calibrated active
  experimental designer: reward the orchestration decision (which experiment reduces
  uncertainty, when to trust the verifier), which is verifiable, not the causal claim,
  which is not.

## The direction

Do not train the reasoner into a biologist. Build the calibrated grounded orchestrator:
a frontier reasoning core; callable specialist foundation models (ESM, AlphaFold,
AlphaMissense, Evo, AlphaGenome, Boltz-2); in-context retrieval; and, the first lever, a
per-input competence reader that feeds the router both the model's calibrated confidence
and the specialist's own per-input uncertainty, with the web-exposure law as the
a-priori prior and notation canonicalization so the model reads the form it grounds. For
discovery, Claude is the calibrated conductor of a closed loop with the data-production
ecosystem, not a solo discoverer.

**The honest definition of superhuman**: not a model that discovers new biology alone,
but a tireless superhuman PI that knows who can do what (calibration), attaches the right
tool or specialist (orchestration), and reads the result faithfully (grounding). That is
already transformative, and grounding-atlas is the ruler that says where it is safe and
where it is not.
