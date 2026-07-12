# Docs index

Design notes, failure-mode taxonomy, and framing for the grounding-atlas program.
For measured outputs see [`../results/`](../results/README.md); for the
consolidated design see [`../PROJECT_DESIGN.md`](../PROJECT_DESIGN.md).

## Read first (framing)
| File | What |
|---|---|
| [`REPORT.md`](REPORT.md) | **the current write-up**: the corrected multi-factor framing, the LLM x SFM post-training landscape, and the measured limits. |
| [`capability_web_exposure.md`](capability_web_exposure.md) | the single-cell study behind the interaction + permissioning figures (engine x fuel; the a-priori deferral lever) |
| [`field_message.md`](field_message.md) | what the placement results say to agentic AI, AI scientists, and AI for science |
| [`FAILURE_MODES.md`](FAILURE_MODES.md) | the grounding failure-mode taxonomy (diagnosis) |
| [`DATA_WAR_THREAD.md`](DATA_WAR_THREAD.md) | the AlphaGenome-era data war and where grounding-atlas sits; synthesis of two rung explorations (both honest negatives) |
| [`RESEARCH_PERSPECTIVE.md`](RESEARCH_PERSPECTIVE.md) | model-agnostic interpretation of the grounding, routing, and placement results, with explicit evidence scope |

## Rung and experiment designs
| File | What |
|---|---|
| [`RL_ENV_PREREG.md`](RL_ENV_PREREG.md) | pre-registered generative comparison: internalized reward post-training vs frozen-model guidance vs base, under matched reward-query budgets and held-out scoring |
| [`BRIDGE_3WAY_PREREG.md`](BRIDGE_3WAY_PREREG.md) | pre-registered comparison of learned bridge, external specialist head, and in-weight LoRA on a shared molecular embedding |
| [`LAYER_LOCALIZATION_PREREG.md`](LAYER_LOCALIZATION_PREREG.md) | **pre-registered**: where the encode-vs-express gap sits by LAYER in two co-primary open 8B LLMs (Qwen3-8B continuity anchor + Llama-3.1-8B bridge substrate; nested-CV unbiased best-layer + selectivity + cluster bootstrap; fixes the prior +0.11 selection bias). The cheap GPU warm-up that tells the calibrated LLM x SFM bridge where to attach the read-out and the calibration |
| [`SINGLE_CELL_RUNG_DESIGN.md`](SINGLE_CELL_RUNG_DESIGN.md) | the descriptive rung with a built-in web-exposure contrast (gene-name vs anon) |
| [`WS3_NONFP_ENDPOINT_DESIGN.md`](WS3_NONFP_ENDPOINT_DESIGN.md) | a non-fingerprint-local endpoint to break the decision-map circularity |
| [`UQ_ROUTING_POC_DESIGN.md`](UQ_ROUTING_POC_DESIGN.md) | inject specialist self-uncertainty into the per-item router to close the measured 0.81 to 0.91 ceiling; the lowest-cost routing follow-up |
| [`DATA_DENSITY_RUNG_DESIGN.md`](DATA_DENSITY_RUNG_DESIGN.md) | web-exposure as a measured covariate via public-data density per cell state. **Outcome: NO-GO** (D and N collinear among commensurable cells; see its finding) |
| [`ALPHAGENOME_CEILING_DESIGN.md`](ALPHAGENOME_CEILING_DESIGN.md) | a regulatory rung with AlphaGenome as the specialist ceiling. **Pipeline works; ceiling needs fine-mapped eQTLs; LLM arm confirmatory** (parked) |

## Landscape and planning
| File | What |
|---|---|
| [`MODALITY_LANDSCAPE.md`](MODALITY_LANDSCAPE.md) | modality landscape for the next grounding rung (deep-research synthesis) |
| [`RL_ENV_DEEPRESEARCH.md`](RL_ENV_DEEPRESEARCH.md) | literature synthesis motivating the matched internalized-RL-vs-external-guidance experiment; read with the executed results in [`REPORT.md`](REPORT.md) |
| [`WS1_BACKLOG.md`](WS1_BACKLOG.md) | WS1 maturity and what more to do |
