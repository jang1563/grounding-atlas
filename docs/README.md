# Docs index

Design notes, failure-mode taxonomy, and framing for the grounding-atlas program.
For measured outputs see [`../results/`](../results/README.md); for the
consolidated design see [`../PROJECT_DESIGN.md`](../PROJECT_DESIGN.md).

## Read first (framing)
| File | What |
|---|---|
| [`REPORT.md`](REPORT.md) | **the current write-up** (brief HF-report / blog style): the corrected two-factor framing, the LLM x SFM post-training landscape, and where this work sits. The heavier journal-style [`MANUSCRIPT.md`](MANUSCRIPT.md) is shelved (predates the two-factor correction). |
| [`capability_web_exposure.md`](capability_web_exposure.md) | the single-cell study behind the interaction + permissioning figures (engine x fuel; the a-priori deferral lever) |
| [`field_message.md`](field_message.md) | what the placement results say to agentic AI, AI scientists, and AI for science |
| [`FAILURE_MODES.md`](FAILURE_MODES.md) | the grounding failure-mode taxonomy (diagnosis) |
| [`DATA_WAR_THREAD.md`](DATA_WAR_THREAD.md) | the AlphaGenome-era data war and where grounding-atlas sits; synthesis of two rung explorations (both honest negatives) |
| [`POSITION_SUPERHUMAN_SCIENTIST.md`](POSITION_SUPERHUMAN_SCIENTIST.md) | one-page position: the path to a superhuman scientist is calibration and orchestration, not knowledge (the whole project's evidence, distilled) |

## Rung and experiment designs
| File | What |
|---|---|
| [`BRIDGE_3WAY_PREREG.md`](BRIDGE_3WAY_PREREG.md) | **pre-registered** (experiment 2): the 3-way calibrated LLM x SFM bridge - learned bridge vs external-orchestration vs in-weight LoRA, on one shared frozen embedding (molecular FM x 7 ADMET endpoints) with a held-out-PROPERTY transfer eval + per-arm-calibrated permissioning. Refutation paths (H1b/H2b) carry the prior's thresholds, so it fairly tests "route, don't train". Staged v1 (Qwen / 1 fold / ChemBERTa) -> v2 (full matrix) |
| [`LAYER_LOCALIZATION_PREREG.md`](LAYER_LOCALIZATION_PREREG.md) | **pre-registered**: where the encode-vs-express gap sits by LAYER in two co-primary open 8B LLMs (Qwen3-8B continuity anchor + Llama-3.1-8B bridge substrate; nested-CV unbiased best-layer + selectivity + cluster bootstrap; fixes the prior +0.11 selection bias). The cheap GPU warm-up that tells the calibrated LLM x SFM bridge where to attach the read-out and the calibration |
| [`SINGLE_CELL_RUNG_DESIGN.md`](SINGLE_CELL_RUNG_DESIGN.md) | the descriptive rung with a built-in web-exposure contrast (gene-name vs anon) |
| [`WS3_NONFP_ENDPOINT_DESIGN.md`](WS3_NONFP_ENDPOINT_DESIGN.md) | a non-fingerprint-local endpoint to break the decision-map circularity |
| [`UQ_ROUTING_POC_DESIGN.md`](UQ_ROUTING_POC_DESIGN.md) | inject specialist self-uncertainty into the per-item router to close the measured 0.81 to 0.91 ceiling; the position's first-lever experiment |
| [`DATA_DENSITY_RUNG_DESIGN.md`](DATA_DENSITY_RUNG_DESIGN.md) | web-exposure as a measured covariate via public-data density per cell state. **Outcome: NO-GO** (D and N collinear among commensurable cells; see its finding) |
| [`ALPHAGENOME_CEILING_DESIGN.md`](ALPHAGENOME_CEILING_DESIGN.md) | a regulatory rung with AlphaGenome as the specialist ceiling. **Pipeline works; ceiling needs fine-mapped eQTLs; LLM arm confirmatory** (parked) |

## Landscape and planning
| File | What |
|---|---|
| [`MODALITY_LANDSCAPE.md`](MODALITY_LANDSCAPE.md) | modality landscape for the next grounding rung (deep-research synthesis) |
| [`WS1_BACKLOG.md`](WS1_BACKLOG.md) | WS1 maturity and what more to do |
