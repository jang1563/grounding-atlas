# grounding-atlas

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Status: active execution](https://img.shields.io/badge/status-active%20execution-brightgreen)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Approach: measurement-first](https://img.shields.io/badge/approach-measurement--first-8A2BE2)

**Contents:** [Why](#why-this-project-exists) · [Thesis](#the-thesis-one-line) · [Workstreams](#three-workstreams) · [Results at a glance](#results-at-a-glance) · [Repository map](#repository-map) · [Setup](#setup) · [Cite](#cite) · [Hard rules](#hard-rules-see-guardrailsmd)

**Does a language model do biology by the *content* of a specialist model's output (sequence, structure, identifier, numeric prediction), or just by its *name*? Measure it, manufacture verifiable signal to close it, and map where each capability should live.**

A measurement-first research project toward a **grounded biology orchestrator**. Capability-focused (make a model better at biology), not safety. Synthesizes several existing projects (NMSE, the LLM x SFM over-trust instrument, NullAtlas/NegBioRL, LabCraft) into one program tailored to an Anthropic Research Scientist, Life Sciences direction.

Status: **active execution** (updated 2026-06-13). Thesis, failure-mode taxonomy, and the WS1 spec are settled (`docs/FAILURE_MODES.md`, `eval/README.md`). The instrument is built and has produced results across the modality ladder (small molecules, proteins, variants, methylation, histopathology, single-cell, and more) — see `results/SYNTHESIS.md` and `docs/field_message.md`. WS2 signal generators span ~18 modality families under `signal/`; the WS3 placement map is measured (`results/decision_map_placement.md`), with the per-item calibration extension in `calibration_discovery/`. Assets this builds on are catalogued in `ASSET_REUSE_MAP.md`.

---

## Why this project exists
1. **Standalone research.** The grounding gap is real: a measured name-vs-content recognition gap (name ~100% vs accession ~2-28%), plus the open question of whether the model surfaces what a probe reads from a representation (encoding vs expression). Closing it is a genuine path to a better science model.
2. **Application artifact.** It is the project JK brings to the Anthropic RS Life Sciences application ("here is what I would build, here is the preliminary data I already have"). The adjacent layers are filling in (June 2026): BioMysteryBench measures task solve-rate through tools; gget virus / VirBench (2026-06-08) measures agent retrieval accuracy. Neither measures content-grounding or trust-calibration. **This is the layer those evals do not run** - the chain is retrieval -> content-grounding (this) -> downstream. Eric Kauderer-Abrams names the obstacle as "no single unambiguous source of truth for the training signal" (and is acknowledged on the VirBench work).

## The thesis (one line)
A science model is only as good as it grounds the *content* of a specialist model's output, not its *name*. Today that grounding is decided by assertion. This project makes it measured.

## Three workstreams
- **WS1 - the instrument (MEASURE).** Does the model ground a representation by content or by name? The core is the content-grounding axis (probe-vs-LLM + LLM-activation probe + content-sensitivity), with identity-resolution and channel/action-policy as measured supporting axes. Deterministic, non-LLM-judge, matched controls. Negative-evidence coverage is NullAtlas's (WS2), cited not absorbed.
- **WS2 - the engine (MAKE SIGNAL).** Extend the negative-evidence approach to grounding: generate matched (representation, verifiable-property) pairs where the representation itself is the ground truth, so grounding becomes trainable/evaluable where positive-only literature gives no signal.
- **WS3 - the decision map (MAP THE LINE).** Per capability, measure train (weights) vs retrieve (MCP/RAG) vs orchestrate (call the SFM). Principle: train the skill, retrieve the knowledge, orchestrate the heavy specialist. Local open-weight PoC for the first data points.

Full design: `PROJECT_DESIGN.md`.

## Results at a glance

*Pilot-scale; see [`results/SYNTHESIS.md`](results/SYNTHESIS.md) for the full 17-representation master table and caveats, and [`results/`](results/README.md) for every writeup.*

**The law.** LLMs encode far more biology than they verbalize. A linear probe on an open model's hidden states recovers the property near a specialist ceiling (the *encoding* gap is under 0.10 for 13 of 17 representations), but the verbalized output lags far behind (the *verbalization* gap runs 0.12 to 0.49). What sets that gap is how web-documented the representation-to-property mapping is, not the modality.

| representation → property | ceiling | probe (encode) | output (verbalize) | reads out? |
|---|---|---|---|---|
| MSA column → conserved | 0.999 | 1.000 | 0.795 | grounds (web-rich) |
| single-cell → T cell (gene names) | 0.989 | 0.983 | 0.50 → opus 0.99 | closes with scale |
| single-cell → T cell (anon ids) | 0.989 | 0.964 | 0.497 | invariant (web-zero) |
| methylation → age | 0.701 | 0.685 | 0.487 | invariant (web-zero numbers) |
| histopathology H&E → tumor | ~0.90 | 0.827 | 0.463 | partial, plateau ~0.65 |
| 3D coords → hERG | 0.826 | 0.669 | 0.490 | encoding-limited |

The methylation / MSA pair is the controlled proof: identical task shape, both encoded to ceiling, opposite output (MSA 0.795 vs methylation 0.487) — the only difference is whether the mapping is web-documented.

**The prescription.** Because the frontier is *calibrated* about where it grounds (opus self-confidence tracks actual grounding at corr +0.90), the same map is a routing policy: routing on continuous self-confidence reaches 0.893 mean AUROC, matching the oracle (0.894), versus 0.700 answering everything itself. The web-exposure tag, known a priori before any model call, is itself a competitive deferral prior. Details in [`results/calibration_routing.md`](results/calibration_routing.md) and [`results/decision_map_placement.md`](results/decision_map_placement.md).

## Repository map

**Documents**
| File | What |
|---|---|
| `PROJECT_DESIGN.md` | thesis, the gap, WS1-3 in detail, local PoC, fit, honest scoping |
| `ASSET_REUSE_MAP.md` | which existing project feeds which workstream (real paths) + reuse vs adapt |
| `PRELIMINARY_DATA.md` | exact existing results + numbers + source paths (the "data I already have") |
| `GUARDRAILS.md` | novelty claim/do-not-claim + citations, disclosure-first, style rules |
| `RESEARCH_STATEMENT.md` | the application-facing research statement |

**Code and outputs**
| Path | What |
|---|---|
| `eval/` | WS1 instrument: probe-vs-LLM head-to-head, LLM-activation probe, content-sensitivity (`eval/README.md`) |
| `signal/` | WS2 verifiable-signal generators across ~18 modality families (admet, affinity, methyl, msa, ppi, single_cell, structure3d, computable, ...) |
| `decision_map/` | WS3 train / retrieve / orchestrate placement |
| `calibration_discovery/` | per-item selective-prediction / calibration extension |
| `protein_grounding/`, `variant_grounding/` | modality branches (each with own `data/`, `eval/`, `results/`) |
| `results/` | measured outputs: writeups (`.md`), data (`.json`/`.jsonl`), figures (`.png`) |
| `docs/` | design docs, failure-mode taxonomy, the field message |
| `data/` | shared curated inputs (large/re-fetchable reference DBs are gitignored) |

## Where to start (reading order)
1. Read `ASSET_REUSE_MAP.md` + `PRELIMINARY_DATA.md`, then the source READMEs they point to (especially `FRT_Pilot_Execution/` aggregate outcomes and `Narrow_Model_Safety_Eval/results`).
2. **WS1 first** (the durable, defensible lead): build the content-grounding instrument (axis B) in `eval/` - the probe-vs-LLM head-to-head, the LLM-activation probe, and content-sensitivity - reusing the FRT harness (axes A, E) reframed safety -> capability. See `eval/README.md` for the spec.
3. **WS2 spec** in parallel: write the grounding-signal dataset spec in `signal/` (the (representation, verifiable-property) pair generator), reusing the NullAtlas/NegBioRL machinery.
4. WS3 (local PoC) comes after WS1+WS2 produce a runnable eval and a signal sample.

## Setup

```bash
# 1. Dependencies (research code; versions unpinned)
pip install -r requirements.txt          # or: pip install -e .

# 2. Data. Large public reference DBs (AlphaMissense, ClinVar, UniProt, ProteinGym
#    DMS; ~2.3G) are gitignored and re-fetched per branch:
bash variant_grounding/eval/setup_data_cayuga.sh
bash protein_grounding/eval/setup_data_cayuga.sh
```

LLM clients read API keys from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); keys are never committed. The activation/probe sweeps are GPU jobs (`eval/run_activation_*_cayuga.sh`).

Some ADMET scripts read a SQLite DB from the sibling `Negative_result_DB` project; the path defaults to `../../Negative_result_DB/data/negbiodb_admet.db` and can be overridden with `NEGBIODB_ADMET`.

## Cite

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button). In short:

> JangKeun Kim. *grounding-atlas: a measurement-first map of biological content-grounding in language models.* 2026.

## Hard rules (see `GUARDRAILS.md`)
- **Capability-first.** Lead with the instrument + the verifiable-signal substrate. Safety is one supporting line ("the same instrument flags where a grounded model is unsafe").
- **Claim measurement + signal-engineering; do NOT claim the multimodal build or the general weights-vs-retrieve framing as novel.** Cite the ancestors (Ovadia 2312.05934; In-Tool Learning 2508.20755; NatureLM 2502.07527; Mozi 2603.03655).
- **Numeric over-trust = a verbalization/calibration gap** (the magnitude is in the LLM's activations; do not overstate as "LLMs cannot represent numbers"). **Do NOT extend this to the NMSE AUROC-0.981 result** - that probe is on ESM-2, not the LLM, so it is an encoding-vs-expression question that is still unmeasured.
- **Disclosure-first.** The `FRT_Pilot_Execution/disclosure/` raw bypass material stays out of this capability project. Use AGGREGATE results only. No operational content.
- **No em dashes** in any writeup or application-facing text. Verified facts only.
