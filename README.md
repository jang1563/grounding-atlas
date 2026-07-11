# grounding-atlas

[![Code: Apache 2.0](https://img.shields.io/badge/Code-Apache_2.0-blue.svg)](LICENSE)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/Data-CC--BY--SA--4.0-blue.svg)](DATA_SOURCES.md)
![Status: active execution](https://img.shields.io/badge/status-active%20execution-brightgreen)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Approach: measurement-first](https://img.shields.io/badge/approach-measurement--first-8A2BE2)

**Contents:** [Why](#why-this-project-exists) · [Thesis](#the-thesis-one-line) · [Workstreams](#three-workstreams) · [Results at a glance](#results-at-a-glance) · [Artifact map](#artifact-map) · [Repository map](#repository-map) · [Dataset](#companion-hugging-face-dataset) · [Setup](#setup) · [Cite](#cite) · [Scope & claims](#scope-and-claims)

**Does a language model do biology by the *content* of a specialist model's output (sequence, structure, identifier, numeric prediction), or just by its *name*? Measure it, manufacture verifiable signal to close it, and map where each capability should live.**

**Bottom line:** today's language models internally represent far more biology than they can put into words, and *where* they fall silent is largely predictable a priori — from how familiar the representation's tokens are and how well-documented its representation-to-property mapping is (a capability-dependent mix, not a single web-exposure law). That tells an AI agent when to trust the model and when to call a specialist tool.

![Two-axis decomposition of the grounding gap: encoding (does the model represent the property internally) vs verbalization (does it state it), across 17 representations.](results/synthesis_figure.png)

A measurement-first research project toward a **grounded biology orchestrator**. Capability-focused (make a model better at biology), not safety. Builds on and unifies several of my own prior projects (a frozen-embedding separability study, an LLM over-trust instrument, NullAtlas/NegBioRL, LabCraft).

*Author: **JangKeun Kim** — postdoctoral researcher, computational biology, Weill Cornell Medicine (Mason Lab). Single-cell and spatial genomics, space biology, and AI evaluation for biology. [github.com/jang1563](https://github.com/jang1563)*

Status: **active execution** (updated 2026-07-02; the benchmark surface is GroundBench, 23 tasks / 9 modalities / 3 models + a cheap-head baseline, see [`docs/GROUNDBENCH.md`](docs/GROUNDBENCH.md)). Thesis, failure-mode taxonomy, and the WS1 spec are settled (`docs/FAILURE_MODES.md`, `eval/README.md`). The instrument is built and has produced results across the modality ladder (small molecules, proteins, variants, methylation, histopathology, single-cell, and more) — see `results/SYNTHESIS.md` and `docs/field_message.md`. WS2 signal generators span ~18 modality families under `signal/`; the WS3 placement map is measured (`results/decision_map_placement.md`), with the per-item calibration extension in `calibration_discovery/`. Most recently, WS3's train-vs-orchestrate question was carried from discriminative read-out into the **generative / RL regime** — a pre-registered post-training RL environment on a molecular generator — and **"route, don't train" holds there too** ([`docs/REPORT.md`](docs/REPORT.md), [`results/benchmark/rl_env/`](results/benchmark/rl_env/)).

| Component | State |
|---|---|
| WS1 instrument (encode vs verbalize) | built; 17 representations measured on one 3-arm instrument |
| WS2 verifiable-signal generators | ~18 modality families |
| WS3 placement + calibration | measured: train/retrieve/orchestrate map + per-item routing |
| WS3 generative extension (RL environment) | pre-registered 3-arm generative head-to-head; formal CIs across 3 reward-quality cells (Cayuga + SDSC Expanse) |
| Scale | pilot (n ~80-1500 per rung); ceilings are cheap specialists or cited foundation models |

---

## Artifact map

| Surface | Human entry point | Machine-readable entry point |
|---|---|---|
| **Visual report** | [`docs/report.html`](docs/report.html) — the one-page map: the law, the decision, the generative extension | self-contained HTML (no external assets) |
| GitHub source | [`github.com/jang1563/grounding-atlas`](https://github.com/jang1563/grounding-atlas) | [`pyproject.toml`](pyproject.toml), [`codemeta.json`](codemeta.json), [`CITATION.cff`](CITATION.cff) |
| Hugging Face dataset | [`datasets/jang1563/grounding-atlas`](https://huggingface.co/datasets/jang1563/grounding-atlas) | Parquet configs with dataset-card YAML front matter |
| Results | [`results/SYNTHESIS.md`](results/SYNTHESIS.md), [`results/README.md`](results/README.md) | sibling `.json` / `.jsonl` files under [`results/`](results/) |
| GroundBench | [`docs/GROUNDBENCH.md`](docs/GROUNDBENCH.md) (run it), [`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md) (contract), [`results/benchmark/LEADERBOARD.md`](results/benchmark/LEADERBOARD.md) | [`eval/run_grounding_eval.py`](eval/run_grounding_eval.py) (`--model X`); per-model `scorecard.json` / `manifest.json` / `raw.jsonl` + `LEADERBOARD.md` |
| Data provenance | [`DATA_SOURCES.md`](DATA_SOURCES.md) | per-config source/license table plus HF card metadata |
| Safety and exclusions | [`SECURITY.md`](SECURITY.md) | explicit gitignore boundaries for secrets, raw DBs, and excluded generated scores |

---

## GroundBench: run it on your model

The benchmark surface of this project is **GroundBench**: 23 tasks across 9 modalities x 3 models, plus a
reproducible cheap-head baseline, on one GPU-free output-arm harness. The leaderboard
([`results/benchmark/LEADERBOARD.md`](results/benchmark/LEADERBOARD.md)) is organized by the a-priori
web-exposure tag, which predicts the verbalization floor: `web=zero` rows sit near chance, `web=rich`
rows ground, and the cheap-head column grounds even where the models are at chance.

```bash
pip install -e .
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, or OPENAI_BASE_URL for any OpenAI-compatible server
python eval/run_grounding_eval.py --model claude-opus-4-8     # or --dry-run to validate with no API
```

Quickstart and bring-your-own-model instructions: [`docs/GROUNDBENCH.md`](docs/GROUNDBENCH.md). Task
schema, adding a task, and the mandatory orientation audit: [`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md).
Putting your model on the leaderboard: [`SUBMITTING.md`](SUBMITTING.md).

## Why this project exists
The grounding gap is real: models recognize a biological entity by name far more reliably than they resolve or ground its concrete content — sequence, database accession, numeric representation — a name-vs-content recognition gap established in a prior recognition study, plus the question of whether the model surfaces what a probe reads from a representation (encoding vs expression). Closing it is a genuine path to a better science model.

It is also a distinct layer in the agentic-bio stack. Adjacent evals measure other things: BioMysteryBench measures task solve-rate through tools, and gget virus / VirBench (2026-06-08) measures agent retrieval accuracy against deterministic ground truth. Neither measures whether the model grounds the *content* of what a specialist emits, nor whether it calibrates trust on that output. The complementary chain is **retrieval -> content-grounding (this project) -> downstream**.

## The thesis (one line)
A science model is only as good as it grounds the *content* of a specialist model's output, not its *name*. Today that grounding is decided by assertion. This project makes it measured.

## Three workstreams
- **WS1 - the instrument (MEASURE).** Does the model ground a representation by content or by name? The core is the content-grounding axis (probe-vs-LLM + LLM-activation probe + content-sensitivity), with identity-resolution and channel/action-policy as measured supporting axes. Deterministic, non-LLM-judge, matched controls. Negative-evidence coverage is NullAtlas's (WS2), cited not absorbed.
- **WS2 - the engine (MAKE SIGNAL).** Extend the negative-evidence approach to grounding: generate matched (representation, verifiable-property) pairs where the representation itself is the ground truth, so grounding becomes trainable/evaluable where positive-only literature gives no signal. The ADMET and computable pairs (55,703 rows) are packaged as a public dataset, [`jang1563/grounding-atlas`](https://huggingface.co/datasets/jang1563/grounding-atlas) (CC BY-SA 4.0).
- **WS3 - the decision map (MAP THE LINE).** Per capability, measure train (weights) vs retrieve (MCP/RAG) vs orchestrate (call the SFM). Principle: train the skill, retrieve the knowledge, orchestrate the heavy specialist. Local open-weight PoCs for the first data points, now carried into the **generative / RL regime** by a pre-registered post-training RL environment (internalized RL vs external guidance on a frozen molecular generator) — where the same *route, don't train* verdict holds. See [`docs/REPORT.md`](docs/REPORT.md).

Full design: `PROJECT_DESIGN.md`.

## Results at a glance

*Pilot-scale; see [`results/SYNTHESIS.md`](results/SYNTHESIS.md) for the full 17-representation master table and caveats, and [`results/`](results/README.md) for every writeup.*

**Encode ≫ verbalize.** LLMs encode far more biology than they verbalize. A linear probe on an open model's hidden states recovers the property near a specialist ceiling (the *encoding* gap is under 0.10 for 13 of 17 representations), but the verbalized output lags far behind (the *verbalization* gap runs 0.12 to 0.49). What sets that gap is not the modality but a **capability-dependent mix of two factors** — how familiar the representation's tokens are (reasoning) and how well-documented the representation-to-property mapping is — with the a-priori web-exposure tag a strong predictor of the floor, not a single law.

| representation → property | ceiling | probe (encode) | output (verbalize) | reads out? |
|---|---|---|---|---|
| MSA column → conserved | 0.999 | 1.000 | 0.795 | grounds (web-rich) |
| single-cell → T cell (gene names) | 0.989 | 0.983 | 0.50 → opus 0.99 | closes with scale |
| single-cell → T cell (anon ids) | 0.989 | 0.964 | 0.497 | invariant (web-zero) |
| methylation → age | 0.701 | 0.685 | 0.487 | invariant (web-zero numbers) |
| histopathology H&E → tumor | ~0.90 | 0.827 | 0.463 | partial, plateau ~0.65 |
| 3D coords → hERG | 0.826 | 0.669 | 0.490 | encoding-limited |

> **The sharpest result (a controlled pair).** methylation and MSA have identical task shape and are both encoded to the specialist ceiling, yet opposite output: MSA verbalizes at 0.795 while methylation stays at chance (0.487). The main thing that differs is whether the representation-to-property mapping is web-documented — clean evidence for the **mapping-documentation** factor of the gap (the other factor, token-familiarity / reasoning, is isolated separately by the single-cell name-vs-anon control).

**The prescription.** Because the frontier is *calibrated* about where it grounds (opus self-confidence tracks actual grounding at corr +0.90), the same map is a routing policy. Confidence is good at *knowing when it cannot* — it defers the web-zero tasks — but against **real per-item specialists** it does **not** reach the per-item oracle: routed 0.81 vs oracle 0.91, because self-confidence cannot flag the ~10% of items where the LLM beats the specialist. (An earlier per-rung figure — routing 0.893 ≈ oracle 0.894 — was a ceiling-as-specialist upper bound, corrected in [`calibration_discovery/`](calibration_discovery/results/RESULTS.md).) The durable, model-free win is the **a-priori web-exposure tag**: known before any model call, it is a competitive deferral prior in its own right. Details in [`results/calibration_routing.md`](results/calibration_routing.md) and [`results/decision_map_placement.md`](results/decision_map_placement.md).

**The generative regime.** The train-vs-orchestrate line holds when the model *generates*, not just scores. A pre-registered post-training RL environment puts internalized RL (fine-tune a frozen molecular generator toward a property reward) head-to-head against external inference-time guidance of the same frozen generator — at matched reward-query budget, judged only on a scaffold-disjoint held-out oracle. Internalized RL **ties** external guidance across three reward-quality regimes (formal scaffold-clustered two-sample CIs; the reward drives the gain — a shuffled-reward control collapses to baseline). Training the weights buys nothing over routing a frozen prior through the reward, so **"route, don't train" spans discriminative read-out *and* generation.** (One honest exception, held to its evidence: at high training budget RL pulls *modestly, sub-threshold-ly* ahead by shifting the distribution past the frozen model's guidance ceiling — a caveat that survived every feasible independence check, not an artifact.) Design and result: [`docs/RL_ENV_PREREG.md`](docs/RL_ENV_PREREG.md), [`results/benchmark/rl_env/v1_herg.md`](results/benchmark/rl_env/v1_herg.md).

**The negative class too.** The same encode-but-cannot-verbalize gap holds for confirmed NEGATIVES (this compound is inactive / safe): an open 8B encodes confirmed-inactive near the Morgan ceiling yet verbalizes it at chance, replicated cross-family (Qwen3-8B + OLMo-2-7B), so the known "no negative data leads to excessive false positives" failure is itself an *expression* gap. See [`results/negative_expression_gap.md`](results/negative_expression_gap.md). The verifiability gate that certifies signal also generalizes to 19 modality cells (17/19 PASS) and doubles as a signal-side memorization detector that flags PPI-by-name as recall, not grounding ([`signal/verifiability_multimodal.md`](signal/verifiability_multimodal.md)).

## Repository map

**Documents**
| File | What |
|---|---|
| `PROJECT_DESIGN.md` | thesis, the gap, WS1-3 in detail, scope and honest caveats |

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

## Companion Hugging Face dataset

The public companion dataset is [`jang1563/grounding-atlas`](https://huggingface.co/datasets/jang1563/grounding-atlas). The default config contains 55,703 uniform ADMET + computable rows; the additional configs expose modality-specific rungs as Parquet tables.

```python
from datasets import load_dataset

ds = load_dataset("jang1563/grounding-atlas", split="train")
methyl = load_dataset("jang1563/grounding-atlas", "methyl", split="train")
cells = load_dataset("jang1563/grounding-atlas", "single_cell", split="train")
```

Use the GitHub repository for the measurement instrument and result writeups; use the Hugging Face dataset for training/evaluation rows, schema inspection, and downstream loaders.

## Where to start (reading order)
1. [`results/SYNTHESIS.md`](results/SYNTHESIS.md) - the law, the 17-representation master table, and the orchestrator it prescribes.
2. [`docs/field_message.md`](docs/field_message.md) - the framing: a frontier model's job is to ground and route, not to know.
3. [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md) - the full design and the three workstreams.
4. [`eval/README.md`](eval/README.md) and [`signal/README.md`](signal/README.md) - the instrument and the signal generators; [`results/`](results/README.md) and [`docs/`](docs/README.md) index everything else.

## Setup

```bash
# 1. Dependencies (research code; versions unpinned)
pip install -e .            # light: runs GroundBench (output arm + cheap-head baseline), no GPU
pip install -e ".[full]"    # everything: regenerate signal, ceilings, activation probes, figures
#   (requirements.txt also pins the full stack, e.g. for the activation/probe GPU jobs)

# 2. Data. Large public reference DBs (AlphaMissense, ClinVar, UniProt,
#    ProteinGym DMS; ~2.3G) are gitignored and re-fetched by the branch
#    setup scripts under variant_grounding/eval/ and protein_grounding/eval/.
```

LLM clients read provider credentials from environment variables such as `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`; keys are never committed. The activation/probe sweeps use GPU job templates under `eval/`.

Some ADMET scripts read labels from a local SQLite database (an internal negative-results ADMET source); set its path with the `NEGRES_ADMET` environment variable.

## Cite

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button). In short:

> JangKeun Kim. *grounding-atlas: a measurement-first map of biological content-grounding in language models.* 2026.

## Scope and claims
- **Capability-first, measurement-first.** The contribution is the instrument and the verifiable-signal substrate, not a multimodal model build; the same instrument also flags where a grounded model is unsafe.
- **What is novel vs cited.** The cross-representation grounding measurement and the signal engineering are the contribution; the train-vs-retrieve-vs-orchestrate framing and the encoding-vs-expression decomposition build on prior work (Ovadia 2312.05934; In-Tool Learning 2508.20755; NatureLM 2502.07527; Mozi 2603.03655; Inside-Out 2503.15299).
- **Numeric over-trust is a verbalization/calibration gap**, not an inability to represent numbers (the signal is in the activations). This is not extended to the ESM-2 probe result, which is an encoding question measured on the specialist, not the LLM.
- **Public-safe release.** The committed tree contains evaluation code, derived benchmark rows, and aggregate outputs; secrets, large re-fetchable databases, and excluded generated scores stay out of git (see [`SECURITY.md`](SECURITY.md)).

## License

Code is Apache-2.0 ([`LICENSE`](LICENSE)). The datasets (the `signal/` tables here and the companion Hugging Face dataset) are **CC-BY-SA 4.0**, because some ADMET labels derive from ChEMBL (CC-BY-SA, share-alike). Per-source attribution is in [`DATA_SOURCES.md`](DATA_SOURCES.md). AlphaGenome-derived scores are not redistributed (non-commercial terms).
