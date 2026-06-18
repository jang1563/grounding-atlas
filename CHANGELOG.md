# Changelog

All notable changes to grounding-atlas. Format based on
[Keep a Changelog](https://keepachangelog.com/); dates are ISO 8601.

## [Unreleased]

### Fixed
- **ames label-direction bug** (`eval/analyze_ames.py`, `eval/fix_ames_orientation.py`): a
  structural-alert audit showed the Ames label was inverted — label-0 is the nitroaromatic-rich
  (mutagenic) class, and all models' P(mutagenic) correlates +0.7 to +0.85 with aromatic-nitro
  (correct toxicology). ames is now oriented `oppose`; the three scorecards were re-scored from
  the committed raw outputs (zero API). The "ames anti-grounding ~0.32" result was an artifact;
  corrected, all models ground ames at ~0.68. The prior `eval/output_arm_admet.py` carried the
  same inverted direction.

### Added
- **grounding-atlas-eval harness** (`eval/run_grounding_eval.py`, design
  `docs/BENCHMARK_DESIGN.md`): a model-agnostic benchmark that scores grounding
  (output AUROC + gap to ceiling), calibration (ECE / AURC / selective-accuracy),
  and memorization-transparency (`memo_delta` = AUROC(matched) - AUROC(scrambled))
  on the verifiable-signal pairs. Inspect-style Dataset / Solver / Scorer; one
  versioned prompt, fixed decode, raw outputs + provenance manifest released, every
  metric with a bootstrap CI, no single-number reduction (per EleutherAI
  arXiv:2405.14782). GPU-free output arm; `--dry-run` validates the pipeline with no
  API. Outputs under `results/benchmark/<model>/` plus an aggregated `LEADERBOARD.md`.
  Elicitation forces a bare number via a system constraint + `Probability:` anchor
  (a reasoning model otherwise preambles past the token budget); empirical endpoints
  carry a directional clause and an a priori label orientation (ported from
  `eval/output_arm_admet.py`) so AUROC measures grounding regardless of assay polarity,
  and the raw model text is saved per item for re-scoring.
- **Specialist ceilings** (`eval/compute_ceilings.py`, `results/benchmark/ceilings.json`):
  the cheap-specialist decodability ceiling per rung that populates the harness `gap`
  column. Molecular rungs use Morgan(2048, r2) + logistic regression, out-of-fold 5-fold
  CV (same recipe as `results/SYNTHESIS.md`); computable rungs are 1.0 by construction.
  Reproducible (seeded CV, deterministic featurizer); committed with method + n provenance.

## [0.1.0] - 2026-06-13

Initial versioned release of the research repository.

### Added
- Repository scaffolding: Apache-2.0 `LICENSE`, `CITATION.cff`, `pyproject.toml`,
  `requirements.txt`, `.editorconfig`, `.gitattributes`, `SECURITY.md`.
- Navigation: README "Results at a glance"; `results/` and `docs/` indexes.
- Tooling: `ruff.toml`, `.pre-commit-config.yaml`, `Makefile`, and GitHub Actions
  CI (ruff + `CITATION.cff` validation + committed-JSON validation).
- Analysis - computable-property row (`signal/generate_computable.py`,
  `eval/output_arm_computable.py`, `eval/bridge_test_pi.py`, `eval/computable_scale_sweep.py`;
  `results/computable_property_row.md`): the encode-vs-verbalize control showing computable
  properties are snap-impossible but reasoning-solvable, which bounds the web-exposure law to
  empirical properties.
- Analysis - SFM-embedding rung (`eval/sfm_embed_meltome.py`, `eval/sfm_embedding_output.py`,
  `eval/sfm_embedding_activation.py`; `results/sfm_embedding_rung.md`): a specialist embedding fed
  to the LLM is read at chance (zero-shot and ICL); the 8B activation only partially encodes it,
  so orchestration needs a trained head on the embedding, not the prompt.
- Analysis - WS3 train-placement (`eval/ws3_lora.py` cell-parameterized, `eval/prep_lora_cells.py`,
  `eval/firm_spectra_ms.py`; `results/ws3_train_placement.md`): LoRA on the weak-cheap-specialist
  cells (variant-sequence, MS spectra) confirms TRAIN wins nowhere; the live placements are
  retrieve and orchestrate.
- GPU job templates (`eval/cayuga_sfm_activation.sbatch`, `eval/cayuga_ws3_lora_cells.sbatch`).

### Changed
- Reproducibility: removed hardcoded personal paths from scripts and output JSONs;
  the ADMET DB path is now the `NEGBIODB_ADMET` environment variable with a
  repo-relative default.
- README reframed as a research front door (a "Scope and claims" section replaces
  the internal authoring rules).

### Security
- Internal working notes (strategy, application framing, session checkpoints) moved
  to a gitignored `internal/` directory; sensitive infra details (API-key commands,
  SSH host, personal interpreter paths, private external paths) scrubbed from
  tracked files.

### Fixed
- `eval/ceiling_gate.py` referenced `os.*` without importing `os` (caught by ruff
  F821).
