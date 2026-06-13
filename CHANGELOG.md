# Changelog

All notable changes to grounding-atlas. Format based on
[Keep a Changelog](https://keepachangelog.com/); dates are ISO 8601.

## [0.1.0] - 2026-06-13

Initial versioned release of the research repository.

### Added
- Repository scaffolding: Apache-2.0 `LICENSE`, `CITATION.cff`, `pyproject.toml`,
  `requirements.txt`, `.editorconfig`, `.gitattributes`, `SECURITY.md`.
- Navigation: README "Results at a glance"; `results/` and `docs/` indexes.
- Tooling: `ruff.toml`, `.pre-commit-config.yaml`, `Makefile`, and GitHub Actions
  CI (ruff + `CITATION.cff` validation + committed-JSON validation).

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
