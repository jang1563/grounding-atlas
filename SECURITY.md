# Security and data handling

This is a public research repository on biological content-grounding evaluation:
measuring whether language models ground the *content* of specialist outputs.
It holds evaluation data and writeups, not operational content.

## What is intentionally excluded
- **Large public reference databases** (AlphaMissense, ClinVar, UniProt, ProteinGym
  DMS) are gitignored and re-fetched via the per-branch `setup_data_*.sh` scripts.
- **No secrets** are committed. LLM clients read API keys from the environment
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); a `detect-private-key` pre-commit hook
  and a `*.key`/`.env` gitignore guard against accidents.
- **Public-safe scope.** The repository contains evaluation code, derived benchmark
  rows, and aggregate outputs only; excluded source artifacts and generated scores
  stay out of git.

## Reporting
For sensitive issues, contact the author directly rather than opening a public
issue.
