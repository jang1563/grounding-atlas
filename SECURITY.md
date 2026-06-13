# Security and data handling

This is a research repository on a defensive AI-safety topic: measuring whether
language models ground the *content* of biological specialist outputs. It holds
evaluation data and writeups, not operational content.

## What is intentionally excluded
- **Large public reference databases** (AlphaMissense, ClinVar, UniProt, ProteinGym
  DMS) are gitignored and re-fetched via the per-branch `setup_data_*.sh` scripts.
- **No secrets** are committed. LLM clients read API keys from the environment
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); a `detect-private-key` pre-commit hook
  and a `*.key`/`.env` gitignore guard against accidents.
- **Disclosure-first.** Any raw bypass or operational material from upstream safety
  projects is kept out of this capability-focused repository; only aggregate results
  are used.

## Reporting
This is currently a private repository. For sensitive issues, contact the author
directly rather than opening a public issue.
