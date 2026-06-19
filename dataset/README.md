---
license: cc-by-sa-4.0
pretty_name: GroundBench
tags:
- biology
- grounding
- language-models
- foundation-models
- benchmark
task_categories:
- text-classification
configs:
- config_name: default
  data_files: groundbench.parquet
---

# GroundBench

GroundBench measures whether a language (or vision-language) model can **verbalize a property from a
specialist representation** it is shown, across 24 tasks in 9 modalities. Each row is a
(representation, verifiable binary property) item with an a-priori `web` tag (`rich` / `zero` / `mixed`).

This is the data. The runnable benchmark, leaderboard, and contract live in the code repository
(`docs/GROUNDBENCH.md`, `docs/GROUNDBENCH_SPEC.md`).

## Columns
`task`, `modality`, `kind` (pairs/twocol/emb/image), `web`, `orientation` (align/oppose), `ceiling`
(cheap or cited specialist), `rep_type` (text/embedding/image_path), `representation`, `label`, `id`.
Embeddings are stored as space-joined floats; image rows store a relative path to the patch.

## Load
```python
from datasets import load_dataset
ds = load_dataset("<owner>/grounding-atlas", "default", split="train")
herg = ds.filter(lambda r: r["task"] == "admet/herg")
```

## What it measures, honestly
The `web` tag predicts the snap-verbalization floor and is budget-robust, but it is
necessary-not-sufficient. The single-cell name/anon gap (our cleanest signal) decomposes into
**token-familiarity / reasoning** and **mapping-documentation** in a **capability-dependent mix**: the
token-familiarity share rises monotonically with capability (Haiku 0.32 < Sonnet 0.49 < Opus 0.80), so
weaker models recall documented markers while the frontier reasons over familiar representations. Cite
the name/anon effect as real and large, but its mechanism as a two-factor capability-dependent mix, not a
single web-exposure axis. Details: `results/benchmark/{token_familiarity,capability_trend,budget_arm}.md`.

## License and attribution
Data CC-BY-SA-4.0 (some ADMET labels derive from ChEMBL, share-alike). Per-source attribution in the
repository `DATA_SOURCES.md`. Labels are evaluation targets, not generation guidance.
