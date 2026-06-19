"""Export GroundBench as one self-describing table (Phase C / discoverability).

Writes:
  dataset/groundbench.parquet  -- every task's balanced items, columns:
       task, modality, kind, web, orientation, ceiling, rep_type, representation, label, id
  dataset/tasks.json           -- per-task metadata (modality, web, orientation, ceiling, prompt, n)
  dataset/README.md            -- the Hugging Face dataset card (corrected two-factor framing)

The user uploads dataset/ to the companion Hugging Face dataset; eval/make_croissant.py emits the
machine-readable metadata over the same parquet. No API. Run: python eval/export_dataset.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS, task_items  # noqa: E402
from run_grounding_eval import _ceilings  # noqa: E402

CEIL = _ceilings()
# Modality = the data domain (9 total, matching the spec). hERG-as-graph/NMR/3D are alt-representations
# WITHIN the molecular modality, so admet and herg both map to "molecule"; the SMILES-vs-graph/NMR/3D
# distinction is carried by the task id, not a separate modality.
PREFIX_MODALITY = {"admet": "molecule", "herg": "molecule",
                   "single_cell": "single-cell expression", "variant": "protein variant",
                   "methyl": "DNA methylation", "msa": "protein MSA",
                   "materials": "materials composition", "rna": "RNA sequence"}


def modality(tid, kind):
    if kind == "emb":
        return "SFM embedding"
    if kind == "image":
        return "histopathology image"
    return PREFIX_MODALITY.get(tid.split("/")[0], tid.split("/")[0])


def resolve_ceiling(c):
    return CEIL.get(c) if isinstance(c, str) else c


def rep_of(it, kind):
    if kind == "image":
        return os.path.relpath(it["image"], ROOT), "image_path"
    return it["rep"], ("embedding" if kind == "emb" else "text")


CARD = """---
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
specialist representation** it is shown, across {ntasks} tasks in {nmod} modalities. Each row is a
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
single web-exposure axis. Details: `results/benchmark/{{token_familiarity,capability_trend,budget_arm}}.md`.

## License and attribution
Data CC-BY-SA-4.0 (some ADMET labels derive from ChEMBL, share-alike). Per-source attribution in the
repository `DATA_SOURCES.md`. Labels are evaluation targets, not generation guidance.
"""


def main():
    rng = np.random.default_rng(0)
    rows, meta = [], {}
    for tid in CORE:
        t = TASKS[tid]
        items, _ = task_items(tid, 100000, rng)
        mod = modality(tid, t["kind"])
        ceil = resolve_ceiling(t["ceiling"])
        for it in items:
            rep, rt = rep_of(it, t["kind"])
            rows.append(dict(task=tid, modality=mod, kind=t["kind"], web=t["web"],
                             orientation=t["orient"], ceiling=ceil, rep_type=rt,
                             representation=rep, label=int(it["label"]), id=str(it["id"])))
        meta[tid] = dict(modality=mod, kind=t["kind"], web=t["web"], orientation=t["orient"],
                         ceiling=ceil, n=len(items), prompt=t["prompt"])
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "dataset")
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "groundbench.parquet"), index=False)
    json.dump(meta, open(os.path.join(out, "tasks.json"), "w"), indent=2)
    open(os.path.join(out, "README.md"), "w").write(
        CARD.format(ntasks=df["task"].nunique(), nmod=df["modality"].nunique()))
    print(f"wrote dataset/groundbench.parquet ({len(df)} rows, {df['task'].nunique()} tasks, "
          f"{df['modality'].nunique()} modalities), tasks.json, README.md")
    print(df.groupby("web").size().to_string())


if __name__ == "__main__":
    main()
