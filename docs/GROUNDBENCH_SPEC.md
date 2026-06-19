# grounding-atlas-eval: benchmark spec

*Working name "GroundBench"; final name TBD. Draft 2026-06-19. The contract for adding tasks and
models, so the benchmark is extensible by others, not a one-off. No em dashes.*

## What it measures

The content-grounding layer between retrieval and downstream, the slot adjacent evals leave open
(VirBench = retrieval accuracy; BioMysteryBench = tool solve-rate). For each (representation,
verifiable binary property) task, any LLM is scored on three things, plus an a-priori tag:

- **grounding**: output AUROC, and the gap to a cheap-specialist ceiling.
- **calibration**: ECE (10-bin), AURC (risk-coverage), selective accuracy at 50% coverage.
- **memorization-transparency**: `memo_delta` = AUROC(matched) - AUROC(scrambled); structural
  dependence, where a scrambled control exists.
- **web-exposure tag** (`rich` / `zero` / `mixed`): the a-priori, input-derived label that predicts
  whether to trust the model or defer to a specialist, knowable before any model call.

No single-number reduction: a model is a row of per-task scores, read across the three families and
the web tag.

## Task schema (eval/benchmark_tasks.py)

A task is one entry in the `TASKS` registry, decoupled from data format:

```
"admet/herg": dict(kind="pairs",  data="admet/herg/pairs.jsonl",
                   prompt=_mol("blocks the hERG potassium channel ..."),
                   orient="align", web="rich", ceiling="admet/herg"),
"single_cell/cd8t_nk:anon": dict(kind="twocol", data="single_cell/cd8t_nk.csv", col="anon",
                   prompt=_cell("CD8+ T cell", "NK cell"), orient="align", web="zero", ceiling=0.992),
```

- `kind`: `pairs` (a `signal/.../pairs.jsonl` with matched + scrambled conditions, enabling
  memo_delta) or `twocol` (a CSV with two representation columns, a web-rich vs web-zero contrast).
- `prompt`: a versioned template with a `{rep}` slot, ending in a numeric anchor (`Probability:`).
- `orient`: `align` (positive = label 1) or `oppose` (positive = label 0). Anchors AUROC direction;
  must be set a priori from the assay/label semantics, never tuned to the model.
- `web`: the web-exposure tag.
- `ceiling`: a float, or a key into `results/benchmark/ceilings.json` (Morgan+LR for molecular rungs;
  1.0 for computable).

## How to add a task

1. Drop the data under `signal/<modality>/` as either a `pairs.jsonl`
   (`{id, representation, label, condition}`) or a two-column CSV (`label, <rep_col>, <anon_col>`).
2. Compute a cheap-specialist ceiling (`eval/compute_ceilings.py` for molecular; otherwise a held-out
   CV AUROC) and record it.
3. Set the label direction with `eval/audit_orientations.py` (an independent physicochemical / alert
   prior, not the model's output). Label provenance is the benchmark's largest risk: an inverted Ames
   label produced a spurious "anti-grounding" until a structural-alert audit caught it.
4. Add one `TASKS` entry. It now appears in `evaluate()` and the leaderboard.

## How to run / add a model

```
from eval.run_grounding_eval import evaluate
evaluate("claude-opus-4-8")            # the CORE task set, writes results/benchmark/<model>/ + leaderboard
```
or `python eval/run_grounding_eval.py --model gpt-4o --tasks core --n 100`. Dispatch covers
Anthropic and OpenAI (extensible); `--dry-run` validates the pipeline with no API. Each run writes a
`scorecard.json`, a provenance `manifest.json` (model, prompt version, data commit, seed, decode,
date), and `raw.jsonl` (every item's id, label, parsed probability, raw text) so anyone can re-score.
Every metric carries a 95% bootstrap CI.

## Submission and leaderboard

`results/benchmark/LEADERBOARD.md` aggregates every committed `results/benchmark/<model>/scorecard.json`
into a per-task table with the web tag. A submission is a PR adding a `<model>/` directory (scorecard +
manifest + raw). The harness is deterministic given the data commit and seed.

## Contamination and the web-exposure policy

The benchmark MEASURES web-exposure rather than only avoiding it. The matched / re-notation / scrambled
conditions and the web-rich / web-zero pairs are the "pristine vs defended" split: a high score on a
web-documented item is recall, not grounding, so `memo_delta` and the web tag are reported alongside
accuracy. The same-data web-rich vs web-zero contrast (single-cell gene-names vs anonymized ids) is the
cleanest control: equal information, equal specialist ceiling, only the names differ.

## Versioning, scope, naming

Prompts are versioned constants; the data version is the git commit. Current coverage: **17 tasks across
7 modalities x 3 models** (n=100/task): 6 ADMET (SMILES); 4 single-cell (CD8-T/NK and CD14+/CD16+
monocyte, each web-rich NAME and web-zero ANON); variant effect (web-rich HGVS text + web-poor protein
sequence); DNA methylation -> age (web-zero numeric) and MSA-column -> conserved (web-rich), a controlled
pair; materials metal-vs-not (web-rich formula + web-zero anonymized elements, generality beyond
biology); and the **SFM leg**: an ESM-2 protein embedding -> thermostability, the LLM x SFM interface,
where every model reads the raw embedding at chance (0.50-0.53) while a read-out head on the same
embedding reaches 0.633 (the orchestrate-via-a-trained-head baseline, not prompt-pasting). Three
controlled web-exposure pairs span the leaderboard. One caveat measured here: the materials anonymized
form preserves stoichiometry (`elem_X: count`), so it is a leakier web-zero control than single-cell
anon (one model, gpt-4o, reads composition statistics from it: 0.60 vs chance). Roadmap:
hERG-as-{graph,NMR,3D} (same property, other representations); more SFMs (scGPT cell, Evo2 genomic);
image / histopathology via a VLM arm; Croissant metadata + a public leaderboard. The activation arm
(open-weight probe) is an optional GPU plug-in. Honest scope: pilot n per task; the specialist ceiling
is a cheap or cited model; the encoding arm is open-weight-only.
