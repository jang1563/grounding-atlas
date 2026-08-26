# Submitting a model to the GroundBench leaderboard

The leaderboard ([`results/benchmark/LEADERBOARD.md`](results/benchmark/LEADERBOARD.md)) is built from
every committed `results/benchmark/<model>/scorecard.json`. A submission is a pull request that adds one
`<model>__v5/` directory. The harness is deterministic given the data commit and seed, so a submission is
reproducible from its `manifest.json`, `raw.jsonl`, and registered paired-comparison artifact.

## Steps

```bash
# 1. Run the full CORE set on your model (see docs/GROUNDBENCH.md for provider options).
python eval/run_grounding_eval.py --model <your-model-id> \
  --provider <provider> --model-revision <immutable-provider-or-checkpoint-revision>

# 2. Validate the result directory (exit 0 = ready to submit).
python eval/validate_submission.py results/benchmark/<your-model-id>__v5

# 3. Open a PR adding ONLY results/benchmark/<your-model-id>__v5/
#    (scorecard.json + manifest.json + raw.jsonl + pair_group_comparisons.json).
#    The leaderboard regenerates from it.
```

## What a valid submission must have

- **All 13 active CORE tasks** (the validator lists any missing). The 10 exploratory tasks—the
  provisional ADMET/hERG assay renderings and the ESM task—can be requested explicitly; the invalid
  variant-sequence task is quarantined. Partial runs are possible with `--allow-partial`, but only
  full-CORE entries are comparable on the main leaderboard.
- **The current prompt version** in the manifest (the validator checks it; an old version is not
  comparable, so re-run).
- **Real, not `--dry-run`** results, with `scorecard.json`, `manifest.json` (provenance: model, prompt
  version, immutable revision, provider, decode, seed, data/task checksums, clean code-data state,
  date, registry status, and sampling contract), and `raw.jsonl`
  (source ID, entity ID and scope, condition, source and oriented target labels, parse status,
  probability, and raw text, so anyone can re-score), plus `pair_group_comparisons.json`
  (entity-paired name/anonymous or alternate-representation contrasts). Snapshot-local IDs are not
  upstream-resolvable biological record IDs.

## The rules (so the numbers mean something)

- **Do not tune the prompt or the label orientation to your model.** Prompts are versioned constants and
  each task's orientation is fixed a priori from the assay semantics (see
  [`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md)); an inverted label silently manufactures fake
  anti-grounding. If you change a prompt, you have made a new benchmark version, not a submission.
- **Report every task, no cherry-picking.** A model is a full row; there is no single-number reduction.
- **Do not train on the evaluation rows.** Training on released pairs makes a high score uninterpretable.
  The `web` tag is descriptive metadata, and `memo_delta` is a paired input-perturbation diagnostic—not
  proof of memorization or absence of contamination.
- **The `baseline-cheap-head` column is a non-submittable diagnostic** (`eval/head_baseline.py`).
  Its random-CV estimate is an optimistic surface-readability pilot, not a ceiling or a routing rule.

## PR checklist

- [ ] `python eval/validate_submission.py results/benchmark/<model>__v5` exits 0.
- [ ] Only the new `results/benchmark/<model>__v5/` directory is added; no prompts/tasks/orientations changed.
- [ ] `manifest.json` names the provider/model id and the decode, and `dry_run` is false.
- [ ] The model was not trained on the released GroundBench / grounding-atlas rows.

## Adding a task or a model provider (not a submission)

Adding a **task** is a different contribution: follow the task schema, input-matched reference score,
truth-level/provenance contract, stable entity IDs, and **mandatory orientation audit** in
[`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md). Adding a **provider** (a new
model backend) needs no core change: use `register_provider(...)` or `evaluate(..., complete_fn=...)`, or
the `oai:` + `OPENAI_BASE_URL` path documented in [`docs/GROUNDBENCH.md`](docs/GROUNDBENCH.md).
