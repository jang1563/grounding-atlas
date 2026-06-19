# Submitting a model to the GroundBench leaderboard

The leaderboard ([`results/benchmark/LEADERBOARD.md`](results/benchmark/LEADERBOARD.md)) is built from
every committed `results/benchmark/<model>/scorecard.json`. A submission is a pull request that adds one
`<model>/` directory. The harness is deterministic given the data commit and seed, so a submission is
reproducible from its `manifest.json` + `raw.jsonl`.

## Steps

```bash
# 1. Run the full CORE set on your model (see docs/GROUNDBENCH.md for provider options).
python eval/run_grounding_eval.py --model <your-model-id>

# 2. Validate the result directory (exit 0 = ready to submit).
python eval/validate_submission.py results/benchmark/<your-model-id>

# 3. Open a PR adding ONLY results/benchmark/<your-model-id>/
#    (scorecard.json + manifest.json + raw.jsonl). The leaderboard regenerates from it.
```

## What a valid submission must have

- **All CORE tasks** (the validator lists any missing). Partial runs are possible with
  `--allow-partial`, but only full-CORE entries are comparable on the main leaderboard.
- **The current prompt version** in the manifest (the validator checks it; an old version is not
  comparable, so re-run).
- **Real, not `--dry-run`** results, with `scorecard.json`, `manifest.json` (provenance: model, prompt
  version, decode, seed, data commit, date), and `raw.jsonl` (every item's id, label, parsed
  probability, raw text, so anyone can re-score).

## The rules (so the numbers mean something)

- **Do not tune the prompt or the label orientation to your model.** Prompts are versioned constants and
  each task's orientation is fixed a priori from the assay semantics (see
  [`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md)); an inverted label silently manufactures fake
  anti-grounding. If you change a prompt, you have made a new benchmark version, not a submission.
- **Report every task, no cherry-picking.** A model is a full row; there is no single-number reduction.
- **Do not train on the evaluation rows.** GroundBench measures web-exposure on purpose; training on the
  released pairs is contamination and makes a high score meaningless. The `web` tag and `memo_delta` are
  there to keep recall honest.
- **The `baseline-cheap-head` column is fixed** (`eval/head_baseline.py`); it is the orchestration
  reference, not a submittable model.

## PR checklist

- [ ] `python eval/validate_submission.py results/benchmark/<model>` exits 0.
- [ ] Only the new `results/benchmark/<model>/` directory is added; no prompts/tasks/orientations changed.
- [ ] `manifest.json` names the provider/model id and the decode, and `dry_run` is false.
- [ ] The model was not trained on the released GroundBench / grounding-atlas rows.

## Adding a task or a model provider (not a submission)

Adding a **task** is a different contribution: follow the task schema, ceiling, and **mandatory
orientation audit** in [`docs/GROUNDBENCH_SPEC.md`](docs/GROUNDBENCH_SPEC.md). Adding a **provider** (a new
model backend) needs no core change: use `register_provider(...)` or `evaluate(..., complete_fn=...)`, or
the `oai:` + `OPENAI_BASE_URL` path documented in [`docs/GROUNDBENCH.md`](docs/GROUNDBENCH.md).
