# grounding-atlas-eval results

Output of `eval/run_grounding_eval.py` (design: [`docs/BENCHMARK_DESIGN.md`](../../docs/BENCHMARK_DESIGN.md)).
A model-agnostic ruler for **grounding**, **calibration**, and **memorization-transparency**
on the verifiable-signal pairs, runnable on any model release.

## Layout

```
results/benchmark/
  LEADERBOARD.md          per-rung AUROC across scored models (stratified, no single number)
  ceilings.json           optional: {rung: specialist_ceiling} to populate the gap column (Phase 2)
  <model>/scorecard.json  per-rung AUROC + CI, ECE, AURC, selective-acc@50, gap, memo_delta
  <model>/manifest.json   model id, prompt version, data git commit, seed, decode, date, n
  <model>/raw.jsonl       every item's id, label, parsed probability (re-scorable)
```

`<model>/` for real runs is committed (provenance the design promises to release). The
synthetic `--dry-run` output (`dry/`) is gitignored: it validates the pipeline with no API.

## Reproduce

```bash
python eval/run_grounding_eval.py --dry-run                  # validate pipeline, no API
python eval/run_grounding_eval.py --model claude-opus-4-8 --rungs all --n 120
python eval/run_grounding_eval.py --model gpt-4o --rungs admet/herg,computable/smiles/n_carbon
```

Every metric carries a 95% bootstrap CI; the prompt is a versioned constant and decode is
fixed, so a re-run on the same data commit reproduces the scorecard.

## How to read it

- **grounding** — `output_auroc` and `gap` (ceiling - AUROC). High AUROC = the model
  verbalizes the property from the representation.
- **calibration** — `ece` (10-bin), `aurc` (risk-coverage), `sel_acc_50` (accuracy at 50%
  coverage). Low AURC = confidence ranks correctness, so abstention recovers accuracy.
- **memorization-transparency** — `memo_delta` = AUROC(matched) - AUROC(scrambled). A large
  positive delta flags recall of web-documented items over grounding of the representation.
