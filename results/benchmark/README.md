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
- **memorization-transparency** — `memo_delta` = AUROC(matched) - AUROC(scrambled), where
  scrambled shuffles the SMILES characters (structure destroyed, composition kept). A large
  *positive* delta means the score depends on the real structure = genuine grounding; near zero
  means it rests on surface composition or chance.

## Label-orientation audit

Each empirical endpoint's label direction is verified against an independent physicochemical
prior (`eval/audit_orientations.py`): mutagens are nitroaromatic-rich (ames), hERG blockers and
CYP inhibitors are lipophilic / basic, soluble compounds have lower logP, permeable compounds
lower TPSA. **All six rungs are consistent** after the ames correction — ames was found inverted
(`align` → `oppose`) by a structural-alert audit (`eval/analyze_ames.py`) and re-scored from raw
outputs (`eval/fix_ames_orientation.py`). A benchmark's largest risk is label provenance, so the
audit is part of the instrument.
