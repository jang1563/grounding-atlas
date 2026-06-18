# grounding-atlas-eval: design

*2026-06-15. A model-agnostic benchmark that scores a model's grounding, calibration, and
memorization-transparency on the verifiable-signal pairs. Design is grounded in published
eval best practice. No em dashes.*

## Why this benchmark is different

Most benchmarks try to AVOID contamination. This one MEASURES it: the web-exposure law says
high scores on web-documented items are recall, not grounding. So the benchmark reports a
memorization-transparency signal alongside accuracy, turning the project's own conditions
(matched / re_notation / scrambled, web-rich / web-poor) into the "pristine vs defended"
split that contamination best practice recommends. It scores three things the field's
adjacent evals do not: does the model GROUND the property, is it CALIBRATED about it, and is
the score grounding or memorization.

## Best practice it follows

- Reproducible evaluation (EleutherAI "Lessons from the Trenches", arXiv:2405.14782):
  versioned exact prompts, fixed decode config, **raw model outputs released** (not just
  scores), per-rung breakdown with **bootstrap CI** (no single-number reduction), full
  provenance manifest, model-agnostic interface.
- Architecture (Inspect AI): clean **Dataset / Solver / Scorer** separation, one command
  across providers, structured per-sample logs.
- Calibration: ECE (report bin count; it is bias-variance sensitive) + Brier + a
  risk-coverage curve and AURC + selective accuracy at fixed coverage.
- Contamination (Transparency-Card practice): pristine + defended splits, report the
  web-exposure stratification explicitly.

## Architecture

- **Dataset**: the `signal/` pairs (`{id, modality, property, condition, representation,
  label, ...}`), already carrying the control conditions. Versioned by git commit.
- **Solver**: model-agnostic output arm. One `complete(model, prompt)` dispatch
  (Anthropic / OpenAI now, extensible), a single **versioned prompt template**, fixed decode
  (temperature 0, capped tokens). GPU-free. The activation arm (open-weight, GPU) is an
  optional plug-in, not required.
- **Scorer**, per rung, every number with bootstrap CI:
  - grounding: output AUROC, and the gap to the specialist ceiling.
  - calibration: ECE, Brier, AURC (risk-coverage), selective accuracy at 50% coverage.
  - memorization-transparency: `memo_delta` = AUROC(matched) - AUROC(scrambled), where
    scrambled shuffles the representation's characters (structure destroyed, composition
    kept). A large positive delta means the score depends on the real structure = genuine
    grounding; near zero means it rests on surface composition or chance. (A separate
    web-rich minus web-poor contrast is the one that flags recall of documented items.)

## Outputs (provenance-complete)

- `results/benchmark/<model>/scorecard.json`: per-rung metrics + CIs.
- `results/benchmark/<model>/raw.jsonl`: every item's prompt-id, label, raw model output,
  parsed probability (lets anyone re-score).
- `results/benchmark/<model>/manifest.json`: model id, prompt version, data git commit,
  seed, decode config, date, n per rung (full reproducibility).
- `LEADERBOARD.md`: per-model summary table, scores stratified, never a single number.

## Reproducibility rules (enforced in code)

Exact prompt is a versioned constant; decode is fixed; the data version is the git commit;
raw outputs are saved; every metric carries a bootstrap CI; the manifest records everything
needed to reproduce. Adding a model is `--model X`; nothing else changes.

## Build phases

1. **Output-arm harness (GPU-free), this scaffold**: `eval/run_grounding_eval.py` over the
   ceiling-bearing rungs (ADMET + computable first), full scorecard + manifest + leaderboard.
   `--dry-run` validates the pipeline with no API.
2. Specialist-ceiling computation per rung (cheap CV specialists; reuse `per_item_router`).
3. Calibration arm (ECE/AURC) wired to a self-reported confidence, not only the prediction.
4. Activation arm (open-weight, GPU) as an optional plug-in for the encoding axis.
5. Croissant metadata + a public leaderboard pass.

## Honest scope

Pilot-scale per rung; the specialist ceiling is a cheap or cited model, not exhaustive;
the activation arm is open-weight-only (closed models expose no hidden states). The point is
a reusable, contamination-transparent grounding+calibration ruler that runs on any model
release, not a single leaderboard number.
