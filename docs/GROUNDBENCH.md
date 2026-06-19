# GroundBench: evaluate your model

GroundBench measures whether a language (or vision-language) model can **verbalize a property from a
specialist representation** it is shown, across 23 tasks in 9 modalities. It is a GPU-free output-arm
benchmark: one versioned prompt per task, fixed decode, raw outputs saved, every metric with a bootstrap
CI, and **no single-number reduction**. Each task carries an a-priori `web-exposure` tag (`rich` / `zero`
/ `mixed`) that predicts, before any model call, whether the model should ground or defer to a specialist.

The contract for the task schema, adding a task, and the mandatory orientation audit is
[`GROUNDBENCH_SPEC.md`](GROUNDBENCH_SPEC.md). This file is the how-to-run.

## Evaluate your model in 3 steps

```bash
# 1. Install (output arm + cheap-head baseline; no GPU). Add ".[full]" only to regenerate signal.
pip install -e .

# 2. Provide a model. Pick ONE:
export ANTHROPIC_API_KEY=...        # for --model claude-*
export OPENAI_API_KEY=...           # for --model gpt-* / o1-* / o3-*
#   or point at any OpenAI-compatible server (vLLM / Ollama / together / groq / local):
export OPENAI_BASE_URL=http://localhost:8000/v1   # then --model oai:<served-name>

# 3. Run (writes results/benchmark/<model>/ and updates LEADERBOARD.md)
python eval/run_grounding_eval.py --model claude-opus-4-8            # full CORE set (23 tasks)
python eval/run_grounding_eval.py --model gpt-4o --tasks admet/herg,single_cell/cd8t_nk:name --n 100
python eval/run_grounding_eval.py --dry-run                          # validate the pipeline, no API
```

`--tasks` takes `core` (default), `all`, or a comma-separated list of task ids. `--n` is the balanced
sample per task (default 100). Runs are **incremental**: re-running a subset merges into the existing
scorecard, so you can add tasks or fix one without re-running the rest.

## Three ways to attach a model

1. **Built-in providers** — `claude*` (Anthropic), `gpt*` / `o1*` / `o3*` (OpenAI). Vision is handled
   automatically for the image task.
2. **Any OpenAI-compatible server** — set `OPENAI_BASE_URL` and use `--model oai:<served-name>`. This
   covers vLLM, Ollama, llama.cpp, together, groq, and most local servers, with no code changes.
3. **Bring your own** — wrap your inference in a callable and pass it in, or register a provider:

```python
from eval.run_grounding_eval import evaluate, register_provider

def my_complete(model, prompt, image=None):   # image is a PNG path or None
    return my_model.generate(prompt)           # must return a string

# one-off:
evaluate("my-model-v1", complete_fn=my_complete)

# or register so the CLI/dispatch can find it by name:
register_provider(lambda m: m.startswith("mylab:"), my_complete)
```

A text-only `complete_fn` simply skips the image task. The decode contract is fixed
(`temperature=0`, `max_tokens=16`); the system prompt forces a bare number and the parser takes the
last number in the reply, so a model that emits `Probability: 0.42` or `42%` both parse correctly.

## Run from Inspect (inspect_ai)

GroundBench is also an Inspect eval (`eval/groundbench_inspect.py`), reusing the same tasks, prompts,
and parser, so you can run it with Inspect's model providers, logging, and viewer:

```bash
pip install -e ".[inspect]"
inspect eval eval/groundbench_inspect.py@groundbench -T task_id=msa/conservation --model anthropic/claude-opus-4-8
inspect eval eval/groundbench_inspect.py@groundbench_all --model openai/gpt-4o   # all CORE, AUROC per task
```

The reported metric is per-task grounding AUROC (with the registry's a-priori orientation). For the full
scorecard (calibration, gap-to-ceiling, memo_delta, bootstrap CIs) and the leaderboard, use the
standalone harness above.

## What you get, and how to read it

Each run writes `results/benchmark/<model>/`:
- `scorecard.json` — per task: `output_auroc` + 95% bootstrap CI, `ceiling` and `gap`, calibration
  (`ece`, `aurc`, `sel_acc_50`), `memo_delta` (structural dependence, where a scrambled control exists),
  `web_exposure`, and `orientation`.
- `manifest.json` — provenance: model, prompt version, decode, seed, data commit, date, tasks run.
- `raw.jsonl` — every item's id, label, parsed probability, and raw model text, so anyone can re-score.

Read a model as a **row of per-task scores**, not one number: grounding (AUROC, gap to ceiling),
calibration (ECE / AURC), and memorization-transparency (`memo_delta`) together, alongside the web tag.
The leaderboard (`results/benchmark/LEADERBOARD.md`) sorts by the web tag: `web=zero` rows tend to sit at
chance (the representation-to-property mapping is undocumented), `web=rich` rows tend to ground.

## The cheap-head baseline (the orchestration reference)

`python eval/head_baseline.py` adds a `baseline-cheap-head` column: a reproducible, GPU-free,
API-free logistic-regression head on the **same representation** each model is shown (char n-gram
hashing for text, the raw vector for embeddings, color statistics for images). It is the
"orchestrate via a trained head" reference. On the `web=zero` rows the head often grounds where every
LLM is at chance, which is the prescription the benchmark exists to make concrete: when a model cannot
verbalize a representation, the information is usually still present and should be read out with a head,
not prompt-pasted.

## Reproducibility, scope, submission

The harness is deterministic given the data commit and seed; prompts are versioned constants. Honest
scope: pilot `n` per task; the specialist ceiling is a cheap or cited model; the open-weight activation
("encode") arm and the SFM-embedding generators (ESM, Nucleotide Transformer) live in separate scripts
and, for the SFM legs, a separate transformers-4.x environment (see `eval/sfm_embed_nt.py`). The
committed embeddings let those tasks run without that environment.

To put your model on the public leaderboard, see [`../SUBMITTING.md`](../SUBMITTING.md): run the CORE
set, validate with `python eval/validate_submission.py results/benchmark/<model>`, and open a PR adding
the `<model>/` directory. To add a new task, follow [`GROUNDBENCH_SPEC.md`](GROUNDBENCH_SPEC.md),
including the mandatory a-priori orientation audit (an inverted label silently produces fake
anti-grounding).
