"""GroundBench (grounding-atlas-eval): a model-agnostic grounding + calibration +
memorization-transparency benchmark over the verifiable-signal task registry
(eval/benchmark_tasks.py; design docs/GROUNDBENCH_SPEC.md).

Dataset / Solver / Scorer (Inspect-style), reproducible per EleutherAI "Lessons from the
Trenches": one versioned prompt per task, fixed decode, raw outputs + manifest saved, every
metric with a bootstrap CI, no single-number reduction. GPU-free output arm.

One-line use:
  from eval.run_grounding_eval import evaluate
  evaluate("claude-opus-4-8")              # the CORE task set
CLI:
  python eval/run_grounding_eval.py --dry-run                       # no API, validates pipeline
  python eval/run_grounding_eval.py --model gpt-4o --tasks core --n 100
  python eval/run_grounding_eval.py --model claude-opus-4-8 --tasks admet/herg,single_cell/cd8t_nk:name
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS, task_items  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "benchmark")
PROMPT_VERSION = "v3"   # prompt content unchanged from v3; v4-structure = task registry
DECODE = {"temperature": 0.0, "max_tokens": 16}

SYSTEM = (
    "You are a property predictor. Respond with ONLY a single decimal number between 0 and 1 "
    "(for example: 0.42). No words, no explanation, no analysis, no units. Your entire reply "
    "must be just the number."
)


# ---------- Solver (model-agnostic) ----------

def _b64(path):
    import base64
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def _anthropic(model, prompt, image=None):
    import anthropic
    client = anthropic.Anthropic()
    content = prompt if image is None else [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(image)}},
        {"type": "text", "text": prompt}]
    kw = dict(model=model, max_tokens=DECODE["max_tokens"], system=SYSTEM,
              messages=[{"role": "user", "content": content}])
    try:
        m = client.messages.create(temperature=DECODE["temperature"], **kw)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise
        m = client.messages.create(**kw)   # newer models (e.g. opus-4-8) deprecate temperature
    return "".join(b.text for b in m.content if b.type == "text")


def _openai(model, prompt, image=None, base_url=None):
    import openai
    client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
    user = prompt if image is None else [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}}]
    r = client.chat.completions.create(
        model=model, max_tokens=DECODE["max_tokens"], temperature=DECODE["temperature"],
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content


# Ordered (predicate, handler) registry. Handler signature: (model, prompt, image) -> str.
# Built-ins cover Anthropic, OpenAI, and ANY OpenAI-compatible server (vLLM / Ollama / together /
# groq / a local llama.cpp endpoint, ...) via the `oai:` prefix plus the OPENAI_BASE_URL env var,
# e.g. --model oai:meta-llama/Llama-3.1-8B-Instruct with OPENAI_BASE_URL=http://localhost:8000/v1.
PROVIDERS = [
    (lambda m: m.startswith(("claude", "anthropic")), _anthropic),
    (lambda m: m.startswith("oai:"),
     lambda m, p, i=None: _openai(m[4:], p, i, base_url=os.environ.get("OPENAI_BASE_URL"))),
    (lambda m: m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")), _openai),
]


def register_provider(predicate, handler, front=True):
    """Add a provider so you can evaluate ANY model without editing this file.

    predicate(model:str)->bool selects which model ids you handle; handler(model, prompt, image)->str
    runs inference (image is a PNG path or None). front=True gives it priority over the built-ins.
    For a one-off, prefer evaluate(..., complete_fn=your_fn). See docs/GROUNDBENCH.md.
    """
    PROVIDERS.insert(0 if front else len(PROVIDERS), (predicate, handler))


def complete(model, prompt, image=None):
    for pred, fn in PROVIDERS:
        if pred(model):
            return fn(model, prompt, image)
    raise SystemExit(
        f"No provider for model '{model}'. Built-ins: Anthropic (claude*), OpenAI (gpt*/o1*/o3*), "
        "and any OpenAI-compatible server ('oai:<name>' + OPENAI_BASE_URL). For a local or custom "
        "model, register_provider(pred, handler) or call evaluate(model, complete_fn=your_fn).")


def parse_prob(text):
    """Anchored parse: take the LAST number; [0,1] is a probability, (1,100] a percent."""
    for tok in reversed(re.findall(r"\d*\.?\d+", text or "")):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def solve(model, items, prompt_tmpl, dry, rng, complete_fn=None):
    """Returns (probs, raw_texts). Raw text is saved so anyone can re-score. complete_fn overrides the
    built-in provider dispatch (bring-your-own model: a callable (model, prompt, image=None) -> str)."""
    fn = complete_fn or complete
    probs, texts = [], []
    for it in items:
        if dry:
            p = min(1.0, max(0.0, 0.30 + 0.40 * int(it["label"]) + rng.normal(0, 0.18)))
            probs.append(p); texts.append(f"{p:.3f} (dry)")
        else:
            t = fn(model, prompt_tmpl.format(rep=it.get("rep", "")), image=it.get("image"))
            texts.append(t); probs.append(parse_prob(t))
    return np.array(probs), texts


# ---------- Scorer ----------

def ece(prob, y, bins=10):
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob > 0.5).astype(int) == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def aurc(prob, y):
    conf = np.maximum(prob, 1 - prob)
    err = ((prob > 0.5).astype(int) != y).astype(float)
    order = np.argsort(-conf)
    return float((np.cumsum(err[order]) / (np.arange(len(err)) + 1)).mean())


def sel_acc(prob, y, cov=0.5):
    conf = np.maximum(prob, 1 - prob)
    top = np.argsort(-conf)[:max(1, int(len(y) * cov))]
    return float(((prob[top] > 0.5).astype(int) == y[top]).mean())


def auroc(prob, y):
    return roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")


def ci(fn, prob, y, rng, b=500):
    vals = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(prob[idx], y[idx]))
        except Exception:
            pass
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return round(float(lo), 3), round(float(hi), 3)


def score_task(prob, y, scr_prob, scr_y, ceiling, rng):
    a = auroc(prob, y)
    rec = {"n": len(y), "output_auroc": round(a, 3), "output_auroc_ci": ci(auroc, prob, y, rng),
           "ece": round(ece(prob, y), 3), "aurc": round(aurc(prob, y), 3),
           "sel_acc_50": round(sel_acc(prob, y), 3),
           "ceiling": ceiling, "gap": round(ceiling - a, 3) if ceiling is not None else None,
           "memo_delta": None}
    if scr_y is not None and len(scr_y) and len(set(scr_y)) > 1:
        rec["memo_delta"] = round(a - auroc(scr_prob, scr_y), 3)
    return rec


def update_leaderboard():
    models = []
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        sc = os.path.join(d, "scorecard.json")
        name = os.path.basename(d)
        if name != "dry" and os.path.isfile(sc):
            models.append((name, json.load(open(sc))))
    out = ["# GroundBench leaderboard", "",
           "Per-task output-arm AUROC (95% bootstrap CI), with the a-priori web-exposure tag. No "
           "single-number reduction: read grounding (AUROC, gap to ceiling), calibration (ECE / "
           "AURC), and memorization-transparency (memo_delta) together. Generated by "
           "`eval/run_grounding_eval.py` over `eval/benchmark_tasks.py`.", "",
           "The `baseline-cheap-head` column is not an LLM: it is a reproducible cheap-featurizer "
           "logistic-regression head on the same representation (`eval/head_baseline.py`), the "
           "orchestrate-via-a-trained-head reference. Where it grounds but the models are at chance "
           "(the web-zero rows), the information is present in the representation and should be "
           "orchestrated with a head, not prompt-pasted.", ""]
    if not models:
        out.append("_No models scored yet. Run `python eval/run_grounding_eval.py --model <id>`._")
    else:
        tasks = sorted({t for _, sc in models for t in sc})
        out += ["| task | web | ceiling | " + " | ".join(f"{m} AUROC [CI]" for m, _ in models) + " |",
                "|" + "---|" * (len(models) + 3)]
        for t in tasks:
            web = next((sc[t].get("web_exposure", "") for _, sc in models if t in sc), "")
            ceil = next((sc[t]["ceiling"] for _, sc in models
                         if t in sc and sc[t].get("ceiling") is not None), None)
            cells = []
            for _, sc in models:
                r = sc.get(t)
                cells.append(f"{r['output_auroc']} {tuple(r['output_auroc_ci'])}" if r else "-")
            out.append(f"| `{t}` | {web} | {ceil if ceil is not None else '-'} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "LEADERBOARD.md"), "w").write("\n".join(out) + "\n")


def _ceilings():
    cf = os.path.join(OUT, "ceilings.json")
    if not os.path.exists(cf):
        return {}
    return {k: (v["ceiling"] if isinstance(v, dict) else v) for k, v in json.load(open(cf)).items()}


def evaluate(model="dry", tasks=None, n=100, seed=0, dry=False, merge=True, complete_fn=None):
    """Run GroundBench for one model over a list of task ids (default CORE). With merge=True the
    run's tasks are ADDED to any existing scorecard (incremental: add a task without re-running the
    rest). complete_fn=(model, prompt, image=None)->str evaluates ANY model with no code edits
    (bring-your-own; otherwise the PROVIDERS dispatch handles the model id). Writes
    results/benchmark/<model>/{scorecard,manifest,raw} + LEADERBOARD.md; returns the scorecard."""
    rng = np.random.default_rng(seed)
    task_ids = list(CORE) if tasks in (None, "core", "all") else tasks
    ceilings = _ceilings()
    model_dir = os.path.join(OUT, model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    scorecard, raw = {}, []
    for tid in task_ids:
        if tid not in TASKS:
            print(f"  skip {tid} (not in registry)"); continue
        t = TASKS[tid]
        items, scr = task_items(tid, n, rng)
        if not items:
            continue
        prob, texts = solve(model, items, t["prompt"], dry, rng, complete_fn)
        y = np.array([it["label"] for it in items])
        sprob = solve(model, scr, t["prompt"], dry, rng, complete_fn)[0] if scr else None
        sy = np.array([it["label"] for it in scr]) if scr else None
        if t["orient"] == "oppose":
            y = 1 - y
            sy = None if sy is None else 1 - sy
        ceil = ceilings.get(t["ceiling"]) if isinstance(t["ceiling"], str) else t["ceiling"]
        rec = score_task(prob, y, sprob, sy, ceil, rng)
        rec["orientation"], rec["web_exposure"] = t["orient"], t["web"]
        scorecard[tid] = rec
        for it, p, tx in zip(items, prob, texts):
            raw.append({"task": tid, "id": it["id"], "label": int(it["label"]),
                        "prob": round(float(p), 4), "output": tx})
        print(f"  {tid:28s} web={t['web']:4s} AUROC={rec['output_auroc']} {rec['output_auroc_ci']} "
              f"ECE={rec['ece']} memo_delta={rec['memo_delta']}", flush=True)

    scp, rawp = os.path.join(model_dir, "scorecard.json"), os.path.join(model_dir, "raw.jsonl")
    full = scorecard
    if merge and os.path.exists(scp):
        full = json.load(open(scp))
        full.update(scorecard)                 # this run's tasks override / extend
    if merge and os.path.exists(rawp):
        kept = [r for r in (json.loads(line) for line in open(rawp)) if r.get("task") not in scorecard]
        raw = kept + raw
    manifest = {"model": model, "prompt_version": PROMPT_VERSION, "decode": DECODE, "seed": seed,
                "n_per_task": n, "dry_run": dry, "tasks": list(full), "last_run": list(scorecard),
                "data_commit": subprocess.getoutput("git rev-parse --short HEAD"),
                "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    json.dump(full, open(scp, "w"), indent=2)
    json.dump(manifest, open(os.path.join(model_dir, "manifest.json"), "w"), indent=2)
    with open(rawp, "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    update_leaderboard()
    print(f"\nwrote {model_dir}/  [{len(full)} tasks total, commit {manifest['data_commit']}, prompt {PROMPT_VERSION}]")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dry")
    ap.add_argument("--tasks", default="core", help="'core', 'all', or a comma list of task ids")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tasks = args.tasks if args.tasks in ("core", "all") else [t.strip() for t in args.tasks.split(",")]
    evaluate(model=args.model, tasks=tasks, n=args.n, seed=args.seed, dry=args.dry_run)


if __name__ == "__main__":
    main()
