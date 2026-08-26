"""GroundBench (grounding-atlas-eval): a model-agnostic grounding + calibration +
memorization-transparency benchmark over the verifiable-signal task registry
(eval/benchmark_tasks.py; design docs/GROUNDBENCH_SPEC.md).

Dataset / Solver / Scorer (Inspect-style), reproducible per EleutherAI "Lessons from the
Trenches": one versioned prompt per task, fixed decode, raw outputs + manifest saved, primary
discrimination and proper-score metrics with bootstrap CIs, no single-number reduction.
Current CIs use an item-level pilot bootstrap; a release analysis still needs biological
dependency-group identifiers. GPU-free output arm.

One-line use:
  from eval.run_grounding_eval import evaluate
  evaluate("claude-opus-4-8")              # the CORE task set
CLI:
  python eval/run_grounding_eval.py --dry-run                       # no API, validates pipeline
  python eval/run_grounding_eval.py --model gpt-4o --tasks core --n 100
  python eval/run_grounding_eval.py --model claude-opus-4-8 --tasks admet/herg,single_cell/cd8t_nk:name
"""
import argparse
import csv
import glob
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import (  # noqa: E402
    CORE,
    EXPLORATORY,
    QUARANTINED,
    TASKS,
    TRUTH_TAXONOMY_VERSION,
    GroundBenchSampler,
)

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "benchmark")
PROMPT_VERSION = "v5"   # target-contract, entity-pairing, and truth-level correction
# Schema 5 separates the machine-facing T0-T5 truth-level code from descriptive target-source
# provenance. Prompt v5 is unchanged. Earlier artifacts remain historical and are intentionally
# excluded from merge/leaderboard validation.
SCORE_SCHEMA = 5
METADATA_CONTRACT_VERSION = "phase0-v2"
DECODE = {"temperature": 0.0, "max_tokens": 16}

SYSTEM = (
    "You are a property predictor. Respond with ONLY a single decimal number between 0 and 1 "
    "(for example: 0.42). No words, no explanation, no analysis, no units. Your entire reply "
    "must be just the number."
)


def _task_rng(seed, task_id, purpose):
    """Deterministic task-local RNG, independent of task execution order."""
    key = f"{int(seed)}\0{task_id}\0{purpose}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def _repository_state():
    """Fingerprint tracked code/data state while excluding result files written by a run."""
    paths = ["eval", "signal", "dataset", "pyproject.toml"]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=ROOT,
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *paths],
        cwd=ROOT,
    )
    clean = not status.strip()
    state_material = commit.encode() + b"\0" + status + b"\0" + diff
    return {
        "data_commit": commit,
        "working_tree_clean": clean,
        "code_data_fingerprint": hashlib.sha256(state_material).hexdigest(),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_data_checksums():
    """Hash every registered source file and the image payloads referenced by image tasks."""
    checksums = {}
    for task_id, task in TASKS.items():
        data_path = os.path.join(ROOT, "signal", task["data"])
        digest = hashlib.sha256()
        digest.update(task["data"].encode())
        digest.update(b"\0")
        digest.update(_sha256_file(data_path).encode())
        if task["kind"] == "image":
            with open(data_path, newline="") as handle:
                for row in csv.DictReader(handle):
                    relative_image_path = row[task["col"]]
                    digest.update(b"\0")
                    digest.update(relative_image_path.encode())
                    digest.update(b"\0")
                    digest.update(_sha256_file(os.path.join(ROOT, relative_image_path)).encode())
        checksums[task_id] = digest.hexdigest()
    return checksums


def _run_provenance(model, provider, model_revision):
    dataset_path = os.path.join(ROOT, "dataset", "groundbench.parquet")
    reference_path = os.path.join(OUT, "ceilings.json")
    return {
        "provider": provider or (
            "custom"
            if model.startswith("custom:")
            else "anthropic"
            if model.startswith(("claude", "anthropic"))
            else "openai-compatible"
            if model.startswith("oai:")
            else "openai"
            if model.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))
            else "custom"
        ),
        "model_revision": model_revision or "unspecified",
        "dataset_sha256": _sha256_file(dataset_path) if os.path.isfile(dataset_path) else None,
        "task_data_sha256": _task_data_checksums(),
        "reference_registry_sha256": (
            _sha256_file(reference_path) if os.path.isfile(reference_path) else None
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


# ---------- Solver (model-agnostic) ----------

def _b64(path):
    import base64
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def _anthropic(model, prompt, image=None, system=SYSTEM):
    import anthropic
    client = anthropic.Anthropic()
    content = prompt if image is None else [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(image)}},
        {"type": "text", "text": prompt}]
    kw = dict(model=model, max_tokens=DECODE["max_tokens"], system=system,
              messages=[{"role": "user", "content": content}])
    try:
        m = client.messages.create(temperature=DECODE["temperature"], **kw)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise
        m = client.messages.create(**kw)   # newer models (e.g. opus-4-8) deprecate temperature
    return "".join(b.text for b in m.content if b.type == "text")


def _openai(model, prompt, image=None, base_url=None, system=SYSTEM):
    import openai
    client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
    user = prompt if image is None else [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}}]
    r = client.chat.completions.create(
        model=model, max_tokens=DECODE["max_tokens"], temperature=DECODE["temperature"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content


# Ordered (predicate, handler) registry. Handler signature: (model, prompt, image) -> str.
# Built-ins cover Anthropic, OpenAI, and ANY OpenAI-compatible server (vLLM / Ollama / together /
# groq / a local llama.cpp endpoint, ...) via the `oai:` prefix plus the OPENAI_BASE_URL env var,
# e.g. --model oai:meta-llama/Llama-3.1-8B-Instruct with OPENAI_BASE_URL=http://localhost:8000/v1.
PROVIDERS = [
    (lambda m: m.startswith(("claude", "anthropic")), _anthropic),
    (lambda m: m.startswith("oai:"),
     lambda m, p, i=None, system=SYSTEM: _openai(
         m[4:],
         p,
         i,
         base_url=os.environ.get("OPENAI_BASE_URL"),
         system=system,
     )),
    (lambda m: m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")), _openai),
]


def register_provider(predicate, handler, front=True):
    """Add a provider so you can evaluate ANY model without editing this file.

    predicate(model:str)->bool selects which model ids you handle; handler(model, prompt, image)->str
    runs inference (image is a PNG path or None). front=True gives it priority over the built-ins.
    For a one-off, prefer evaluate(..., complete_fn=your_fn). See docs/GROUNDBENCH.md.
    """
    PROVIDERS.insert(0 if front else len(PROVIDERS), (predicate, handler))


def complete(model, prompt, image=None, system=None):
    for pred, fn in PROVIDERS:
        if pred(model):
            if system is None:
                return fn(model, prompt, image)
            try:
                return fn(model, prompt, image, system=system)
            except TypeError as error:
                raise TypeError(
                    "the selected custom provider does not accept a system= override required "
                    "by this evaluation arm"
                ) from error
    raise SystemExit(
        f"No provider for model '{model}'. Built-ins: Anthropic (claude*), OpenAI (gpt*/o1*/o3*), "
        "and any OpenAI-compatible server ('oai:<name>' + OPENAI_BASE_URL). For a local or custom "
        "model, register_provider(pred, handler) or call evaluate(model, complete_fn=your_fn).")


def parse_prob_with_status(text):
    """Return ``(probability, parsed)`` under the prompt's exact-output contract.

    Invalid/refusal responses retain the historical neutral value 0.5 for score compatibility,
    but are explicitly marked unparsed and reported separately. Explanations, multiple numbers,
    bare percentages, and out-of-range values are noncompliant rather than heuristically repaired.
    """
    token = (text or "").strip()
    if re.fullmatch(r"(?:0(?:\.\d*)?|1(?:\.0*)?|\.\d+)", token):
        value = float(token)
        if 0.0 <= value <= 1.0:
            return value, True
    return 0.5, False


def parse_prob(text):
    """Backward-compatible probability-only parser."""
    return parse_prob_with_status(text)[0]


def solve(model, items, prompt_tmpl, dry, rng, complete_fn=None):
    """Return probabilities, raw text, and parse-validity flags for every item."""
    fn = complete_fn or complete
    probs, texts, parsed = [], [], []
    for it in items:
        if dry:
            target_label = int(it.get("target_label", it["label"]))
            p = min(1.0, max(0.0, 0.30 + 0.40 * target_label + rng.normal(0, 0.18)))
            probs.append(p)
            texts.append(repr(float(p)))
            parsed.append(True)
        else:
            t = fn(model, prompt_tmpl.format(rep=it.get("rep", "")), image=it.get("image"))
            p, ok = parse_prob_with_status(t)
            texts.append(t)
            probs.append(p)
            parsed.append(ok)
    return np.array(probs), texts, np.asarray(parsed, dtype=bool)


# ---------- Scorer ----------

def ece(prob, y, bins=10):
    """Legacy top-label confidence ECE (retained for scorecard compatibility)."""
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob > 0.5).astype(int) == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def probability_ece(prob, y, bins=10):
    """Binary probability calibration error: |mean prediction - empirical prevalence| per bin."""
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        right = prob <= edges[i + 1] if i == bins - 1 else prob < edges[i + 1]
        mask = (prob >= edges[i]) & right
        if mask.any():
            e += mask.mean() * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return float(e)


def aurc(prob, y):
    """Tie-averaged area under the risk-coverage curve.

    Confidence ties are averaged over all within-tie orderings, so the result cannot depend on CSV
    row order or class-blocked source files.
    """
    conf = np.maximum(prob, 1 - prob)
    err = ((prob > 0.5).astype(int) != y).astype(float)
    cumulative_n, cumulative_errors = 0, 0.0
    risks = []
    for confidence in sorted(np.unique(conf), reverse=True):
        group_errors = err[conf == confidence]
        group_n = len(group_errors)
        error_rate = float(group_errors.mean())
        for rank_in_group in range(1, group_n + 1):
            expected_errors = cumulative_errors + rank_in_group * error_rate
            risks.append(expected_errors / (cumulative_n + rank_in_group))
        cumulative_n += group_n
        cumulative_errors += float(group_errors.sum())
    return float(np.mean(risks))


def sel_acc(prob, y, cov=0.5):
    """Tie-averaged accuracy at the requested confidence coverage."""
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob > 0.5).astype(int) == y).astype(float)
    target_n = max(1, int(np.ceil(len(y) * cov)))
    selected_n, expected_correct = 0, 0.0
    for confidence in sorted(np.unique(conf), reverse=True):
        group_correct = correct[conf == confidence]
        take = min(len(group_correct), target_n - selected_n)
        expected_correct += take * float(group_correct.mean())
        selected_n += take
        if selected_n == target_n:
            break
    return float(expected_correct / selected_n)


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


def paired_auroc_delta_ci(prob, scr_prob, y, rng, b=500):
    """Paired item bootstrap CI for AUROC(matched) - AUROC(scrambled)."""
    vals = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(auroc(prob[idx], y[idx]) - auroc(scr_prob[idx], y[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def _brier(prob, y):
    return float(brier_score_loss(y, prob))


def _log_loss(prob, y):
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(log_loss(y, np.column_stack([1 - clipped, clipped]), labels=[0, 1]))


def score_task(prob, y, scr_prob, scr_y, reference, rng, parsed=None, scr_parsed=None,
               matched_memo_prob=None, reference_comparability="context_only_external_cohort_or_split"):
    a = auroc(prob, y)
    parsed = np.ones(len(y), dtype=bool) if parsed is None else np.asarray(parsed, dtype=bool)
    rec = {"n": len(y), "output_auroc": round(a, 3), "output_auroc_ci": ci(auroc, prob, y, rng),
           "brier": round(_brier(prob, y), 3), "brier_ci": ci(_brier, prob, y, rng),
           "log_loss": round(_log_loss(prob, y), 3), "log_loss_ci": ci(_log_loss, prob, y, rng),
           "probability_ece": round(probability_ece(prob, y), 3),
           "confidence_ece": round(ece(prob, y), 3), "ece": round(ece(prob, y), 3),
           "aurc": round(aurc(prob, y), 3),
           "sel_acc_50": round(sel_acc(prob, y), 3),
           "valid_response_rate": round(float(parsed.mean()), 3),
           "n_invalid": int((~parsed).sum()),
           "sample_prevalence": round(float(np.mean(y)), 6),
           "calibration_scope": "balanced_benchmark_distribution_only",
           "uncertainty_method": "iid_entity_item_bootstrap_pilot",
           "uncertainty_unit": "entity_id_row_not_biological_dependency_cluster",
           "reference_score": reference,
           "reference_comparability": reference_comparability,
           "reference_gap": (
               round(reference - a, 3)
               if reference is not None and reference_comparability == "same_entities_same_split"
               else None
           ),
           # Deprecated aliases retained so historical consumers do not break.
           "ceiling": reference,
           "gap": (
               round(reference - a, 3)
               if reference is not None and reference_comparability == "same_entities_same_split"
               else None
           ),
           "memo_n": 0, "memo_delta": None, "memo_delta_ci": None}
    if parsed.any() and len(set(y[parsed])) > 1:
        rec["output_auroc_valid_only"] = round(auroc(prob[parsed], y[parsed]), 3)
    else:
        rec["output_auroc_valid_only"] = None
    if scr_y is not None and len(scr_y) and len(set(scr_y)) > 1:
        if matched_memo_prob is None:
            if len(scr_y) != len(y) or not np.array_equal(np.asarray(scr_y), np.asarray(y)):
                raise ValueError("memo_delta requires explicitly paired matched predictions")
            matched_memo_prob = prob
        matched_memo_prob = np.asarray(matched_memo_prob)
        if len(matched_memo_prob) != len(scr_y):
            raise ValueError("matched and scrambled memo predictions must have identical lengths")
        rec["memo_n"] = int(len(scr_y))
        rec["memo_delta"] = round(auroc(matched_memo_prob, scr_y) - auroc(scr_prob, scr_y), 3)
        rec["memo_delta_ci"] = paired_auroc_delta_ci(matched_memo_prob, scr_prob, scr_y, rng)
        scr_parsed = np.ones(len(scr_y), dtype=bool) if scr_parsed is None else np.asarray(scr_parsed, dtype=bool)
        rec["scrambled_valid_response_rate"] = round(float(scr_parsed.mean()), 3)
    return rec


def build_pair_group_comparisons(scorecard, raw, seed):
    """Compute registered cross-task paired representation comparisons.

    These are entity-paired descriptive comparisons, not independent task-score subtraction.
    Their pilot CIs resample entity IDs and do not claim biological-cluster uncertainty.
    """
    matched_by_task = {}
    for row in raw:
        if row["condition"] == "matched" and row["task"] in scorecard:
            matched_by_task.setdefault(row["task"], {})[row["entity_id"]] = row

    groups = {}
    for task_id in scorecard:
        pair_group = TASKS[task_id].get("pair_group")
        if pair_group and task_id in matched_by_task:
            groups.setdefault(pair_group, []).append(task_id)

    comparisons = []
    for pair_group, task_ids in sorted(groups.items()):
        for task_a, task_b in itertools.combinations(sorted(task_ids), 2):
            rows_a = matched_by_task[task_a]
            rows_b = matched_by_task[task_b]
            if set(rows_a) != set(rows_b):
                raise ValueError(
                    f"pair_group {pair_group}: {task_a} and {task_b} do not share entity IDs"
                )
            entity_ids = sorted(rows_a)
            y_a = np.asarray([rows_a[entity_id]["target_label"] for entity_id in entity_ids])
            y_b = np.asarray([rows_b[entity_id]["target_label"] for entity_id in entity_ids])
            source_a = [rows_a[entity_id]["source_label"] for entity_id in entity_ids]
            source_b = [rows_b[entity_id]["source_label"] for entity_id in entity_ids]
            if not np.array_equal(y_a, y_b) or source_a != source_b:
                raise ValueError(
                    f"pair_group {pair_group}: {task_a} and {task_b} do not share labels"
                )
            prob_a = np.asarray([rows_a[entity_id]["prob"] for entity_id in entity_ids])
            prob_b = np.asarray([rows_b[entity_id]["prob"] for entity_id in entity_ids])
            if len(set(y_a)) > 1:
                auc_a = auroc(prob_a, y_a)
                auc_b = auroc(prob_b, y_a)
                delta = auc_a - auc_b
                delta_ci = paired_auroc_delta_ci(
                    prob_a,
                    prob_b,
                    y_a,
                    _task_rng(seed, f"{pair_group}:{task_a}:{task_b}", "pair-group-bootstrap"),
                )
            else:
                auc_a = auc_b = delta = None
                delta_ci = None
            comparisons.append({
                "pair_group": pair_group,
                "task_a": task_a,
                "task_b": task_b,
                "n": len(entity_ids),
                "task_a_auroc": None if auc_a is None else round(auc_a, 3),
                "task_b_auroc": None if auc_b is None else round(auc_b, 3),
                "task_a_minus_task_b_auroc": None if delta is None else round(delta, 3),
                "task_a_minus_task_b_auroc_ci": delta_ci,
                "uncertainty_method": "paired_iid_entity_bootstrap_pilot",
                "uncertainty_unit": "entity_id_not_biological_dependency_cluster",
            })
    return {
        "comparison_schema": 1,
        "prompt_version": PROMPT_VERSION,
        "comparisons": comparisons,
    }


def update_leaderboard():
    models = []
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        sc = os.path.join(d, "scorecard.json")
        manifest_path = os.path.join(d, "manifest.json")
        name = os.path.basename(d)
        if not os.path.isfile(manifest_path):
            continue
        manifest = json.load(open(manifest_path))
        if (
            name != f"dry__{PROMPT_VERSION}"
            and manifest.get("prompt_version") == PROMPT_VERSION
            and manifest.get("score_schema") == SCORE_SCHEMA
            and os.path.isfile(sc)
        ):
            models.append((name, json.load(open(sc))))
    out = ["# GroundBench leaderboard", "",
           "Per-task native-output AUROC (95% bootstrap CI), with the descriptive a-priori web tag. No "
           "single-number reduction: read output AUROC, an input-matched reference score where "
           "available, calibration/proper scores, invalid-response rate, and paired perturbation delta "
           "together. Generated by "
           "`eval/run_grounding_eval.py` over `eval/benchmark_tasks.py`.", "",
           "Calibration and proper scores describe the deliberately balanced benchmark sample, not "
           "deployment prevalence. Current intervals use an IID entity-item pilot bootstrap; they are "
           "not release-grade biological-cluster intervals.", "",
           "The `baseline-cheap-head` column is not an LLM: it is a reproducible cheap-featurizer "
           "logistic-regression head on the evaluated representation (`eval/head_baseline.py`). It measures "
           "supervised representation-label predictability, not an upper bound or latent knowledge.", "",
           "The `web` tag is a descriptive hypothesis, not a law. The current admissible artifact "
           f"contract is prompt {PROMPT_VERSION}, score schema {SCORE_SCHEMA}, metadata "
           f"{METADATA_CONTRACT_VERSION}. It enforces shared entity sampling for registered paired "
           "representations and explicit biological-question, task-family, split-group, intervention-pair, "
           "and descriptive factor metadata. Earlier artifacts are historical and are excluded rather than "
           "silently pooled. See docs/GROUNDBENCH_SPEC.md.", ""]
    if not models:
        out.append("_No models scored yet. Run `python eval/run_grounding_eval.py --model <id>`._")
    else:
        tasks = sorted({t for _, sc in models for t in sc})
        out += ["| task | web | reference | " + " | ".join(f"{m} AUROC [CI]" for m, _ in models) + " |",
                "|" + "---|" * (len(models) + 3)]
        for t in tasks:
            web = next((sc[t].get("web_exposure", "") for _, sc in models if t in sc), "")
            reference = next((
                sc[t].get("reference_score", sc[t].get("ceiling"))
                for _, sc in models
                if t in sc and sc[t].get("reference_score", sc[t].get("ceiling")) is not None
            ), None)
            cells = []
            for _, sc in models:
                r = sc.get(t)
                cells.append(f"{r['output_auroc']} {tuple(r['output_auroc_ci'])}" if r else "-")
            out.append(
                f"| `{t}` | {web} | {reference if reference is not None else '-'} | " + " | ".join(cells) + " |"
            )
    open(os.path.join(OUT, "LEADERBOARD.md"), "w").write("\n".join(out) + "\n")


def _ceilings():
    cf = os.path.join(OUT, "ceilings.json")
    if not os.path.exists(cf):
        return {}
    return {k: (v["ceiling"] if isinstance(v, dict) else v) for k, v in json.load(open(cf)).items()}


def evaluate(model="dry", tasks=None, n=100, seed=0, dry=False, merge=True, complete_fn=None,
             provider=None, model_revision=None):
    """Run GroundBench for one model over a list of task ids (default CORE). With merge=True the
    run's tasks are ADDED to any existing scorecard (incremental: add a task without re-running the
    rest). complete_fn=(model, prompt, image=None)->str evaluates ANY model with no code edits
    (bring-your-own; otherwise the PROVIDERS dispatch handles the model id). Writes
    results/benchmark/<model>/{scorecard,manifest,raw} + LEADERBOARD.md; returns the scorecard."""
    if tasks in (None, "core"):
        task_ids = list(CORE)
    elif tasks == "all":
        task_ids = list(CORE) + list(EXPLORATORY)
    else:
        task_ids = list(tasks)
    repository_state = _repository_state()
    run_provenance = _run_provenance(model, provider, model_revision)
    references = _ceilings()
    sampler = GroundBenchSampler(seed=seed)
    model_dir = os.path.join(OUT, f"{model.replace('/', '_')}__{PROMPT_VERSION}")
    os.makedirs(model_dir, exist_ok=True)
    existing_manifest_path = os.path.join(model_dir, "manifest.json")
    existing_scorecard_path = os.path.join(model_dir, "scorecard.json")
    if merge and os.path.exists(existing_scorecard_path):
        if not os.path.exists(existing_manifest_path):
            raise ValueError("cannot merge: existing scorecard has no manifest; use merge=False")
        existing_manifest = json.load(open(existing_manifest_path))
        current_contract = {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "score_schema": SCORE_SCHEMA,
            "metadata_contract_version": METADATA_CONTRACT_VERSION,
            "truth_taxonomy_version": TRUTH_TAXONOMY_VERSION,
            "decode": DECODE,
            "seed": seed,
            "n_per_task": n,
            "dry_run": dry,
            **repository_state,
            **run_provenance,
        }
        mismatches = [
            key for key, value in current_contract.items()
            if existing_manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "cannot merge runs with different contracts "
                f"({', '.join(mismatches)}); use merge=False or a distinct model id"
            )
    scorecard, raw = {}, []
    for tid in task_ids:
        if tid not in TASKS:
            print(f"  skip {tid} (not in registry)")
            continue
        t = TASKS[tid]
        if t["status"] == "quarantined":
            raise ValueError(f"{tid} is quarantined: {t['status_reason']}")
        condition_items = sampler.task_condition_items(tid, n)
        items = condition_items["matched"]
        if not items:
            continue

        target_by_condition = {}
        for condition, rows in condition_items.items():
            labels = np.asarray([item["label"] for item in rows], dtype=int)
            targets = 1 - labels if t["orient"] == "oppose" else labels
            target_by_condition[condition] = targets
            for item, target_label in zip(rows, targets):
                item["target_label"] = int(target_label)

        y = target_by_condition["matched"]
        inference_rng = _task_rng(seed, tid, "dry-inference")
        prob, texts, parsed = solve(model, items, t["prompt"], dry, inference_rng, complete_fn)
        predictions = {
            "matched": {
                "items": items,
                "target": y,
                "prob": prob,
                "texts": texts,
                "parsed": parsed,
                "matched_prob": prob,
            }
        }
        matched_index = {item["entity_id"]: i for i, item in enumerate(items)}
        for condition in sorted(set(condition_items) - {"matched"}):
            rows = condition_items[condition]
            if not rows:
                continue
            condition_prob, condition_texts, condition_parsed = solve(
                model,
                rows,
                t["prompt"],
                dry,
                inference_rng,
                complete_fn,
            )
            indices = [matched_index[item["entity_id"]] for item in rows]
            matched_prob = prob[indices]
            matched_target = y[indices]
            condition_target = target_by_condition[condition]
            if not np.array_equal(matched_target, condition_target):
                raise ValueError(
                    f"{tid} matched/{condition} target labels are not entity-paired"
                )
            predictions[condition] = {
                "items": rows,
                "target": condition_target,
                "prob": condition_prob,
                "texts": condition_texts,
                "parsed": condition_parsed,
                "matched_prob": matched_prob,
            }

        scrambled = predictions.get("scrambled")
        reference_key = t["reference"]
        reference = references.get(reference_key) if isinstance(reference_key, str) else reference_key
        score_rng = _task_rng(seed, tid, "bootstrap")
        rec = score_task(
            prob,
            y,
            None if scrambled is None else scrambled["prob"],
            None if scrambled is None else scrambled["target"],
            reference,
            score_rng,
            parsed=parsed,
            scr_parsed=None if scrambled is None else scrambled["parsed"],
            matched_memo_prob=None if scrambled is None else scrambled["matched_prob"],
            reference_comparability=t["reference_comparability"],
        )
        for condition, result in predictions.items():
            if condition == "matched":
                continue
            condition_y = result["target"]
            condition_parsed = result["parsed"]
            rec[f"{condition}_n"] = int(len(condition_y))
            rec[f"{condition}_valid_response_rate"] = round(float(condition_parsed.mean()), 3)
            if len(set(condition_y)) > 1:
                condition_auc = auroc(result["prob"], condition_y)
                matched_subset_auc = auroc(result["matched_prob"], condition_y)
                rec[f"{condition}_auroc"] = round(condition_auc, 3)
                rec[f"matched_minus_{condition}_auroc"] = round(
                    matched_subset_auc - condition_auc,
                    3,
                )
                rec[f"matched_minus_{condition}_auroc_ci"] = paired_auroc_delta_ci(
                    result["matched_prob"],
                    result["prob"],
                    condition_y,
                    _task_rng(seed, tid, f"{condition}-bootstrap"),
                )
            else:
                rec[f"{condition}_auroc"] = None
                rec[f"matched_minus_{condition}_auroc"] = None
                rec[f"matched_minus_{condition}_auroc_ci"] = None
        if scrambled is not None:
            rec["corruption_n"] = rec["memo_n"]
            rec["corruption_delta"] = rec["matched_minus_scrambled_auroc"]
            rec["corruption_delta_ci"] = rec["matched_minus_scrambled_auroc_ci"]
        rec.update({
            "orientation": t["orient"],
            "web_exposure": t["web"],
            "truth_level_code": t["truth_level_code"],
            "target_source_kind": t["target_source_kind"],
            "truth_level": t["truth_level"],
            "task_status": t["status"],
            "biological_question_id": t["biological_question_id"],
            "task_family_id": t["task_family_id"],
            "split_group_scope": t["split_group_scope"],
            "intervention_pair_field": t["intervention_pair_field"],
            "intervention_pair_id": t["intervention_pair_id"],
            "factor_levels": t["factor_levels"],
            "pair_group": t.get("pair_group"),
        })
        scorecard[tid] = rec
        for condition, result in predictions.items():
            for item, target_label, item_prob, text, ok in zip(
                result["items"],
                result["target"],
                result["prob"],
                result["texts"],
                result["parsed"],
            ):
                raw.append({
                    "task": tid,
                    "id": item["id"],
                    "entity_id": item["entity_id"],
                    "source_id": item.get("source_id"),
                    "entity_id_scope": item.get("entity_id_scope"),
                    "truth_level_code": item["truth_level_code"],
                    "target_source_kind": item["target_source_kind"],
                    "truth_level": item["truth_level"],
                    "biological_question_id": item["biological_question_id"],
                    "task_family_id": item["task_family_id"],
                    "split_group_id": item["split_group_id"],
                    "split_group_scope": item["split_group_scope"],
                    "intervention_pair_id": item["intervention_pair_id"],
                    "factor_levels": item["factor_levels"],
                    "condition": condition,
                    "source_label": int(item["label"]),
                    "target_label": int(target_label),
                    "label": int(target_label),
                    "prob": float(item_prob),
                    "prompt_version": PROMPT_VERSION,
                    "parse_valid": bool(ok),
                    "parse_status": "parsed" if ok else "invalid_or_refusal",
                    "output": text,
                })
        print(f"  {tid:28s} web={t['web']:4s} AUROC={rec['output_auroc']} {rec['output_auroc_ci']} "
              f"valid={rec['valid_response_rate']:.3f} "
              f"scrambled_delta={rec.get('matched_minus_scrambled_auroc')} "
              f"renotation_delta={rec.get('matched_minus_re_notation_auroc')}", flush=True)

    scp, rawp = os.path.join(model_dir, "scorecard.json"), os.path.join(model_dir, "raw.jsonl")
    full = scorecard
    if merge and os.path.exists(scp):
        full = json.load(open(scp))
        full.update(scorecard)                 # this run's tasks override / extend
    if merge and os.path.exists(rawp):
        kept = [r for r in (json.loads(line) for line in open(rawp)) if r.get("task") not in scorecard]
        raw = kept + raw
    pair_comparisons = build_pair_group_comparisons(full, raw, seed)
    pair_comparison_filename = "pair_group_comparisons.json"
    manifest = {"model": model, "prompt_version": PROMPT_VERSION, "score_schema": SCORE_SCHEMA,
                "metadata_contract_version": METADATA_CONTRACT_VERSION,
                "truth_taxonomy_version": TRUTH_TAXONOMY_VERSION,
                "decode": DECODE, "seed": seed,
                "n_per_task": n, "dry_run": dry, "tasks": list(full), "last_run": list(scorecard),
                "task_truth_metadata": {
                    task_id: {
                        "truth_level_code": TASKS[task_id]["truth_level_code"],
                        "target_source_kind": TASKS[task_id]["target_source_kind"],
                    }
                    for task_id in full
                },
                "registry": {"core": list(CORE), "exploratory": list(EXPLORATORY),
                             "quarantined": list(QUARANTINED)},
                "sampling_contract": "shared entity intersection within pair_group; balanced labels",
                "calibration_scope": "balanced_benchmark_distribution_only",
                "uncertainty_method": "iid_entity_item_bootstrap_pilot",
                "uncertainty_unit": "entity_id_row_not_biological_dependency_cluster",
                "pair_group_comparisons_file": pair_comparison_filename,
                **repository_state,
                **run_provenance,
                "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    json.dump(full, open(scp, "w"), indent=2)
    json.dump(manifest, open(os.path.join(model_dir, "manifest.json"), "w"), indent=2)
    json.dump(
        pair_comparisons,
        open(os.path.join(model_dir, pair_comparison_filename), "w"),
        indent=2,
    )
    with open(rawp, "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    if not dry:
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
    ap.add_argument("--provider", default=None, help="provider name recorded in the run manifest")
    ap.add_argument("--model-revision", default=None,
                    help="immutable provider/checkpoint revision recorded in the run manifest")
    args = ap.parse_args()
    tasks = args.tasks if args.tasks in ("core", "all") else [t.strip() for t in args.tasks.split(",")]
    evaluate(
        model=args.model,
        tasks=tasks,
        n=args.n,
        seed=args.seed,
        dry=args.dry_run,
        provider=args.provider,
        model_revision=args.model_revision,
    )


if __name__ == "__main__":
    main()
