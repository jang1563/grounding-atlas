"""GroundBench cheap-head pilot: an optimistic surface-readability diagnostic.

For every task a cheap, representation-agnostic head is cross-validated on the SAME
representation the LLM is shown:
  - emb tasks         -> the raw embedding vector
  - image tasks       -> per-channel color statistics
  - numeric reps      -> the parsed value vector (e.g. methylation betas)
  - everything else   -> char n-gram hashing (a uniform cheap text head)
This asks whether a generic supervised classifier can predict the benchmark label from
the presented representation. It is not a specialist ceiling, proof of biological
grounding, or a supervision-matched comparison with native output. The current random
StratifiedKFold and IID item bootstrap are optimistic pilot estimates where biological
dependency groups exist.

No API, no GPU. Writes results/benchmark/baseline-cheap-head/scorecard.json and
regenerates the leaderboard.  Run:  python eval/head_baseline.py
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS, GroundBenchSampler  # noqa: E402
from run_grounding_eval import (  # noqa: E402
    OUT,
    PROMPT_VERSION,
    SCORE_SCHEMA,
    _task_rng,
    auroc,
    ci,
    update_leaderboard,
)

NUMERIC = {"methyl/age"}  # reps that are key:value numeric panels, parsed to a value vector
N = int(os.environ.get("GROUNDBENCH_N", "100"))
SEED = int(os.environ.get("GROUNDBENCH_SEED", "0"))


def _img_feats(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    f = []
    for c in range(3):
        ch = a[:, :, c].ravel()
        f += [ch.mean(), ch.std(), np.percentile(ch, 25), np.percentile(ch, 50), np.percentile(ch, 75)]
    return np.array(f)


def _numeric_vec(items):
    vecs = [[float(tok.split(":")[-1]) for tok in it["rep"].split()] for it in items]
    m = min(len(v) for v in vecs)
    return np.array([v[:m] for v in vecs])


def featurize(task, items):
    kind = TASKS[task]["kind"]
    if kind == "emb":
        return np.array([[float(x) for x in it["rep"].split()] for it in items])
    if kind == "image":
        return np.array([_img_feats(it["image"]) for it in items])
    if task in NUMERIC:
        return _numeric_vec(items)
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18,
                           alternate_sign=False, norm="l2")
    return hv.transform([it["rep"] for it in items])


def main():
    sampler = GroundBenchSampler(seed=SEED)
    sc = {}
    for task in CORE:
        items = sampler.task_condition_items(task, N)["matched"]
        if not items:
            continue
        try:
            X = featurize(task, items)
            y = np.array([it["label"] for it in items])
            if TASKS[task]["orient"] == "oppose":
                y = 1 - y
            p = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  method="predict_proba")[:, 1]
            a = auroc(p, y)
        except Exception as e:
            print(f"  skip {task}: {e}", flush=True)
            continue
        sc[task] = {"n": int(len(y)), "output_auroc": round(float(a), 3),
                    "output_auroc_ci": ci(
                        auroc,
                        p,
                        y,
                        _task_rng(SEED, task, "cheap-head-bootstrap"),
                    ),
                    "reference_score": None, "ceiling": None, "web_exposure": TASKS[task]["web"],
                    "orientation": TASKS[task]["orient"],
                    "method": "optimistic-pilot-supervised-surface-readability",
                    "split_contract": "random_stratified_5fold_not_biologically_grouped",
                    "uncertainty_method": "iid_entity_item_bootstrap_pilot",
                    "interpretation": "predictability diagnostic; not ceiling or grounding proof",
                    "sampling_contract": "same matched GroundBench v5 entities; different CV contract"}
        print(f"  {task:28s} web={TASKS[task]['web']:5s} head AUROC={sc[task]['output_auroc']} "
              f"{sc[task]['output_auroc_ci']}", flush=True)
    d = os.path.join(OUT, f"baseline-cheap-head__{PROMPT_VERSION}")
    os.makedirs(d, exist_ok=True)
    json.dump(sc, open(os.path.join(d, "scorecard.json"), "w"), indent=2)
    manifest = {
        "model": "baseline-cheap-head",
        "prompt_version": PROMPT_VERSION,
        "score_schema": SCORE_SCHEMA,
        "artifact_kind": "diagnostic_non_submission",
        "submission_eligible": False,
        "seed": SEED,
        "n_per_task": N,
        "tasks": list(sc),
        "sampling_contract": "same matched GroundBench v5 entities; random-CV diagnostic only",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    update_leaderboard()
    print(f"\nwrote {d}/scorecard.json [{len(sc)} tasks]")


if __name__ == "__main__":
    main()
