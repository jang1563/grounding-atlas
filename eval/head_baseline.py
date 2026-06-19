"""GroundBench cheap-head baseline: the "orchestrate via a trained head" reference.

For every task a cheap, representation-agnostic head is cross-validated on the SAME
representation the LLM is shown:
  - emb tasks         -> the raw embedding vector
  - image tasks       -> per-channel color statistics
  - numeric reps      -> the parsed value vector (e.g. methylation betas)
  - everything else   -> char n-gram hashing (a uniform cheap text head)
This is the "can a dumb specialist read this representation" baseline. The point it
makes is the orchestration prescription: on the web-zero tasks the cheap head often
GROUNDS (the information is present in the representation) where the LLM verbalizes at
chance. So you orchestrate by putting a trained head on the representation, you do not
prompt-paste it. Where the head is weak but the cited ceiling is high (numeric NMR/3D,
variant sequence), a modality-specific specialist is needed; the uniform head is a floor.

No API, no GPU. Writes results/benchmark/baseline-cheap-head/scorecard.json and
regenerates the leaderboard.  Run:  python eval/head_baseline.py
"""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS, task_items  # noqa: E402
from run_grounding_eval import OUT, auroc, ci, update_leaderboard  # noqa: E402

NUMERIC = {"methyl/age"}  # reps that are key:value numeric panels, parsed to a value vector


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
    rng = np.random.default_rng(0)
    sc = {}
    for task in CORE:
        items, _ = task_items(task, 100000, rng)  # all available, balanced
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
                    "output_auroc_ci": ci(auroc, p, y, rng),
                    "ceiling": None, "web_exposure": TASKS[task]["web"],
                    "orientation": TASKS[task]["orient"],
                    "method": "cheap-head: char3-5 hashing / value-vector / color"}
        print(f"  {task:28s} web={TASKS[task]['web']:5s} head AUROC={sc[task]['output_auroc']} "
              f"{sc[task]['output_auroc_ci']}", flush=True)
    d = os.path.join(OUT, "baseline-cheap-head")
    os.makedirs(d, exist_ok=True)
    json.dump(sc, open(os.path.join(d, "scorecard.json"), "w"), indent=2)
    update_leaderboard()
    print(f"\nwrote {d}/scorecard.json [{len(sc)} tasks]")


if __name__ == "__main__":
    main()
