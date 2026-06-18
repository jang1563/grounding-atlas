"""Calibration/routing arm (Part A, no API): does the model's confidence say when to defer?

For each model, pools the corrected ADMET items and asks the project's prescription question:
if you route each compound to the model when it is confident and to a cheap Morgan specialist
when it is not, do you beat always-answering-yourself and always-calling-the-specialist? Uses
the model probabilities already in results/benchmark/<model>/raw.jsonl (implicit confidence
|P-0.5|) and a per-item out-of-fold Morgan+LR specialist on the oriented label. Read-only.

Run:  python eval/routing_arm.py
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compute_ceilings import SIGNAL, matched, morgan  # noqa: E402
from run_grounding_eval import CLAUSES, OUT  # noqa: E402

MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]
RUNGS = ["ames", "cyp2d6", "cyp3a4", "herg", "permeability", "solubility"]


def specialist_oof(rung):
    """id -> P(clause-positive) from a Morgan+LR 5-fold out-of-fold specialist on the oriented label."""
    rows = matched(os.path.join(SIGNAL, "admet", rung, "pairs.jsonl"))
    orient = CLAUSES.get(rung, (None, "align"))[1]
    X, keep = morgan([r["representation"] for r in rows])
    ids = [rows[i]["id"] for i in keep]
    y = np.array([int(rows[i]["label"]) for i in keep])
    if orient == "oppose":
        y = 1 - y
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return dict(zip(ids, p))


def model_items(model):
    """rung -> list of (id, model_prob, oriented_label) from raw.jsonl."""
    out = {}
    for line in open(os.path.join(OUT, model, "raw.jsonl")):
        r = json.loads(line)
        rung = r["rung"].split("/")[-1]
        if rung not in RUNGS:
            continue
        orient = CLAUSES.get(rung, (None, "align"))[1]
        y = r["label"] if orient == "align" else 1 - r["label"]
        out.setdefault(rung, []).append((r["id"], r["prob"], y))
    return out


def aurc(conf, correct):
    """Area under the risk-coverage curve (model alone, abstain=drop). Lower is better."""
    order = np.argsort(-conf)
    risk = np.cumsum(1 - correct[order]) / (np.arange(len(correct)) + 1)
    return float(risk.mean())


def main():
    spec = {rung: specialist_oof(rung) for rung in RUNGS}
    report = {}
    print(f"{'model':20s} {'self':>5} {'spec':>5} {'oracle':>6} {'routed':>6} {'@cov':>5} "
          f"{'AURC':>5} {'uniq%':>6}")
    for model in MODELS:
        items = model_items(model)
        mp, sp, y = [], [], []
        for rung, lst in items.items():
            for iid, prob, lab in lst:
                if iid in spec[rung]:
                    mp.append(prob); sp.append(spec[rung][iid]); y.append(lab)
        mp, sp, y = np.array(mp), np.array(sp), np.array(y)
        mcorr = ((mp > 0.5).astype(int) == y).astype(float)
        scorr = ((sp > 0.5).astype(int) == y).astype(float)
        self_acc, spec_acc = mcorr.mean(), scorr.mean()
        oracle = np.maximum(mcorr, scorr).mean()
        conf = np.abs(mp - 0.5)
        order = np.argsort(-conf)   # most-confident first keep the model's answer
        n = len(y)
        curve = [(k / n, (mcorr[order[:k]].sum() + scorr[order[k:]].sum()) / n) for k in range(n + 1)]
        best_cov, best_acc = max(curve, key=lambda t: t[1])
        uniq = float((mcorr.astype(bool) & ~scorr.astype(bool)).mean())   # model right, specialist wrong
        report[model] = {"n": n, "self_acc": round(self_acc, 3), "spec_acc": round(spec_acc, 3),
                         "oracle_acc": round(oracle, 3), "routed_acc": round(best_acc, 3),
                         "routed_coverage": round(best_cov, 3), "model_alone_aurc": round(aurc(conf, mcorr), 3),
                         "model_uniquely_right_frac": round(uniq, 3),
                         "routing_beats_specialist": bool(best_acc > spec_acc + 1e-9)}
        r = report[model]
        print(f"{model:20s} {r['self_acc']:5.3f} {r['spec_acc']:5.3f} {r['oracle_acc']:6.3f} "
              f"{r['routed_acc']:6.3f} {r['routed_coverage']:5.2f} {r['model_alone_aurc']:5.3f} "
              f"{100 * r['model_uniquely_right_frac']:5.1f}%")
    json.dump(report, open(os.path.join(OUT, "routing_implicit.json"), "w"), indent=2)
    print(f"\nself=always-model  spec=always-specialist  oracle=route-by-truth  "
          f"routed=best confidence-routing (@cov = model coverage)")
    print(f"wrote {OUT}/routing_implicit.json")


if __name__ == "__main__":
    main()
