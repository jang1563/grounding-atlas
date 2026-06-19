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
from benchmark_tasks import CLAUSES  # noqa: E402
from compute_ceilings import SIGNAL, matched, morgan  # noqa: E402
from run_grounding_eval import OUT  # noqa: E402

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


def route(conf, mcorr, scorr):
    """Best confidence-routing accuracy (keep model on most-confident items, route rest to
    specialist) and the coverage that achieves it, plus the model-alone AURC on this signal."""
    order = np.argsort(-conf)
    n = len(conf)
    best_acc, best_cov = max(
        ((mcorr[order[:k]].sum() + scorr[order[k:]].sum()) / n, k / n) for k in range(n + 1))
    return best_acc, best_cov, aurc(conf, mcorr)


def load_conf(model):
    """id -> explicit self-reported confidence, or None if not elicited yet."""
    p = os.path.join(OUT, model, "confidence.jsonl")
    if not os.path.exists(p):
        return None
    return {r["id"]: r["confidence"] for r in (json.loads(line) for line in open(p))}


def main():
    spec = {rung: specialist_oof(rung) for rung in RUNGS}
    report = {}
    print(f"{'model':20s} {'self':>5} {'spec':>5} {'oracle':>6} | {'imp_rt':>6} {'impAURC':>7} | "
          f"{'exp_rt':>6} {'expAURC':>7} {'exp_r':>6}   (rt=best routed acc)")
    for model in MODELS:
        items = model_items(model)
        mp, sp, y, ids = [], [], [], []
        for rung, lst in items.items():
            for iid, prob, lab in lst:
                if iid in spec[rung]:
                    mp.append(prob); sp.append(spec[rung][iid]); y.append(lab); ids.append(iid)
        mp, sp, y = np.array(mp), np.array(sp), np.array(y)
        mcorr = ((mp > 0.5).astype(int) == y).astype(float)
        scorr = ((sp > 0.5).astype(int) == y).astype(float)
        self_acc, spec_acc, oracle = mcorr.mean(), scorr.mean(), np.maximum(mcorr, scorr).mean()
        imp_acc, imp_cov, imp_aurc = route(np.abs(mp - 0.5), mcorr, scorr)
        rec = {"n": len(y), "self_acc": round(self_acc, 3), "spec_acc": round(spec_acc, 3),
               "oracle_acc": round(oracle, 3),
               "model_uniquely_right_frac": round(float((mcorr.astype(bool) & ~scorr.astype(bool)).mean()), 3),
               "implicit": {"routed_acc": round(imp_acc, 3), "coverage": round(imp_cov, 3),
                            "aurc": round(imp_aurc, 3)}}
        cmap = load_conf(model)
        exp_str = f"{'-':>6} {'-':>7} {'-':>6}"
        if cmap is not None:
            econf = np.array([cmap.get(i, 0.5) for i in ids])
            exp_acc, exp_cov, exp_aurc = route(econf, mcorr, scorr)
            r_corr = float(np.corrcoef(econf, mcorr)[0, 1]) if econf.std() > 0 else float("nan")
            rec["explicit"] = {"routed_acc": round(exp_acc, 3), "coverage": round(exp_cov, 3),
                               "aurc": round(exp_aurc, 3), "conf_vs_correct_corr": round(r_corr, 3)}
            exp_str = f"{exp_acc:6.3f} {exp_aurc:7.3f} {r_corr:6.3f}"
        report[model] = rec
        print(f"{model:20s} {self_acc:5.3f} {spec_acc:5.3f} {oracle:6.3f} | "
              f"{imp_acc:6.3f} {imp_aurc:7.3f} | {exp_str}")
    json.dump(report, open(os.path.join(OUT, "routing.json"), "w"), indent=2)
    print("\nself=always-model  spec=always-specialist  oracle=route-by-truth | imp=implicit |P-0.5| "
          "routing, exp=explicit self-confidence routing (exp_r = corr of stated confidence with being right)")
    print(f"wrote {OUT}/routing.json")


if __name__ == "__main__":
    main()
