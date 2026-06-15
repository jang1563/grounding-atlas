"""Follow-up to uq_competence.py: do OTHER model-superiority signals close the per-item gap?
(docs/UQ_ROUTING_POC_DESIGN.md)

uq_competence.py showed specialist self-uncertainty does not beat always-call-the-specialist.
Here we test the indicated next signals: model-specialist disagreement, a per-item
web-exposure flag, and per-context (per-rung) specialist reliability. No API calls. No em dashes.

Run:  PYTHONPATH=calibration_discovery/eval python calibration_discovery/eval/uq_signals.py
"""
import csv
import os

import numpy as np
from per_item_router import PER_ITEM, N, specialist_proba
from selective_eval import RUNGS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

MODEL = os.environ.get("UQ_MODEL", "claude-opus-4-8")


def main():
    rows = list(csv.DictReader(open(PER_ITEM)))
    rungs = [r for r in RUNGS if r in {x["rung"] for x in rows}]
    mp, cf, sp, lb, rg, tg = [], [], [], [], [], []
    for rung in rungs:
        p, y = specialist_proba(rung, N)
        sub = [r for r in rows if r["model"] == MODEL and r["rung"] == rung]
        if len(sub) != len(y):
            continue
        mp += [float(r["pred"]) for r in sub]
        cf += [float(r["conf"]) for r in sub]
        sp += list(p); lb += list(y); rg += [rung] * len(y)
        tg += [r.get("tag", "") for r in sub]
    mp, cf, sp, lb = (np.array(a) for a in (mp, cf, sp, lb))
    rg = np.array(rg)
    mc = (mp > 0.5) == (lb == 1)
    sc = (sp > 0.5) == (lb == 1)
    U = 1 - 2 * np.abs(sp - 0.5)
    dis = np.abs(mp - sp)

    def acc(pred):
        return float(np.mean((pred > 0.5) == (lb == 1)))

    print(f"always-spec={acc(sp):.3f}  oracle={(mc | sc).mean():.3f}\n[per-rung model vs spec]")
    for rung in rungs:
        m = rg == rung
        flag = "  <- model better" if mc[m].mean() > sc[m].mean() else ""
        print(f"  {rung:16s} model={mc[m].mean():.2f} spec={sc[m].mean():.2f}{flag}")

    # per-context (per-rung) reliability routing: in-sample upper bound and CV
    better = {r: mc[rg == r].mean() > sc[rg == r].mean() for r in rungs}
    ru_up = acc(np.where(np.array([better[r] for r in rg]), mp, sp))
    rng = np.random.RandomState(0)
    routed = np.empty(len(lb))
    for r in rungs:
        idx = np.where(rg == r)[0]; rng.shuffle(idx); h = len(idx) // 2
        for tr, te in [(idx[:h], idx[h:]), (idx[h:], idx[:h])]:
            routed[te] = mp[te] if mc[tr].mean() > sc[tr].mean() else sp[te]
    print(f"\n[rung-aware routing] in-sample(upper)={ru_up:.3f}  CV={acc(routed):.3f}  vs always-spec={acc(sp):.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oh = np.array([[r == x for x in rungs] for r in rg], float)
    wr = np.array([("rich" in t) for t in tg], float)

    def router(F):
        g = cross_val_predict(LogisticRegression(max_iter=1000), F, mc.astype(int),
                              cv=cv, method="predict_proba")[:, 1]
        return acc(np.where(g > 0.5, mp, sp))

    print("\n[feature routers, CV] (none beats always-spec)")
    for name, F in [("disagreement", dis[:, None]), ("conf+U+dis", np.c_[cf, U, dis]),
                    ("rung-onehot", oh), ("conf+U+dis+rung", np.c_[cf, U, dis, oh]),
                    ("+webflag", np.c_[cf, U, dis, oh, wr])]:
        print(f"   {name:18s} {router(F):.3f}")


if __name__ == "__main__":
    main()
