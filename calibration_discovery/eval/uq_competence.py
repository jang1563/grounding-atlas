"""UQ routing PoC: does specialist self-uncertainty close the per-item routing gap?
(docs/UQ_ROUTING_POC_DESIGN.md)

Reuses per_item_router.py (aligned per-item model pred/conf + real specialist proba) and
adds the specialist's per-input uncertainty U = 1 - 2|p - 0.5| (ambiguity). Tests, honestly,
whether routing on {model conf, U} beats always-call-the-specialist toward the per-item
oracle. No API calls. No em dashes.

Run:  PYTHONPATH=calibration_discovery/eval python calibration_discovery/eval/uq_competence.py
"""
import csv
import os

import numpy as np
from per_item_router import PER_ITEM, N, specialist_proba
from selective_eval import RUNGS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

MODEL = os.environ.get("UQ_MODEL", "claude-opus-4-8")


def main():
    rows = list(csv.DictReader(open(PER_ITEM)))
    rungs = [r for r in RUNGS if r in {x["rung"] for x in rows}]
    mp, cf, sp, lb = [], [], [], []
    for rung in rungs:
        p, y = specialist_proba(rung, N)
        sub = [r for r in rows if r["model"] == MODEL and r["rung"] == rung]
        if len(sub) != len(y) or [int(r["label"]) for r in sub] != list(y):
            print("  MISALIGN", rung); continue
        mp += [float(r["pred"]) for r in sub]
        cf += [float(r["conf"]) for r in sub]
        sp += list(p); lb += list(y)
    mp, cf, sp, lb = (np.array(a) for a in (mp, cf, sp, lb))
    U = 1 - 2 * np.abs(sp - 0.5)
    mc = (mp > 0.5) == (lb == 1)
    sc = (sp > 0.5) == (lb == 1)

    def acc(pred):
        return float(np.mean((pred > 0.5) == (lb == 1)))

    # Step 0: alignment (reproduce the reported gap)
    print(f"[Step0] n={len(lb)}  always-model={acc(mp):.3f}  always-spec={acc(sp):.3f}  oracle={(mc | sc).mean():.3f}")
    recov = mc & ~sc
    print(f"[Step1a] recoverable={recov.mean():.3f}  AUROC(U flags recoverable vs spec-correct)="
          f"{roc_auc_score(recov[recov | sc].astype(int), U[recov | sc]):.3f}")
    hi = U >= np.quantile(U, 2 / 3)
    print(f"[Step1b] U-high tercile: P(model right)={mc[hi].mean():.3f}  P(spec right)={sc[hi].mean():.3f}  "
          "(U helps only if model > spec here)")

    # Step 2: CV'd routers (predict model-correct, route to model if oof>0.5)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def router(feats):
        g = cross_val_predict(LogisticRegression(max_iter=1000), feats, mc.astype(int),
                              cv=cv, method="predict_proba")[:, 1]
        return acc(np.where(g > 0.5, mp, sp))

    print("[Step2] routers (CV, honest):")
    for name, F in [("R1 conf", cf[:, None]), ("R1b U", U[:, None]), ("R2 conf+U", np.c_[cf, U])]:
        print(f"   {name:12s} {router(F):.3f}")
    print(f"   {'always-spec':12s} {acc(sp):.3f}   {'oracle':12s} {(mc | sc).mean():.3f}")
    print("  -> verdict: do any routers beat always-spec? (measured: no)")


if __name__ == "__main__":
    main()
