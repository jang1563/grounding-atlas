"""Phase 2 of grounding-atlas-eval: the specialist ceiling per rung (populates the `gap`
column in run_grounding_eval.py).

CEILING = is the property decodable from the raw representation at all, by a cheap specialist.
  - computable rungs: 1.0 by construction (RDKit / Biopython compute them exactly).
  - molecular (SMILES) rungs: Morgan fingerprint + logistic regression, out-of-fold 5-fold CV
    AUROC on the matched pairs (the same cheap-specialist recipe as results/SYNTHESIS.md).
  - sequence / spectrum rungs (variant_seq, spectra_ms): specialist not wired here; left null.

Writes results/benchmark/ceilings.json: {rung: {ceiling, method, n}}. Reproducible (seeded CV,
deterministic featurizer); committed as a provenance artifact.

Run:  python eval/compute_ceilings.py
"""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIGNAL = os.path.join(ROOT, "signal")
OUT = os.path.join(ROOT, "results", "benchmark")
SEED = 0


def matched(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if r.get("condition", "matched") == "matched"]


def morgan(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as fpg
    gen = fpg.GetMorganGenerator(radius=2, fpSize=2048)
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X.append(np.array(gen.GetFingerprint(m), dtype=np.int8))
            keep.append(i)
    return np.array(X), keep


def cv_auroc(X, y):
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    ceilings = {}
    for path in sorted(glob.glob(os.path.join(SIGNAL, "**", "pairs.jsonl"), recursive=True)):
        rung = os.path.relpath(os.path.dirname(path), SIGNAL)
        if rung.startswith("computable/"):
            ceilings[rung] = {"ceiling": 1.0, "method": "deterministic (RDKit/Biopython exact)",
                              "n": len(matched(path))}
            print(f"  {rung:34s} 1.000  (computable, exact)")
            continue
        rows = matched(path)
        mod = rows[0].get("modality") if rows else None
        if mod != "smiles":
            print(f"  {rung:34s}  skip (no cheap specialist wired for modality={mod})")
            continue
        X, keep = morgan([r["representation"] for r in rows])
        y = np.array([int(rows[i]["label"]) for i in keep])
        if len(set(y.tolist())) < 2:
            print(f"  {rung:34s}  skip (one class)"); continue
        a = cv_auroc(X, y)
        ceilings[rung] = {"ceiling": round(a, 3),
                          "method": "Morgan(2048,r2) + LR, 5-fold out-of-fold CV", "n": int(len(y))}
        print(f"  {rung:34s} {a:.3f}  (Morgan+LR CV, n={len(y)})")

    os.makedirs(OUT, exist_ok=True)
    json.dump(ceilings, open(os.path.join(OUT, "ceilings.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/ceilings.json ({len(ceilings)} rungs)")


if __name__ == "__main__":
    main()
