"""Variant-branch ceiling gate (axis-B): is pathogenicity in the variant CONTENT?

The specialist analog of ../../eval/ceiling_gate.py (Morgan FP) and
../../protein_grounding/eval/ceiling_gate_protein.py (ESM2). Here the specialist is
AlphaMissense (precomputed, ClinVar AUROC ~0.94 in Science 2023). The score IS a trained
classifier on the variant content (protein sequence context + substitution), so the ceiling is
simply AUROC(label, am_pathogenicity); LogReg on the 1-D score is monotone and gives the same
AUROC, reported for parity.

The control that matters here is NOT a train/test split (AlphaMissense is precomputed) but the
TEMPORAL stratification: does the specialist still separate variants first added to ClinVar
AFTER the model cutoff? If AlphaMissense holds on the post-cutoff slice while the LLM output arm
collapses there, that contrast (specialist generalizes, recall does not) is the headline. So we
report AUROC overall and stratified by review stars and first_seen bin.

ESM-1v wild-type-marginal LLR is the secondary ceiling (single forward pass, no MSA, the purest
specialist, mirrors the hERG fingerprint): see ceiling_esm1v_variant.py (GPU).

Env: VG_CSV (default ../data/variant_clinvar_full.csv).
"""
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CSV = os.environ.get("VG_CSV", os.path.join(os.path.dirname(__file__), "..", "data", "variant_clinvar_full.csv"))


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("am") in (None, ""):
                continue
            try:
                am = float(r["am"])
            except ValueError:
                continue
            rows.append({
                "y": int(r["label"]), "am": am, "stars": int(r["stars"]),
                "first_seen": r["first_seen"], "post": int(r["post_cutoff"]),
            })
    return rows


def bootstrap_ci(y, s, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def auroc_row(name, sub):
    y = [r["y"] for r in sub]
    s = [r["am"] for r in sub]
    n, npos = len(y), sum(y)
    if npos == 0 or npos == n:
        print(f"  {name:24s} n={n:5d}  pos={npos:5d}  (single-class, AUROC undefined)", flush=True)
        return
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    lo, hi = bootstrap_ci(y, s)
    print(f"  {name:24s} n={n:5d}  pos={npos:5d} ({npos/n:.0%})  "
          f"AUROC={auc:.3f} [{lo:.3f},{hi:.3f}]  AUPRC={ap:.3f}", flush=True)


def main():
    rows = load(CSV)
    y = np.array([r["y"] for r in rows])
    s = np.array([r["am"] for r in rows])
    print(f"csv={os.path.basename(CSV)}  n_with_AM={len(rows)}  P={int(y.sum())} B={int((1-y).sum())}\n", flush=True)

    print("AlphaMissense ceiling (AUROC of the precomputed score):", flush=True)
    auroc_row("ALL", rows)

    # LogReg-on-score parity (the branches' 'LogReg ceiling' convention); monotone -> same AUROC
    if len(set(y)) > 1:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        proba = cross_val_predict(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            s.reshape(-1, 1), y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        print(f"  {'LogReg(score) 5-fold CV':24s} n={len(y):5d}  AUROC={roc_auc_score(y, proba):.3f}  "
              f"(monotone in the score: same as raw)", flush=True)

    print("\nBy review stars:", flush=True)
    for lab, pred in [("star 1", lambda r: r["stars"] == 1),
                      ("star 2", lambda r: r["stars"] == 2),
                      ("star 3+", lambda r: r["stars"] >= 3),
                      ("star 2+ (expert)", lambda r: r["stars"] >= 2)]:
        auroc_row(lab, [r for r in rows if pred(r)])

    print("\nBy first-seen bin (temporal holdout):", flush=True)
    for lab in ["le_2025_06", "2025H2", "post_2026_01"]:
        auroc_row(lab, [r for r in rows if r["first_seen"] == lab])
    auroc_row("post_2026_01 (strict)", [r for r in rows if r["post"] == 1])

    print("\nRead: the ceiling should HOLD across stars and across the temporal holdout. A flat "
          "AUROC on the post-2026-01 slice (novel variants) is the point: the specialist grounds "
          "from content, so it generalizes where memorized recall cannot. Compare the output arm "
          "(output_arm_variant.py), which is predicted to collapse on that same slice.", flush=True)


if __name__ == "__main__":
    main()
