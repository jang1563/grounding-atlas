"""Build the CD8-T vs NK cell-sentence dataset (harder than T-vs-not; lit-review fix).

Each cell -> a cell-sentence (top-k expressed gene SYMBOLS) plus a global-consistent ANON
version (same gene -> same arbitrary token across all cells, so the expression vector is intact
and a probe still separates the classes ~equally; only the human-readable name is removed). This
is the correct control for the "encoded-equal, web-exposure-severed" dissociation -- per-instance
anonymization would destroy the cross-cell structure and break the probe ceiling.

Writes signal/single_cell/cd8t_nk.csv (label, cell_sentence, anon) with label 1 = CD8 T, 0 = NK.
Prints diagnostics: class counts, a specialist-ceiling AUROC (is the task separable but non-trivial?),
and sample sentences. Run: python signal/single_cell/build_cd8t_nk.py
"""
import os
import re

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Uninformative high-expression genes that otherwise dominate every cell-sentence: ribosomal
# (RPL/RPS/MRPL/MRPS), mitochondrial (MT-), and MALAT1. Filtered before ranking so the markers
# surface. This is a fixed a-priori gene filter, not cross-cell peeking.
HOUSEKEEPING = re.compile(r"^(RP[LS]|MRP[LS]|MT-|MALAT1)")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cd8t_nk.csv")
TOPK = 50
CLASSES = {"CD8 T cells": 1, "NK cells": 0}


def main():
    a = sc.datasets.pbmc3k_processed()
    raw = a.raw.to_adata() if a.raw is not None else a   # log-normalized abundance, full genes
    hvg = set(map(str, a.var_names))                     # the 1838 highly-variable genes
    genes_all = np.array([str(g) for g in raw.var_names], dtype=object)
    mask = np.array([(g in hvg) and HOUSEKEEPING.match(g) is None for g in genes_all])
    genes = genes_all[mask].astype(str)
    keep = a.obs["louvain"].isin(CLASSES).values
    X = raw.X[keep]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X[:, mask]
    y = np.array([CLASSES[c] for c in a.obs["louvain"].values[keep]])
    print(f"cells: CD8 T={int((y == 1).sum())}  NK={int((y == 0).sum())}  "
          f"informative HVGs={len(genes)} (HVG and non-housekeeping)")

    # specialist ceiling: how separable are the two classes from the full expression vector?
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    print(f"specialist ceiling (LogReg 5-fold CV AUROC) = {roc_auc_score(y, p):.3f}")

    # global-consistent anon map over all genes that ever appear in a top-k sentence
    order = np.argsort(-X, axis=1)[:, :TOPK]
    used = sorted({int(j) for row in order for j in row})
    anon = {int(j): f"feat_{i}" for i, j in enumerate(used)}

    rows = []
    for i in range(len(y)):
        idx = order[i]
        rows.append((int(y[i]),
                     " ".join(genes[j] for j in idx),
                     " ".join(anon[int(j)] for j in idx)))
    with open(OUT, "w") as f:
        f.write("label,cell_sentence,anon\n")
        for lab, cs, an in rows:
            f.write(f"{lab},{cs},{an}\n")
    print(f"wrote {OUT}  ({len(rows)} cells, top-{TOPK} genes, global-consistent anon)")
    for lab in (1, 0):
        ex = next(r for r in rows if r[0] == lab)
        print(f"  {'CD8T' if lab else 'NK  '} sample: {ex[1][:90]}")


if __name__ == "__main__":
    main()
