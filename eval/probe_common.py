"""Shared probing machinery for the layer-localization arms (docs/LAYER_LOCALIZATION_PREREG.md).

Single source of truth so the DNA / single-cell / MSA / hERG / SFM activation arms compute the
SAME nested-CV unbiased best-layer, the SAME per-layer curve + selectivity, the SAME bootstrap CI,
and dump the SAME task-tagged JSON. The nested-CV here RETURNS the out-of-fold probe vector and the
per-fold picked layer (the prior arms computed the OOF vector then discarded it at
activation_arm.py:160; H3 calibration needs it). No em dashes.

The locked probe (Section 4.2) is balanced_lr: standardized features, L2 C=1.0, class_weight
balanced, used identically in the per-layer curve AND the nested-CV factory. Splitter is GroupKFold
when a group vector is given (hERG scaffold, SFM cluster) and a shuffled StratifiedKFold when it is
not (single-cell, MSA, DNA), so the same code serves both without forking.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_JOBS = 5


def results_path(fname):
    """Absolute path under the repo's results/ (robust to the Cayuga `cd ~/bge` working dir);
    creates the directory if missing."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def balanced_lr():
    """The locked probe (LAYER_LOCALIZATION_PREREG.md Section 4.2): StandardScaler + L2 C=1.0 +
    class_weight balanced. The SAME factory is used for the per-layer curve and the nested CV so the
    operating point ECE/sel_acc read is consistent with the AUROC."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))


def _kfold(n_splits, groups):
    if groups is not None:
        return GroupKFold(n_splits)
    return StratifiedKFold(n_splits, shuffle=True, random_state=0)


def _cvp(clf, X, y, cv, groups):
    """cross_val_predict probability of class 1; pass groups only to a GroupKFold."""
    g = groups if isinstance(cv, GroupKFold) else None
    return cross_val_predict(clf, X, y, cv=cv, groups=g, method="predict_proba", n_jobs=N_JOBS)[:, 1]


def layer_curve(H, y, groups=None, clf_factory=balanced_lr, n_splits=5):
    """Per-layer CROSS-VALIDATED probe AUROC (the curve) + the biased max-over-layers.

    Returns (aucs: list[float], best_L: int, best_oof: np.ndarray). best_* is the selection-BIASED
    number, kept only so the optimism gap (naive minus nested) can be reported.
    """
    y = np.asarray(y)
    cv = _kfold(n_splits, groups)
    aucs, best, best_L, best_oof = [], -1.0, -1, None
    for L in range(len(H)):
        p = _cvp(clf_factory(), np.asarray(H[L]), y, cv, groups)
        a = roc_auc_score(y, p)
        aucs.append(float(a))
        if a > best:
            best, best_L, best_oof = a, L, p
    return aucs, best_L, best_oof


def nested_layer_auroc(H, y, groups=None, clf_factory=balanced_lr, n_splits=5):
    """UNBIASED best-layer AUROC by nested CV (Cawley-Talbot JMLR 2010): pick the layer on the inner
    folds of the outer-TRAIN rows only, score the untouched outer-test fold at that layer. Removes
    the max-over-layers selection bias (the prior +0.11 at 0.5B).

    Returns a dict:
      auroc        pooled out-of-fold AUROC at the per-fold picked layer
      auroc_fold   mean of the outer-test-fold AUROCs (the design's headline; Section 4.3 step 4)
      fold_aucs    the per-outer-fold test AUROCs
      picked       per-outer-fold selected layer (across-fold spread = the H1 localization)
      band         per-outer-fold within-1-SE layer set (the band form of H1)
      oof          out-of-fold probability vector at the per-fold picked layer (THE FIX for H3)

    groups=None uses a shuffled StratifiedKFold (e.g. single-cell); a group vector uses GroupKFold.
    """
    y = np.asarray(y)
    Harr = [np.asarray(h) for h in H]
    layers = len(Harr)
    g_all = None if groups is None else np.asarray(groups)
    oof = np.zeros(len(y), dtype=float)
    picked, bands, fold_aucs = [], [], []
    outer = _kfold(n_splits, groups)
    split_args = (Harr[0], y) if groups is None else (Harr[0], y, g_all)
    for tr, te in outer.split(*split_args):
        g_tr = None if g_all is None else g_all[tr]
        n_in = n_splits if g_tr is None else min(n_splits, len(np.unique(g_tr)))
        inner = _kfold(n_in, g_tr)
        layer_scores = []
        for L in range(layers):
            p = _cvp(clf_factory(), Harr[L][tr], y[tr], inner, g_tr)
            layer_scores.append(roc_auc_score(y[tr], p))
        layer_scores = np.asarray(layer_scores)
        bL = int(layer_scores.argmax())
        se = layer_scores.std(ddof=1) / np.sqrt(layers) if layers > 1 else 0.0
        band = [int(L) for L in range(layers) if layer_scores[L] >= layer_scores[bL] - se]
        clf = clf_factory().fit(Harr[bL][tr], y[tr])
        pte = clf.predict_proba(Harr[bL][te])[:, 1]
        oof[te] = pte
        picked.append(bL)
        bands.append(band)
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(float(roc_auc_score(y[te], pte)))
    return {
        "auroc": float(roc_auc_score(y, oof)),
        "auroc_fold": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "fold_aucs": fold_aucs,
        "picked": [int(x) for x in picked],
        "band": bands,
        "oof": oof,
    }


def selectivity_at(H, y, L, groups=None, clf_factory=balanced_lr, n_splits=5, seed=123):
    """Hewitt-Liang control (1909.03368): refit the SAME probe at layer L on PERMUTED labels under
    the same split. Returns (control_auroc, selectivity = real - control). Near 0 = the probe just
    fits; high = it reads real signal."""
    y = np.asarray(y)
    Xl = np.asarray(H[L])
    cv = _kfold(n_splits, groups)
    real = roc_auc_score(y, _cvp(clf_factory(), Xl, y, cv, groups))
    ys = np.random.RandomState(seed).permutation(y)
    ctrl = roc_auc_score(ys, _cvp(clf_factory(), Xl, ys, _kfold(n_splits, groups), groups))
    return float(ctrl), float(real - ctrl)


def cluster_boot(y, p, groups=None, n_boot=1000, seed=0):
    """95 percent bootstrap CI for AUROC. CLUSTER (block) bootstrap when groups are given (resample
    GROUPS with replacement, take all their rows) so recurring entities (hERG scaffolds, homolog
    families) do not deflate the CI; iid resampling otherwise (Section 4.6)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.RandomState(seed)
    aucs = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            gs = rng.choice(uniq, len(uniq), replace=True)
            b = np.concatenate([idx_by_g[g] for g in gs])
            if len(np.unique(y[b])) > 1:
                aucs.append(roc_auc_score(y[b], p[b]))
    else:
        idx = np.arange(len(y))
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(y[b])) > 1:
                aucs.append(roc_auc_score(y[b], p[b]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# --- calibration / selective-prediction metrics (mirror run_grounding_eval.py for H3) ---

def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        hi = p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * (m.sum() / len(y))
    return float(e)


def aurc(y, p):
    """Area under the risk-coverage curve. Confidence = |p-0.5|; abstain on the least confident.
    Lower = a better confidence ranking (errors concentrate on abstained items)."""
    y = np.asarray(y)
    p = np.asarray(p)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    correct = ((p >= 0.5).astype(int) == y).astype(float)[order]
    risk = np.cumsum(1.0 - correct) / np.arange(1, len(y) + 1)
    return float(risk.mean())


def sel_acc(y, p, cov=0.5):
    """Accuracy on the most-confident `cov` fraction (selective accuracy at coverage cov)."""
    y = np.asarray(y)
    p = np.asarray(p)
    conf = np.abs(p - 0.5)
    k = max(1, int(round(cov * len(y))))
    keep = np.argsort(-conf)[:k]
    return float((((p >= 0.5).astype(int) == y)[keep]).mean())


def dump_layerloc(path, task, model, y, layer_aucs, nested, sel, *, output=None, ceiling=None,
                  groups=None, extra=None):
    """Task-tagged JSON for an arm: per-layer curve, the naive-minus-nested optimism, the nested
    headline (auroc, picked, band, OOF vector), selectivity, and (if given) the verbalize vector +
    H3 calibration (probe ECE/AURC vs output AURC) so the analysis in Section 5 reads one schema."""
    y = np.asarray(y)
    naive = float(max(layer_aucs))
    rec = {
        "task": task, "model": model, "n": int(len(y)), "pos": int(np.sum(y)),
        "layer_auroc": [round(float(a), 4) for a in layer_aucs],
        "naive_max_auroc": round(naive, 4),
        "naive_best_layer": int(np.argmax(layer_aucs)),
        "nested_auroc": round(nested["auroc"], 4),
        "nested_auroc_fold": round(nested["auroc_fold"], 4),
        "nested_fold_aucs": [round(a, 4) for a in nested["fold_aucs"]],
        "picked_layers": nested["picked"],
        "picked_band": nested["band"],
        "selection_bias": round(naive - nested["auroc"], 4),
        "selectivity": round(float(sel), 4),
        "oof": [round(float(v), 4) for v in nested["oof"]],
        "label": [int(v) for v in y],
    }
    if output is not None:
        output = np.asarray(output)
        rec["output"] = [round(float(v), 4) for v in output]
        rec["output_auroc"] = round(float(roc_auc_score(y, output)), 4)
        rec["raw_gap"] = round(rec["nested_auroc"] - rec["output_auroc"], 4)
        rec["probe_ece_10"] = round(ece(y, nested["oof"], 10), 4)
        rec["probe_ece_5"] = round(ece(y, nested["oof"], 5), 4)
        rec["probe_aurc"] = round(aurc(y, nested["oof"]), 4)
        rec["output_aurc"] = round(aurc(y, output), 4)
    if ceiling is not None:
        rec["ceiling"] = round(float(ceiling), 4)
    if extra:
        rec.update(extra)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec
