"""Per-item router with REAL per-item specialists (closes the SYNTHESIS.md open refinement).

The calibration_routing result routed on a per-RUNG ceiling (the specialist assumed perfect when
called) and per-rung defer majority, a per-rung UPPER BOUND. Here we replace that with a real
per-item specialist prediction for every rung (a cheap cross-validated out-of-fold classifier on
the raw representation: Morgan FP for SMILES, an LR clock on the methylation betas, k-mers for DNA,
column-stats for MSA, a bag-of-genes for single-cell, binned m/z for NMR, AlphaMissense `am` for the
variant) and ask the per-item question: does routing on the model's own continuous CONF, to a real
(imperfect) specialist, recover the per-item ORACLE and beat both always-model and the model's own
binary DEFER.

No API calls: the model's per-item PRED/CONF/ROUTE are read from results/per_item.csv (v1 run),
the specialist is computed here, and the two are aligned by load() order (same seed/N). No em dashes.
"""
import csv
import json
import os
import re

import numpy as np
from selective_eval import ROOT, RUNGS  # reuse identical item sampling (load_raw below mirrors load())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N = int(os.environ.get("SE_N", "80"))
PER_ITEM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "per_item.csv")
RES = os.path.dirname(PER_ITEM)
MODELS = os.environ.get("SE_MODELS", "claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001").split(",")
CV = StratifiedKFold(5, shuffle=True, random_state=0)


# ---------- per-rung featurizers (return dense or sparse X for the loaded repr_texts) ----------

def feat_morgan(texts):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    X = np.zeros((len(texts), 1024), dtype=float)
    for i, s in enumerate(texts):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            X[i] = np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")
    return X


def feat_beta(texts):
    parsed = [dict((p.split(":")[0], float(p.split(":")[1])) for p in t.split() if ":" in p) for t in texts]
    keys = sorted({k for d in parsed for k in d})
    return np.array([[d.get(k, 0.0) for k in keys] for d in parsed])


def feat_kmer(texts, k=4):
    bases = "ACGT"
    idx = {"".join(p): i for i, p in enumerate(__import__("itertools").product(bases, repeat=k))}
    X = np.zeros((len(texts), len(idx)), dtype=float)
    for i, s in enumerate(texts):
        s = s.upper()
        for j in range(len(s) - k + 1):
            km = s[j:j + k]
            if km in idx:
                X[i, idx[km]] += 1
    return X


def feat_msa(texts):
    rows = []
    for t in texts:
        toks = t.split()
        d = len(toks) or 1
        cnt = {}
        for x in toks:
            cnt[x] = cnt.get(x, 0) + 1
        rows.append([d, len(set(toks)), toks.count("-") / d, max(cnt.values()) / d])
    return np.array(rows, dtype=float)


def feat_nmr(texts):
    X = np.zeros((len(texts), 22), dtype=float)
    for i, t in enumerate(texts):
        vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", t)]
        if vals:
            X[i], _ = np.histogram(vals, bins=22, range=(0, 220))
    return X


def feat_bag(texts):
    return CountVectorizer(token_pattern=r"\S+", min_df=2).fit_transform(texts)


FEATURIZERS = {
    "smiles_herg": (feat_morgan, False),   # (fn, needs_scaling)
    "methyl_age": (feat_beta, True),
    "dna_promoter": (feat_kmer, False),
    "msa_conserv": (feat_msa, True),
    "sc_cellsentence": (feat_bag, False),
    "sc_anon": (feat_bag, False),
    "nmr_herg": (feat_nmr, True),
}


def load_raw(rung, n):
    """Raw (UNtruncated) rows + labels in the exact order load() produces (same seed/N), so they
    align with per_item.csv. The specialist gets the FULL representation, not the LLM-truncated
    prompt text (methyl betas are cut to 100 CpGs for the prompt; the specialist should see all)."""
    spec = RUNGS[rung]
    rows = list(csv.DictReader(open(os.path.join(ROOT, spec["csv"]))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = pos[:k] + neg[:k]
    return data, np.array([int(r["label"]) for r in data])


def specialist_proba(rung, n):
    """Out-of-fold per-item specialist probability on the full raw representation. Variant = `am`."""
    data, y = load_raw(rung, n)
    if rung.startswith("variant"):
        am = []
        for r in data:
            try:
                am.append(float(r["am"]))
            except (ValueError, TypeError, KeyError):
                am.append(0.5)
        return np.array(am), y
    fn, scale = FEATURIZERS[rung]
    X = fn([r[RUNGS[rung]["field"]] for r in data])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if scale else LogisticRegression(max_iter=2000)
    p = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
    return p, y


def acc(pred, y):
    return float(np.mean((np.asarray(pred) > 0.5) == (np.asarray(y) == 1)))


def main():
    # model per-item table: (model, rung) -> ordered list of (label, pred, conf, route)
    rows = list(csv.DictReader(open(PER_ITEM)))
    rungs = [r for r in RUNGS if r in set(x["rung"] for x in rows)]   # rungs present in the v1 csv
    print(f"per_item.csv rungs: {rungs}\n", flush=True)

    result = {"specialist_auroc": {}, "models": {}}   # saved for plot_router.py
    # specialist per item (model-independent), aligned to load() order
    spec = {}
    print(f"{'rung':16s} {'specialist AUROC (real per-item)':>34s}", flush=True)
    for rung in rungs:
        p, y = specialist_proba(rung, N)
        spec[rung] = (p, y)
        a = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
        result["specialist_auroc"][rung] = round(a, 3)
        print(f"{rung:16s} {a:34.3f}", flush=True)

    print(f"\n{'model':26s} {'always-model':>12s} {'always-spec':>11s} {'own-DEFER':>10s} {'CONF-route':>11s} {'oracle':>8s} {'spec-call%':>10s}", flush=True)
    for model in MODELS:
        mp, cf, rt, lb, sp = [], [], [], [], []
        for rung in rungs:
            sub = [r for r in rows if r["model"] == model and r["rung"] == rung]
            p, y = spec[rung]
            assert len(sub) == len(y), f"{model}/{rung}: {len(sub)} vs {len(y)}"
            assert [int(r["label"]) for r in sub] == list(y), f"label misalign {model}/{rung}"
            mp += [float(r["pred"]) for r in sub]
            cf += [float(r["conf"]) for r in sub]
            rt += [r["route"] for r in sub]
            lb += list(y)
            sp += list(p)
        mp, cf, lb, sp = np.array(mp), np.array(cf), np.array(lb), np.array(sp)
        rt = np.array(rt)

        model_acc = acc(mp, lb)
        spec_acc = acc(sp, lb)
        m_correct = (mp > 0.5) == (lb == 1)
        s_correct = (sp > 0.5) == (lb == 1)
        oracle_acc = float(np.mean(m_correct | s_correct))               # per-item upper bound
        # policy: model's own binary DEFER -> use specialist when ROUTE==DEFER
        own = np.where(rt == "DEFER", sp, mp)
        own_acc = acc(own, lb)
        # policy: CONF threshold (use model if conf>=t else specialist), best t over the conf grid
        best_acc, best_t, best_callrate = -1, None, None
        for t in np.unique(np.r_[cf, 1.01]):
            routed = np.where(cf >= t, mp, sp)
            a = acc(routed, lb)
            if a > best_acc:
                best_acc, best_t, best_callrate = a, t, float(np.mean(cf < t))
        result["models"][model] = dict(always_model=round(model_acc, 3), always_spec=round(spec_acc, 3),
                                        own_defer=round(own_acc, 3), conf_route=round(best_acc, 3),
                                        oracle=round(oracle_acc, 3), spec_call_rate=round(best_callrate, 3))
        print(f"{model:26s} {model_acc:12.3f} {spec_acc:11.3f} {own_acc:10.3f} {best_acc:11.3f} {oracle_acc:8.3f} {best_callrate*100:9.0f}%", flush=True)

    json.dump(result, open(os.path.join(RES, "router_results.json"), "w"), indent=2)
    print("\n  always-model = answer everything in-model; always-spec = call the specialist every time;", flush=True)
    print("  own-DEFER = use the specialist when the model itself said DEFER; CONF-route = threshold the", flush=True)
    print("  continuous confidence (best t); oracle = per-item best of {model, specialist} (upper bound).", flush=True)
    print("  spec-call% = fraction routed to the specialist at the best CONF threshold.", flush=True)
    print(f"  [wrote {RES}/router_results.json]", flush=True)


if __name__ == "__main__":
    main()
