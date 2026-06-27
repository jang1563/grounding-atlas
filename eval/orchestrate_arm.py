"""Experiment 2, Arm B (EXTERNAL ORCHESTRATION): a trained read-out head on the frozen molecular-FM
embedding, the LLM untouched. The closed-weight-friendly placement and the transfer reference.

Per docs/BRIDGE_3WAY_PREREG.md Section 5 (Arm B) + Section 3 (H2):
- WITHIN-property: balanced_lr on the ChemBERTa embedding, scaffold GroupKFold AUROC per endpoint, with
  a shuffled-label selectivity guard (>= 0.10).
- HELD-OUT-PROPERTY transfer (LPO headline fold: train {ames,cyp3a4,cyp2d6,solubility,permeability},
  test {herg, clearance}): a head fit on the POOLED train-endpoint embeddings, scored on the held-out
  endpoint's frozen scaffold-test split. The FLOOR is the SAME pooled-train transfer on a cheap Morgan
  featurizer (the cross-property transfer with no learned representation, ~0.52). NT =
  (transfer - floor) / (in_property - floor); in_property = a head trained on the held-out endpoint's
  OWN train split, scored on the SAME test items (no estimator mismatch).
Writes results/orchestrate_arm.json and the shared signal/admet/folds/lpo_herg_clearance.json (test_ids
all GPU arms must reuse). CPU only. No em dashes.
"""
import json
import os

import numpy as np
from probe_common import balanced_lr, cluster_boot
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "signal", "sfm_embedding")
ALL = ["herg", "cyp3a4", "cyp2d6", "ames", "solubility", "permeability", "clearance"]
HELD_OUT = ["herg", "clearance"]            # LPO headline fold
TRAIN_EP = [e for e in ALL if e not in HELD_OUT]   # the 5 train endpoints
_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load(e):
    d = np.load(os.path.join(EMB, f"chemberta_{e}.npz"), allow_pickle=True)
    return d["emb"], d["y"].astype(int), d["groups"], d["ids"], d["smiles"]


def morgan(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(str(s))
        out.append(_MGEN.GetFingerprintAsNumPy(m) if m is not None else np.zeros(2048, dtype=np.int8))
    return np.asarray(out)


def morgan_lr():
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, class_weight="balanced"))


def within_property(e):
    """Scaffold-GroupKFold AUROC + shuffled-label selectivity."""
    emb, y, g, _, _ = load(e)
    cv = GroupKFold(5)
    p = cross_val_predict(balanced_lr(), emb, y, cv=cv, groups=g, method="predict_proba", n_jobs=5)[:, 1]
    auc = roc_auc_score(y, p)
    ys = np.random.RandomState(123).permutation(y)
    pc = cross_val_predict(balanced_lr(), emb, ys, cv=GroupKFold(5), groups=g, method="predict_proba", n_jobs=5)[:, 1]
    return float(auc), float(auc - roc_auc_score(ys, pc)), len(y), int(y.sum())


def held_out_split(e):
    """Frozen scaffold split of a held-out endpoint into (train, test) = GroupKFold fold 0 as test."""
    emb, y, g, ids, smis = load(e)
    tr, te = next(iter(GroupKFold(5).split(emb, y, g)))
    return emb, y, g, ids, smis, tr, te


def pooled_train(featurizer):
    """Concatenate the 5 train endpoints; featurizer = 'emb' (ChemBERTa) or 'morgan'."""
    X, Y = [], []
    for e in TRAIN_EP:
        emb, y, _, _, smis = load(e)
        X.append(emb if featurizer == "emb" else morgan(smis))
        Y.append(y)
    return np.concatenate(X, 0), np.concatenate(Y, 0)


def main():
    out = {"arm": "orchestrate", "fm": "ChemBERTa-77M-MTR", "within": {}, "transfer": {},
           "held_out": HELD_OUT, "train_endpoints": TRAIN_EP}

    print("== WITHIN-property (orchestrate head, scaffold GroupKFold) ==", flush=True)
    for e in ALL:
        auc, sel, n, pos = within_property(e)
        out["within"][e] = {"auroc": round(auc, 4), "selectivity": round(sel, 4), "n": n, "pos": pos}
        print(f"  {e:12s} AUROC={auc:.3f} selectivity={sel:+.3f} n={n} pos={pos}", flush=True)

    print("\n== HELD-OUT-PROPERTY transfer (pooled-train head -> held-out test split) ==", flush=True)
    Xemb_tr, Yemb_tr = pooled_train("emb")
    Xmgn_tr, Ymgn_tr = pooled_train("morgan")
    fold = {"fold": "lpo_herg_clearance", "train_endpoints": TRAIN_EP, "held_out": {}}
    for e in HELD_OUT:
        emb, y, g, ids, smis, tr, te = held_out_split(e)
        yte = y[te]
        # in-property: head on this endpoint's OWN train split -> test split
        inp = balanced_lr().fit(emb[tr], y[tr])
        a_in = roc_auc_score(yte, inp.predict_proba(emb[te])[:, 1])
        # transfer: ChemBERTa head on the pooled train endpoints -> this endpoint's test split
        trf = balanced_lr().fit(Xemb_tr, Yemb_tr)
        p_tr = trf.predict_proba(emb[te])[:, 1]
        a_tr = roc_auc_score(yte, p_tr)
        # floor: same pooled-train transfer on cheap Morgan fingerprints (no learned representation)
        flr = morgan_lr().fit(Xmgn_tr, Ymgn_tr)
        a_fl = roc_auc_score(yte, flr.predict_proba(morgan(smis[te]))[:, 1])
        nt = (a_tr - a_fl) / (a_in - a_fl) if (a_in - a_fl) > 1e-6 else float("nan")
        lo, hi = cluster_boot(yte, p_tr, groups=g[te])
        out["transfer"][e] = {"in_property": round(float(a_in), 4), "transfer": round(float(a_tr), 4),
                              "floor_morgan": round(float(a_fl), 4), "NT": round(float(nt), 4),
                              "transfer_ci": [round(lo, 4), round(hi, 4)], "n_test": int(len(te))}
        fold["held_out"][e] = {"test_ids": [str(x) for x in ids[te]], "n_test": int(len(te))}
        band = "SKILL" if nt >= 0.90 else ("PARTIAL" if nt >= 0.60 else "MEMORIZES")
        print(f"  {e:12s} in-prop={a_in:.3f} transfer={a_tr:.3f} [floor(Morgan)={a_fl:.3f}] "
              f"NT={nt:+.3f} ({band})  n_test={len(te)}", flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "orchestrate_arm.json"), "w"), indent=2)
    os.makedirs(os.path.join(ROOT, "signal", "admet", "folds"), exist_ok=True)
    json.dump(fold, open(os.path.join(ROOT, "signal", "admet", "folds", "lpo_herg_clearance.json"), "w"), indent=2)
    print("\nsaved -> results/orchestrate_arm.json + signal/admet/folds/lpo_herg_clearance.json", flush=True)
    print("NOTE orchestrate transfer IS the H2 reference; NT_orch>0 means ChemBERTa carries cross-property "
          "structure above the Morgan floor. Bridge/LoRA (GPU) reuse these test_ids and compare here.", flush=True)


if __name__ == "__main__":
    main()
