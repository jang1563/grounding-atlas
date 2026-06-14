"""SFM-embedding rung, stage 1: generate ESM-2 embeddings for Meltome + verify the ceiling.

The "widest-open" rung (PROJECT_DESIGN 7.4): feed an SFM EMBEDDING to the LLM and ask whether it
can read a property out of it (the orchestrate-condition input). Capability-neutral property =
thermostability Tm (Meltome), NOT the prior hazard-flagged panel. This stage only makes the embeddings and
confirms the ceiling is real (the property IS in the embedding); the LLM output / ICL arm is
stage 2 (eval/sfm_embedding_output.py), gated on a high ceiling here.

Local: transformers ESM-2 on MPS/CPU (no fair-esm, no HPC). esm2_t30_150M (640-dim) by default.
Ceiling = logistic probe under a CLUSTER GroupKFold (Meltome's homology cluster column = leakage
control) + shuffled-label selectivity. No em dashes.
Env: SFM_N (320), SFM_MODEL (facebook/esm2_t30_150M_UR50D), SFM_BATCH (16).
"""
import csv
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MELTOME = os.path.join(ROOT, "protein_grounding", "data", "protein_meltome.csv")
OUTDIR = os.path.join(ROOT, "signal", "sfm_embedding")
MODEL = os.environ.get("SFM_MODEL", "facebook/esm2_t30_150M_UR50D")
N = int(os.environ.get("SFM_N", "320"))
BATCH = int(os.environ.get("SFM_BATCH", "16"))


def load(n):
    rows = [r for r in csv.DictReader(open(MELTOME)) if r.get("tm") not in (None, "")]
    tm = np.array([float(r["tm"]) for r in rows])
    med = float(np.median(tm))
    rng = np.random.RandomState(42)
    idx = np.arange(len(rows)); rng.shuffle(idx)
    pos = [i for i in idx if float(rows[i]["tm"]) > med]
    neg = [i for i in idx if float(rows[i]["tm"]) <= med]
    k = min(n // 2, len(pos), len(neg))
    sel = pos[:k] + neg[:k]
    seqs = [rows[i]["sequence"] for i in sel]
    y = np.array([1 if float(rows[i]["tm"]) > med else 0 for i in sel])
    grp = np.array([rows[i].get("cluster", rows[i]["id"]) for i in sel])
    ids = [rows[i]["id"] for i in sel]
    tms = np.array([float(rows[i]["tm"]) for i in sel])
    return seqs, y, grp, ids, tms, med


def embed(seqs):
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={dev} model={MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    out = []
    for b in range(0, len(seqs), BATCH):
        chunk = [s[:1022] for s in seqs[b:b + BATCH]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state            # (B, L, D)
        m = enc["attention_mask"].unsqueeze(-1).float()   # (B, L, 1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)    # mean-pool over real tokens
        out.append(pooled.float().cpu().numpy())
        print(f"  embedded {min(b + BATCH, len(seqs))}/{len(seqs)}", flush=True)
    return np.concatenate(out, 0)


def ceiling(X, y, groups):
    cv = GroupKFold(min(5, len(set(groups))))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, p)
    ys = np.random.RandomState(123).permutation(y)
    ps = cross_val_predict(clf, X, ys, cv=cv, groups=groups, method="predict_proba", n_jobs=-1)[:, 1]
    ctrl = roc_auc_score(ys, ps)
    return auc, ctrl


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    seqs, y, grp, ids, tms, med = load(N)
    print(f"== SFM embed Meltome :: n={len(seqs)} pos={int(y.sum())} clusters={len(set(grp))} "
          f"Tm median={med:.1f} ==", flush=True)
    X = embed(seqs)
    print(f"  embeddings: {X.shape}", flush=True)
    npz = os.path.join(OUTDIR, "meltome_esm2.npz")
    np.savez(npz, emb=X, y=y, groups=grp, ids=np.array(ids), tm=tms, tm_median=med, model=MODEL)
    auc, ctrl = ceiling(X, y, grp)
    meta = {"model": MODEL, "n": len(seqs), "dim": int(X.shape[1]), "tm_median": med,
            "ceiling_auroc": round(float(auc), 3), "control_auroc": round(float(ctrl), 3),
            "selectivity": round(float(auc - ctrl), 3), "split": "cluster_groupkfold", "npz": npz}
    json.dump(meta, open(os.path.join(OUTDIR, "meltome_esm2_ceiling.json"), "w"), indent=2)
    print(f"\nCEILING (ESM-2 {X.shape[1]}-dim -> Tm bin, cluster GroupKFold): AUROC={auc:.3f} "
          f"(shuffled-label control={ctrl:.3f}, selectivity={auc - ctrl:+.3f})", flush=True)
    print("VIABLE for the rung" if auc >= 0.65 and (auc - ctrl) >= 0.10 else
          "WEAK ceiling: pick a stronger ESM-2 (t33_650M) or a different property before the LLM arm")
    print(f"saved -> {npz}", flush=True)


if __name__ == "__main__":
    main()
