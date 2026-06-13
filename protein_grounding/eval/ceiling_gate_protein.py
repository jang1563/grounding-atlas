"""Protein-branch ceiling gate (axis-B candidate screening), the ESM2 analog of
../../eval/ceiling_gate.py.

Question: is the property predictable from the representation CONTENT (here the amino-acid
sequence, via a protein SFM)? A high supervised ceiling means the signal is in the
representation, so a probe-vs-LLM head-to-head is meaningful. A low ceiling means there is
nothing for the LLM to fail to surface.

Specialist features = ESM2 (facebook/esm2_t33_650M_UR50D) mean-pooled over residues, the
protein analog of the Morgan fingerprint. Probes: logistic regression + random forest.
Two splits: random StratifiedKFold and cluster GroupKFold (MMseqs2 identity clusters, the
leakage control = the protein analog of the Murcko scaffold split). A large random->cluster
drop is the DTI trap (apparent ceiling from near-duplicate homologs); a small drop means
the signal is genuine sequence content.

Env: PG_CSV, ESM_MODEL, PG_BATCH.
"""
import csv
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

CSV = os.environ.get("PG_CSV", "protein_meltome.csv")
ESM_MODEL = os.environ.get("ESM_MODEL", "facebook/esm2_t33_650M_UR50D")
BATCH = int(os.environ.get("PG_BATCH", "8"))


def load(csv_path):
    ids, seqs, y, groups = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            seqs.append(row["sequence"])
            y.append(int(row["label"]))
            groups.append(row["cluster"])
    return ids, seqs, np.array(y), np.array(groups)


def esm_embed(seqs, model_name, batch_size):
    """Mean-pool ESM2 last_hidden_state over true residues (cls/eos/pad masked out)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    specials = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id}
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state  # (B, L, H)
        res = torch.ones_like(enc["input_ids"], dtype=torch.bool)
        for sp in specials:
            if sp is not None:
                res &= enc["input_ids"] != sp
        m = res.unsqueeze(-1).float()
        vec = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(vec.float().cpu().numpy())
        if (i + batch_size) % 200 == 0:
            print(f"  embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return np.concatenate(out, 0)


def evaluate(X, y, splits, clf):
    proba = cross_val_predict(clf, X, y, cv=splits, method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(y, proba), average_precision_score(y, proba)


def main():
    ids, seqs, y, groups = load(CSV)
    print(
        f"csv={CSV}  n={len(y)}  pos={int(y.sum())} ({y.mean():.1%})  "
        f"n_clusters={len(set(groups))}  model={ESM_MODEL}",
        flush=True,
    )
    X = esm_embed(seqs, ESM_MODEL, BATCH)
    print(f"ESM2 embeddings: {X.shape}", flush=True)

    rand = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    clust = list(GroupKFold(5).split(X, y, groups=groups))
    for name, clf in [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)),
    ]:
        ra, rp = evaluate(X, y, rand, clf)
        sa, sp = evaluate(X, y, clust, clf)
        print(
            f"  {name:7s} random AUROC={ra:.3f} AUPRC={rp:.3f}   "
            f"cluster AUROC={sa:.3f} AUPRC={sp:.3f}   (baseline AUPRC={y.mean():.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
