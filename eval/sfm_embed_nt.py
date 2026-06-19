"""2nd SFM leg (genomic): Nucleotide Transformer embeddings of the RNA coding sequences.

Parallels the ESM-2 protein leg. Feed an NT embedding to the LLM and measure whether it
can read coding-vs-noncoding from the abstract vector (expected chance), while a read-out
head on the same embedding is the ceiling. Generalizes the SFM-embedding finding from a
protein FM (ESM) to a genomic FM (NT), and gives a within-modality contrast on RNA: the
raw sequence-as-text verbalizes (ORF heuristic, 0.84-0.96) but the SFM embedding should not.

Writes signal/sfm_embedding/rna_nt.npz {emb, y, model} + prints a read-out head ceiling.
Run on CPU/MPS:  python eval/sfm_embed_nt.py
"""
import csv
import os
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Compatibility shim: NT v2's bundled modeling code was written for transformers 4.x and imports
# find_pruneable_heads_and_indices, which was removed in 5.x. Re-add the canonical implementation so
# the custom module imports cleanly (a version shim, not a behavior change).
import transformers.pytorch_utils as _pu  # noqa: E402

if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[int(head)] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
    _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal/rna/coding.csv"))))
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    random.Random(0).shuffle(pos)
    random.Random(1).shuffle(neg)
    k = min(len(pos), len(neg), 300)
    sel = pos[:k] + neg[:k]
    seqs = [r["smiles"].upper().replace("U", "T") for r in sel]
    y = np.array([int(r["label"]) for r in sel])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev}  model={MODEL}  n={len(seqs)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL, output_hidden_states=True, trust_remote_code=True).to(dev).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            batch = seqs[i:i + 16]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96).to(dev)
            h = model(**enc).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1)
            mean = (h * mask).sum(1) / mask.sum(1)
            embs.append(mean.cpu().float().numpy())
            print(f"  embedded {min(i + 16, len(seqs))}/{len(seqs)}", flush=True)
    emb = np.concatenate(embs)
    out = os.path.join(ROOT, "signal/sfm_embedding/rna_nt.npz")
    np.savez(out, emb=emb, y=y, model=MODEL)
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                          emb, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    print(f"\nwrote {out}", flush=True)
    print(f"NT embedding head ceiling AUROC={roc_auc_score(y, p):.3f} (n={len(y)}, dim={emb.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
