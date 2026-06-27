"""Experiment 2 (the 3-way bridge), keystone: embed the ADMET endpoints with a molecular foundation
model so all three arms (bridge / orchestrate / LoRA) consume ONE shared frozen embedding.

Per docs/BRIDGE_3WAY_PREREG.md Section 5: ChemBERTa-77M-MTR (384-dim, mean-pooled) over the canonical
('matched') molecule of each signal/admet/<endpoint>/pairs.jsonl, with Murcko-scaffold groups for the
leakage-controlled GroupKFold (P4). Writes signal/sfm_embedding/chemberta_<endpoint>.npz with
emb / y / groups (scaffold) / smiles / ids. MoLFormer-XL is the v2 robustness substrate (CHEM_MODEL
override + trust_remote_code). Local Mac MPS, no GPU needed. No em dashes.
Env: CHEM_MODEL (default DeepChem/ChemBERTa-77M-MTR), CHEM_ENDPOINTS (comma list, default all 7),
CHEM_BATCH (64).
"""
import json
import os

import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("CHEM_MODEL", "DeepChem/ChemBERTa-77M-MTR")
ENDPOINTS = os.environ.get(
    "CHEM_ENDPOINTS", "herg,cyp3a4,cyp2d6,ames,solubility,permeability,clearance").split(",")
BATCH = int(os.environ.get("CHEM_BATCH", "64"))
TRUST = os.environ.get("CHEM_TRUST", "1") == "1"   # MoLFormer needs trust_remote_code


def scaffold(smi):
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smi)
        return s or smi
    except Exception:
        return smi


def load_matched(endpoint):
    """The canonical molecule per id (condition == matched); unique (smiles, label, id)."""
    seen, smis, y, ids = set(), [], [], []
    for line in open(os.path.join(ROOT, "signal", "admet", endpoint, "pairs.jsonl")):
        r = json.loads(line)
        if r.get("condition") != "matched":
            continue
        s = r["representation"]
        if s in seen:
            continue
        seen.add(s)
        smis.append(s)
        y.append(int(r["label"]))
        ids.append(r["id"])
    return smis, np.array(y), ids


def embed(smis, tok, model, dev):
    out = []
    for i in range(0, len(smis), BATCH):
        enc = tok(smis[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state               # (B, T, D)
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)      # mean over real tokens
        out.append(pooled.float().cpu().numpy())
        if (i + BATCH) % 512 < BATCH:
            print(f"    {min(i + BATCH, len(smis))}/{len(smis)}", flush=True)
    return np.concatenate(out, 0)


def main():
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=TRUST)
    mkw = {"trust_remote_code": TRUST}
    if "MoLFormer" in MODEL:
        mkw["deterministic_eval"] = True
    model = AutoModel.from_pretrained(MODEL, **mkw).to(dev).eval()
    tag = MODEL.split("/")[-1]
    outdir = os.path.join(ROOT, "signal", "sfm_embedding")
    os.makedirs(outdir, exist_ok=True)
    print(f"MODEL={MODEL}  device={dev}  endpoints={ENDPOINTS}", flush=True)
    for e in ENDPOINTS:
        smis, y, ids = load_matched(e)
        emb = embed(smis, tok, model, dev)
        groups = np.array([scaffold(s) for s in smis])
        path = os.path.join(outdir, f"chemberta_{e}.npz" if "ChemBERTa" in MODEL else f"{tag}_{e}.npz")
        np.savez_compressed(path, emb=emb.astype(np.float32), y=y, groups=groups,
                            smiles=np.array(smis), ids=np.array(ids), model=MODEL)
        print(f"  {e}: n={len(y)} pos={int(y.sum())} dim={emb.shape[1]} scaffolds={len(set(groups))} -> {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
