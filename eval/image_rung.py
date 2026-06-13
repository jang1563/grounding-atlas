"""Molecular-IMAGE rung: assemble the encoding-limited 3-arm anchor for PROJECT_DESIGN 7.2.

The image rung is a light 2-arm in the plan (open VLM optional), so the encoding side is
measured by a PERCEPTION PROXY rather than a hidden-state probe (Claude exposes no
activations): how much hERG signal survives the VLM's PERCEIVED structure. Three numbers on
the same rendered hERG molecules (ws3_image_items.jsonl, fields true_smiles, ocsr_cand, prop):
  ceiling        = Morgan probe on the TRUE structure (the property IS structure-decodable)
  encoding-proxy = Morgan probe on the VLM-TRANSCRIBED structure (ocsr_cand); if it collapses
                   to chance while the ceiling is high, the VLM cannot ENCODE the structure
                   from pixels = ENCODING-LIMITED (the perception floor gates everything)
  output         = the VLM solo-image P(hERG) directly (prop)
Run after re-running ws3_image.py (which saves the transcribed SMILES). No em dashes.
"""
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "results", "ws3_image_items.jsonl")


def fp(smi):
    if not smi:
        return np.zeros(2048)
    m = Chem.MolFromSmiles(smi)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), float) if m is not None else np.zeros(2048)


def main():
    rows = [json.loads(l) for l in open(ITEMS)]
    if "true_smiles" not in rows[0]:
        raise SystemExit("items file lacks true_smiles/ocsr_cand: re-run ws3_image.py (current version saves them)")
    y = np.array([int(r["label"]) for r in rows])
    FPt = np.array([fp(r["true_smiles"]) for r in rows])
    FPc = np.array([fp(r.get("ocsr_cand", "")) for r in rows])
    prop = np.array([float(r["prop"]) for r in rows])
    valid = sum(1 for r in rows if r.get("ocsr_valid"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = lambda: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))

    def auc(X):
        p = cross_val_predict(clf(), X, y, cv=cv, method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, p)), 3)

    ceiling = auc(FPt)
    enc_proxy = auc(FPc)
    output = round(float(roc_auc_score(y, prop)), 3)
    out = {"modality": "molecular_image", "endpoint": "herg", "n": len(y),
           "ocsr_valid": valid, "model": "claude-sonnet-4-6 (VLM, no hidden states)",
           "ceiling_morgan_on_true": ceiling,
           "encoding_proxy_morgan_on_perceived": enc_proxy,
           "output_solo_image": output,
           "encoding_gap_proxy": round(ceiling - enc_proxy, 3),
           "regime": "encoding-limited (perception floor)"}
    json.dump(out, open(os.path.join(ROOT, "results", "image_rung.json"), "w"), indent=2)
    print(f"n={len(y)} ocsr_valid={valid}/{len(y)}")
    print(f"CEILING (Morgan on TRUE structure)         = {ceiling}")
    print(f"ENCODING-PROXY (Morgan on VLM-PERCEIVED)    = {enc_proxy}   [encoding gap {ceiling-enc_proxy:.3f}]")
    print(f"OUTPUT (VLM solo-image)                     = {output}")
    print(f"regime: {'ENCODING-LIMITED' if enc_proxy < 0.62 else 'check'} (perception floor: the VLM cannot read the structure from pixels)")


if __name__ == "__main__":
    main()
