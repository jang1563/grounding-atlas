"""Audit every ADMET rung's label-orientation against an INDEPENDENT physicochemical prior.

The ames anti-grounding turned out to be an inverted label (eval/analyze_ames.py). This checks
the other endpoints the same way: for each rung, a structural/physicochemical descriptor that a
chemist expects to differ between the positive and negative class is computed, and we test
whether the class the current orientation calls "positive" actually has the expected direction.
A mismatch flags a possibly-inverted label. Read-only.

Run:  python eval/audit_orientations.py
"""
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMET = os.path.join(ROOT, "signal", "admet")

NITRO = Chem.MolFromSmarts("[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]")
BASIC_N = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=*);!$([NX3](=O))]")  # basic amine (hERG pharmacophore)

# rung -> (orientation, prior descriptor, expected sign of (positive_class - negative_class),
#          chemical rationale)
PRIORS = {
    "ames":         ("oppose", "nitro_alert", +1, "mutagens are nitroaromatic-rich"),
    "herg":         ("align",  "logp",        +1, "hERG blockers are lipophilic/basic"),
    "herg_basic":   ("align",  "basic_n",     +1, "hERG blockers carry a basic amine"),
    "cyp3a4":       ("align",  "logp",        +1, "CYP3A4 inhibitors trend lipophilic"),
    "cyp2d6":       ("align",  "basic_n",     +1, "CYP2D6 inhibitors are often basic amines"),
    "solubility":   ("oppose", "logp",        -1, "soluble compounds have LOWER logP"),
    "permeability": ("oppose", "tpsa",        -1, "permeable compounds have LOWER TPSA"),
}


def desc(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return {"logp": Crippen.MolLogP(m), "tpsa": rdMolDescriptors.CalcTPSA(m),
            "mw": Descriptors.MolWt(m),
            "nitro_alert": float(m.HasSubstructMatch(NITRO)),
            "basic_n": float(len(m.GetSubstructMatches(BASIC_N)))}


def main():
    print(f"{'rung':12s} {'orient':7s} {'prior':11s} {'pos':>7} {'neg':>7} {'Δ(pos-neg)':>11} "
          f"{'expect':>7}  verdict")
    for key, (orient, feat, sign, why) in PRIORS.items():
        ep = key.replace("_basic", "")
        path = os.path.join(ADMET, ep, "pairs.jsonl")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        rows = [r for r in rows if r.get("condition", "matched") == "matched"]
        vals = {0: [], 1: []}
        for r in rows:
            d = desc(r["representation"])
            if d is not None:
                vals[int(r["label"])].append(d[feat])
        pos_label = 1 if orient == "align" else 0   # orientation defines the positive class
        pos = np.mean(vals[pos_label])
        neg = np.mean(vals[1 - pos_label])
        delta = pos - neg
        ok = (np.sign(delta) == sign) or abs(delta) < 1e-6
        verdict = "consistent" if ok else "*** FLAG: possibly inverted ***"
        exp = "pos>neg" if sign > 0 else "pos<neg"
        print(f"{ep:12s} {orient:7s} {feat:11s} {pos:7.2f} {neg:7.2f} {delta:+11.2f} {exp:>7}  {verdict}")
        print(f"             rationale: {why}")


if __name__ == "__main__":
    main()
