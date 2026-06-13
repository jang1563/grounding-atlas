"""DECIMER OCSR baseline to calibrate the molecular-image perception floor.

ws3_image.py found Claude transcribes a rendered molecule to a half-right SMILES
(mean Tanimoto 0.54, exact 0.16). Is that "the model cannot perceive structure," or
"320px RDKit renders are just hard"? A dedicated OCSR tool (DECIMER) on the SAME 120
images settles it: if DECIMER is much better (Tanimoto ~0.9), the renders are legible
and Claude's perception is genuinely poor (floor confirmed, orchestrate-a-specialist
justified); if DECIMER is also ~0.5, the renders are the limit (a rendering statement).

Runs on Cayuga in the `decimer` conda env (needs decimer + rdkit). Same 120-molecule
balanced hERG set, seed 42, same RDKit MolDraw2DCairo 320px render as ws3_image.py.
No em dashes.
"""
import os
import json
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")


def load(n):
    by = defaultdict(list)
    for line in open(PAIRS):
        r = json.loads(line)
        if r["condition"] == "matched":
            by[int(r["label"])].append(r["representation"])
    rng = np.random.RandomState(42)
    out = []
    for lab in (0, 1):
        it = by[lab][:]
        rng.shuffle(it)
        out += [(s, lab) for s in it[:n // 2]]
    return out


def render(smi, path, size=320):
    m = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    d.DrawMolecule(m)
    d.FinishDrawing()
    with open(path, "wb") as fh:
        fh.write(d.GetDrawingText())
    return m


def main():
    from DECIMER import predict_SMILES
    data = load(int(os.environ.get("DEC_N", "120")))
    tani, exact, flat, valid = [], 0, 0, 0
    os.makedirs("/tmp/decimg", exist_ok=True)
    for i, (smi, lab) in enumerate(data):
        p = f"/tmp/decimg/mol_{i}.png"
        mol = render(smi, p)
        try:
            pred = predict_SMILES(p)
        except Exception:
            pred = ""
        cm = Chem.MolFromSmiles(pred) if pred else None
        if cm is not None:
            valid += 1
            tani.append(DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
                AllChem.GetMorganFingerprintAsBitVect(cm, 2, 2048)))
            if Chem.MolToSmiles(cm) == Chem.MolToSmiles(mol):
                exact += 1
            if Chem.MolToSmiles(cm, isomericSmiles=False) == Chem.MolToSmiles(mol, isomericSmiles=False):
                flat += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(data)}", flush=True)
    n = len(data)
    out = {"tool": "DECIMER", "n": n,
           "valid_rate": round(valid / n, 3),
           "mean_tanimoto": round(float(np.mean(tani)), 3) if tani else None,
           "exact_rate": round(exact / n, 3),
           "exact_flat_rate": round(flat / n, 3),
           "claude_ref": {"valid": 0.725, "mean_tanimoto": 0.544, "exact": 0.158}}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "decimer_ocsr.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nDECIMER OCSR (same 120 images): valid={out['valid_rate']} "
          f"mean_Tanimoto={out['mean_tanimoto']} exact={out['exact_rate']} "
          f"exact_flat={out['exact_flat_rate']}")
    print(f"Claude was: valid 0.725, mean_Tanimoto 0.544, exact 0.158")


if __name__ == "__main__":
    main()
