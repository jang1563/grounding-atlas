"""Why does Ames mutagenicity anti-ground? Diagnostic on the committed n=100 ames items.

Joins each model's emitted P(mutagenic) (results/benchmark/<model>/raw.jsonl) with the SMILES
and label, (1) confirms the label direction independently via Kazius/Bursi mutagenicity
structural-alert enrichment, (2) asks what each model's probability actually tracks (Spearman
vs label, alerts, size/lipophilicity), and (3) prints the most confidently misranked structures.
Read-only.

Run:  python eval/analyze_ames.py
"""
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMES = os.path.join(ROOT, "signal/admet/ames/pairs.jsonl")
BENCH = os.path.join(ROOT, "results/benchmark")
MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]

# Canonical Ames mutagenicity structural alerts (Kazius/Bursi 2005, the dominant ones).
ALERTS = {
    "aromatic_nitro": "[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "aromatic_amine": "[NX3;H2,H1;!$(NC=O);!$(N=*)][c]",
    "three_member_NO": "[O,N]1[CX4][CX4]1",
    "azo": "[#6][NX2]=[NX2][#6]",
    "nitroso": "[#6][NX2]=O",
    "aliphatic_halide": "[CX4][Cl,Br,I]",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in ALERTS.items()}


def load_ames():
    d = {}
    for line in open(AMES):
        r = json.loads(line)
        if r.get("condition", "matched") == "matched":
            d[r["id"]] = (r["representation"], int(r["label"]))
    return d


def load_probs(model):
    d = {}
    for line in open(os.path.join(BENCH, model, "raw.jsonl")):
        r = json.loads(line)
        if r["rung"] == "admet/ames":
            d[r["id"]] = r["prob"]
    return d


def feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    f = {"mw": Descriptors.MolWt(m), "logp": Descriptors.MolLogP(m),
         "arom_rings": rdMolDescriptors.CalcNumAromaticRings(m),
         "heavy": m.GetNumHeavyAtoms()}
    for k, p in PATS.items():
        f[k] = int(m.HasSubstructMatch(p)) if p is not None else 0
    f["any_alert"] = int(any(f[k] for k in ALERTS))
    return f


def main():
    ames = load_ames()
    probs = {m: load_probs(m) for m in MODELS}
    ids = sorted(set(ames) & set(probs[MODELS[0]]))
    rows = []
    for i in ids:
        smi, lab = ames[i]
        f = feats(smi)
        if f is None:
            continue
        f.update({"id": i, "label": lab, "smi": smi})
        for m in MODELS:
            f["P_" + m] = probs[m].get(i, np.nan)
        rows.append(f)
    y = np.array([r["label"] for r in rows])
    print(f"n={len(rows)}  mutagenic(label1)={int(y.sum())}  non(label0)={int((1 - y).sum())}")

    print("\n[1] structural-alert rate (independent label-direction check), label1 vs label0:")
    for k in list(ALERTS) + ["any_alert"]:
        a1 = np.mean([r[k] for r in rows if r["label"] == 1])
        a0 = np.mean([r[k] for r in rows if r["label"] == 0])
        print(f"  {k:18s} label1={a1:.2f}  label0={a0:.2f}  enrichment={a1 - a0:+.2f}")

    print("\n[2] each model's P(mutagenic): AUROC and Spearman rho with features:")
    for m in MODELS:
        P = np.array([r["P_" + m] for r in rows])
        print(f"\n  {m}  AUROC={roc_auc_score(y, P):.3f}")
        for k in ["label", "any_alert", "aromatic_nitro", "aromatic_amine", "mw", "logp", "arom_rings"]:
            x = np.array([float(r[k]) for r in rows])
            print(f"    rho(P, {k:14s}) = {spearmanr(P, x).correlation:+.3f}")

    m = MODELS[0]
    rs = sorted(rows, key=lambda r: r["P_" + m])
    print(f"\n[3] {m}: mutagenic(label1) compounds rated LOWEST P(mutagenic):")
    for r in [r for r in rs if r["label"] == 1][:5]:
        al = [k for k in ALERTS if r[k]]
        print(f"    P={r['P_' + m]:.2f} MW={r['mw']:.0f} alerts={al or '-'}  {r['smi'][:48]}")
    print(f"  {m}: non-mutagenic(label0) compounds rated HIGHEST P(mutagenic):")
    for r in [r for r in reversed(rs) if r["label"] == 0][:5]:
        al = [k for k in ALERTS if r[k]]
        print(f"    P={r['P_' + m]:.2f} MW={r['mw']:.0f} alerts={al or '-'}  {r['smi'][:48]}")


if __name__ == "__main__":
    main()
