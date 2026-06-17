"""Firm the spectra_ms near-miss: is the LoRA-8B 0.706 a real win over the best non-train placement,
or noise / dominated? Reproduces the EXACT ws3_lora train/test split (deterministic seed-42 load +
scaffold GroupShuffleSplit) and, on the SAME held-out test molecules, measures with bootstrap CIs:
  - cheap binned-m/z probe (no structure)               = the bar TRAIN edged (rung 0.667)
  - structure Morgan probe (true SMILES)                = the PERFECT-elucidation orchestrate ceiling
and compares both to the fixed LoRA-8B finetuned 0.706 (results/ws3_lora_cells.json).

Read: if 0.706 sits inside the cheap probe's 95% CI, the +0.04 edge is not real (near-miss = noise).
If the structure ceiling is clearly above 0.706, orchestrate-via-elucidation beats train WHEN
elucidation works (real MS elucidation, SpecTUS ~65% / GPT-4o exact 1.4%, sits between cheap and the
structure ceiling). Local (rdkit + sklearn), no GPU. No em dashes.
"""
import csv
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERG = os.path.join(ROOT, "data", "herg.csv")
N = 1250
LORA_FT = 0.706   # spectra_ms finetuned (results/ws3_lora_cells.json)


def peaks_scaf(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    masses = {round(Descriptors.ExactMolWt(m), 1)}
    try:
        for f in BRICS.BRICSDecompose(m):
            fm = Chem.MolFromSmiles(f)
            if fm is not None:
                masses.add(round(Descriptors.ExactMolWt(fm), 1))
    except Exception:
        pass
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smi
    except Exception:
        scaf = smi
    return ", ".join(str(x) for x in sorted(masses)), scaf


def mz_hist(pl):                       # cheap surface: binned m/z, 10-Da bins to 1000
    v = np.zeros(100)
    for t in pl.split(","):
        try:
            x = float(t)
        except ValueError:
            continue
        if 0 <= x < 1000:
            v[int(x // 10)] += 1
    return v


def boot_ci(yt, sc, n=2000):
    rs = np.random.RandomState(0)
    a = []
    for _ in range(n):
        b = rs.choice(len(yt), len(yt), replace=True)
        if len(set(yt[b])) > 1:
            a.append(roc_auc_score(yt[b], sc[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    # rebuild records in herg.csv order (matches prep_lora_cells.py / the jsonl), carrying SMILES
    recs = []
    for r in csv.DictReader(open(HERG)):
        ps = peaks_scaf(r["smiles"])
        if ps is None:
            continue
        recs.append((ps[0], r["smiles"], ps[1], int(float(r["label"]))))
    # replicate ws3_lora.load(N): by_label in order, ONE rng(42) shuffles class 0 then class 1
    by = {0: [], 1: []}
    for rec in recs:
        by[rec[3]].append(rec)
    rng = np.random.RandomState(42)
    sel = []
    for lab in (0, 1):
        it = by[lab][:]
        rng.shuffle(it)
        sel += it[:N // 2]
    reps = [s[0] for s in sel]; smis = [s[1] for s in sel]
    grp = [s[2] for s in sel]; y = np.array([s[3] for s in sel])
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(reps, y, grp))
    print(f"replicated split: train={len(tr)} test={len(te)}  (LoRA had 769/481; match confirms same split)")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    Xmz = np.array([mz_hist(reps[i]) for i in range(len(reps))])
    Xfp = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None
                    else np.zeros(2048) for s in smis])
    yte = y[te]

    def arm(X, name):
        clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        sc = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(yte, sc); lo, hi = boot_ci(yte, sc)
        print(f"  {name:48s} AUROC={auc:.3f} [95% {lo:.3f}, {hi:.3f}]")
        return auc, lo, hi

    print(f"\nSame-split arms on the n={len(te)} held-out test set:")
    c_auc, c_lo, c_hi = arm(Xmz, "cheap binned-m/z probe (no structure)")
    s_auc, s_lo, s_hi = arm(Xfp, "structure Morgan (perfect-elucidation ceiling)")
    print(f"  {'LoRA-8B finetuned (Cayuga, fixed point estimate)':48s} AUROC={LORA_FT:.3f}")

    print("\nVERDICT:")
    print(f"  train vs cheap: 0.706 {'INSIDE' if c_lo <= LORA_FT <= c_hi else 'OUTSIDE'} cheap 95% CI "
          f"[{c_lo:.3f},{c_hi:.3f}] -> edge {'within noise (near-miss not real)' if LORA_FT <= c_hi else 'real'}")
    print(f"  train vs elucidation ceiling: structure {s_auc:.3f} {'>' if s_auc > LORA_FT else '<='} 0.706 -> "
          f"{'orchestrate-if-elucidation-works beats train' if s_auc > LORA_FT else 'train competitive even vs perfect elucidation'}")
    print("  (real MS elucidation, SpecTUS ~65%, lies between cheap and the structure ceiling.)")


if __name__ == "__main__":
    main()
