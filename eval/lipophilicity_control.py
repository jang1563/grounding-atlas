"""Two decisive local controls for the 'encodes chemistry' soft bedrock (no GPU/API).

A deep review flagged that the hERG structure signal (Morgan probe ~0.82, activation
probe ~0.79, char-n-gram ~0.80) may be (i) a coarse LIPOPHILICITY axis, not hERG-specific
chemistry, and (ii) SURFACE-STRING orthography, not structure. These are testable on the
structure/orthography probes locally (the activation arm needs GPU and is queued separately):

  A. Lipophilicity: how much hERG AUROC do just logP / MW / TPSA give, vs Morgan FP?
     descriptors-only ~ Morgan  ->  the 'structure ceiling' is mostly lipophilicity.
  B. Orthography: does the char-n-gram probe (no chemistry) survive RANDOMIZED SMILES?
     drops on randomized  ->  the 0.80 was canonical-string orthography, not robust.

All under the same Murcko-scaffold GroupKFold as the activation arm. Morgan/descriptors
also residualized: train on descriptor-residualized Morgan to see if structure adds beyond
lipophilicity. Run: python eval/lipophilicity_control.py
No em dashes.
"""
import json
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")


def load(n=2000):
    by = defaultdict(list)
    for line in open(PAIRS):
        r = json.loads(line)
        if r["condition"] == "matched":
            by[int(r["label"])].append(r["representation"])
    rng = np.random.RandomState(42)
    smis, ys = [], []
    for lab in (0, 1):
        it = by[lab][:]
        rng.shuffle(it)
        for s in it[:n // 2]:
            smis.append(s); ys.append(lab)
    return smis, np.array(ys)


def featurize(smis):
    morgan, desc, scaf, rand = [], [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=float))
        desc.append([Descriptors.MolLogP(m), Descriptors.MolWt(m), Descriptors.TPSA(m)])
        scaf.append(MurckoScaffold.MurckoScaffoldSmiles(s) or s)
        try:
            rand.append(Chem.MolToSmiles(m, doRandom=True))
        except Exception:
            rand.append(s)
    return np.array(morgan), np.array(desc), scaf, rand


def cv_auroc(X, y, groups, clf):
    p = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(5),
                          method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 3)


def main():
    smis, y = load(2000)
    morgan, desc, scaf, rand = featurize(smis)
    groups = scaf
    print(f"n={len(y)} pos={int(y.sum())} scaffolds={len(set(scaf))}", flush=True)

    lr = lambda: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    lrd = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # A. lipophilicity vs structure
    morgan_auc = cv_auroc(morgan, y, groups, lr())
    desc_auc = cv_auroc(desc, y, groups, lrd())
    both_auc = cv_auroc(np.hstack([morgan, desc]), y, groups, lrd())
    # residualize Morgan columns on the 3 descriptors, then probe the residuals
    resid = morgan - LinearRegression().fit(desc, morgan).predict(desc)
    resid_auc = cv_auroc(resid, y, groups, lrd())

    # B. orthography: char-n-gram on canonical vs randomized SMILES
    cv = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    Xc = cv.fit_transform(smis)
    Xr = TfidfVectorizer(analyzer="char", ngram_range=(2, 5)).fit_transform(rand)
    cgram_canon = cv_auroc(Xc, y, groups, LogisticRegression(max_iter=1000))
    cgram_rand = cv_auroc(Xr, y, groups, LogisticRegression(max_iter=1000))

    out = {"n": len(y),
           "A_lipophilicity": {
               "descriptors_only_logP_MW_TPSA": desc_auc,
               "morgan_fp": morgan_auc,
               "morgan_plus_descriptors": both_auc,
               "morgan_residualized_on_descriptors": resid_auc},
           "B_orthography": {
               "char_ngram_canonical": cgram_canon,
               "char_ngram_randomized": cgram_rand},
           "refs": {"activation_probe": 0.787, "output_arm": 0.453}}
    json.dump(out, open(os.path.join(ROOT, "results", "lipophilicity_control.json"), "w"), indent=2)
    print("\n== A. lipophilicity vs structure (scaffold GroupKFold AUROC) ==")
    print(f"  descriptors only (logP/MW/TPSA) = {desc_auc}")
    print(f"  Morgan FP                       = {morgan_auc}")
    print(f"  Morgan + descriptors            = {both_auc}")
    print(f"  Morgan residualized on desc     = {resid_auc}  (structure signal beyond lipophilicity)")
    print("== B. orthography (char-n-gram, no chemistry) ==")
    print(f"  char-n-gram canonical SMILES    = {cgram_canon}")
    print(f"  char-n-gram randomized SMILES   = {cgram_rand}  (drop => canonical-orthography)")


if __name__ == "__main__":
    main()
