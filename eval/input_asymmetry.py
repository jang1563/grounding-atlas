"""Input-asymmetry control (eval/README confound 2), cheap + local, no GPU.

The objection: the structure-probe reads an SFM embedding (Morgan FP / ESM-2), already
processed by a chemistry/biology model, while the LLM reads raw text. So the probe-minus-LLM
gap could be "the SFM pre-processed the input into a decodable form" rather than
"the property is internally present in the LLM but not surfaced".

The control: probe the property from a RAW-TEXT featurization that uses NO chemistry and NO
SFM, just character n-grams of the SMILES STRING the LLM itself reads, with a linear model on
top. If that still beats the LLM output by a wide margin, the signal is linearly decodable from
the very same raw string the LLM sees, so the LLM's near-chance output is an EXPRESSION gap, not
an artifact of the probe's better (SFM-processed) input. Same balanced sample, same Murcko
scaffold GroupKFold, shuffled-label selectivity control. Reference: ../results/selection_bias.md.
"""
import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

CSV = "../data/herg.csv"
N = 1250            # balanced 625/625, matches activation_arm.py
LLM_OUTPUT_AUROC = 0.453   # measured 8B zero-shot output (results/head_to_head.md)


def load(path, n):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["smiles"], int(r["label"])))
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smi) or smi
    except Exception:
        return smi


def cv_auc(make_X, smis, y, groups, label):
    X = make_X(smis)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    auc = roc_auc_score(y, p)
    # shuffled-label selectivity control
    yr = np.random.RandomState(123).permutation(y)
    pr = cross_val_predict(clf, X, yr, cv=GroupKFold(5), groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    ctrl = roc_auc_score(yr, pr)
    print(f"  {label:34s} AUROC={auc:.3f}  (shuffled-label control={ctrl:.3f}, selectivity={auc-ctrl:+.3f})", flush=True)
    return auc


def morgan(smis):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(2048, dtype=np.int8))
    return np.asarray(out)


def char_ngram(smis):
    # pure surface text of the SMILES string: char 2-5 grams, no chemistry, no SFM
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3).fit_transform(smis)


def main():
    data = load(CSV, N)
    smis = [s for s, _ in data]
    y = np.array([l for _, l in data])
    groups = np.array([scaffold_of(s) for s in smis])
    print(f"n={len(y)} pos={int(y.sum())} scaffolds={len(set(groups))}  (balanced, Murcko scaffold GroupKFold)\n")

    print("Probe arms (Murcko scaffold GroupKFold, shuffled-label selectivity):")
    auc_morgan = cv_auc(morgan, smis, y, groups, "SFM ceiling (Morgan FP, chemistry)")
    auc_char = cv_auc(char_ngram, smis, y, groups, "RAW-TEXT (char 2-5 gram, no SFM)")

    print(f"\n  LLM output arm (8B zero-shot, same task)   AUROC={LLM_OUTPUT_AUROC:.3f}  (results/head_to_head.md)")
    print("\nRead:")
    print(f"  raw-text char-ngram probe = {auc_char:.3f}  vs  LLM output = {LLM_OUTPUT_AUROC:.3f}  "
          f"(gap {auc_char-LLM_OUTPUT_AUROC:+.3f})")
    if auc_char - LLM_OUTPUT_AUROC > 0.10:
        print("  => the hERG signal is linearly decodable from the RAW SMILES STRING the LLM also reads,")
        print("     with no chemistry and no SFM. So the probe-minus-LLM gap is NOT an SFM input advantage;")
        print("     the LLM sees the same string and (per the activation arm) encodes it internally but does")
        print("     not surface it. Input-asymmetry confound controlled: the gap is EXPRESSION, not input.")
    else:
        print("  => raw-text probe does not beat the LLM; the SFM input advantage cannot be ruled out here.")


if __name__ == "__main__":
    main()
