"""WS1 axis-B ceiling gate.

Question: is the property predictable from the representation CONTENT (here SMILES)?
A high supervised ceiling means the signal is in the representation, so a probe-vs-LLM
head-to-head is meaningful there. A low ceiling means there is nothing for the LLM to
fail to surface, so the domain is not a head-to-head candidate.

This is the gate, not the head-to-head. Random (cold-compound) split; scaffold split
is a stricter follow-up. See eval/README.md Phase 2.
"""
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

DB = os.environ.get("NEGBIODB_ADMET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Negative_result_DB", "data", "negbiodb_admet.db"))
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "herg"


def load(endpoint):
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        SELECT c.canonical_smiles, r.outcome
        FROM admet_results r
        JOIN admet_assays a ON r.assay_id = a.id
        JOIN admet_compounds c ON r.compound_id = c.id
        WHERE a.endpoint = ? AND r.outcome IN ('pass','fail')
          AND c.canonical_smiles IS NOT NULL AND c.canonical_smiles != ''
        """,
        (endpoint,),
    ).fetchall()
    con.close()
    # compound-level aggregate: any fail -> fail (1), else pass (0)
    agg = defaultdict(list)
    for smi, out in rows:
        agg[smi].append(out)
    return [(smi, 1 if "fail" in outs else 0) for smi, outs in agg.items()]


def featurize(data):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, y, scaf = [], [], []
    for smi, label in data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        X.append(gen.GetFingerprintAsNumPy(mol))
        y.append(label)
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smi)
        except Exception:
            scaf.append(smi)
    return np.asarray(X), np.asarray(y), np.asarray(scaf)


def evaluate(X, y, splits, clf):
    proba = cross_val_predict(clf, X, y, cv=splits, method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(y, proba), average_precision_score(y, proba)


def main():
    data = load(ENDPOINT)
    X, y, scaf = featurize(data)
    print(
        f"endpoint={ENDPOINT}  n_compounds={len(y)}  pos(fail)={int(y.sum())} "
        f"({y.mean():.1%})  n_scaffolds={len(set(scaf))}"
    )
    rand = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    scaff = list(GroupKFold(5).split(X, y, groups=scaf))
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)),
    ]:
        ra, rp = evaluate(X, y, rand, clf)
        sa, sp = evaluate(X, y, scaff, clf)
        print(
            f"  {name:7s} random AUROC={ra:.3f} AUPRC={rp:.3f}   "
            f"scaffold AUROC={sa:.3f} AUPRC={sp:.3f}   (baseline={y.mean():.3f})"
        )


if __name__ == "__main__":
    main()
