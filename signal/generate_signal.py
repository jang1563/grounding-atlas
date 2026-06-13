"""WS2 verifiable-signal generator + verifiability gate.

Generalizes the per-branch ceiling pipelines (eval/ceiling_gate.py, the protein/variant
ceiling scripts) into ONE reusable tool, the WS2 deliverable PROJECT_DESIGN asks for:

  ingest a (content, property) source
    -> emit standardized (representation, verifiable-property) pairs + content-sensitivity
       condition variants (matched / scrambled / re-notation; mismatched + content-only are
       LLM-arm prompt variants, see ../eval/README.md)
    -> run the VERIFIABILITY GATE: a content-feature probe under a leakage-controlled
       (cold) split + a shuffled-label selectivity control
    -> PASS only if the property is genuinely in the content and survives the cold split.

The gate is the point: it tells the B-axis instrument which (representation, property) tasks
are real signal (probe ceiling high, leakage-free) versus the DTI trap (high on a random
split, collapses cold). Borrows NullAtlas's verifiable-signal method (Negative_result_DB);
NullAtlas's negative-evidence-coverage result (rho=-0.70) is CITED, not re-measured here.

Modality-general by design: a Source supplies (content, label, modality) + a featurizer + a
leakage-control grouper + optional notation variants. SMILES/ADMET is implemented and gated
locally (rdkit + sklearn, CPU). Sequence/variant/metabolite plug in the same way with an
SFM-embedding featurizer, gated on GPU (Expanse); see ../signal/README.md.

Usage:  python generate_signal.py [endpoint ...]   (default: all ADMET endpoints)
"""
import os
import sys
import json
import sqlite3
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

ADMET_DB = os.environ.get("NEGBIODB_ADMET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Negative_result_DB", "data", "negbiodb_admet.db"))
OUTDIR = os.path.join(os.path.dirname(__file__), "admet")
ALL_ENDPOINTS = ["herg", "cyp3a4", "cyp2d6", "ames", "solubility", "permeability", "clearance"]

# Gate thresholds: the property must be readable from CONTENT under the COLD split (not just
# the random split), and the probe must be selective (beat a shuffled-label control).
GATE_COLD_AUROC = 0.65       # signal survives a scaffold (cold-compound) split
GATE_SELECTIVITY = 0.10      # probe reads real signal, not label noise
MAX_PAIRS = 4000             # cap emitted pairs per endpoint


# ---- source: NegBioDB ADMET (SMILES -> binary outcome) -----------------------------
def load_admet(endpoint):
    con = sqlite3.connect(ADMET_DB)
    rows = con.execute(
        """SELECT c.canonical_smiles, r.outcome
           FROM admet_results r
           JOIN admet_assays a ON r.assay_id = a.id
           JOIN admet_compounds c ON r.compound_id = c.id
           WHERE a.endpoint = ? AND r.outcome IN ('pass','fail')
             AND c.canonical_smiles IS NOT NULL AND c.canonical_smiles != ''""",
        (endpoint,),
    ).fetchall()
    con.close()
    agg = defaultdict(list)                      # compound-level: any fail -> fail
    for smi, out in rows:
        agg[smi].append(out)
    return [(smi, 1 if "fail" in outs else 0) for smi, outs in agg.items()]


# ---- featurizer + leakage-control grouper (SMILES) ---------------------------------
def featurize_smiles(data):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, y, grp, kept = [], [], [], []
    for smi, label in data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        X.append(gen.GetFingerprintAsNumPy(mol))
        y.append(label)
        try:
            grp.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smi)
        except Exception:
            grp.append(smi)
        kept.append((smi, label))
    return np.asarray(X), np.asarray(y), np.asarray(grp), kept


# ---- content-sensitivity notation variants (SMILES) --------------------------------
def randomized_smiles(smi):
    """A VALID alternate notation of the SAME molecule (re-notation / invariance test)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except Exception:
        return None


def scramble_smiles(smi, seed=0):
    """Corrupt the string so the molecular signal is destroyed (a true scramble, not a
    re-ordering): shuffle the characters. A content-grounder should degrade on this."""
    rng = np.random.RandomState(seed)
    chars = list(smi)
    rng.shuffle(chars)
    return "".join(chars)


# ---- the verifiability gate --------------------------------------------------------
def gate(X, y, groups):
    out = {"n": int(len(y)), "pos": int(y.sum()), "baseline": round(float(y.mean()), 3),
           "n_groups": int(len(set(groups)))}
    rand = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    cold = list(GroupKFold(5).split(X, y, groups=groups))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    pr = cross_val_predict(clf, X, y, cv=rand, method="predict_proba", n_jobs=-1)[:, 1]
    pc = cross_val_predict(clf, X, y, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    pc_rf = cross_val_predict(rf, X, y, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    # shuffled-label selectivity (Hewitt-Liang) on the cold split
    ys = np.random.RandomState(123).permutation(y)
    ps = cross_val_predict(clf, X, ys, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    out["random_auroc"] = round(float(roc_auc_score(y, pr)), 3)
    out["cold_auroc"] = round(float(roc_auc_score(y, pc)), 3)
    out["cold_auroc_rf"] = round(float(roc_auc_score(y, pc_rf)), 3)
    out["cold_auprc"] = round(float(average_precision_score(y, pc)), 3)
    out["control_auroc"] = round(float(roc_auc_score(ys, ps)), 3)
    out["selectivity"] = round(out["cold_auroc"] - out["control_auroc"], 3)
    out["leakage_drop"] = round(out["random_auroc"] - out["cold_auroc"], 3)  # big -> DTI trap
    out["passed"] = bool(out["cold_auroc"] >= GATE_COLD_AUROC and out["selectivity"] >= GATE_SELECTIVITY)
    return out


# ---- emit standardized signal records ----------------------------------------------
def emit_pairs(endpoint, kept, path, n_variants=200):
    rng = np.random.RandomState(7)
    idx = list(range(len(kept)))
    rng.shuffle(idx)
    idx = idx[:MAX_PAIRS]
    vset = set(idx[:n_variants])                 # a subset also gets notation variants
    with open(path, "w") as f:
        for i in idx:
            smi, label = kept[i]
            rec = {"id": f"{endpoint}_{i}", "modality": "smiles", "property": endpoint,
                   "condition": "matched", "representation": smi, "label": int(label),
                   "source": "negbiodb_admet"}
            f.write(json.dumps(rec) + "\n")
            if i in vset:
                rn = randomized_smiles(smi)
                if rn:
                    f.write(json.dumps({**rec, "condition": "re_notation", "representation": rn}) + "\n")
                f.write(json.dumps({**rec, "condition": "scrambled",
                                    "representation": scramble_smiles(smi)}) + "\n")


def main():
    endpoints = sys.argv[1:] or ALL_ENDPOINTS
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    for ep in endpoints:
        data = load_admet(ep)
        if len(data) < 100:
            print(f"{ep:14s} SKIP (n={len(data)} too small)", flush=True)
            continue
        X, y, grp, kept = featurize_smiles(data)
        if len(set(y)) < 2:
            print(f"{ep:14s} SKIP (single class)", flush=True)
            continue
        g = gate(X, y, grp)
        epdir = os.path.join(OUTDIR, ep)
        os.makedirs(epdir, exist_ok=True)
        emit_pairs(ep, kept, os.path.join(epdir, "pairs.jsonl"))
        with open(os.path.join(epdir, "verifiability.json"), "w") as f:
            json.dump({"endpoint": ep, "modality": "smiles", "featurizer": "morgan_r2_2048",
                       "split": "murcko_scaffold_groupkfold5", **g}, f, indent=2)
        g["endpoint"] = ep
        report.append(g)
        print(f"{ep:14s} n={g['n']:6d} base={g['baseline']:.2f}  "
              f"random={g['random_auroc']:.3f} cold={g['cold_auroc']:.3f} "
              f"(drop {g['leakage_drop']:+.3f})  selectivity={g['selectivity']:+.3f}  "
              f"-> {'PASS' if g['passed'] else 'fail'}", flush=True)
    with open(os.path.join(OUTDIR, "verifiability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    n_pass = sum(r["passed"] for r in report)
    print(f"\n{n_pass}/{len(report)} endpoints PASS the verifiability gate "
          f"(cold AUROC >= {GATE_COLD_AUROC} AND selectivity >= {GATE_SELECTIVITY}).")


if __name__ == "__main__":
    main()
