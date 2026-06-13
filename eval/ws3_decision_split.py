"""Clean within-split decision map for hERG: all four placements on ONE scaffold split.

The decision-map numbers were on mixed models and splits (weights = 8B on the LoRA scaffold
split, retrieve = sonnet on its own split, k-NN on another). This computes the two non-LLM
placements (a no-LLM neighbor-mean k-NN and the Morgan-fingerprint specialist probe) on the
EXACT train/test scaffold split that ws3_lora.py used (same seed, same load), so they line
up apples-to-apples with the LoRA solo (0.575) and weights (0.856) already measured there.

No API, no GPU: rdkit Morgan + sklearn on the reproduced split. Run with the miniconda
python (rdkit + sklearn). No em dashes.
"""
import json
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smi)
    except Exception:
        return smi


def load(n=2000):
    # EXACT replica of eval/ws3_lora.py load(): balanced-up-to-n//2 per label, seed 42,
    # then GroupShuffleSplit(test=0.3, rs=42) by Murcko scaffold.
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
    groups = [scaffold_of(s) for s in smis]
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(smis, ys, groups))
    return ([smis[i] for i in tr], [ys[i] for i in tr],
            [smis[i] for i in te], [ys[i] for i in te])


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def main():
    tr_s, tr_y, te_s, te_y = load(2000)
    tr_fp = [fp(s) for s in tr_s]
    te_fp = [fp(s) for s in te_s]
    print(f"train={len(tr_s)} (pos {sum(tr_y)})  test={len(te_s)} (pos {sum(te_y)})", flush=True)

    # placement 1: no-LLM neighbor-mean k-NN (k=10), the cheap retrieval baseline
    knn = []
    for qf in te_fp:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qf, tr_fp))
        nn = np.argsort(sims)[::-1][:10]
        knn.append(float(np.mean([tr_y[j] for j in nn])))
    knn_auc = round(float(roc_auc_score(te_y, knn)), 3)

    # placement 2: orchestrate the Morgan-fingerprint specialist (logistic probe)
    X_tr = np.array([np.array(f) for f in tr_fp])
    X_te = np.array([np.array(f) for f in te_fp])
    probe = LogisticRegression(max_iter=1000).fit(X_tr, tr_y)
    orch_auc = round(float(roc_auc_score(te_y, probe.predict_proba(X_te)[:, 1])), 3)

    out = {"split": "ws3_lora scaffold split (n=2000, seed 42)",
           "n_train": len(tr_s), "n_test": len(te_s),
           "solo_8b_output": 0.575, "weights_lora_output": 0.856,
           "retrieve_knn_neighbor_mean": knn_auc,
           "orchestrate_morgan_probe": orch_auc}
    json.dump(out, open(os.path.join(ROOT, "results", "ws3_decision_split.json"), "w"), indent=2)
    print("\nhERG decision map, ALL on the same scaffold split (test n={}):".format(len(te_s)))
    print("  solo (8B output)            = 0.575   (from ws3_lora base)")
    print("  weights (LoRA output)       = 0.856   (from ws3_lora finetuned)")
    print(f"  retrieve (no-LLM k-NN mean) = {knn_auc}")
    print(f"  orchestrate (Morgan probe)  = {orch_auc}")


if __name__ == "__main__":
    main()
