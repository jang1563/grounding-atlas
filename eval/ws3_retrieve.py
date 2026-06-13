"""WS3 decision-map: the RETRIEVE placement for an expression-limited capability.

The decision map (PROJECT_DESIGN WS3) asks, per capability, which placement wins:
train into weights / retrieve / orchestrate. For hERG (expression-limited: the model
encodes it at 0.79 but verbalizes at chance), this measures the RETRIEVE arm: give the
model the k nearest labeled molecules (Morgan Tanimoto) in context and ask for the query.

Prediction: retrieval helps little, because the bottleneck is READING the query SMILES,
not lacking examples. If the model cannot map the query structure to the retrieved
neighbors (it reads SMILES at chance), nearest-neighbor context lands near random few-shot
(~0.49), not near the ceiling. That would be the decision-map verdict: an expression-
limited capability cannot be retrieved into place, so it routes to weights or orchestrate.

Run: /Users/jak4013/miniconda3-arm64/bin/python eval/ws3_retrieve.py  (rdkit + anthropic)
Env: WS3_K (neighbors, default 10), WS3_N (balanced test size, default 150), WS3_MODEL,
WS3_DRY. No em dashes.
"""
import os
import re
import json
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAIRS = os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")

SYSTEM = ("You are a molecular property predictor. Respond with ONLY a single decimal "
          "number between 0 and 1 (for example: 0.42). No words, no explanation.")


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smi)
    except Exception:
        return smi


def load():
    by = defaultdict(list)
    for line in open(PAIRS):
        r = json.loads(line)
        if r.get("condition") != "matched":
            continue
        f = fp(r["representation"])
        if f is not None:
            by[int(r["label"])].append((r["representation"], f, scaffold(r["representation"])))
    return by


def parse(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        v = float(tok)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100
    return 0.5


def main():
    K = int(os.environ.get("WS3_K", "10"))
    N = int(os.environ.get("WS3_N", "150"))
    model = os.environ.get("WS3_MODEL", "claude-sonnet-4-6")
    dry = os.environ.get("WS3_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    holdout = os.environ.get("WS3_SCAFFOLD_HOLDOUT", "0") == "1"
    flip = os.environ.get("WS3_FLIP_LABELS", "0") == "1"  # adversarial: show neighbors' OPPOSITE labels

    by = load()
    rng = np.random.RandomState(42)
    test, pool = [], []
    for lab in (0, 1):
        items = by[lab][:]
        rng.shuffle(items)
        k = N // 2
        test += [(s, f, sc, lab) for s, f, sc in items[:k]]
        pool += [(s, f, sc, lab) for s, f, sc in items[k:]]
    cap = int(os.environ.get("WS3_POOL_CAP", "0"))
    if cap and cap < len(pool):  # sparse-pool regime: few labeled neighbors available
        idx = rng.choice(len(pool), cap, replace=False)
        pool = [pool[j] for j in idx]
    pool_fps = [f for _, f, _, _ in pool]
    pool_scaf = [sc for _, _, sc, _ in pool]
    mode = ("scaffold_holdout" if holdout else "random_split") + (f"_pool{cap}" if cap else "") + ("_FLIPPED" if flip else "")
    print(f"test={len(test)} pool={len(pool)} K={K} mode={mode}")

    def neighbors(qf, qscaf):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qf, pool_fps))
        if holdout:  # exclude same-Murcko-scaffold pool entries
            sims[[j for j, sc in enumerate(pool_scaf) if sc == qscaf]] = -1.0
        nn = np.argsort(sims)[::-1][:K]
        nn = [int(j) for j in nn if sims[j] >= 0]  # drop holdout/invalid if pool starved
        return nn, sims

    # MANDATORY control: the no-LLM neighbor-label-mean baseline (never reads the query).
    # If this saturates, the retrieve AUROC measures neighbor-label purity, not model skill.
    yb, sb, pur = [], [], []
    for qs, qf, qscaf, lab in test:
        nn, _ = neighbors(qf, qscaf)
        labs = [pool[j][3] for j in nn]
        yb.append(lab); sb.append(float(np.mean(labs)) if labs else 0.5)
        pur.append(max(labs.count(0), labs.count(1)) / len(labs) if labs else 0.5)
    base_auroc = round(float(roc_auc_score(yb, sb)), 3) if len(set(yb)) > 1 else None
    mean_purity = round(float(np.mean(pur)), 3)
    print(f"NO-LLM neighbor-mean baseline AUROC={base_auroc}  mean_neighbor_purity={mean_purity}")

    if dry:
        qs, qf, qscaf, ql = test[0]
        nn, sims = neighbors(qf, qscaf)
        print(f"example query (label={ql}): {qs[:50]}")
        print(f"top-{K} neighbor sims: {[round(float(sims[i]),2) for i in nn]}")
        print(f"neighbor labels: {[pool[i][3] for i in nn]}")
        print("DRY: retrieval wiring only, no API.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    y, p = [], []
    for i, (qs, qf, qscaf, lab) in enumerate(test):
        nn, _ = neighbors(qf, qscaf)
        shots = "\n".join(
            f"SMILES: {pool[j][0]}  hERG_blocker: {'yes' if (pool[j][3] == 1) != flip else 'no'}" for j in nn)
        prompt = (f"Here are {K} molecules most similar to the query, with their hERG status:\n"
                  f"{shots}\n\nNow estimate the probability (0 to 1) that the QUERY blocks hERG.\n"
                  f"QUERY SMILES: {qs}")
        msg = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        y.append(lab)
        p.append(parse(txt))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test)}")
    auroc = round(float(roc_auc_score(y, p)), 3) if len(set(y)) > 1 else None
    out = {"placement": "retrieve", "mode": mode, "endpoint": "herg", "model": model, "k": K,
           "n_test": len(y), "retrieve_auroc": auroc,
           "neighbor_mean_baseline_auroc": base_auroc, "mean_neighbor_purity": mean_purity,
           "llm_minus_baseline": round(auroc - base_auroc, 3) if (auroc and base_auroc) else None,
           "ref_solo_output_8b": 0.453, "ref_solo_output_frontier": 0.633,
           "ref_readout_weights_scaffoldCV_8b": 0.787, "ref_orchestrate_ceiling_cold": 0.895}
    path = os.path.join(ROOT, "results", f"ws3_retrieve_{mode}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nRETRIEVE ({mode}) LLM AUROC={auroc}  vs  no-LLM neighbor-mean={base_auroc}  "
          f"(LLM minus baseline = {out['llm_minus_baseline']})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
