"""WS3 #4: the decision-map circularity-breaker on a fingerprint-WEAK, knowledge-heavy endpoint.

Drug market-WITHDRAWAL (safety) is the endpoint WS2's gate would reject: a balanced
scaffold-CV Morgan probe is ~0.61 and a Tanimoto k-NN ~0.64 (confirmed), because why a
drug was pulled from market is a clinical/historical fact with little local-substructure
basis. So no cheap structure specialist exists. The LLM has another route: it has read the
withdrawal history of named drugs. This measures whether the LLM-NAME arm beats the
structure tools, which would be the first decision-map cell where the LLM is the right tool.

Five arms on ONE balanced sample (seed 42), AUROC vs label-1 = withdrawn:
  1 morgan      structure probe, Murcko-scaffold GroupKFold (the cheap specialist)
  2 knn         no-LLM neighbor-mean, Tanimoto, scaffold CV (mandatory baseline)
  3 llm_smiles  LLM-output from SMILES only (matched input to the structure tools)
  4 llm_name    LLM-output from the DRUG NAME (the knowledge route) -> the circularity-breaker
  5 llm_fake    LLM-output from a DERANGED (shuffled) name -> withdrawal-specificity control
Honest decomposition: llm_name >> llm_smiles means the win is name-knowledge, not structure;
llm_name >> llm_fake means it is drug-SPECIFIC knowledge, not a generic "is-a-drug" prior.
(The activation arm runs separately on GPU: activation_arm.py ACT_CSV=withdrawn.csv.)

Data: signal/withdrawn/withdrawn.csv (name,smiles,label,inchikey) from the WITHDRAWN /
DrugBank / ChEMBL / NCATS agreement set (Mazuz et al., eyalmazuz/DrugWithdrawn).
DRY-RUN: WD_DRY=1 (no API). REAL: set ANTHROPIC_API_KEY in the environment. Env: WD_N (balanced total, default 500),
WD_MODEL (default claude-sonnet-4-6), WD_WORKERS (default 8). No em dashes.
"""
import csv
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "signal", "withdrawn", "withdrawn.csv")
OUT = os.path.join(ROOT, "results", "withdrawn_endpoint.json")

N = int(os.environ.get("WD_N", "500"))
MODEL = os.environ.get("WD_MODEL", "claude-sonnet-4-6")
WORKERS = int(os.environ.get("WD_WORKERS", "8"))
DRY = os.environ.get("WD_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")

SYSTEM = ("You are a drug-safety predictor. Respond with ONLY a single decimal number "
          "between 0 and 1 (for example: 0.42). No words, no explanation. Just the number.")
P_SMILES = ("You are given a molecule as a SMILES string. Estimate the probability (a single "
            "number between 0 and 1) that this drug was WITHDRAWN or DISCONTINUED from the market "
            "for SAFETY reasons (toxicity). Judge only from the structure. Reply with ONLY the "
            "number.\n\nSMILES: {x}\nProbability:")
P_NAME = ("Estimate the probability (a single number between 0 and 1) that the drug '{x}' was "
          "WITHDRAWN or DISCONTINUED from the market for SAFETY reasons (toxicity). Reply with "
          "ONLY the number.\n\nDrug: {x}\nProbability:")


def parse_prob(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "parsed"
        if 1.0 < v <= 100.0:
            return v / 100.0, "percent"
    return 0.5, "fallback"


def load_balanced(n):
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if Chem.MolFromSmiles(r["smiles"]) is not None:
                rows.append((r["name"], r["smiles"], int(r["label"])))
    pos = [x for x in rows if x[2] == 1]
    neg = [x for x in rows if x[2] == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = pos[:k] + neg[:k]
    rng.shuffle(data)
    return data


def boot_ci(y, p, nb=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    a = []
    for _ in range(nb):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)


def structure_arms(data):
    smis = [s for _, s, _ in data]
    y = np.array([l for _, _, l in data])
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smis]
    FP = np.array([np.array(fp, dtype=float) for fp in fps])
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) or s for s in smis]
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    pm = cross_val_predict(clf, FP, y, groups=scaf, cv=GroupKFold(5), method="predict_proba")[:, 1]
    # no-LLM k-NN neighbor-mean, Tanimoto, scaffold CV
    pk = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(FP, y, scaf):
        for i in te:
            sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in tr]
            top = np.argsort(sims)[-5:]
            pk[i] = np.mean([y[tr[t]] for t in top])
    return y, {"morgan": (pm, scaf), "knn": (pk, None)}


def call(client, system, prompt):
    msg = client.messages.create(model=MODEL, max_tokens=16, system=system,
                                 messages=[{"role": "user", "content": prompt}])
    t = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return parse_prob(t[0]) if t else (0.5, "empty")


def llm_arm(client, prompts):
    res = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, client, SYSTEM, p): i for i, p in enumerate(prompts)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res[i] = fut.result()
            except Exception as e:
                res[i] = (0.5, f"err:{type(e).__name__}")
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(prompts)}", flush=True)
    return np.array([r[0] for r in res]), [r[1] for r in res]


def main():
    data = load_balanced(N)
    y, sa = structure_arms(data)
    out = {"endpoint": "market_withdrawal_safety", "n": len(y), "pos": int(y.sum()),
           "model": MODEL, "arms": {}}
    preds = {}  # arm -> per-item prediction array (data order), for the per-drug agreement test
    for name, (p, _) in sa.items():
        preds[name] = p
        lo, hi = boot_ci(y, p)
        out["arms"][name] = {"auroc": round(float(roc_auc_score(y, p)), 3),
                             "auprc": round(float(average_precision_score(y, p)), 3),
                             "ci95": [lo, hi]}
        print(f"{name:11s} AUROC={out['arms'][name]['auroc']:.3f} CI={lo,hi}", flush=True)

    if DRY:
        print("\nDRY-RUN: skipping LLM arms. Sample prompts:")
        print(" SMILES:", P_SMILES.format(x=data[0][1])[:120])
        print(" NAME  :", P_NAME.format(x=data[0][0]))
        json.dump(out, open(OUT, "w"), indent=2)
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    names = [n for n, _, _ in data]
    smis = [s for _, s, _ in data]
    # deranged names for the specificity control (no molecule keeps its own name)
    rng = random.Random(7)
    perm = list(range(len(names)))
    while any(perm[i] == i for i in range(len(perm))):
        rng.shuffle(perm)
    fake_names = [names[perm[i]] for i in range(len(names))]

    for arm, prompts in [("llm_smiles", [P_SMILES.format(x=s) for s in smis]),
                         ("llm_name", [P_NAME.format(x=n) for n in names]),
                         ("llm_fake", [P_NAME.format(x=n) for n in fake_names])]:
        print(f"\n== {arm} ({len(prompts)} calls) ==", flush=True)
        p, kinds = llm_arm(client, prompts)
        preds[arm] = p
        lo, hi = boot_ci(y, p)
        out["arms"][arm] = {"auroc": round(float(roc_auc_score(y, p)), 3),
                            "auprc": round(float(average_precision_score(y, p)), 3),
                            "ci95": [lo, hi], "parse": dict(Counter(kinds))}
        print(f"{arm:11s} AUROC={out['arms'][arm]['auroc']:.3f} CI={lo,hi} parse={dict(Counter(kinds))}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)  # kill-safe incremental save

    s = out["arms"]
    print("\n=== SUMMARY (AUROC vs withdrawn) ===")
    for a in ["morgan", "knn", "llm_smiles", "llm_name", "llm_fake"]:
        if a in s:
            print(f"  {a:11s} {s[a]['auroc']:.3f}  CI{tuple(s[a]['ci95'])}")
    if "llm_name" in s:
        print(f"\nCIRCULARITY-BREAKER: llm_name {s['llm_name']['auroc']:.3f} vs structure max "
              f"{max(s['morgan']['auroc'], s['knn']['auroc']):.3f}; "
              f"knowledge-not-structure: llm_name vs llm_smiles {s['llm_smiles']['auroc']:.3f}; "
              f"specificity: llm_name vs llm_fake {s['llm_fake']['auroc']:.3f}")
    json.dump(out, open(OUT, "w"), indent=2)

    # per-item dump keyed by SMILES, for the per-drug agreement test vs the 8B activation arm
    peritem = [{"smiles": data[i][1], "name": data[i][0], "label": int(y[i]),
                **{arm: round(float(preds[arm][i]), 4) for arm in preds}} for i in range(len(y))]
    json.dump(peritem, open(os.path.join(ROOT, "results", "withdrawn_peritem.json"), "w"))
    print(f"wrote {len(peritem)} per-item predictions to results/withdrawn_peritem.json")


if __name__ == "__main__":
    main()
