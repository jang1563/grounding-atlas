"""B (the sole non-overlapping cell): negative-class expression gap.

Question: does the LLM ENCODE confirmed-INACTIVE compounds (a Morgan probe separates
active/inactive at a high ceiling) yet OVER-CALL them as active in its OUTPUT (a false-positive
asymmetry)? This is the LLM-expression-layer decomposition of NullAtlas's headline
"models without negative data produce excessive false positives": NullAtlas measures it at the
ML/L4 output level; we add whether the signal is INTERNALLY present but mis-verbalized, and
whether the over-call is asymmetric (inactive->active >> active->inactive).

3-arm on the negative class:
  ceiling   = Morgan r2/2048 logistic, active-vs-inactive AUROC (signal is in the structure)
  activation= CITED from the existing ADMET 3-arm (hERG encode ~0.78 > output ~0.46); the
              negative-class hidden-state probe is a Cayuga follow-up, not run here
  output    = frontier LLM active-probability per compound -> AUROC + FP(inactive->active) +
              FN(active->inactive) + asymmetry(FP-FN), per endpoint (web-exposure varies) and
              per model (haiku vs opus = scale curve)

Data: NegResultDB admet.db, compound-level (any fail -> active=1, else inactive=0), best tier per
compound. set ANTHROPIC_API_KEY in the environment. No em dashes.
"""
import json
import os
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("NEGRES_ADMET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "NegResultDB", "data", "negres_admet.db"))

# endpoint -> (named target phrasing, anonymized generic phrasing)
ENDPOINTS = {
    "herg":   ("a hERG potassium channel (Kv11.1) blocker", "active against a cardiac ion channel"),
    "cyp3a4": ("a CYP3A4 enzyme inhibitor", "active against a drug-metabolizing enzyme"),
    "cyp2d6": ("a CYP2D6 enzyme inhibitor", "active against a drug-metabolizing enzyme"),
    "ames":   ("a mutagen (positive in the Ames test)", "active in a standard toxicity assay"),
}
N_PER_CLASS = int(os.environ.get("NEG_N", "120"))
DO_LLM = os.environ.get("NEG_LLM", "1") == "1"
MODELS = os.environ.get("NEG_MODELS", "claude-haiku-4-5-20251001,claude-opus-4-8").split(",")
ANON = os.environ.get("NEG_ANON", "0") == "1"
WORKERS = int(os.environ.get("NEG_WORKERS", "8"))


def load(ep):
    c = sqlite3.connect(DB)
    rows = c.execute(
        """select cp.canonical_smiles, r.outcome from admet_results r
           join admet_assays a on r.assay_id = a.id
           join admet_compounds cp on r.compound_id = cp.id
           where a.endpoint = ? and r.outcome in ('pass','fail')
             and cp.canonical_smiles is not null and cp.canonical_smiles != ''""",
        (ep,)).fetchall()
    c.close()
    agg = defaultdict(list)
    for smi, out in rows:
        agg[smi].append(out)
    data = [(smi, 1 if "fail" in o else 0) for smi, o in agg.items()]   # any fail -> active
    pos = [d for d in data if d[1] == 1]
    neg = [d for d in data if d[1] == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(N_PER_CLASS, len(pos), len(neg))
    return pos[:k] + neg[:k], len(pos), len(neg)


def morgan(smis):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, keep = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            keep.append(False); continue
        X.append(gen.GetFingerprintAsNumPy(m)); keep.append(True)
    return np.asarray(X, dtype=float), np.array(keep)


def ceiling(X, y):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba", n_jobs=-1)[:, 1]
    return float(roc_auc_score(y, p))


def llm_prob(client, model, smi, phrasing):
    prompt = (f"Compound (SMILES): {smi}\n\nIs this compound {phrasing}? Give the probability "
              "from 0 to 1 that it IS active. Reply with ONLY the number.")
    try:
        m = client.messages.create(model=model, max_tokens=10,
                                    messages=[{"role": "user", "content": prompt}])
        t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
        for tok in re.findall(r"\d*\.?\d+", t):
            v = float(tok)
            if 0 <= v <= 1:
                return v
            if 1 < v <= 100:
                return v / 100
    except Exception:
        pass
    return 0.5


def run_llm(client, model, smis, phrasing):
    res = [0.5] * len(smis)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(llm_prob, client, model, smis[i], phrasing): i for i in range(len(smis))}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
    return np.array(res)


def main():
    client = None
    if DO_LLM:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    cond = "anon" if ANON else "named"
    print(f"phrasing condition = {cond};  N/class = {N_PER_CLASS}\n", flush=True)
    print(f"{'endpoint':9s} {'arm/model':22s} {'n':>5s}  AUROC  FP(inact>act)  FN(act>inact)  asym   ", flush=True)
    print("-" * 92, flush=True)
    out = []
    for ep, (named, generic) in ENDPOINTS.items():
        data, npos, nneg = load(ep)
        smis = [d[0] for d in data]
        y0 = np.array([d[1] for d in data])
        X, keep = morgan(smis)
        y = y0[keep]
        smis = [s for s, k in zip(smis, keep) if k]
        cauroc = ceiling(X, y)
        print(f"{ep:9s} {'ceiling(Morgan)':22s} {len(y):5d}  {cauroc:.3f}  {'-':>12s}  {'-':>12s}   pool act={npos} inact={nneg}", flush=True)
        for model in (MODELS if DO_LLM else []):
            phrasing = generic if ANON else named
            p = run_llm(client, model, smis, phrasing)
            auroc = float(roc_auc_score(y, p))
            pred = (p > 0.5).astype(int)
            fp = float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
            fn = float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
            mshort = model.split("-")[1] if "-" in model else model
            print(f"{ep:9s} {('output:' + mshort + '/' + cond):22s} {len(y):5d}  {auroc:.3f}  {fp:>12.3f}  {fn:>12.3f}   {fp - fn:>+5.3f}", flush=True)
            out.append(dict(endpoint=ep, model=model, cond=cond, n=int(len(y)),
                            ceiling=round(cauroc, 3), output_auroc=round(auroc, 3),
                            fp=round(fp, 3), fn=round(fn, 3), asym=round(fp - fn, 3)))
    if out:
        tag = os.environ.get("NEG_TAG", "")
        tag = ("_" + tag) if tag else ""
        path = os.path.join(ROOT, "results", "negative_expression_gap%s%s.json" % ("_anon" if ANON else "", tag))
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    print("\nKEY: asym > 0 = LLM over-calls confirmed-INACTIVE compounds as active = the "
          "false-positive bias (NullAtlas) localized to the LLM expression layer, where the "
          "Morgan ceiling shows the inactive/active boundary IS in the structure.", flush=True)


if __name__ == "__main__":
    main()
