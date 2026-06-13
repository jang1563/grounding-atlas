"""T2 (apply) PROPOSE mode: can the model generate molecules with a property.

T2-solve (output_arm_admet.py) asks the model to READ a property off a structure.
PROPOSE is the generative dual: ask it to GENERATE molecules that have the property,
then judge each proposal with the WS2 specialist probe (eval/README Bridge to T2:
"propose a candidate, scored against ground truth"). The probe is the deterministic
judge, so no LLM-judge is used.

Endpoints: CYP3A4 / CYP2D6 inhibition, where label-1 = inhibitor (fail) so "propose an
inhibitor" needs no direction flip, and proposing an enzyme inhibitor carries no acute-
toxicity refusal confound (unlike hERG block or Ames mutagenicity).

Pipeline per endpoint: ask for K SMILES -> RDKit validity/uniqueness -> train a Morgan
LogisticRegression probe on the WS2 matched data (label-1=inhibitor) -> score each
proposal P(inhibitor). Reference points: the probe's mean P on the WS2 actives (label-1)
and inactives (label-0), so the proposals' active-rate is read against the base rates.

Run: /Users/jak4013/miniconda3-arm64/bin/python eval/t2_propose.py  (rdkit+sklearn+anthropic)
Env: PROP_MODEL (default claude-sonnet-4-6), PROP_K (default 15), PROP_DRY. No em dashes.
"""
import os
import re
import json

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog("rdApp.*")  # silence the per-molecule MorganGenerator deprecation spam

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADMET = os.path.join(ROOT, "signal", "admet")

ENDPOINTS = {
    "cyp3a4": "inhibit the CYP3A4 enzyme",
    "cyp2d6": "inhibit the CYP2D6 enzyme",
}


def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    return np.array(fp, dtype=np.int8)


def load_ws2(ep):
    X, y = [], []
    for line in open(os.path.join(ADMET, ep, "pairs.jsonl")):
        r = json.loads(line)
        if r.get("condition") != "matched":
            continue
        f = morgan(r["representation"])
        if f is not None:
            X.append(f)
            y.append(int(r["label"]))
    return np.array(X), np.array(y)


def ask_proposals(client, model, clause, k):
    prompt = (
        f"Propose {k} distinct, real, drug-like small molecules that {clause}. "
        "Output ONLY a JSON array of SMILES strings, nothing else."
    )
    msg = client.messages.create(model=model, max_tokens=1200,
        messages=[{"role": "user", "content": prompt}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [s for s in arr if isinstance(s, str)]
    except Exception:
        return re.findall(r'"([^"]+)"', m.group(0))


def main():
    dry = os.environ.get("PROP_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("PROP_MODEL", "claude-sonnet-4-6")
    K = int(os.environ.get("PROP_K", "15"))

    client = None
    if not dry:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    for ep, clause in ENDPOINTS.items():
        X, y = load_ws2(ep)
        probe = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
        base_active = float(probe.predict_proba(X[y == 1])[:, 1].mean())   # probe P on real actives
        base_inactive = float(probe.predict_proba(X[y == 0])[:, 1].mean())  # on real inactives

        if dry:
            print(f"[{ep}] WS2 n={len(y)} pos={int(y.sum())}  probe P(active): "
                  f"actives={base_active:.3f} inactives={base_inactive:.3f}  (clause: {clause})")
            continue

        proposals = ask_proposals(client, model, clause, K)
        parsed = [(s, morgan(s)) for s in proposals]
        valid = [(s, f) for s, f in parsed if f is not None]
        uniq = list({Chem.MolToSmiles(Chem.MolFromSmiles(s)): f for s, f in valid}.items())  # canonical dedup
        if uniq:
            P = probe.predict_proba(np.array([f for _, f in uniq]))[:, 1]
            mean_p = float(P.mean())
            active_rate = float((P > 0.5).mean())
        else:
            mean_p, active_rate = None, None
        r = {
            "endpoint": ep, "model": model, "asked": K, "returned": len(proposals),
            "valid_smiles": len(valid), "unique_valid": len(uniq),
            "proposal_mean_P_active": round(mean_p, 3) if mean_p is not None else None,
            "proposal_active_rate": round(active_rate, 3) if active_rate is not None else None,
            "ref_P_active_on_real_actives": round(base_active, 3),
            "ref_P_active_on_real_inactives": round(base_inactive, 3),
            "example_proposals": [s for s, _ in uniq[:5]],
        }
        results.append(r)
        print(f"[{ep}] returned={r['returned']} valid={r['valid_smiles']} uniq={r['unique_valid']}  "
              f"proposal meanP={r['proposal_mean_P_active']} active_rate={r['proposal_active_rate']}  "
              f"(ref actives={r['ref_P_active_on_real_actives']} inactives={r['ref_P_active_on_real_inactives']})")

    if not dry and results:
        with open(os.path.join(ROOT, "results", "t2_propose.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {os.path.join(ROOT, 'results', 't2_propose.json')}")


if __name__ == "__main__":
    main()
