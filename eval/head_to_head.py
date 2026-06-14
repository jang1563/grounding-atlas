"""WS1 axis-B head-to-head: LLM-output arm + content-sensitivity.

Give the LLM a molecule, ask for hERG-block probability, and compare AUROC against the
probe ceiling (eval/ceiling_gate.py: hERG scaffold ~0.91). Caveat: that LLMs are near-
chance at zero-shot property prediction from SMILES is PRIOR ART. Our
contribution is the instrument, the validity-vs-property dissociation, and the activation
arm; this script is the LLM-output arm.

Single-condition (HH_PAIRED unset), env HH_COND:
  real        canonical SMILES (the content-only baseline)
  randomized  same molecule, random valid SMILES (representation robustness)
  scrambled   character-shuffled SMILES (usually invalid; the model often refuses)
  mismatched  swapped for a VALID molecule with the OPPOSITE label

Paired (HH_PAIRED=1): ask real and HH_COND on the same items; report the gap.
  real AUROC near 0.91 -> reads structure; near 0.5 -> does not.
  mismatched paired is the clean content-sensitivity test: if the answer barely changes
  when the molecule is swapped for one with the opposite property (small mean|delta|,
  high correlation), the LLM is not reading the structure. Unlike scrambled, mismatched
  stays valid, so it avoids the invalid-SMILES refusal that contaminates the signal.

Only the LLM-output arm. The LLM-activation arm (encoding vs expression) needs hidden
states and runs on an open-weight model on Cayuga/Expanse. Env: HH_MODEL, HH_N, HH_COND, HH_PAIRED.
"""
import os
import random
import re
import sqlite3
from collections import Counter, defaultdict

import anthropic
import numpy as np
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score

DB = os.environ.get("NEGBIODB_ADMET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Negative_result_DB", "data", "negbiodb_admet.db"))
MODEL = os.environ.get("HH_MODEL", "claude-sonnet-4-5-20250929")
N = int(os.environ.get("HH_N", "20"))
COND = os.environ.get("HH_COND", "real")
PAIRED = os.environ.get("HH_PAIRED", "0") == "1"

PROMPT = (
    "You are given a molecule as a SMILES string. Estimate the probability "
    "(a single number between 0 and 1) that it BLOCKS the hERG potassium "
    "channel (cardiotoxicity risk). Judge only from the structure. "
    "Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:"
)


def load_sample(n):
    con = sqlite3.connect(DB)
    rows = con.execute(
        """SELECT c.canonical_smiles, r.outcome FROM admet_results r
           JOIN admet_assays a ON r.assay_id=a.id
           JOIN admet_compounds c ON r.compound_id=c.id
           WHERE a.endpoint='herg' AND r.outcome IN ('pass','fail')
             AND c.canonical_smiles IS NOT NULL AND c.canonical_smiles!=''"""
    ).fetchall()
    con.close()
    agg = defaultdict(list)
    for smi, out in rows:
        agg[smi].append(out)
    data = [(smi, 1 if "fail" in o else 0) for smi, o in agg.items()]
    pos = [d for d in data if d[1] == 1]
    neg = [d for d in data if d[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = n // 2
    return pos[:k] + neg[:k]


def alt_smiles(smi, label, cond, seed, pos_pool, neg_pool):
    if cond == "scrambled":
        ch = list(smi)
        random.Random(seed).shuffle(ch)
        return "".join(ch)
    if cond == "randomized":
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                return Chem.MolToSmiles(m, doRandom=True)
            except Exception:
                return smi
        return smi
    if cond == "mismatched":
        pool = neg_pool if label == 1 else pos_pool  # opposite label, valid molecule
        return pool[seed % len(pool)]
    return smi  # real


def parse_prob(txt):
    """Anchored: take the LAST number; map [0,1] directly, (1,100] as percent.
    Identical to activation_arm.py so the output arm is parsed the same way."""
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


def ask(client, smi):
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": PROMPT.format(smiles=smi)}],
        )
    except Exception:
        return 0.5, "error"
    texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not texts:
        return 0.5, "empty"
    return parse_prob(texts[0])


def main():
    data = load_sample(N)
    pos_pool = [smi for smi, lab in data if lab == 1]
    neg_pool = [smi for smi, lab in data if lab == 0]
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com"
    )

    if PAIRED:
        y, pr, pa, kinds = [], [], [], []
        for i, (smi, label) in enumerate(data):
            rp, rk = ask(client, smi)
            ap, ak = ask(client, alt_smiles(smi, label, COND, i, pos_pool, neg_pool))
            y.append(label)
            pr.append(rp)
            pa.append(ap)
            kinds += [rk, ak]
            print(f"[{i+1}/{len(data)}] label={label} real={rp:.2f} {COND}={ap:.2f}")
        y, pr, pa = np.array(y), np.array(pr), np.array(pa)
        print(f"\nMODEL={MODEL}  n={len(y)}  paired(real vs {COND})")
        print(f"  parse: {dict(Counter(kinds))}  (fallback/error/empty all map to 0.5)")
        if len(set(y)) > 1:
            print(f"  real AUROC={roc_auc_score(y, pr):.3f}  (probe ceiling ~0.91)")
        md = float(np.abs(pr - pa).mean())
        if COND == "mismatched":
            print(f"  mean|delta|={md:.3f}  [null=0 structure-ignored .. ceiling=1 opposite-labels-fully-separated]")
        elif COND == "randomized":
            print(f"  mean|delta|={md:.3f}  [0=representation-invariant .. larger=brittle]")
        else:
            print(f"  mean|delta|={md:.3f}")
        print(f"  corr(real, {COND})={np.corrcoef(pr, pa)[0,1]:.3f}  (high = structure ignored)")
        return

    y, p, kinds = [], [], []
    for i, (smi, label) in enumerate(data):
        val, kind = ask(client, alt_smiles(smi, label, COND, i, pos_pool, neg_pool))
        p.append(val)
        y.append(label)
        kinds.append(kind)
        print(f"[{i+1}/{len(data)}] label={label} llm_p={p[-1]:.2f}")
    y, p = np.array(y), np.array(p)
    print(f"\nMODEL={MODEL}  n={len(y)}  condition={COND}")
    print(f"  parse: {dict(Counter(kinds))}  (fallback/error/empty all map to 0.5 -> AUROC noise)")
    if len(set(y)) > 1:
        print(
            f"LLM-output AUROC={roc_auc_score(y, p):.3f}  "
            f"AUPRC={average_precision_score(y, p):.3f}   (probe ceiling ~0.91)"
        )


if __name__ == "__main__":
    main()
