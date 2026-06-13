"""Grounding vs memorization control for T2-solve, on the BALANCED signal-bearing set.

The R3 frontier sweep "reads" some ADMET properties on a BALANCED 100/100 set (CYP3A4
0.61). The reviewer's objection: is that reading the STRUCTURE, or pattern-matching the
canonical SMILES STRING (memorization), or a non-structural prior? A first attempt used
the WS2 `re_notation` rows, but that subset is imbalanced (27/200 pos) and at chance, so it
could not probe the reading (canonical ~ randomized ~ scrambled, all ~0.5 = floor, not
grounding). This version fixes that: take the SAME balanced sample the sweep scores, and
GENERATE the notations here, so all three arms run on the signal-bearing molecules:

  canonical  = RDKit canonical SMILES (the sweep's input)
  randomized = RDKit doRandom valid SMILES of the SAME molecule (notation change, structure kept)
  scrambled  = character-shuffled canonical (structure destroyed, usually invalid)

Reads: canonical ~ randomized AND canonical >> scrambled -> reads the STRUCTURE (grounding).
canonical ~ randomized ~ scrambled (all > 0.5) -> non-structural prior / not reading structure.
canonical drops on randomized -> canonical-string memorization. Bootstrap CI on each AUROC
and on the paired drops, because run-to-run sampling noise is ~0.03 at this n.

Run: python eval/notation_control.py
Env: NOTE_EP (default 'cyp3a4'), NOTE_N (balanced total, default 200), NOTE_MODEL. No em dashes.
"""
import json
import os
import random as _random
import re
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADMET = os.path.join(ROOT, "signal", "admet")

SYSTEM = ("You are a molecular property predictor. Respond with ONLY a single decimal "
          "number between 0 and 1 (for example: 0.42). No words, no explanation.")
CLAUSE = {
    "herg": "BLOCKS the hERG potassium channel (cardiotoxicity risk)",
    "cyp3a4": "INHIBITS the CYP3A4 enzyme",
    "cyp2d6": "INHIBITS the CYP2D6 enzyme",
}
PROMPT = ("You are given a molecule as a SMILES string. Estimate the probability "
          "(a single number between 0 and 1) that it {clause}. Judge only from the "
          "structure. Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:")


def parse_prob(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        v = float(tok)
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def load_balanced(ep, n):
    by = defaultdict(list)
    for line in open(os.path.join(ADMET, ep, "pairs.jsonl")):
        r = json.loads(line)
        if r["condition"] == "matched":
            m = Chem.MolFromSmiles(r["representation"])
            if m is not None:
                by[int(r["label"])].append(Chem.MolToSmiles(m))
    rng = np.random.RandomState(42)
    pos, neg = by[1][:], by[0][:]
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return [(s, 1) for s in pos[:k]] + [(s, 0) for s in neg[:k]]


def randomized(canon):
    m = Chem.MolFromSmiles(canon)
    try:
        return Chem.MolToSmiles(m, doRandom=True) if m else canon
    except Exception:
        return canon


def scrambled(canon, seed):
    ch = list(canon)
    _random.Random(seed).shuffle(ch)
    return "".join(ch)


def boot_ci(y, p, n=2000):
    y, p = np.array(y), np.array(p)
    rng = np.random.RandomState(0)
    a = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            a.append(roc_auc_score(y[idx], p[idx]))
    return (round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)) if a else (None, None)


def main():
    eps = [e.strip() for e in os.environ.get("NOTE_EP", "cyp3a4").split(",")]
    n = int(os.environ.get("NOTE_N", "200"))
    model = os.environ.get("NOTE_MODEL", "claude-sonnet-4-6")
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def ask(clause, smi):
        m = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT.format(clause=clause, smiles=smi)}])
        t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
        return parse_prob(t)

    results = []
    for ep in eps:
        from concurrent.futures import ThreadPoolExecutor
        data = load_balanced(ep, n)
        clause = CLAUSE[ep]
        y, p_can, p_rand, p_scr = [], [], [], []
        itf = open(os.path.join(ROOT, "results", f"notation_items_{ep}.jsonl"), "w")  # timeout-safe
        with ThreadPoolExecutor(max_workers=6) as ex:  # 3 conditions concurrent (3x throughput)
            for k, (csmi, lab) in enumerate(data):
                fc = ex.submit(ask, clause, csmi)
                fr = ex.submit(ask, clause, randomized(csmi))
                fs = ex.submit(ask, clause, scrambled(csmi, k))
                vc, vr, vs = fc.result(), fr.result(), fs.result()
                y.append(lab); p_can.append(vc); p_rand.append(vr); p_scr.append(vs)
                itf.write(json.dumps({"label": lab, "can": vc, "rand": vr, "scr": vs}) + "\n")
                itf.flush()
                if (k + 1) % 50 == 0:
                    print(f"  [{ep}] {k+1}/{len(data)}", flush=True)
        itf.close()
        y = np.array(y)
        auc = lambda p: round(float(roc_auc_score(y, p)), 3)
        r = {"endpoint": ep, "model": model, "n": len(y), "balanced": True,
             "canonical_auroc": auc(p_can), "canonical_ci": boot_ci(y, p_can),
             "randomized_auroc": auc(p_rand), "randomized_ci": boot_ci(y, p_rand),
             "scrambled_auroc": auc(p_scr), "scrambled_ci": boot_ci(y, p_scr),
             "drop_canonical_minus_randomized": round(auc(p_can) - auc(p_rand), 3),
             "drop_canonical_minus_scrambled": round(auc(p_can) - auc(p_scr), 3)}
        results.append(r)
        print(f"[{ep}] n={r['n']}  canonical={r['canonical_auroc']}{r['canonical_ci']}  "
              f"randomized={r['randomized_auroc']}  scrambled={r['scrambled_auroc']}  "
              f"drop_renote={r['drop_canonical_minus_randomized']} drop_scram={r['drop_canonical_minus_scrambled']}", flush=True)

    p = os.path.join(ROOT, "results", "notation_control.json")
    merged = {}
    if os.path.isfile(p):
        for r in json.load(open(p)):
            merged[r["endpoint"]] = r
    for r in results:
        merged[r["endpoint"]] = r
    json.dump([merged[k] for k in sorted(merged)], open(p, "w"), indent=2)
    print(f"\nwrote {p} ({len(merged)} endpoints)")


if __name__ == "__main__":
    main()
