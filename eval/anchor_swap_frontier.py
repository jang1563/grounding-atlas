"""Phase 1 pre-step (training-free): methylation anchor-token swap on frontier output.

Renders the SAME methylation beta-values two ways - gene-named (web-rich tokens) vs anonymized
feature IDs (web-zero) - and asks a frontier model to estimate age. The web-exposure law predicts
the gene-named rendering verbalizes age better than the anonymized one, with numeric content
identical. Mirrors the single-cell gene-name vs anon contrast, now within methylation. Continuous
MAE + Pearson and binary AUROC. set ANTHROPIC_API_KEY in the environment. No em dashes.
"""
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "signal", "methyl", "methyl_anchor.csv")
N = int(os.environ.get("AS_N", "400"))
K = int(os.environ.get("AS_K", "80"))
WORKERS = int(os.environ.get("AS_WORKERS", "8"))
MODELS = os.environ.get("AS_MODELS", "claude-opus-4-8,claude-haiku-4-5-20251001").split(",")
PROMPT = ("Below are DNA methylation beta values (token:value, value in [0,1]) measured from a human "
          "blood sample. Estimate this person's age in YEARS. Judge only from the values. Reply with "
          "ONLY a single number.\n\n{x}\n\nAge in years:")


def parse_age(t):
    for tok in re.findall(r"\d+\.?\d*", t):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0 <= v <= 120:
            return v
    return 50.0


def trunc(text, k):
    return " ".join(text.split()[:k])


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def run(client, model, texts):
    def call(x):
        try:
            m = client.messages.create(model=model, max_tokens=12,
                                       messages=[{"role": "user", "content": PROMPT.format(x=x)}])
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return parse_age(t)
        except Exception:
            return 50.0
    res = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, x): i for i, x in enumerate(texts)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return np.array(res)


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    ages = np.array([float(r["age"]) for r in data])
    fields = {"gene-named (web-rich)": [trunc(r["text_gene"], K) for r in data],
              "anonymized (web-zero)": [trunc(r["text_anon"], K) for r in data]}
    print(f"n={len(y)} probes_shown={K}  reference: 8B output AUROC 0.487 (chance), probe 0.685", flush=True)
    for model in MODELS:
        print(f"\n## {model}", flush=True)
        for name, texts in fields.items():
            pred = run(client, model, texts)
            auc = roc_auc_score(y, pred)
            mae = float(np.mean(np.abs(pred - ages)))
            r = float(np.corrcoef(pred, ages)[0, 1])
            print(f"  {name:24s} AUROC={auc:.3f}  MAE={mae:.1f}y  Pearson={r:+.3f}", flush=True)


if __name__ == "__main__":
    main()
