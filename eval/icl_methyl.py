"""Few-shot ICL: does in-context learning close the methylation->age gap WITHOUT training?

The free upper bound on SFT, and a test of whether the gap is closable at all without weight
updates. k age-stratified examples are prepended, then the model estimates age for held-out
queries. Run on gene-named and anon renderings: if ICL needs the gene anchor, the gap is
web-anchored; if ICL closes both, it is an in-context computation the model can do once shown
examples. source ~/.api_keys. No em dashes.
"""
import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "signal", "methyl", "methyl_anchor.csv")
N = int(os.environ.get("ICL_N", "200"))
KP = int(os.environ.get("ICL_KPROBE", "60"))
SHOT = int(os.environ.get("ICL_SHOT", "12"))
WORKERS = int(os.environ.get("ICL_WORKERS", "8"))
MODELS = os.environ.get("ICL_MODELS", "claude-opus-4-8,claude-haiku-4-5-20251001").split(",")


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


def load():
    rows = list(csv.DictReader(open(CSV)))
    rng = np.random.RandomState(0)
    rng.shuffle(rows)
    pool = rows[:40]
    test = rows[40:40 + N]
    return pool, test


def pick_examples(pool, shot):
    sp = sorted(pool, key=lambda r: float(r["age"]))
    idx = np.linspace(0, len(sp) - 1, shot).astype(int)
    return [sp[i] for i in idx]


def build(exs, q, field):
    s = ("Estimate a person's age in years from their blood DNA methylation beta values "
         "(token:value). Here are labeled examples:\n\n")
    for e in exs:
        s += f"Values: {trunc(e[field], KP)}\nAge: {float(e['age']):.0f}\n\n"
    s += f"Now estimate the age for this sample. Reply with ONLY a number.\nValues: {trunc(q[field], KP)}\nAge:"
    return s


def run(client, model, prompts):
    def call(p):
        try:
            m = client.messages.create(model=model, max_tokens=12,
                                       messages=[{"role": "user", "content": p}])
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return parse_age(t)
        except Exception:
            return 50.0
    res = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return np.array(res)


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pool, test = load()
    exs = pick_examples(pool, SHOT)
    y = np.array([int(r["label"]) for r in test])
    ages = np.array([float(r["age"]) for r in test])
    print(f"{SHOT}-shot ICL  n_test={len(y)} probes={KP}  ref: 0-shot frontier ~chance, 8B output 0.487, ceiling(age-CpG) 0.957", flush=True)
    for model in MODELS:
        print(f"\n## {model}", flush=True)
        for field in ["text_gene", "text_anon"]:
            pred = run(client, model, [build(exs, q, field) for q in test])
            auc = roc_auc_score(y, pred)
            mae = float(np.mean(np.abs(pred - ages)))
            r = float(np.corrcoef(pred, ages)[0, 1])
            tag = "gene-named" if field == "text_gene" else "anonymized"
            print(f"  {tag:12s} AUROC={auc:.3f}  MAE={mae:.1f}y  Pearson={r:+.3f}", flush=True)


if __name__ == "__main__":
    main()
