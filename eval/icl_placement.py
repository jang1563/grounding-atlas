"""Few-shot ICL placement test: for each capability, does in-context (retrieve) close the gap?

Fills the RETRIEVE column of the decision map (train / retrieve / orchestrate). If k-shot ICL
closes a rung, that capability is served by retrieve/in-context and does not need weight training.
If ICL does not close it (and a specialist tool does), the placement is orchestrate. Binary rungs.
Compares against the known 0-shot output and the specialist ceiling. source ~/.api_keys. No em dashes.
"""
import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("ICL_N", "200"))
SHOT = int(os.environ.get("ICL_SHOT", "16"))
KC = int(os.environ.get("ICL_KCHAR", "400"))
WORKERS = int(os.environ.get("ICL_WORKERS", "8"))
MODELS = os.environ.get("ICL_MODELS", "claude-opus-4-8").split(",")

# rung: (csv, field, property phrase, 0-shot output ref, ceiling ref)
RUNGS = {
    "ecg5000": ("signal/ecg/ecg5000.csv", "series",
                "this ECG heartbeat trace is ABNORMAL (not normal sinus rhythm)", "~0.5", "5-NN 0.994 / ECG-FM"),
}


def parse_prob(t):
    for tok in reversed(re.findall(r"\d*\.?\d+", t)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def load(csvp, n):
    rows = list(csv.DictReader(open(os.path.join(ROOT, csvp))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos, neg


def build(exs, q, field, prop):
    s = (f"Estimate the probability (a single number between 0 and 1) that {prop}. "
         f"Here are labeled examples (1 = yes, 0 = no):\n\n")
    for e in exs:
        s += f"Input: {e[field][:KC]}\nProbability: {int(e['label'])}\n\n"
    s += f"Now give the probability for this input. Reply with ONLY a number.\nInput: {q[field][:KC]}\nProbability:"
    return s


def run(client, model, prompts):
    def call(p):
        try:
            m = client.messages.create(model=model, max_tokens=10,
                                       messages=[{"role": "user", "content": p}])
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return parse_prob(t)
        except Exception:
            return 0.5
    res = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return np.array(res)


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    print(f"{SHOT}-shot ICL placement (retrieve column of the decision map)", flush=True)
    for rung, (csvp, field, prop, oref, cref) in RUNGS.items():
        pos, neg = load(csvp, N)
        k = SHOT // 2
        exs = []
        for i in range(k):
            exs += [pos[i], neg[i]]
        test = pos[k:k + N // 2] + neg[k:k + N // 2]
        y = np.array([int(r["label"]) for r in test])
        print(f"\n## {rung}  (0-shot output {oref}; ceiling {cref})", flush=True)
        for model in MODELS:
            pred = run(client, model, [build(exs, q, field, prop) for q in test])
            auc = roc_auc_score(y, pred)
            print(f"  {model:28s} {SHOT}-shot ICL AUROC={auc:.3f}  -> {'CLOSES (retrieve)' if auc > 0.7 else 'does NOT close (train/orchestrate)'}", flush=True)


if __name__ == "__main__":
    main()
