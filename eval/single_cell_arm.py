"""Single-cell capability x web-exposure arm: classify CD8-T vs NK from a cell-sentence, in two
conditions -- real gene NAMES (web-documented) vs global-consistent ANON ids (same vector, names
removed). The named condition should rise with model capability; the anon condition should stay
at chance at every scale. Reuses the harness dispatch + number parser.

Run:  python eval/single_cell_arm.py --model claude-sonnet-4-6 --n 60
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_grounding_eval import OUT, complete, parse_prob  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "signal", "single_cell", "cd8t_nk.csv")
PROMPT = ("These are the most distinctively expressed genes of one human immune cell (PBMC), "
          "highest first:\n{s}\nIs it a CD8+ T cell or an NK cell? Reply with ONLY a number "
          "between 0 and 1 = the probability that it is a CD8+ T cell.")


def load(n, rng):
    rows = list(csv.DictReader(open(DATA)))
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    k = min(n // 2, len(pos), len(neg))
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:k] + neg[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--conditions", default="name,anon")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    items = load(args.n, rng)
    y = np.array([int(r["label"]) for r in items])
    out = {"model": args.model, "n": len(y)}
    for cond in args.conditions.split(","):
        col = "cell_sentence" if cond == "name" else "anon"
        prob = np.array([parse_prob(complete(args.model, PROMPT.format(s=r[col]))) for r in items])
        auroc = roc_auc_score(y, prob) if len(set(y.tolist())) > 1 else float("nan")
        out[cond] = round(float(auroc), 3)
        print(f"{args.model:20s} {cond:5s} AUROC={auroc:.3f}  n={len(y)}", flush=True)
    d = os.path.join(OUT, "single_cell")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, args.model.replace("/", "_") + ".json"), "w"), indent=2)
    print(out, flush=True)


if __name__ == "__main__":
    main()
