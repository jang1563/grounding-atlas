"""Methylation rung: the 3-arm on a web-ZERO numeric vector (epigenetic clock).

DNA-methylation beta values (cgXXXX:0.NN per CpG) -> chronological age. This is the purest
web-zero representation with a STRONG external ceiling: methylation predicts age extremely
well (the epigenetic clock), but the representation is pure numbers that never appear in text
bound to "age", so it tests the encoding floor. ceiling = logistic regression on the numeric
beta vector (the cheap clock); activation = Qwen3-8B hidden-state probe on the beta-TEXT;
output = 8B verbalized age estimate. Prediction: ceiling high, activation + output at chance
(encoding-limited), mirroring single-cell-anon. Contrast: single-cell with gene NAMES grounds,
methylation probe values do not. Data: signal/methyl/methyl_age.csv. Env: ACT_MODEL, ACT_N,
ACT_K (probes shown to the LLM). No em dashes.
"""
import csv
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.environ.get("ACT_CSV", os.path.join(ROOT, "signal", "methyl", "methyl_age.csv"))
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "600"))
K = int(os.environ.get("ACT_K", "100"))   # probes shown to the LLM (fits context)
PROMPT = ("Below are DNA methylation beta values (probe_id:value, value in [0,1]) measured from a "
          "human blood sample. Estimate this person's age in YEARS. Judge only from the values. "
          "Reply with ONLY a single number.\n\n{x}\n\nAge in years:")


def parse_age(t):
    for tok in re.findall(r"\d+\.?\d*", t):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0 <= v <= 120:
            return v
    return 50.0


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def vec_of(text, k):
    toks = text.split()[:k]
    return np.array([float(t.split(":")[1]) for t in toks]), " ".join(toks)


def chat(tok, x):
    m = [{"role": "user", "content": PROMPT.format(x=x)}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def main():
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    ages = np.array([float(r["age"]) for r in data])
    parsed = [vec_of(r["beta_text"], K) for r in data]
    X = np.array([v for v, _ in parsed])
    texts = [t for _, t in parsed]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    # ceiling: logistic regression on the numeric beta vector (the cheap clock)
    c_auc = roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                               X, y, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
    print(f"CEILING (LR on {K} beta values) AUROC={c_auc:.3f}  (epigenetic clock ref ~0.9+)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
    layers, H, pred_age = None, None, []
    for i, t in enumerate(texts):
        inp = tok(chat(tok, t), return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            fwd = model(**inp, output_hidden_states=True)
        vec = [h[0, -1].float().cpu().numpy() for h in fwd.hidden_states]
        if H is None:
            layers = len(vec)
            H = [[] for _ in range(layers)]
        for L in range(layers):
            H[L].append(vec[L])
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        pred_age.append(parse_age(tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    pred_age = np.array(pred_age)
    o_auc = roc_auc_score(y, pred_age)   # does verbalized age rank old>young?
    age_mae = float(np.mean(np.abs(pred_age - ages)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best = max(roc_auc_score(y, cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, method="predict_proba", n_jobs=5)[:, 1]) for L in range(layers))
    print(f"MODEL={MODEL}  n={len(y)}  K={K}", flush=True)
    print(f"SUMMARY (methylation age):  ceiling={c_auc:.3f} | ACTIVATION={best:.3f} | OUTPUT={o_auc:.3f}  (verbalized-age MAE={age_mae:.1f}y)", flush=True)
    print(f"gaps: encoding = {c_auc - best:.3f} | expression = {best - o_auc:.3f}", flush=True)


if __name__ == "__main__":
    main()
