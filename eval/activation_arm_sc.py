"""Single-cell rung: the 3-arm on a cell's expression rendered two ways (Qwen3-8B, GPU).

Each cell is the SAME entity in a web-rich and a web-zero form, the cleanest web-exposure test:
  cell_sentence = top-50 gene SYMBOLS rank-ordered by expression (web-rich; gene names known)
  anon          = the SAME genes' fixed anonymized IDs, same order (web-zero; identity removed)
Both forms carry the signal (a bag-of-tokens probe gets ~0.99 on either, a consistent renaming),
so the contrast isolates GENE IDENTITY = web exposure. Specialist ceiling (supervised LogReg on
the full expression vector) = 0.989, precomputed locally. For each form: ceiling_surface (bag of
tokens), activation (LLM hidden-state probe), output (LLM verbalized P(T cell)). Prediction:
cell_sentence output >> anon output (only gene names let the model verbalize the cell type), and
cell_sentence activation > anon activation. Env: ACT_MODEL, ACT_N. No em dashes.
"""
import csv
import os
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "384"))
CSV = os.environ.get("ACT_CSV", "pbmc_Tcell.csv")
GEN = 12
CEILING_SUPERVISED = 0.989  # supervised LogReg on the full expression vector (precomputed local)
PROMPT = ("Below are the most highly expressed genes of a single human PBMC cell, in descending order "
          "of expression. Estimate the probability (a single number between 0 and 1) that this cell is "
          "a T lymphocyte (T cell). Reply with ONLY the number.\n\nGenes: {x}\nProbability:")


def parse_prob(t):
    for tok in reversed(re.findall(r"\d*\.?\d+", t)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "parsed"
        if 1.0 < v <= 100.0:
            return v / 100.0, "percent"
    return 0.5, "fallback"


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def chat(tok, x):
    m = [{"role": "user", "content": PROMPT.format(x=x)}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def main():
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
    print(f"MODEL={MODEL}  n={len(y)}  pos={int(y.sum())}  supervised_ceiling={CEILING_SUPERVISED}", flush=True)

    for field in ("cell_sentence", "anon"):
        texts = [r[field] for r in data]
        surf = roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000),
                CountVectorizer(token_pattern=r"[^ ]+").fit_transform(texts), y, cv=cv, method="predict_proba")[:, 1])
        layers, H, outp = None, None, []
        for i, t in enumerate(texts):
            inp = tok(chat(tok, t), return_tensors="pt", truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                fwd = model(**inp, output_hidden_states=True)
            vec = [h[0, -1].float().cpu().numpy() for h in fwd.hidden_states]
            if H is None:
                layers = len(vec)
                H = [[] for _ in range(layers)]
            for L in range(layers):
                H[L].append(vec[L])
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            outp.append(parse_prob(tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))[0])
        outp = np.array(outp)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        best = max(roc_auc_score(y, cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
                   for L in range(layers))
        o_auc = roc_auc_score(y, outp)
        print(f"\n== {field} ==", flush=True)
        print(f"  ceiling_surface(bag-of-tokens)={surf:.3f} | ACTIVATION(max)={best:.3f} | OUTPUT={o_auc:.3f}", flush=True)
        print(f"  vs supervised ceiling {CEILING_SUPERVISED}: enc gap {CEILING_SUPERVISED-best:.3f} | exp gap {best-o_auc:.3f}", flush=True)


if __name__ == "__main__":
    main()
