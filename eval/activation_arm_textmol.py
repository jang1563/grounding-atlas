"""General molecular-text 3-arm (Qwen3-8B, GPU): hERG from a non-SMILES text rendering.

One script for the molecular-representation rungs (graph, NMR, 3D coords) that share the same
hERG molecules and the Morgan ceiling, differing only in how the molecule is rendered as text.
ceiling = Morgan FP on the SMILES; activation = LLM hidden-state probe on the text field;
output = LLM verbalized P(hERG). Env: ACT_CSV, ACT_TEXT_FIELD (graph|nmr|xyz), ACT_MODEL, ACT_N.
No em dashes.
"""
import os
import re
import csv
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "600"))
CSV = os.environ.get("ACT_CSV", "herg_graph.csv")
FIELD = os.environ.get("ACT_TEXT_FIELD", "graph")
GEN = 12
PROMPTS = {
    "graph": ("Below is a molecular graph (an atom list and a bond edge list) of a molecule. Estimate the "
              "probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel "
              "(cardiotoxicity risk). Reply with ONLY the number.\n\n{x}\nProbability:"),
    "nmr": ("Below is the simulated 13C NMR spectrum (a list of carbon chemical shifts in ppm) of a molecule. "
            "Estimate the probability (a single number between 0 and 1) that it BLOCKS the hERG potassium "
            "channel (cardiotoxicity risk). Reply with ONLY the number.\n\nShifts (ppm): {x}\nProbability:"),
    "xyz": ("Below are the 3D atomic coordinates (element, x, y, z in Angstroms) of a molecule. Estimate the "
            "probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel "
            "(cardiotoxicity risk). Reply with ONLY the number.\n\nCoordinates:\n{x}\nProbability:"),
}
PROMPT = PROMPTS[FIELD]


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
    smis = [r["smiles"] for r in data]
    texts = [r[FIELD] for r in data]
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) or s for s in smis])
    cv = GroupKFold(5)
    FP = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048), float) for s in smis])
    c_auc = roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000)),
                                               FP, y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
    print(f"CEILING (Morgan FP on SMILES) AUROC={c_auc:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
    layers, H, outp = None, None, []
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
            g = model.generate(**inp, max_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)
        outp.append(parse_prob(tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))[0])
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best = max(roc_auc_score(y, cross_val_predict(clf, np.asarray(H[L]), y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1]) for L in range(layers))
    print(f"MODEL={MODEL}  field={FIELD}  n={len(y)}", flush=True)
    print(f"SUMMARY ({FIELD} hERG):  ceiling(Morgan)={c_auc:.3f} | ACTIVATION={best:.3f} | OUTPUT={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = {c_auc - best:.3f} | expression = {best - o_auc:.3f}", flush=True)


if __name__ == "__main__":
    main()
