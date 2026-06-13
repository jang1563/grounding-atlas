"""3D-structure rung: the 3-arm on a molecule's 3D COORDINATES (Qwen3-8B, GPU).

Same hERG molecules as the SMILES rung, but presented as raw XYZ atomic coordinates (web-zero,
geometry-bound) instead of the SMILES string. Tests whether 3D structure is the genuine
ENCODING-LIMITED anchor the modality ladder lacked: the property (hERG) IS in the structure
the coordinates encode (Morgan ceiling ~0.82), but recovering it from raw coordinates needs 3D
geometry parsing a forward pass cannot do. ceiling = Morgan FP on the SMILES (the structure the
coords encode); activation = LLM hidden-state probe on the XYZ text; output = LLM verbalized.
Reading: activation LOW (near chance) = encoding-limited (the model cannot form the structure
from coordinates); activation MODERATE (surface) = expression-limited like the others.
Frontier output already measured at chance (0.49 to 0.54). Env: ACT_MODEL, ACT_N. No em dashes.
"""
import csv
import os
import re

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "400"))
CSV = os.environ.get("ACT_CSV", "herg_xyz.csv")
GEN = 12
PROMPT = ("Below are the 3D atomic coordinates (element, x, y, z in Angstroms) of a molecule. Estimate "
          "the probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel "
          "(cardiotoxicity risk). Reply with ONLY the number.\n\nCoordinates:\n{x}\nProbability:")


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
    xyzs = [r["xyz"] for r in data]
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) or s for s in smis])
    cv = GroupKFold(5)

    # ceiling: Morgan FP on the SMILES (the structure the coordinates encode)
    FP = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048), float) for s in smis])
    sp = cross_val_predict(make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000)),
                           FP, y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    c_auc = roc_auc_score(y, sp)
    print(f"CEILING (Morgan FP on SMILES) AUROC={c_auc:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
    layers, H, outp = None, None, []
    for i, t in enumerate(xyzs):
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
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(xyzs)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best = max(roc_auc_score(y, cross_val_predict(clf, np.asarray(H[L]), y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1]) for L in range(layers))
    print(f"MODEL={MODEL}  n={len(y)}", flush=True)
    print(f"SUMMARY (3D coords hERG):  ceiling(Morgan)={c_auc:.3f} | ACTIVATION={best:.3f} | OUTPUT={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = {c_auc - best:.3f} | expression = {best - o_auc:.3f}", flush=True)
    print(f"regime: {'ENCODING-LIMITED (cannot form structure from coordinates)' if best < 0.62 else 'expression-limited (encodes the coordinate surface)'}", flush=True)


if __name__ == "__main__":
    main()
