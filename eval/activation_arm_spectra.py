"""Spectra rung: the 3-arm on a SIMULATED mass spectrum (the non-renderable, hardest modality).

The molecule is presented ONLY as a list of fragment m/z peaks (BRICS fragmentation, a
deterministic crude MS), never as a structure. Two ceilings bracket what is recoverable:
  ceiling_structure = Morgan probe on the TRUE structure (~0.82) = what a perfect structure-
                      elucidation specialist (SpecTUS-class) reaches by inverting the spectrum
  ceiling_surface   = a probe on the binned m/z histogram (~0.64) = the SURFACE peak-statistics
                      signal, recoverable without elucidation
The LLM arms read the peak-list TEXT:
  activation = linear probe on the LLM hidden states
  output     = the LLM verbalized P(hERG)
Reading: activation near ceiling_surface (0.64) but far below ceiling_structure (0.82) = a
genuine ENCODING gap (the LLM reads surface peak stats but cannot elucidate the structure in
a forward pass), the first encoding-limited anchor; activation near chance = even the surface
is unreadable. Env: ACT_MODEL, ACT_N, ACT_CSV (smiles,label). No em dashes.
"""
import os
import re
import csv
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "1500"))
CSV = os.environ.get("ACT_CSV", "herg.csv")
GEN_TOKENS = 12
PROMPT = (
    "The following are the fragment m/z peaks from the mass spectrum of a molecule. Estimate "
    "the probability (a single number between 0 and 1) that the molecule BLOCKS the hERG "
    "potassium channel (cardiotoxicity risk). Reply with ONLY the number.\n\nPeaks (m/z): {x}\nProbability:"
)


def parse_prob(txt):
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


def sim_ms_masses(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    masses = [Descriptors.ExactMolWt(m)]
    try:
        for f in BRICS.BRICSDecompose(m):
            fm = Chem.MolFromSmiles(f)
            if fm is not None:
                masses.append(Descriptors.ExactMolWt(fm))
    except Exception:
        pass
    return sorted(masses)


def ms_text(masses):
    return ", ".join(f"{x:.1f}" for x in masses) if masses else "none"


def ms_hist(masses, nbin=100, hi=1000.0):
    v = np.zeros(nbin)
    for mz in masses:
        b = int(mz / hi * nbin)
        if 0 <= b < nbin:
            v[b] += 1.0
    return v


def boot(y, p, nb=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    a = []
    for _ in range(nb):
        b = rng.choice(idx, len(idx), True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def load(n):
    rows = [(r["smiles"], int(r["label"])) for r in csv.DictReader(open(CSV)) if Chem.MolFromSmiles(r["smiles"]) is not None]
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
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
    smis = [s for s, _ in data]
    y = np.array([l for _, l in data])
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) or s for s in smis])
    cv = GroupKFold(5)
    masses = [sim_ms_masses(s) for s in smis]
    texts = [ms_text(m) for m in masses]

    # two ceilings
    FPs = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048), float) for s in smis])
    HISTs = np.array([ms_hist(m) for m in masses])
    lr = lambda: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    c_struct = roc_auc_score(y, cross_val_predict(lr(), FPs, y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
    c_surf = roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), HISTs, y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
    print(f"CEILING structure (Morgan on true)   = {c_struct:.3f}  (elucidation-specialist ceiling)", flush=True)
    print(f"CEILING surface  (binned m/z probe)  = {c_surf:.3f}  (surface peak-statistics)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device

    layers, H, outp, ptypes = None, None, [], []
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
            g = model.generate(**inp, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        p, kind = parse_prob(txt)
        outp.append(p)
        ptypes.append(kind)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    print(f"MODEL={MODEL}  n={len(y)}  layers={layers}", flush=True)
    print(f"OUTPUT  AUROC={o_auc:.3f}  parse={dict(Counter(ptypes))}", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, bestL, bestp = 0.0, -1, None
    for L in range(layers):
        pr = cross_val_predict(clf, np.asarray(H[L]), y, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, pr)
        if a > best:
            best, bestL, bestp = a, L, pr
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    a_lo, a_hi = boot(y, bestp)
    print(f"\nbest ACTIVATION layer {bestL}: AUROC={best:.3f} [{a_lo:.3f},{a_hi:.3f}]", flush=True)

    ys = np.random.RandomState(123).permutation(y)
    ac = cross_val_predict(clf, np.asarray(H[bestL]), ys, groups=groups, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    print(f"SELECTIVITY: activation {best - roc_auc_score(ys, ac):.3f}", flush=True)

    print(f"\nSUMMARY (spectra/MS hERG, n={len(y)}):  ceiling_structure={c_struct:.3f} | ceiling_surface={c_surf:.3f} | activation={best:.3f} | output={o_auc:.3f}", flush=True)
    print(f"gaps: vs structure(elucidation) = {c_struct - best:.3f} | vs surface = {c_surf - best:.3f} | expression = {best - o_auc:.3f}", flush=True)
    if best <= c_surf + 0.03 and (c_struct - best) > 0.10:
        print("regime: ENCODING-LIMITED vs structure (LLM reads surface peak stats, cannot elucidate the structure)", flush=True)
    elif best > c_surf + 0.05:
        print("regime: the LLM exceeds surface peak stats (encodes more than the histogram)", flush=True)
    else:
        print("regime: check", flush=True)


if __name__ == "__main__":
    main()
