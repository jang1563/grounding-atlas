"""MSA rung: the 3-arm on an alignment column -> evolutionary conservation.

A column of a protein multiple sequence alignment (aligned residues at one position across
homologs) -> is this position conserved. This is the on-thesis POSITIVE contrast to the
methylation rung: here the representation is amino-acid LETTERS (web-rich tokens) and the
property (conservation) is web-documented, so the two-factor law predicts the model grounds it
(small or no gap), whereas methylation's web-zero numeric vector does not. ceiling = logistic
regression on transparent column statistics (gap fraction, number of distinct residues);
activation = Qwen3-8B hidden-state probe on the column text; output = 8B verbalized P(conserved).
Data: signal/msa/msa_conservation.csv (Pfam seed alignments). Env: ACT_MODEL, ACT_N. No em dashes.
"""
import csv
import os
import re

import numpy as np
import torch
from probe_common import control_curve, dump_layerloc, layer_curve, nested_layer_auroc, results_path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.environ.get("ACT_CSV", os.path.join(ROOT, "signal", "msa", "msa_conservation.csv"))
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "600"))
PROMPT = ("Below is a single column of a protein multiple sequence alignment: the aligned amino-acid "
          "residues at one position across homologous proteins ('-' is a gap). Estimate the probability "
          "(a single number between 0 and 1) that this position is evolutionarily CONSERVED (functionally "
          "important, little variation). Reply with ONLY the number.\n\nColumn: {x}\nProbability:")


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
    texts = [r["column"] for r in data]
    Xc = np.array([[float(r["gap_frac"]), float(r["n_distinct"]), int(r["depth"])] for r in data])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    # ceiling: transparent column statistics (the cheap specialist)
    c_auc = roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                               Xc, y, cv=cv, method="predict_proba", n_jobs=5)[:, 1])
    print(f"CEILING (LR on column stats: gap_frac, n_distinct, depth) AUROC={c_auc:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
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
            g = model.generate(**inp, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        outp.append(parse_prob(tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    # per-layer curve + UNBIASED nested-CV best-layer + selectivity (probe_common). MSA is the
    # POSITIVE CONTROL / H2 floor: its nested headline anchors the gap scale for every other task.
    aucs, bestL, bestp = layer_curve(H, y)
    for L, a in enumerate(aucs):
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    naive = max(aucs)
    nb = nested_layer_auroc(H, y)
    cc = control_curve(H, y)
    sel_curve = [a - c for a, c in zip(aucs, cc)]
    peak_sel = int(np.argmax(sel_curve))
    ctrl, sel = cc[bestL], sel_curve[bestL]
    print(f"MODEL={MODEL}  n={len(y)}", flush=True)
    print(f"best ACTIVATION layer {bestL}: AUROC={naive:.3f} (MAX, selection-biased) | HELD-OUT (nested CV) AUROC={nb['auroc']:.3f} | bias {naive - nb['auroc']:+.3f}", flush=True)
    print(f"SELECTIVITY: activation@L{bestL} = {sel:.3f} (control {ctrl:.3f})", flush=True)
    print(f"PEAK-SELECTIVITY layer {peak_sel} (sel={sel_curve[peak_sel]:.3f}, depth {peak_sel / (layers - 1):.2f}) = H1 endpoint", flush=True)
    print(f"SUMMARY (MSA conservation):  ceiling={c_auc:.3f} | ACTIVATION(held-out)={nb['auroc']:.3f} | OUTPUT={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = ceiling - act = {c_auc - nb['auroc']:.3f} | expression = act - output = {nb['auroc'] - o_auc:.3f}", flush=True)
    tag = MODEL.split("/")[-1]
    dump_layerloc(results_path(f"layer_loc_msa_conservation_{tag}.json"), "msa/conservation", MODEL,
                  y, aucs, nb, sel, output=outp, ceiling=c_auc,
                  sel_curve=sel_curve, control_curve=cc, peak_sel_layer=peak_sel)
    print(f"  [layer-loc] wrote results/layer_loc_msa_conservation_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
