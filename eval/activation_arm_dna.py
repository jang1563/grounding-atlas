"""DNA/RNA modality rung: the 3-arm instrument on a DNA content-property (promoter).

Mirrors activation_arm.py but for DNA: the cheap-specialist CEILING is a 6-mer-frequency
logistic regression (the DNA analog of the Morgan fingerprint), and the split is a random
StratifiedKFold (DNA has no Murcko scaffold). Tests the web-exposure law at a NEW point:
a raw DNA sequence -> property mapping is web-poor (sequences rarely appear in text bound to
"promoter"), so the OUTPUT arm should sit near chance; the ACTIVATION arm decides whether
the 8B nonetheless ENCODES promoter-ness (expression gap) or never forms it (encoding gap).

Env: ACT_MODEL (Qwen/Qwen3-8B), ACT_N (balanced total, default 1500), ACT_CSV (smiles col =
DNA sequence), ACT_DUMP (per-item json). Data: dna_promoter.csv (genomic-benchmarks
human_nontata_promoters, 6-mer gating 0.898). No em dashes.
"""
import os
import re
import csv
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

CSV = os.environ.get("ACT_CSV", "dna_promoter.csv")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "1500"))
GEN_TOKENS = 12
PROMPTS = {
    "promoter": ("You are given a DNA sequence. Estimate the probability (a single number between 0 and 1) "
                 "that it is a PROMOTER (a regulatory region that initiates transcription). Judge only from "
                 "the sequence. Reply with ONLY the number.\n\nSequence: {x}\nProbability:"),
    "coding": ("You are given a genomic sequence. Estimate the probability (a single number between 0 and 1) "
               "that it is a protein-CODING sequence (an exon / coding RNA, as opposed to an intergenic "
               "region). Judge only from the sequence. Reply with ONLY the number.\n\nSequence: {x}\nProbability:"),
}
PROMPT = PROMPTS[os.environ.get("ACT_TASK", "promoter")]


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


def kmers(s, k=6):
    return " ".join(s[i:i + k] for i in range(len(s) - k + 1))


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
    rows = [(r["smiles"], int(r["label"])) for r in csv.DictReader(open(CSV))]
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
    seqs = [s for s, _ in data]
    y = np.array([l for _, l in data])
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)

    # CEILING: 6-mer frequency LR (the cheap DNA specialist, analog of Morgan FP)
    Xc = CountVectorizer().fit_transform([kmers(s) for s in seqs])
    lrk = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    sp = cross_val_predict(lrk, Xc, y, cv=cvk, method="predict_proba", n_jobs=5)[:, 1]
    sp_auc = roc_auc_score(y, sp)
    print(f"CEILING (6-mer LR)  AUROC={sp_auc:.3f}  AUPRC={average_precision_score(y, sp):.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device

    layers, H, outp, ptypes = None, None, [], []
    for i, s in enumerate(seqs):
        inp = tok(chat(tok, s), return_tensors="pt", truncation=True, max_length=512).to(dev)
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
            print(f"  {i+1}/{len(seqs)}", flush=True)
    outp = np.array(outp)
    print(f"MODEL={MODEL}  n={len(y)}  pos={int(y.sum())}  layers={layers}", flush=True)

    o_auc = roc_auc_score(y, outp)
    o_lo, o_hi = boot(y, outp)
    pc = Counter(ptypes)
    print(f"OUTPUT  AUROC={o_auc:.3f} [{o_lo:.3f},{o_hi:.3f}]  parse={dict(pc)}", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, bestL, bestp = 0.0, -1, None
    for L in range(layers):
        pr = cross_val_predict(clf, np.asarray(H[L]), y, cv=cvk, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, pr)
        if a > best:
            best, bestL, bestp = a, L, pr
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    a_lo, a_hi = boot(y, bestp)
    print(f"\nbest ACTIVATION layer {bestL}: AUROC={best:.3f} [{a_lo:.3f},{a_hi:.3f}] (max over {layers})", flush=True)

    # selectivity control (Hewitt-Liang): shuffled labels
    ys = np.random.RandomState(123).permutation(y)
    ac = cross_val_predict(clf, np.asarray(H[bestL]), ys, cv=cvk, method="predict_proba", n_jobs=5)[:, 1]
    print(f"SELECTIVITY: activation {best - roc_auc_score(ys, ac):.3f}  (high = reads real signal)", flush=True)

    print(f"\nSUMMARY (DNA promoter, n={len(y)}):  ceiling(6-mer)={sp_auc:.3f} | activation={best:.3f} | output={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = ceiling - activation = {sp_auc - best:.3f} | expression = activation - output = {best - o_auc:.3f}", flush=True)

    if os.environ.get("ACT_DUMP"):
        import json
        json.dump({"best_layer": int(bestL), "model": MODEL, "n": len(y),
                   "summary": {"ceiling": round(sp_auc, 3), "activation": round(best, 3), "output": round(o_auc, 3)},
                   "items": [{"seq": seqs[i], "label": int(y[i]), "act": round(float(bestp[i]), 4),
                              "ceiling": round(float(sp[i]), 4), "output": round(float(outp[i]), 4)} for i in range(len(y))]},
                  open(os.environ["ACT_DUMP"], "w"))
        print(f"  [ACT_DUMP] wrote per-item to {os.environ['ACT_DUMP']}", flush=True)


if __name__ == "__main__":
    main()
