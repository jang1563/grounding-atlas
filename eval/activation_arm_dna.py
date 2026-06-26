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
import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from probe_common import (
    cluster_boot,
    control_curve,
    dump_layerloc,
    layer_curve,
    nested_layer_auroc,
    residualized_cv_auroc,
    results_path,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # per-layer curve (biased max) + UNBIASED nested-CV best-layer + selectivity (probe_common).
    # DNA has no group structure here (StratifiedKFold, shuffle handles the class-block order);
    # the prereg's sequence-cluster GroupKFold + GC-residualization are control-2 follow-ups.
    aucs, bestL, bestp = layer_curve(H, y)
    for L, a in enumerate(aucs):
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    naive = max(aucs)
    a_lo, a_hi = cluster_boot(y, bestp)
    nb = nested_layer_auroc(H, y)
    cc = control_curve(H, y)
    sel_curve = [a - c for a, c in zip(aucs, cc)]
    peak_sel = int(np.argmax(sel_curve))
    ctrl, sel = cc[bestL], sel_curve[bestL]
    print(f"\nbest ACTIVATION layer {bestL}: AUROC={naive:.3f} [{a_lo:.3f},{a_hi:.3f}] (MAX over {layers}, selection-biased)", flush=True)
    print(f"HELD-OUT-LAYER (nested CV, UNBIASED): AUROC={nb['auroc']:.3f} (fold-mean {nb['auroc_fold']:.3f}, picked {nb['picked']}) | selection bias = {naive - nb['auroc']:+.3f}", flush=True)
    print(f"SELECTIVITY: activation@L{bestL} = {sel:.3f} (control {ctrl:.3f}; high = reads real signal)", flush=True)
    print(f"PEAK-SELECTIVITY layer {peak_sel} (sel={sel_curve[peak_sel]:.3f}, depth {peak_sel / (layers - 1):.2f}) = H1 endpoint (where COMPUTED, not the raw-AUROC peak)", flush=True)

    print(f"\nSUMMARY (DNA promoter, n={len(y)}):  ceiling(6-mer)={sp_auc:.3f} | activation(max)={naive:.3f} | activation(held-out)={nb['auroc']:.3f} | output={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = ceiling - act(held-out) = {sp_auc - nb['auroc']:.3f} | expression = act(held-out) - output = {nb['auroc'] - o_auc:.3f}", flush=True)

    # GC-residualization control (prereg 4.4 control 2): is L* reading promoter SEMANTICS or just GC%?
    # The DNA promoter task is strongly GC-separable (a surface confound), so a layer that only encodes
    # composition would score high BY CONSTRUCTION. H2-DNA is KILLED if the residualized probe does not
    # beat the strongest pure-surface predictor (GC-alone or 6-mer LR) by >= +0.10.
    gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
    slen = np.array([len(s) for s in seqs], dtype=float)
    Z = np.column_stack([gc, slen, np.ones(len(y))])
    gc_floor = roc_auc_score(y, cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        gc[:, None], y, cv=cvk, method="predict_proba", n_jobs=5)[:, 1])
    surface_floor = max(gc_floor, sp_auc)
    res_auroc = residualized_cv_auroc(np.asarray(H[bestL]), y, Z)
    gc_margin = res_auroc - surface_floor
    gc_verdict = "PASS" if gc_margin >= 0.10 else "KILL H2-DNA (reads composition, not semantics)"
    print(f"GC-CONTROL: gc%-floor={gc_floor:.3f} 6mer-ceiling={sp_auc:.3f} surface-floor={surface_floor:.3f} | residualized-probe@L{bestL}={res_auroc:.3f} | margin={gc_margin:+.3f} -> {gc_verdict}", flush=True)

    tag = MODEL.split("/")[-1]
    dump_layerloc(results_path(f"layer_loc_dna_promoter_{tag}.json"), "dna/promoter", MODEL,
                  y, aucs, nb, sel, output=outp, ceiling=sp_auc,
                  sel_curve=sel_curve, control_curve=cc, peak_sel_layer=peak_sel,
                  extra={"gc_floor": round(float(gc_floor), 4), "surface_floor": round(float(surface_floor), 4),
                         "residualized_auroc": round(float(res_auroc), 4), "gc_margin": round(float(gc_margin), 4)})
    print(f"  [layer-loc] wrote results/layer_loc_dna_promoter_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
