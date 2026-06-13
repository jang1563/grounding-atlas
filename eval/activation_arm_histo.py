"""Histopathology rung: open-VLM 3-arm on H&E patches (Qwen2.5-VL-7B, GPU).

PatchCamelyon 96x96 H&E patches -> tumor present in the central region. ceiling = a cheap
image-feature classifier (per-channel color statistics, the cheap-specialist analog; a real
pathology FM like CONCH reaches ~0.9); activation = VLM hidden-state probe on the patch;
output = VLM verbalized P(tumor). Tests whether a general VLM grounds histopathology (like the
molecular-image rung but on tissue). Data: signal/histo/pcam.csv (img path, label). Env:
VL_MODEL, VL_N. No em dashes.
"""
import csv
import os
import re

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
N = int(os.environ.get("VL_N", "400"))
CSV = os.environ.get("VL_CSV", os.path.join(ROOT, "signal", "histo", "pcam.csv"))
PROMPT = ("This is a hematoxylin and eosin (H&E) stained histopathology image patch. Estimate the "
          "probability (a single number between 0 and 1) that it contains TUMOR (cancerous) tissue. "
          "Reply with ONLY the number.")


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


def color_feats(img):
    a = np.asarray(img.convert("RGB"), float) / 255.0
    f = []
    for c in range(3):
        ch = a[:, :, c]
        f += [ch.mean(), ch.std(), np.percentile(ch, 25), np.percentile(ch, 75)]
    # hematoxylin (blue/purple nuclei) proxy: blue-minus-red, and overall darkness
    f += [(a[:, :, 2] - a[:, :, 0]).mean(), a.mean(), a.std()]
    return f


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def main():
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    imgs = [Image.open(r["img"]).convert("RGB") for r in data]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    # ceiling: cheap color-feature classifier
    X = np.array([color_feats(im) for im in imgs])
    sp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                           X, y, cv=cv, method="predict_proba")[:, 1]
    c_auc = roc_auc_score(y, sp)
    print(f"CEILING (cheap color-feature classifier) AUROC={c_auc:.3f}  (pathology FM CONCH ~0.9 ref)", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device
    layers, H, outp = None, None, []
    for i, img in enumerate(imgs):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            fwd = model(**inp, output_hidden_states=True)
        vec = [h[0, -1].float().cpu().numpy() for h in fwd.hidden_states]
        if H is None:
            layers = len(vec)
            H = [[] for _ in range(layers)]
        for L in range(layers):
            H[L].append(vec[L])
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=12, do_sample=False)
        txt = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        outp.append(parse_prob(txt)[0])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best = max(roc_auc_score(y, cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, method="predict_proba", n_jobs=5)[:, 1]) for L in range(layers))
    print(f"MODEL={MODEL}  n={len(y)}", flush=True)
    print(f"SUMMARY (histopath tumor):  ceiling={c_auc:.3f} | ACTIVATION={best:.3f} | OUTPUT={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = {c_auc - best:.3f} | expression = {best - o_auc:.3f}", flush=True)


if __name__ == "__main__":
    main()
