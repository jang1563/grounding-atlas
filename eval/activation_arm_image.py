"""Image rung B: open-VLM (Qwen2.5-VL-7B) hidden-state 3-arm on rendered hERG molecules.

The DIRECT test the OCSR proxy could only approximate: do the VLM's HIDDEN STATES encode
hERG from the image? ceiling = Morgan on the true structure; activation = a linear probe on
the VLM's last-token hidden states given the image; output = the VLM's verbalized P(hERG).
Predicted from the OCSR proxy (image_rung.md, 0.759): activation ~0.75 = EXPRESSION-limited
(the VLM encodes hERG from the image but cannot verbalize it), NOT encoding-limited.
Env: VL_MODEL, VL_N (balanced total, default 400), VL_CSV (smiles,label). No em dashes.
"""
import csv
import io
import os
import re
from collections import Counter

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

RDLogger.DisableLog("rdApp.*")
MODEL = os.environ.get("VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
N = int(os.environ.get("VL_N", "400"))
CSV = os.environ.get("VL_CSV", "herg.csv")
PROMPT = ("This image shows a molecule. Estimate the probability (a single number between 0 and 1) "
          "that it BLOCKS the hERG potassium channel (cardiotoxicity risk). Reply with ONLY the number.")


def render(smi, size=336):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    d.DrawMolecule(m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")


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
    rows = [(r["smiles"], int(r["label"])) for r in csv.DictReader(open(CSV)) if Chem.MolFromSmiles(r["smiles"]) is not None]
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def main():
    data = load(N)
    smis = [s for s, _ in data]
    y = np.array([l for _, l in data])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    FP = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048), float) for s in smis])
    lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    sp = cross_val_predict(lr, FP, y, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    sp_auc = roc_auc_score(y, sp)
    print(f"CEILING (Morgan on true structure)  AUROC={sp_auc:.3f}  AUPRC={average_precision_score(y, sp):.3f}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
    dev = next(model.parameters()).device

    layers, H, outp, ptypes = None, None, [], []
    for i, s in enumerate(smis):
        img = render(s)
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
        p, kind = parse_prob(txt)
        outp.append(p)
        ptypes.append(kind)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(smis)}", flush=True)
    outp = np.array(outp)
    o_auc = roc_auc_score(y, outp)
    print(f"MODEL={MODEL}  n={len(y)}  pos={int(y.sum())}  layers={layers}", flush=True)
    print(f"OUTPUT  AUROC={o_auc:.3f}  parse={dict(Counter(ptypes))}", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, bestL, bestp = 0.0, -1, None
    for L in range(layers):
        pr = cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, pr)
        if a > best:
            best, bestL, bestp = a, L, pr
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    print(f"\nbest ACTIVATION layer {bestL}: AUROC={best:.3f} (max over {layers})", flush=True)

    ys = np.random.RandomState(123).permutation(y)
    ac = cross_val_predict(clf, np.asarray(H[bestL]), ys, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    print(f"SELECTIVITY: activation {best - roc_auc_score(ys, ac):.3f}", flush=True)

    print(f"\nSUMMARY (image hERG, n={len(y)}):  ceiling={sp_auc:.3f} | activation={best:.3f} | output={o_auc:.3f}", flush=True)
    print(f"gaps: encoding = ceiling - activation = {sp_auc - best:.3f} | expression = activation - output = {best - o_auc:.3f}", flush=True)
    print(f"regime: {'ENCODING-limited' if best < 0.62 else 'EXPRESSION-limited (VLM encodes hERG from the image, cannot verbalize)'}", flush=True)


if __name__ == "__main__":
    main()
