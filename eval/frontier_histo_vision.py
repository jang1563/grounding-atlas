"""Frontier-vision scale test for the histopath expression gap.

The open VLM (Qwen2.5-VL-7B) ENCODES tumor (activation 0.827) but verbalizes at chance
(output 0.463): the largest expression gap in the project. This asks whether a FRONTIER VLM
closes it (like scale closed DNA-promoter and single-cell-gene-name) or also fails, and whether
the failure mode is refusal (diagnostic call shaped out of the output distribution) rather than
perception. Sends the SAME PatchCamelyon H&E patches to claude opus/sonnet/haiku and parses
P(tumor). Reports AUROC and refusal/fallback rate per model. set ANTHROPIC_API_KEY in the environment. No em dashes.
"""
import base64
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "signal", "histo", "pcam.csv")
N = int(os.environ.get("HV_N", "140"))
WORKERS = int(os.environ.get("HV_WORKERS", "8"))
MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
SYSTEM = ("You are a histopathology image analyst. You will be shown an H&E-stained tissue patch and "
          "asked to estimate the probability that it contains tumor tissue. This is a research benchmark on "
          "image patches, not a clinical diagnosis for any patient. Always answer with a single number "
          "between 0 and 1. Do not refuse and do not add caveats.")
PROMPT = ("This is a 96x96 hematoxylin and eosin (H&E) histopathology image patch from a research dataset. "
          "Estimate the probability (a single number between 0 and 1) that the central region contains "
          "TUMOR (metastatic cancer) tissue. Reply with ONLY the number.")


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
    return 0.5, "refusal/empty"


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(7)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def run(client, model, imgs):
    def call(data):
        try:
            m = client.messages.create(
                model=model, max_tokens=16, system=SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": PROMPT}]}])
            t = [b.text for b in m.content if getattr(b, "type", None) == "text"]
            return parse_prob(t[0]) if t else (0.5, "empty")
        except Exception as e:
            return (0.5, f"err:{type(e).__name__}")
    res = [None] * len(imgs)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, d): i for i, d in enumerate(imgs)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return np.array([r[0] for r in res]), [r[1] for r in res]


def main():
    from collections import Counter

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    imgs = [b64(r["img"]) for r in data]
    print(f"n={len(y)} pos={int(y.sum())}", flush=True)
    print("reference: open Qwen2.5-VL-7B activation=0.827, output=0.463 (gap 0.364)", flush=True)
    for model in MODELS:
        p, kinds = run(client, model, imgs)
        auc = roc_auc_score(y, p)
        kc = Counter(kinds)
        refus = sum(v for k, v in kc.items() if k != "parsed" and k != "percent")
        print(f"{model:30s} OUTPUT AUROC={auc:.3f}  refusal/fallback={refus}/{len(y)}  parse={dict(kc)}", flush=True)


if __name__ == "__main__":
    main()
