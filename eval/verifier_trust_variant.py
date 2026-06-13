"""Move 3 (light): does verifier-trust depend on web-exposure? variant named vs anon.

The LLM is shown a SIMULATED specialist verifier's verdict (PATHOGENIC/BENIGN) on a ClinVar
variant, with the verifier correct on half and deliberately wrong on half. The variant is
rendered named (gene + accession visible = web-rich, self-recallable) vs anon (gene/accession
stripped, only the c./p. change = web-poor). We measure whether the LLM FOLLOWS the verifier or
OVERRIDES it with its own answer, split by verifier-correct vs verifier-wrong.

Prediction: named -> LLM self-recalls and acts on its own knowledge (overrides a WRONG verifier
to the truth = good, but also ignores a CORRECT verifier = Tool-Ignored); anon -> LLM cannot
self-recall and DEPENDS on the verifier (follows it whether right or wrong). The headline is the
verifier-WRONG subset: LLM final accuracy named >> anon means web-exposure drives verifier-reliance.
source <your-keys-file>. No em dashes.
"""
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "signal", "clinvar", "variant_text.csv")
N = int(os.environ.get("VT_N", "400"))
WORKERS = int(os.environ.get("VT_WORKERS", "8"))
MODEL = os.environ.get("VT_MODEL", "claude-opus-4-8")
PROMPT = ("A specialist variant-effect prediction model classifies this genetic variant as {V}. "
          "Variant: {x}\n\nUsing the specialist prediction and your own judgment, estimate the probability "
          "(a single number between 0 and 1) that the variant is PATHOGENIC. Reply with ONLY the number.")


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


def anon(hgvs):
    # strip leading "NM_xxx.x(GENE):" so only the c./p. change remains (gene identity removed)
    s = re.sub(r"^[A-Za-z0-9_.]+\([^)]+\)\s*:\s*", "", hgvs).strip()
    s = re.sub(r"^[NX][MR]_[0-9.]+\s*:\s*", "", s).strip()
    return s if s and s != hgvs else hgvs


def load(n):
    rows = list(csv.DictReader(open(CSV)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = pos[:k] + neg[:k]
    rng.shuffle(data)
    return data


def run(client, texts, vpreds):
    def call(args):
        x, v = args
        try:
            m = client.messages.create(model=MODEL, max_tokens=10,
                                       messages=[{"role": "user", "content": PROMPT.format(V=v, x=x)}])
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return parse_prob(t)
        except Exception:
            return 0.5
    res = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, (texts[i], vpreds[i])): i for i in range(len(texts))}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return np.array(res)


def report(name, pred, y, vp, vcorrect):
    llm = (pred > 0.5).astype(int)
    follow = (llm == vp)            # LLM agrees with the verifier verdict
    correct = (llm == y)            # LLM final answer is right
    for tag, mask in [("verifier-CORRECT", vcorrect), ("verifier-WRONG", ~vcorrect)]:
        m = mask
        print(f"  {name:5s} {tag:16s} n={m.sum():3d}  LLM-correct={correct[m].mean():.3f}  follows-verifier={follow[m].mean():.3f}", flush=True)


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    data = load(N)
    y = np.array([int(r["label"]) for r in data])
    rng = np.random.RandomState(1)
    vp = np.where(rng.rand(len(y)) < 0.5, y, 1 - y)     # verifier verdict: 50% correct, 50% wrong
    vcorrect = (vp == y)
    vwords = np.where(vp == 1, "PATHOGENIC", "BENIGN")
    named = [r["text"] for r in data]
    anon_t = [anon(r["text"]) for r in data]
    print(f"n={len(y)} (verifier correct {int(vcorrect.sum())} / wrong {int((~vcorrect).sum())})  model={MODEL}", flush=True)
    print(f"anon example: '{named[0][:55]}' -> '{anon_t[0][:45]}'", flush=True)
    pred_named = run(client, named, vwords)
    pred_anon = run(client, anon_t, vwords)
    print("\n## results (LLM-correct = final answer right; follows-verifier = agrees with the verdict)", flush=True)
    report("named", pred_named, y, vp, vcorrect)
    report("anon", pred_anon, y, vp, vcorrect)
    print("\nHEADLINE: on verifier-WRONG, named LLM-correct >> anon = web-exposure drives verifier-reliance", flush=True)
    print(f"  named verifier-wrong correct {pred_named[~vcorrect].round().astype(int).__class__}  vs anon", flush=True)


if __name__ == "__main__":
    main()
