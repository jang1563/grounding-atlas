"""WS3 decision map: the molecular-IMAGE point (the encoding-limited corner).

hERG-from-SMILES is expression-limited (the model encodes it, fails to verbalize), so
every non-solo placement recovered it (DECISION_MAP.md). Molecular IMAGE is predicted to
be the opposite corner: ENCODING-limited (the model cannot perceive the structure from
pixels at all, MolVision 2507.03283 image 0.15 vs text 0.71), so solo / retrieve / weights
should ALL fail and only ORCHESTRATE (an OCSR or image specialist, then the structure
probe) works. Claude is a VLM, so this is measured directly with image inputs.

Two arms on rendered hERG molecules:
  solo-image : image -> P(hERG block), AUROC  (can it read the PROPERTY off the image)
  ocsr       : image -> SMILES transcription, validity + Tanimoto to truth  (can it read
               the STRUCTURE at all = the perception floor that gates everything else)

Run: python eval/ws3_image.py
Env: IMG_N (balanced total, default 120), IMG_MODEL (default claude-sonnet-4-6), IMG_DRY.
No em dashes.
"""
import os
import re
import json
import base64
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAIRS = os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")
SYSTEM = ("You are a molecular property predictor. Respond with ONLY a single decimal "
          "number between 0 and 1 (for example: 0.42). No words, no explanation.")


def render_png(smi, size=320):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText(), m  # PNG bytes, mol


def load(n):
    by = defaultdict(list)
    for line in open(PAIRS):
        r = json.loads(line)
        if r.get("condition") != "matched":
            continue
        by[int(r["label"])].append(r["representation"])
    rng = np.random.RandomState(42)
    out = []
    for lab in (0, 1):
        items = by[lab][:]
        rng.shuffle(items)
        out += [(s, lab) for s in items[:n // 2]]
    return out


def parse_prob(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        v = float(tok)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100
    return 0.5


def img_block(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
            "data": base64.b64encode(png).decode()}}


def main():
    N = int(os.environ.get("IMG_N", "120"))
    model = os.environ.get("IMG_MODEL", "claude-sonnet-4-6")
    dry = os.environ.get("IMG_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    data = load(N)
    print(f"n={len(data)} (balanced) model={model}")

    if dry:
        png, _ = render_png(data[0][0])
        p = os.path.join(ROOT, "results", "ws3_image_sample.png")
        with open(p, "wb") as fh:
            fh.write(png)
        print(f"rendered sample -> {p} ({len(png)} bytes). DRY: no API.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    y, p_prop = [], []
    ocsr_valid, ocsr_tani, ocsr_exact = 0, [], 0
    itf = open(os.path.join(ROOT, "results", "ws3_image_items.jsonl"), "w")  # kill-safe per-item
    for i, (smi, lab) in enumerate(data):
        png, mol = render_png(smi)
        if png is None:
            continue
        # arm 1: read the PROPERTY off the image
        m1 = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
            messages=[{"role": "user", "content": [img_block(png),
                {"type": "text", "text": "Estimate the probability (0 to 1) that the molecule "
                 "shown blocks the hERG potassium channel. Judge only from the structure."}]}])
        t1 = "".join(b.text for b in m1.content if getattr(b, "type", None) == "text")
        y.append(lab)
        p_prop.append(parse_prob(t1))
        # arm 2: read the STRUCTURE off the image (OCSR, the perception floor)
        m2 = client.messages.create(model=model, max_tokens=120,
            messages=[{"role": "user", "content": [img_block(png),
                {"type": "text", "text": "Transcribe the molecule shown to a single SMILES "
                 "string. Reply with ONLY the SMILES, no other text."}]}])
        t2 = "".join(b.text for b in m2.content if getattr(b, "type", None) == "text").strip()
        t2c = re.sub(r"```[\w]*", "", t2).strip()  # strip fences without eating content
        parts = t2c.split()
        cand = parts[0] if parts else ""
        cm = Chem.MolFromSmiles(cand) if cand else None
        ti, exact = None, 0
        if cm is not None:
            ocsr_valid += 1
            ti = DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
                AllChem.GetMorganFingerprintAsBitVect(cm, 2, 2048))
            ocsr_tani.append(ti)
            if Chem.MolToSmiles(cm) == Chem.MolToSmiles(mol):
                ocsr_exact += 1
                exact = 1
        # also store stereo-stripped exact-match (rendered stereo is often illegible)
        exact_flat = int(cm is not None and Chem.MolToSmiles(cm, isomericSmiles=False) ==
                         Chem.MolToSmiles(mol, isomericSmiles=False))
        itf.write(json.dumps({"label": lab, "true_smiles": smi, "ocsr_cand": cand,
                              "prop": p_prop[-1], "ocsr_valid": int(cm is not None),
                              "ocsr_tani": ti, "ocsr_exact": exact, "ocsr_exact_flat": exact_flat}) + "\n")
        itf.flush()
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(data)}")
    itf.close()
    n = len(y)
    out = {"point": "molecular_image", "endpoint": "herg", "model": model, "n": n,
           "solo_image_auroc": round(float(roc_auc_score(y, p_prop)), 3),
           "ocsr_valid_rate": round(ocsr_valid / n, 3),
           "ocsr_mean_tanimoto": round(float(np.mean(ocsr_tani)), 3) if ocsr_tani else None,
           "ocsr_exact_rate": round(ocsr_exact / n, 3),
           "ref_solo_text_frontier": 0.633, "ref_orchestrate_ceiling": 0.895,
           "ref_lit_molvision_image": 0.15}
    with open(os.path.join(ROOT, "results", "ws3_image.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSOLO-IMAGE property AUROC={out['solo_image_auroc']}  "
          f"(solo-text 0.633, orchestrate 0.895, lit image 0.15)")
    print(f"OCSR (perception floor): valid={out['ocsr_valid_rate']} "
          f"mean_Tanimoto={out['ocsr_mean_tanimoto']} exact={out['ocsr_exact_rate']}")


if __name__ == "__main__":
    main()
