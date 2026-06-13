"""Supervision-asymmetry control: 8B FEW-SHOT output on hERG.

The activation probe is trained on 1250 labels (supervised) while the output arm is
zero-shot, which is an unfair comparison (eval/README confound 1). This gives the LLM
matched in-context examples and re-measures the output. If few-shot output stays well
below the activation probe (0.787), the probe's advantage is NOT just supervision: the
model is handed labeled examples and still cannot verbalize the property, so the
expression gap is real, not a trained-vs-zero-shot artifact.

Same 1250 query set and seed as activation_arm.py; the K few-shot examples are held out
from the query set. Env: ACT_MODEL, ACT_CSV, ACT_N, ACT_FEWSHOT.
"""
import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

CSV = os.environ.get("ACT_CSV", "herg.csv")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "1250"))
K = int(os.environ.get("ACT_FEWSHOT", "10"))

PROMPT = (
    "You are given a molecule as a SMILES string. Estimate the probability "
    "(a single number between 0 and 1) that it BLOCKS the hERG potassium "
    "channel (cardiotoxicity risk). Judge only from the structure. "
    "Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:"
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


rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append((r["smiles"], int(r["label"])))
pos = [x for x in rows if x[1] == 1]
neg = [x for x in rows if x[1] == 0]
rng = np.random.RandomState(42)
rng.shuffle(pos)
rng.shuffle(neg)
kf = K // 2
# BUGFIX (deep review): reserve the few-shot examples FIRST, then build the query from the
# rest. The old code took query = pos[:k] with k=625=len(pos), so pos[k:k+kf] was EMPTY and
# the "K=10 balanced" few-shot was actually kf negatives + 0 positives (a degenerate prompt).
ex_pos, ex_neg = pos[:kf], neg[:kf]
pos, neg = pos[kf:], neg[kf:]
k = min(N // 2, len(pos), len(neg))
query = pos[:k] + neg[:k]
examples = ex_pos + ex_neg  # kf positives + kf negatives, held out from the query
ex_text = "Here are labeled examples (1.0 = blocks hERG, 0.0 = does not):\n" + "".join(
    f"SMILES: {s}\nProbability: {float(l):.1f}\n" for s, l in examples
) + "\nNow answer for this molecule the same way:\n"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto").eval()
dev = next(model.parameters()).device

y, p, kinds = [], [], []
for i, (smi, lab) in enumerate(query):
    content = ex_text + PROMPT.format(smiles=smi)
    msgs = [{"role": "user", "content": content}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    out = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    v, kind = parse_prob(out)
    p.append(v)
    y.append(lab)
    kinds.append(kind)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(query)}", flush=True)

y = np.array(y)
p = np.array(p)
print(f"MODEL={MODEL}  few-shot K={K}  n={len(y)}", flush=True)
print(f"FEW-SHOT OUTPUT AUROC={roc_auc_score(y, p):.3f}  parse={dict(Counter(kinds))}", flush=True)
print("compare: zero-shot output 0.453 | activation probe 0.787 | structure-probe 0.825", flush=True)
print("read: if few-shot stays near 0.45-0.55, the probe advantage is not supervision; the expression gap is real.", flush=True)
