"""Protein branch, Claude output arm + content-sensitivity + name-vs-content (axis B/A).

The protein copy of ../../eval/head_to_head.py. Give Claude a protein, ask for the
thermostability probability, and compare AUROC against the ESM2 ceiling (~0.70). Tests both
whether Claude UTILIZES the sequence content and whether it grounds by NAME instead.

Conditions (env HH_COND); single, or PAIRED (HH_PAIRED=1, real vs HH_COND on the same items):
  real        sequence only (the content-only baseline -> output AUROC)
  scrambled   residue-shuffled sequence (composition preserved, ORDER destroyed)  [paired]
  mismatched  swapped for a real protein with the OPPOSITE label                  [paired]
  name_only   ONLY the source organism, no sequence (the name/identity cue)        [single]
  matched     source organism + sequence together                                 [single]
  name_conflict  sequence + a FALSE (opposite-thermal) organism: follows wrong name? [single]

The name-vs-content panel (content-only vs name_only vs matched) is the original-plan axis-A
test: meltome thermostability is strongly organism-determined (thermophiles high, psychrophiles
low), so if Claude predicts Tm from the organism NAME alone far better than from the SEQUENCE,
that is direct name-grounding. Needs the organism column (protein_meltome_named.csv).

Env: HH_MODEL, HH_N, HH_COND, HH_PAIRED, HH_WORKERS, HH_CSV. Keys: `set -a; source <your-keys-file>; set +a`.
"""
import csv
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = os.environ.get("HH_CSV", "protein_meltome_named.csv")
MODEL = os.environ.get("HH_MODEL", "claude-sonnet-4-5-20250929")
N = int(os.environ.get("HH_N", "200"))
COND = os.environ.get("HH_COND", "real")
PAIRED = os.environ.get("HH_PAIRED", "0") == "1"
WORKERS = int(os.environ.get("HH_WORKERS", "8"))

PROMPT = (
    "You are given a protein as an amino-acid sequence. Estimate the probability "
    "(a single number between 0 and 1) that its melting temperature is ABOVE {thresh} "
    "degrees Celsius (i.e. that it is relatively thermostable). Judge only from the "
    "sequence. Reply with ONLY the number.\n\nSequence: {seq}\nProbability:"
)
PROMPT_NAME = (
    "A protein's only known attribute is its source organism: {organism}. Estimate the "
    "probability (a single number between 0 and 1) that its melting temperature is ABOVE "
    "{thresh} degrees Celsius (i.e. that it is relatively thermostable). Reply with ONLY the "
    "number.\n\nProbability:"
)
PROMPT_MATCHED = (
    "You are given a protein's source organism and its amino-acid sequence. Estimate the "
    "probability (a single number between 0 and 1) that its melting temperature is ABOVE "
    "{thresh} degrees Celsius (i.e. that it is relatively thermostable). Reply with ONLY the "
    "number.\n\nOrganism: {organism}\nSequence: {seq}\nProbability:"
)


def load_sample(csv_path, n):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["sequence"], int(row["label"]), float(row["tm"]), row.get("organism", "")))
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    thresh = int(round(float(np.median([t for _, _, t, _ in rows]))))
    return pos[:k] + neg[:k], thresh


def alt_seq(seq, label, cond, seed, pos_pool, neg_pool):
    if cond == "scrambled":
        ch = list(seq)
        random.Random(seed).shuffle(ch)
        return "".join(ch)
    if cond == "mismatched":
        pool = neg_pool if label == 1 else pos_pool  # opposite label, real protein
        return pool[seed % len(pool)]
    return seq


# name_conflict: real sequence + a FALSE organism whose thermal class is opposite to the true
# label (thermophile name on a non-thermostable protein, cold name on a thermostable one). If
# Claude follows the name over the content, P(thermostable) anti-correlates with the true label
# and the AUROC inverts below 0.5. The decisive name-vs-content conflict test (FAILURE_MODES
# "mismatched" = real content + wrong name).
THERMO = ["Thermus thermophilus HB27", "Geobacillus stearothermophilus NCA26", "Picrophilus torridus DSM9790"]
COLD = ["Oleispira antarctica RB-8", "Caenorhabditis elegans", "Arabidopsis thaliana seedling"]


def false_organism(label, seed):
    pool = THERMO if label == 0 else COLD  # opposite thermal class to the truth
    return pool[seed % len(pool)]


def content_for(cond, seq, label, organism, thresh, seed, pos_pool, neg_pool):
    if cond == "name_only":
        return PROMPT_NAME.format(organism=organism or "unknown", thresh=thresh)
    if cond == "matched":
        return PROMPT_MATCHED.format(organism=organism or "unknown", seq=seq, thresh=thresh)
    if cond == "name_conflict":
        return PROMPT_MATCHED.format(organism=false_organism(label, seed), seq=seq, thresh=thresh)
    s = alt_seq(seq, label, cond, seed, pos_pool, neg_pool)
    return PROMPT.format(seq=s, thresh=thresh)


def parse_prob(txt):
    """Anchored: take the LAST number; map [0,1] directly, (1,100] as percent.
    Identical to activation_arm.py / head_to_head.py so every arm is parsed the same way."""
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


def bootstrap_ci(y, proba, n_boot=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], proba[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def ask(client, content):
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=20,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        return 0.5, "error"
    texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not texts:
        return 0.5, "empty"
    return parse_prob(texts[0])


def main():
    data, thresh = load_sample(CSV, N)
    pos_pool = [s for s, lab, _, _ in data if lab == 1]
    neg_pool = [s for s, lab, _, _ in data if lab == 0]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com")
    print(f"MODEL={MODEL}  n={len(data)}  thresh={thresh}C  cond={COND}  paired={PAIRED}  workers={WORKERS}", flush=True)

    if PAIRED:
        def run(i_item):
            i, (seq, label, _, org) = i_item
            rp, rk = ask(client, content_for("real", seq, label, org, thresh, i, pos_pool, neg_pool))
            ap, ak = ask(client, content_for(COND, seq, label, org, thresh, i, pos_pool, neg_pool))
            return label, rp, ap, rk, ak
        out = list(ThreadPoolExecutor(WORKERS).map(run, list(enumerate(data))))
        y = np.array([o[0] for o in out]); pr = np.array([o[1] for o in out]); pa = np.array([o[2] for o in out])
        kinds = [k for o in out for k in (o[3], o[4])]
        print(f"  parse: {dict(Counter(kinds))}  (fallback/error/empty -> 0.5)")
        if len(set(y)) > 1:
            print(f"  real AUROC={roc_auc_score(y, pr):.3f}  (ESM ceiling ~0.70)")
        md = float(np.abs(pr - pa).mean())
        tag = "0=content-ignored .. larger=tracks the swap" if COND == "mismatched" else "0=content-invariant .. larger=order-sensitive"
        print(f"  mean|delta|={md:.3f}  [{tag}]")
        print(f"  corr(real,{COND})={np.corrcoef(pr, pa)[0,1]:.3f}  (high = content ignored)")
        return

    def run(i_item):
        i, (seq, label, _, org) = i_item
        val, kind = ask(client, content_for(COND, seq, label, org, thresh, i, pos_pool, neg_pool))
        return label, val, kind
    out = list(ThreadPoolExecutor(WORKERS).map(run, list(enumerate(data))))
    y = np.array([o[0] for o in out]); p = np.array([o[1] for o in out]); kinds = [o[2] for o in out]
    mtag = MODEL.split("/")[-1].replace(".", "-")
    with open(f"hh_preds_{COND}_{mtag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "prob", "kind"])
        for o in out:
            w.writerow([o[0], o[1], o[2]])
    print(f"  parse: {dict(Counter(kinds))}  (fallback/error/empty -> 0.5 -> AUROC noise)")
    if len(set(y)) > 1:
        lo, hi = bootstrap_ci(y, p)
        print(f"LLM-output AUROC={roc_auc_score(y, p):.3f} [95% {lo:.3f},{hi:.3f}]  AUPRC={average_precision_score(y, p):.3f}  (cond={COND}; ESM ceiling ~0.70)")


if __name__ == "__main__":
    main()
