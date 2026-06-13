"""Protein branch, Claude comprehension probe (axis A, the UNDERSTAND half).

Does Claude parse a raw amino-acid sequence as structured domain language at all, or treat it
as "informational noise" (CoKE 2510.23127)? This is the recognition side of the original plan
(PROJECT_DESIGN axis A: name ~100% vs accession ~2%), brought to the protein SEQUENCE: not
"does it know the property" but "does it even read the residues".

We ask Claude three SURFACE questions whose answers are deterministic from the sequence, so
they are scored against ground truth (no labels needed):
  1. length (residue count)
  2. cysteine (C) count
  3. charged-residue percent (D/E/K/R)

The read is correlation, not exact match: LLMs are weak at exact counting, so the question is
whether Claude's estimate TRACKS the truth across sequences (it reads composition) or not (it
guesses from priors). Pairs with head_to_head_protein.py: comprehension here, utilization
there. The expected dissociation mirrors the SMILES O2 (validity-known-but-property-unknown):
Claude reads surface composition yet cannot read the deep property (Tm).

Env: HH_MODEL, PROBE_N, HH_WORKERS, HH_CSV. Keys: `set -a; source ~/.api_keys; set +a`.
"""
import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import anthropic

CSV = os.environ.get("HH_CSV", "protein_meltome.csv")
MODEL = os.environ.get("HH_MODEL", "claude-sonnet-4-5-20250929")
N = int(os.environ.get("PROBE_N", "120"))
WORKERS = int(os.environ.get("HH_WORKERS", "8"))

PROMPT = (
    "You are given a protein amino-acid sequence. Answer three questions about it with numbers "
    "only.\n"
    "1) Exactly how many amino-acid residues are in the sequence (its length)?\n"
    "2) Exactly how many cysteine (C) residues does it contain?\n"
    "3) What percentage of the residues are charged (D, E, K, or R)? A number from 0 to 100.\n"
    "Reply with EXACTLY three numbers separated by commas, in that order, nothing else.\n\n"
    "Sequence: {seq}"
)


def truth(seq):
    n = len(seq)
    return n, seq.count("C"), 100.0 * sum(seq.count(c) for c in "DEKR") / n


def load_sample(csv_path, n):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row["sequence"])
    rng = np.random.RandomState(42)
    rng.shuffle(rows)
    return rows[:n]


def ask(client, seq):
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=30,
            messages=[{"role": "user", "content": PROMPT.format(seq=seq)}],
        )
        txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    except Exception:
        return None
    nums = re.findall(r"\d*\.?\d+", txt)
    if len(nums) < 3:
        return None
    try:
        return float(nums[0]), float(nums[1]), float(nums[2])
    except ValueError:
        return None


def spearman(a, b):
    """Spearman rho = Pearson on ranks (no scipy needed; ties via argsort-of-argsort)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def report(name, pred, true, unit=""):
    pred, true = np.array(pred, float), np.array(true, float)
    pear = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 and pred.std() > 0 else float("nan")
    spr = spearman(pred, true)
    mae = float(np.abs(pred - true).mean())
    print(f"  {name:16s} pearson={pear:+.3f}  spearman={spr:+.3f}  MAE={mae:.2f}{unit}  (truth mean={true.mean():.1f}, pred mean={pred.mean():.1f})", flush=True)
    return pear, spr, mae


def main():
    seqs = load_sample(CSV, N)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com")
    print(f"MODEL={MODEL}  n={len(seqs)}  workers={WORKERS}  (comprehension: surface features, ground-truth scored)", flush=True)

    preds = list(ThreadPoolExecutor(WORKERS).map(lambda s: (ask(client, s), s), seqs))
    plen, pcys, pchg, tlen, tcys, tchg = [], [], [], [], [], []
    fail = 0
    for pred, seq in preds:
        if pred is None:
            fail += 1
            continue
        tl, tc, tg = truth(seq)
        plen.append(pred[0]); pcys.append(pred[1]); pchg.append(pred[2])
        tlen.append(tl); tcys.append(tc); tchg.append(tg)
    print(f"  answered={len(plen)}/{len(seqs)}  unparsed={fail}", flush=True)
    with open("comprehension_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_len", "true_len", "pred_cys", "true_cys", "pred_chg", "true_chg"])
        for i in range(len(plen)):
            w.writerow([plen[i], tlen[i], pcys[i], tcys[i], pchg[i], tchg[i]])
    print("does Claude read the sequence (pearson + spearman of its estimate vs the truth)?", flush=True)
    report("length", plen, tlen, " res")
    report("cysteine count", pcys, tcys)
    report("charged %", pchg, tchg, " pts")
    # exact-match rate for the two integer features (a stricter read)
    le = np.mean([abs(p - t) < 0.5 for p, t in zip(plen, tlen)])
    ce = np.mean([abs(p - t) < 0.5 for p, t in zip(pcys, tcys)])
    print(f"  exact-match: length={le:.1%}  cysteine={ce:.1%}  (LLMs are weak at exact counting; corr is the read)", flush=True)


if __name__ == "__main__":
    main()
