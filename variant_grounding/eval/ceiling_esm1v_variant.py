"""Secondary ceiling: ESM-1v wild-type-marginal LLR (single forward pass, no MSA).

The methodologically purest specialist and the closest analog to the hERG Morgan-fingerprint
probe: a protein language model scoring a missense substitution from sequence alone, no MSA, no
supervision, one forward pass. ESM-1v (Meier et al, NeurIPS 2021) wild-type-marginal score:
    LLR = log p(mut | WT context) - log p(wt | WT context)   at the variant position,
read off the unmasked WT-sequence logits. A more negative LLR = more deleterious. The
pathogenicity score is -LLR; ceiling = AUROC(label, -LLR), stratified by stars and temporal bin
exactly like the AlphaMissense ceiling. Unlike AlphaMissense (which saw ClinVar-adjacent
supervision), ESM-1v is fully unsupervised, so its temporal-holdout AUROC is the cleanest
specialist baseline against which the LLM output collapse is read.

Runs on GPU (Cayuga a40); ESM-1v 650M. Long proteins are windowed to <=1022 residues centered
on the variant so the position stays in context. Mirrors ../../protein_grounding ceiling style.

Env: VG_CSV, ESM1V_MODEL, VG_RAW (for the FASTA), VG_MAXLEN, VG_BATCH.
"""
import csv
import gzip
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

CSV = os.environ.get("VG_CSV", os.path.join(os.path.dirname(__file__), "..", "data", "variant_clinvar.csv"))
RAW = os.environ.get("VG_RAW", os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
MODEL = os.environ.get("ESM1V_MODEL", "facebook/esm1v_t33_650M_UR90S_1")
MAXLEN = int(os.environ.get("VG_MAXLEN", "1022"))
BATCH = int(os.environ.get("VG_BATCH", "8"))
FASTA = os.path.join(RAW, "uniprot_human_sprot.fasta.gz")


def load_seqs(path):
    acc2seq, acc, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    acc2seq[acc] = "".join(buf)
                buf = []
                p = line[1:].split("|")
                acc = p[1] if len(p) > 2 else None
            else:
                buf.append(line.strip())
        if acc:
            acc2seq[acc] = "".join(buf)
    return acc2seq


def window(seq, pos, maxlen):
    """Return (subseq, local_pos_1based) with the variant residue centered, length <= maxlen."""
    if len(seq) <= maxlen:
        return seq, pos
    half = maxlen // 2
    lo = max(0, min(pos - 1 - half, len(seq) - maxlen))
    return seq[lo:lo + maxlen], pos - lo


def main():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    acc2seq = load_seqs(FASTA)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"csv={os.path.basename(CSV)} n={len(rows)} model={MODEL} device={device}", flush=True)

    y, score, stars, first_seen, post = [], [], [], [], []
    skipped = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        subs, locs, keep = [], [], []
        for r in batch:
            seq = acc2seq.get(r["uniprot"], "")
            pos = int(r["pos"])
            if not seq or pos > len(seq) or seq[pos - 1] != r["wt"]:
                skipped += 1
                continue
            sub, lp = window(seq, pos, MAXLEN)
            subs.append(sub)
            locs.append(lp)
            keep.append(r)
        if not subs:
            continue
        enc = tok(subs, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # (B, L, vocab)
        logp = torch.log_softmax(logits.float(), dim=-1)
        for j, r in enumerate(keep):
            ti = locs[j]  # cls at index 0 -> residue at 1-based local pos == token index
            wt_id = tok.convert_tokens_to_ids(r["wt"])
            mut_id = tok.convert_tokens_to_ids(r["mut"])
            llr = (logp[j, ti, mut_id] - logp[j, ti, wt_id]).item()
            y.append(int(r["label"]))
            score.append(-llr)  # pathogenicity = -LLR
            stars.append(int(r["stars"]))
            first_seen.append(r["first_seen"])
            post.append(int(r["post_cutoff"]))
        if (i + BATCH) % 200 == 0:
            print(f"  scored {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)

    y = np.array(y); score = np.array(score)
    stars = np.array(stars); first_seen = np.array(first_seen); post = np.array(post)
    print(f"scored {len(y)} (skipped {skipped} on seq/WT mismatch)", flush=True)

    def row(name, mask):
        ys, ss = y[mask], score[mask]
        if len(ys) == 0 or len(set(ys)) < 2:
            print(f"  {name:24s} n={int(mask.sum()):5d}  (insufficient)", flush=True)
            return
        print(f"  {name:24s} n={int(mask.sum()):5d}  AUROC={roc_auc_score(ys, ss):.3f}  "
              f"AUPRC={average_precision_score(ys, ss):.3f}", flush=True)

    print("\nESM-1v WT-marginal ceiling (AUROC of -LLR):", flush=True)
    row("ALL", np.ones(len(y), bool))
    row("star1", stars == 1)
    row("star2+", stars >= 2)
    for b in ["le_2025_06", "2025H2", "post_2026_01"]:
        row(b, first_seen == b)
    row("post_2026_01 (strict)", post == 1)


if __name__ == "__main__":
    main()
