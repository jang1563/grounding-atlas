"""Protein branch data prep: FLIP meltome -> balanced binary thermostability + cluster split.

Mirrors the SMILES branch's data setup (../../data/herg.csv) for proteins:
- Source: FLIP meltome mixed_split.fasta (Tm per sequence, 27,951 diverse proteins).
- Label: binarize Tm at the median -> 1 (thermostable) / 0, balanced by construction.
- Length filter: keep MINLEN..MAXLEN residues (compute budget + comparability).
- Leakage control: MMseqs2 sequence-identity clustering -> a cluster id per sequence;
  the activation arm does GroupKFold on clusters (the protein analog of the SMILES
  scaffold split). A single-protein DMS would collapse to one cluster, so this needs a
  sequence-diverse assay; meltome spans many species and families on purpose.

Output: data/protein_meltome.csv with columns id,sequence,label,tm,cluster.
Why thermostability: ESM2 reads it from the sequence (the structural ceiling), yet a
melting temperature essentially never appears as "sequence -> Tm" in web text, so it is
the cleanest test of the cross-domain hypothesis (protein encoding gap vs SMILES).

Env: PG_N (balanced total, default 1500), PG_MINLEN, PG_MAXLEN, PG_MINID (mmseqs
     --min-seq-id), PG_COV, MMSEQS_BIN, PG_FASTA_URL, PG_RAW, PG_OUT, PG_SEED.
"""
import csv
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np

URL = os.environ.get(
    "PG_FASTA_URL",
    "http://data.bioembeddings.com/public/FLIP/fasta/meltome/mixed_split.fasta",
)
N = int(os.environ.get("PG_N", "1500"))
MINLEN = int(os.environ.get("PG_MINLEN", "50"))
MAXLEN = int(os.environ.get("PG_MAXLEN", "512"))
MINID = float(os.environ.get("PG_MINID", "0.3"))
COV = float(os.environ.get("PG_COV", "0.8"))
MMSEQS = os.environ.get("MMSEQS_BIN", "mmseqs")
RAW = os.environ.get("PG_RAW", "meltome_mixed.fasta")
OUT = os.environ.get("PG_OUT", "protein_meltome.csv")
SEED = int(os.environ.get("PG_SEED", "42"))

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"using cached {path} ({os.path.getsize(path)} bytes)")
        return
    print(f"downloading {url}")
    subprocess.run(["curl", "-sL", "--max-time", "600", "-o", path, url], check=True)


def parse_fasta(path):
    """Yield (id, header, seq). Meltome header: >SeqID TARGET=<Tm> SET=.. VALIDATION=.."""
    sid, hdr, buf = None, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if sid is not None:
                    yield sid, hdr, "".join(buf)
                parts = line[1:].split(None, 1)
                sid = parts[0]
                hdr = parts[1] if len(parts) > 1 else ""
                buf = []
            else:
                buf.append(line.strip())
        if sid is not None:
            yield sid, hdr, "".join(buf)


def target_of(hdr):
    for tok in hdr.split():
        if tok.upper().startswith("TARGET="):
            try:
                return float(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def cluster(items, min_id, cov, mmseqs_bin):
    """items: list of (id, seq). Returns {id: cluster_rep_id} via MMseqs2 easy-cluster."""
    tmpd = tempfile.mkdtemp(prefix="pgclust_")
    fasta = os.path.join(tmpd, "in.fasta")
    with open(fasta, "w") as f:
        for sid, seq in items:
            f.write(f">{sid}\n{seq}\n")
    pref = os.path.join(tmpd, "clu")
    cmd = [
        mmseqs_bin, "easy-cluster", fasta, pref, os.path.join(tmpd, "tmp"),
        "--min-seq-id", str(min_id), "-c", str(cov), "--cov-mode", "0", "-v", "1",
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    rep_of = {}
    with open(pref + "_cluster.tsv") as f:
        for line in f:
            rep, mem = line.rstrip("\n").split("\t")
            rep_of[mem] = rep
    shutil.rmtree(tmpd, ignore_errors=True)
    return rep_of


def main():
    fetch(URL, RAW)

    rows = []
    for sid, hdr, seq in parse_fasta(RAW):
        tm = target_of(hdr)
        if tm is None:
            continue
        seq = seq.upper()
        if not (MINLEN <= len(seq) <= MAXLEN):
            continue
        if any(c not in VALID_AA for c in seq):
            continue  # drop X/B/Z/U/* and other non-standard residues
        rows.append((sid, seq, tm))

    tms = np.array([t for _, _, t in rows])
    thresh = float(np.median(tms))
    print(
        f"parsed {len(rows)} sequences (len {MINLEN}-{MAXLEN}, standard AA); "
        f"Tm median={thresh:.2f}C  range=[{tms.min():.1f},{tms.max():.1f}]",
        flush=True,
    )

    labeled = [(sid, seq, tm, 1 if tm > thresh else 0) for sid, seq, tm in rows]
    pos = [x for x in labeled if x[3] == 1]
    neg = [x for x in labeled if x[3] == 0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(N // 2, len(pos), len(neg))
    sample = pos[:k] + neg[:k]
    rng.shuffle(sample)
    print(f"balanced sample: {k} pos + {k} neg = {len(sample)}", flush=True)

    rep_of = cluster([(sid, seq) for sid, seq, _, _ in sample], MINID, COV, MMSEQS)
    reps = [rep_of[sid] for sid, _, _, _ in sample]
    csizes = Counter(reps)
    print(
        f"MMseqs2 clusters (min-seq-id={MINID}, cov={COV}): {len(csizes)} clusters; "
        f"largest={max(csizes.values())}; singletons={sum(1 for v in csizes.values() if v == 1)}",
        flush=True,
    )

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence", "label", "tm", "cluster"])
        for (sid, seq, tm, lab) in sample:
            w.writerow([sid, seq, lab, f"{tm:.4f}", rep_of[sid]])
    npos = sum(l for _, _, _, l in sample)
    print(
        f"wrote {OUT}: n={len(sample)} pos={npos} neg={len(sample)-npos} "
        f"clusters={len(csizes)} thresh={thresh:.2f}C",
        flush=True,
    )


if __name__ == "__main__":
    main()
