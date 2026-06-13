"""Variant branch data prep: ClinVar missense P/B -> UniProt seq -> AlphaMissense + temporal flag.

Mirrors the SMILES/protein branches' data setup, adapted to genetic variants. The output row
carries the SAME variant in three self-consistent forms so the dual-form output arm and the
ceiling read the identical variant:
  - text form:     {gene, HGVS protein change}            (web-rich symbolic surface)
  - sequence form: a wild-type protein window + the AA substitution, NO gene name (web-poor)
  - ceiling:       the AlphaMissense pathogenicity score   (specialist, ClinVar AUROC ~0.94)

Pipeline:
  1. UniProt human reviewed (SwissProt) FASTA -> {gene -> [accessions]} and {acc -> sequence}.
  2. ClinVar variant_summary (latest, pinned release): keep GRCh38 single-AA missense, germline
     classification in {Pathogenic, Likely pathogenic} (1) or {Benign, Likely benign} (0),
     review status star >= 1, single gene. Parse the p.HGVS three-letter change.
  3. Map gene -> UniProt sequence; QC: the UniProt canonical residue at the position MUST equal
     ClinVar's wild-type AA (drops isoform/coordinate mismatches, makes every kept variant
     self-consistent with AlphaMissense, which is also UniProt-canonical based).
  4. Temporal holdout: scan older dated ClinVar releases; a variant absent from the 2026-01
     snapshot was first added AFTER it (post Opus-class cutoff). first_seen bin from 2025-06,
     2026-01 membership. This is the single most important leakage control.
  5. AlphaMissense: stream the precomputed aa-substitution scores once, attach am_pathogenicity
     for the (uniprot, one-letter-sub) targets.
  6. Balance to N (label-balanced), write the main eval CSV, the full table, and a balanced
     post-2026-01 (strict temporal holdout) CSV.

Env: VG_RAW (dir of downloads), VG_N (balanced total, default 2000), VG_WIN (window radius,
     default 32 -> 65aa), VG_MINSTAR (default 1), VG_SEED (42), VG_OUT (dir).
No em dashes. Capability framing. See ../README.md, eval/README.md.
"""
import os
import re
import csv
import gzip
import sys
from collections import defaultdict, Counter

import numpy as np

RAW = os.environ.get("VG_RAW", os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
OUTDIR = os.environ.get("VG_OUT", os.path.join(os.path.dirname(__file__), "..", "data"))
N = int(os.environ.get("VG_N", "2000"))
WIN = int(os.environ.get("VG_WIN", "32"))
MINSTAR = int(os.environ.get("VG_MINSTAR", "1"))
SEED = int(os.environ.get("VG_SEED", "42"))

UNIPROT_FASTA = os.path.join(RAW, "uniprot_human_sprot.fasta.gz")
CLINVAR_LATEST = os.path.join(RAW, "clinvar_2026-06.txt.gz")   # pinned "current" release
SNAP_2026_01 = os.path.join(RAW, "clinvar_2026-01.txt.gz")     # Opus-class cutoff boundary
SNAP_2025_06 = os.path.join(RAW, "clinvar_2025-06.txt.gz")     # Sonnet-4.5-class cutoff boundary
ALPHAMISSENSE = os.path.join(RAW, "AlphaMissense_aa_substitutions.tsv.gz")

AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
PROT_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")

# ClinVar ReviewStatus -> star rating
STAR = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
}
PATHO = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN = {"benign", "likely benign", "benign/likely benign"}


def load_uniprot(path):
    """Return gene2accs {GENE: [acc,...]} and acc2seq {acc: sequence} from a SwissProt FASTA."""
    gene2accs, acc2seq = defaultdict(list), {}
    acc, gene, buf = None, None, []
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc is not None:
                    acc2seq[acc] = "".join(buf)
                    if gene:
                        gene2accs[gene].append(acc)
                buf = []
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 2 else None
                m = re.search(r"\bGN=(\S+)", line)
                gene = m.group(1) if m else None
            else:
                buf.append(line.strip())
        if acc is not None:
            acc2seq[acc] = "".join(buf)
            if gene:
                gene2accs[gene].append(acc)
    return gene2accs, acc2seq


def col_index(header_line):
    cols = header_line.rstrip("\n").split("\t")
    cols[0] = cols[0].lstrip("#")
    return {name: i for i, name in enumerate(cols)}


def parse_clinvar_latest(path):
    """Yield candidate dicts: GRCh38 single-AA missense, P/B, star>=MINSTAR, single gene."""
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        idx = col_index(f.readline())
        g = lambda r, k: r[idx[k]] if idx.get(k) is not None and idx[k] < len(r) else ""
        seen = set()
        for line in f:
            r = line.rstrip("\n").split("\t")
            if g(r, "Assembly") != "GRCh38":
                continue
            vid = g(r, "VariationID")
            if not vid or vid in seen:
                continue
            sig = g(r, "ClinicalSignificance").strip().lower()
            label = 1 if sig in PATHO else 0 if sig in BENIGN else None
            if label is None:
                continue
            rev = g(r, "ReviewStatus").strip().lower()
            star = STAR.get(rev, 1 if rev.startswith("criteria provided") else 0)
            if star < MINSTAR:
                continue
            gene = g(r, "GeneSymbol").strip()
            if not gene or gene == "-" or ";" in gene:
                continue
            m = PROT_RE.search(g(r, "Name"))
            if not m:
                continue
            wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
            if wt3 not in AA3TO1 or mut3 not in AA3TO1:
                continue  # excludes Ter/synonymous/non-standard
            wt1, mut1 = AA3TO1[wt3], AA3TO1[mut3]
            if wt1 == mut1:
                continue
            seen.add(vid)
            yield {
                "vid": vid, "gene": gene, "pos": pos, "wt": wt1, "mut": mut1,
                "hgvs_p": f"p.{wt3}{pos}{mut3}", "label": label, "stars": star,
                "rsid": g(r, "RS# (dbSNP)").strip(), "last_eval": g(r, "LastEvaluated").strip(),
            }


def snapshot_vids(path):
    """Set of all GRCh38 VariationIDs present in a dated ClinVar release (membership only)."""
    vids = set()
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        idx = col_index(f.readline())
        ia, iv = idx.get("Assembly"), idx.get("VariationID")
        for line in f:
            r = line.rstrip("\n").split("\t")
            if iv is not None and iv < len(r) and (ia is None or (ia < len(r) and r[ia] == "GRCh38")):
                vids.add(r[iv])
    return vids


def attach_alphamissense(cands, path):
    """Stream AlphaMissense_aa_substitutions.tsv.gz once; set c['am'] for matched (acc, sub)."""
    targets = {}  # (acc, sub1) -> list of cand refs
    for c in cands:
        if c.get("uniprot"):
            targets.setdefault((c["uniprot"], f"{c['wt']}{c['pos']}{c['mut']}"), []).append(c)
    if not os.path.exists(path):
        print(f"  [warn] AlphaMissense file missing ({path}); am column left empty", flush=True)
        return 0
    hit, scanned = 0, 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            key = (p[0], p[1])
            refs = targets.get(key)
            if refs is None:
                continue
            try:
                am = float(p[2])
            except ValueError:
                continue  # header row
            for c in refs:
                c["am"] = am
            hit += 1
            scanned += 1
    return hit


def balance(rows, n, seed, keyfn=lambda r: r["label"]):
    pos = [r for r in rows if keyfn(r) == 1]
    neg = [r for r in rows if keyfn(r) == 0]
    rng = np.random.RandomState(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out, k


FIELDS = ["id", "gene", "hgvs_p", "uniprot", "pos", "wt", "mut", "sub1", "label",
          "stars", "rsid", "first_seen", "post_cutoff", "am", "seq_len", "win_pos", "wt_window"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print(f"raw={RAW}  N={N}  win=+/-{WIN}  minstar={MINSTAR}", flush=True)

    print("loading UniProt human SwissProt FASTA ...", flush=True)
    gene2accs, acc2seq = load_uniprot(UNIPROT_FASTA)
    print(f"  {len(acc2seq)} proteins, {len(gene2accs)} gene symbols", flush=True)

    print("parsing ClinVar latest (missense P/B, star>=%d) ..." % MINSTAR, flush=True)
    cands = list(parse_clinvar_latest(CLINVAR_LATEST))
    print(f"  {len(cands)} candidate missense P/B variants (pre-mapping)", flush=True)

    # map to UniProt + WT-residue consistency QC
    mapped, drop_nogene, drop_nomatch = [], 0, 0
    for c in cands:
        accs = gene2accs.get(c["gene"], [])
        if not accs:
            drop_nogene += 1
            continue
        chosen = None
        for acc in accs:
            seq = acc2seq.get(acc, "")
            if len(seq) >= c["pos"] and seq[c["pos"] - 1] == c["wt"]:
                chosen = (acc, seq)
                break
        if chosen is None:
            drop_nomatch += 1
            continue
        acc, seq = chosen
        lo = max(0, c["pos"] - 1 - WIN)
        hi = min(len(seq), c["pos"] - 1 + WIN + 1)
        c["uniprot"] = acc
        c["seq_len"] = len(seq)
        c["wt_window"] = seq[lo:hi]
        c["win_pos"] = c["pos"] - lo  # 1-based index of the variant residue within the window
        c["sub1"] = f"{c['wt']}{c['pos']}{c['mut']}"
        mapped.append(c)
    print(f"  mapped {len(mapped)} (dropped: no UniProt gene {drop_nogene}, "
          f"WT-residue mismatch {drop_nomatch})", flush=True)

    # temporal holdout via release diff
    for snap, name in [(SNAP_2026_01, "in_2026_01"), (SNAP_2025_06, "in_2025_06")]:
        if os.path.exists(snap):
            print(f"scanning snapshot {os.path.basename(snap)} ...", flush=True)
            vids = snapshot_vids(snap)
            for c in mapped:
                c[name] = c["vid"] in vids
        else:
            print(f"  [warn] snapshot missing: {snap}", flush=True)
            for c in mapped:
                c[name] = None
    for c in mapped:
        if c.get("in_2026_01") is False:
            c["first_seen"], c["post_cutoff"] = "post_2026_01", 1
        elif c.get("in_2025_06") is False:
            c["first_seen"], c["post_cutoff"] = "2025H2", 0
        else:
            c["first_seen"], c["post_cutoff"] = "le_2025_06", 0

    # AlphaMissense ceiling scores
    print("attaching AlphaMissense scores (streaming aa_substitutions) ...", flush=True)
    hit = attach_alphamissense(mapped, ALPHAMISSENSE)
    with_am = sum(1 for c in mapped if "am" in c)
    print(f"  AlphaMissense matched {hit} (coverage {with_am}/{len(mapped)} = "
          f"{with_am/max(1,len(mapped)):.1%})", flush=True)

    # finalize id field
    for c in mapped:
        c["id"] = c["vid"]
        c.setdefault("am", "")

    # summaries
    by_label = Counter(c["label"] for c in mapped)
    by_star = Counter(c["stars"] for c in mapped)
    by_seen = Counter(c["first_seen"] for c in mapped)
    print(f"\nfull mapped set: n={len(mapped)}  P={by_label[1]} B={by_label[0]}", flush=True)
    print(f"  stars: {dict(sorted(by_star.items()))}", flush=True)
    print(f"  first_seen: {dict(by_seen)}", flush=True)
    pc = [c for c in mapped if c["post_cutoff"] == 1]
    print(f"  post-2026-01 (strict holdout): n={len(pc)}  "
          f"P={sum(c['label'] for c in pc)} B={sum(1-c['label'] for c in pc)}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    write_csv(os.path.join(OUTDIR, "variant_clinvar_full.csv"), mapped)

    main_sample, k = balance(mapped, N, SEED)
    write_csv(os.path.join(OUTDIR, "variant_clinvar.csv"), main_sample)
    print(f"\nwrote variant_clinvar.csv: n={len(main_sample)} ({k}/{k})", flush=True)

    post_sample, kp = balance(pc, N, SEED)
    if kp > 0:
        write_csv(os.path.join(OUTDIR, "variant_clinvar_post2026_01.csv"), post_sample)
        print(f"wrote variant_clinvar_post2026_01.csv: n={len(post_sample)} ({kp}/{kp})", flush=True)
    else:
        print("  [warn] no balanced post-2026-01 sample (too few of one class)", flush=True)


if __name__ == "__main__":
    main()
