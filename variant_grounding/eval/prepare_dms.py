"""DMS parallel track: ProteinGym deep-mutational-scan fitness -> the leakage-free comparison.

The central leakage control (`../README.md` #3). ClinVar pathogenicity labels are on the web and
in training data, so a high text-form output number can be memorized recall. A DMS experimental
fitness label is NOT a memorizable clinical tag: it is a continuous assay readout. If the
text-form output arm is high on ClinVar but collapses to chance on DMS for the SAME clinically
famous genes (BRCA1, TP53, PTEN, MSH2), that ClinVar-minus-DMS gap IS the memorization estimate.

Produces the SAME CSV schema as prepare_data.py (so output_arm_variant.py runs unchanged):
  - text form:     {gene, HGVS}            (gene from the ProteinGym UniProt entry name)
  - sequence form: a WT window + the substitution, NO gene name
  - label:         1 = damaging (low fitness), 0 = tolerated. ProteinGym DMS_score_bin = 1 is
                   functional/fit, so damaging = 1 - DMS_score_bin. Orientation is sanity-checked
                   against AlphaMissense (higher = more pathogenic) per dataset and logged.
  - am:            AlphaMissense score (the DMS ceiling; entry name -> accession via the FASTA).

Env: VG_RAW, VG_DMS_IDS (comma list), VG_WIN, VG_PERDS (cap singles per dataset), VG_OUT, VG_SEED.
"""
import csv
import gzip
import os
import re
from collections import defaultdict

import numpy as np

RAW = os.environ.get("VG_RAW", os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
OUTDIR = os.environ.get("VG_OUT", os.path.join(os.path.dirname(__file__), "..", "data"))
WIN = int(os.environ.get("VG_WIN", "32"))
PERDS = int(os.environ.get("VG_PERDS", "400"))   # balanced cap per dataset
SEED = int(os.environ.get("VG_SEED", "42"))
DMS_IDS = os.environ.get("VG_DMS_IDS",
    "BRCA1_HUMAN_Findlay_2018,PTEN_HUMAN_Mighell_2018,P53_HUMAN_Kotler_2018,MSH2_HUMAN_Jia_2020").split(",")

REF = os.path.join(RAW, "DMS_substitutions_reference.csv")
DMS_DIR = os.path.join(RAW, "dms")
UNIPROT_FASTA = os.path.join(RAW, "uniprot_human_sprot.fasta.gz")
ALPHAMISSENSE = os.path.join(RAW, "AlphaMissense_aa_substitutions.tsv.gz")

AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
       "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
       "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
FIELDS = ["id", "gene", "hgvs_p", "uniprot", "pos", "wt", "mut", "sub1", "label",
          "stars", "rsid", "first_seen", "post_cutoff", "am", "seq_len", "win_pos", "wt_window"]


def entry_to_acc(path):
    """Map UniProt entry name (BRCA1_HUMAN) -> accession (P38398) from FASTA headers."""
    m = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split("|")
                if len(parts) > 2:
                    acc = parts[1]
                    entry = parts[2].split()[0]
                    m[entry] = acc
    return m


def load_ref():
    info = {}
    for r in csv.DictReader(open(REF)):
        info[r["DMS_id"]] = {"entry": r["UniProt_ID"], "seq": r["target_seq"]}
    return info


def parse_dms(dms_id, info, e2acc):
    meta = info[dms_id]
    seq, entry = meta["seq"], meta["entry"]
    gene = entry.split("_")[0]
    acc = e2acc.get(entry, "")
    path = os.path.join(DMS_DIR, f"{dms_id}.csv")
    rows, skipped = [], 0
    for r in csv.DictReader(open(path)):
        mut = r["mutant"]
        if ":" in mut:  # multi-mutant
            continue
        m = MUT_RE.match(mut)
        if not m:
            continue
        wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        if wt == alt or wt not in AA3 or alt not in AA3:
            continue
        if pos > len(seq) or seq[pos - 1] != wt:
            skipped += 1
            continue
        try:
            bin_ = int(float(r["DMS_score_bin"]))
            dms = float(r["DMS_score"])
        except (ValueError, KeyError):
            continue
        lo = max(0, pos - 1 - WIN)
        hi = min(len(seq), pos - 1 + WIN + 1)
        rows.append({
            "id": f"{dms_id}:{mut}", "gene": gene, "hgvs_p": f"p.{AA3[wt]}{pos}{AA3[alt]}",
            "uniprot": acc, "pos": pos, "wt": wt, "mut": alt, "sub1": f"{wt}{pos}{alt}",
            "label": 1 - bin_, "stars": 2, "rsid": "", "first_seen": "dms", "post_cutoff": 0,
            "am": "", "seq_len": len(seq), "win_pos": pos - lo, "wt_window": seq[lo:hi],
            "_dms": dms,
        })
    return rows, gene, acc, skipped


def attach_alphamissense(allrows, path):
    targets = defaultdict(list)
    for c in allrows:
        if c.get("uniprot"):
            targets[(c["uniprot"], c["sub1"])].append(c)
    if not (targets and os.path.exists(path)):
        return 0
    hit = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            refs = targets.get((p[0], p[1]))
            if refs is None:
                continue
            try:
                am = float(p[2])
            except ValueError:
                continue
            for c in refs:
                c["am"] = am
            hit += 1
    return hit


def balance(rows, n, seed):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng = np.random.RandomState(seed)
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out, k


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print(f"raw={RAW}  datasets={DMS_IDS}  win=+/-{WIN}  per_ds_cap={PERDS}", flush=True)
    info = load_ref()
    e2acc = entry_to_acc(UNIPROT_FASTA)

    all_balanced = []
    per_ds_rows = {}
    for dms_id in DMS_IDS:
        if dms_id not in info:
            print(f"  [warn] {dms_id} not in reference; skipping", flush=True)
            continue
        rows, gene, acc, skipped = parse_dms(dms_id, info, e2acc)
        per_ds_rows[dms_id] = rows
        npos = sum(r["label"] for r in rows)
        print(f"  {dms_id}: gene={gene} acc={acc or 'NA'}  singles={len(rows)} "
              f"(damaging={npos}, tolerated={len(rows)-npos}, wt-mismatch-skip={skipped})", flush=True)

    flat = [r for rows in per_ds_rows.values() for r in rows]
    print("attaching AlphaMissense (DMS ceiling) ...", flush=True)
    hit = attach_alphamissense(flat, ALPHAMISSENSE)
    with_am = sum(1 for r in flat if r["am"] != "")
    print(f"  AlphaMissense matched {hit}; coverage {with_am}/{len(flat)}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    for dms_id, rows in per_ds_rows.items():
        # orientation sanity check vs AlphaMissense (am higher => more pathogenic => damaging=1)
        amrows = [r for r in rows if r["am"] != ""]
        if len(amrows) > 20:
            from sklearn.metrics import roc_auc_score
            y = [r["label"] for r in amrows]
            if len(set(y)) > 1:
                a = roc_auc_score(y, [float(r["am"]) for r in amrows])
                flag = "OK" if a >= 0.5 else "CHECK direction"
                print(f"  {dms_id}: AlphaMissense vs damaging-label AUROC={a:.3f} ({flag})", flush=True)
        bal, k = balance(rows, PERDS, SEED)
        write_csv(os.path.join(OUTDIR, f"variant_dms_{dms_id}.csv"), bal)
        all_balanced.extend(bal)
        print(f"  wrote variant_dms_{dms_id}.csv: n={len(bal)} ({k}/{k})", flush=True)

    pooled, kp = balance(all_balanced, len(all_balanced), SEED)  # already balanced per ds; reshuffle
    write_csv(os.path.join(OUTDIR, "variant_dms.csv"), all_balanced)
    np_ = sum(r["label"] for r in all_balanced)
    print(f"\nwrote variant_dms.csv (pooled): n={len(all_balanced)} "
          f"damaging={np_} tolerated={len(all_balanced)-np_}", flush=True)


if __name__ == "__main__":
    main()
