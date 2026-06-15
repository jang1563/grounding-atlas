"""AlphaGenome ceiling gate: does the AlphaGenome score track GTEx measured eQTL
effect size? (docs/ALPHAGENOME_CEILING_DESIGN.md)

For each GTEx eQTL (from gtex_eqtl_fetch.py), score the variant with AlphaGenome,
filter to the SAME eGene and the SAME tissue RNA_SEQ track, extract the signed
predicted effect, and correlate it against the measured GTEx `nes`. A high
correlation means the specialist ceiling reads real regulatory effect (the gate
passes); then the rung's LLM output / activation arms are worth building.

The match is gene- and tissue-specific on purpose: AlphaGenome scores every
nearby gene x every track, GTEx `nes` is one eGene in one tissue, so a pooled
score would be meaningless.

Run:
    set -a; source <your-keys-file>; set +a
    python signal/regulatory/eqtl_ceiling_gate.py            # all rows
    AG_VARIANTS=signal/regulatory/gtex_eqtl_Whole_Blood.csv AG_LIMIT=8 \
        python signal/regulatory/eqtl_ceiling_gate.py        # quick check
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = os.environ.get("AG_VARIANTS", os.path.join(HERE, "gtex_eqtl_Whole_Blood.csv"))
OUT = os.path.join(HERE, "eqtl_ceiling_gate.csv")
LIMIT = int(os.environ.get("AG_LIMIT", "0")) or None
H = 524288  # half of the 2^20 context


def main():
    key = os.environ.get("ALPHA_GENOME_API_KEY")
    if not key:
        sys.exit("set ALPHA_GENOME_API_KEY in the environment")
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    rows = list(csv.DictReader(open(VARIANTS)))
    if LIMIT:
        rows = rows[:LIMIT]
    model = dna_client.create(key)
    scorers = list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values())

    out = []
    for i, r in enumerate(rows, 1):
        pos = int(r["pos"])
        egene = r["gene"].split(".")[0]          # unversioned ENSG
        onto = r["ontology"]                      # UBERON:0013756 (whole blood)
        v = genome.Variant(chromosome=r["chrom"], position=pos,
                           reference_bases=r["ref"], alternate_bases=r["alt"])
        iv = genome.Interval(chromosome=r["chrom"], start=pos - H, end=pos + H)
        df = variant_scorers.tidy_scores(
            model.score_variant(interval=iv, variant=v, variant_scorers=scorers))
        # match GTEx-to-GTEx: the AlphaGenome GTEx whole-blood RNA track for this eGene
        m = df[(df["output_type"].astype(str) == "RNA_SEQ")
               & (df["gene_id"].astype(str) == egene)
               & (df["gene_type"].astype(str) == "protein_coding")
               & (df["gtex_tissue"].astype(str) == r["tissue"])]
        if not len(m):
            print(f"  [{i}/{len(rows)}] {r['gene_symbol']:8s} no matched track, skip")
            continue
        ag = float(m["raw_score"].mean())
        agq = float(m["quantile_score"].mean())
        out.append({"rsid": r["rsid"], "gene_symbol": r["gene_symbol"],
                    "nes": float(r["nes"]), "ag_raw": round(ag, 6),
                    "ag_quantile": round(agq, 6), "n_tracks": len(m)})
        print(f"  [{i}/{len(rows)}] {r['gene_symbol']:8s} nes={float(r['nes']):+.3f} "
              f"ag={ag:+.4f}  (n_tracks={len(m)})")

    if len(out) < 3:
        sys.exit(f"\nonly {len(out)} matched; cannot assess the gate")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    nes = [o["nes"] for o in out]
    ag = [o["ag_raw"] for o in out]
    try:
        from scipy.stats import spearmanr
        sr, sp = spearmanr(ag, nes)
        # direction is the meaningful gate (nes is normalized, ag is LFC, so magnitudes
        # are not on the same scale); restrict concordance to clear effects.
        clear = [(a, e) for a, e in zip(ag, nes) if abs(e) >= 0.1]
        conc = sum(1 for a, e in clear if (a > 0) == (e > 0)) / len(clear) if clear else float("nan")
        print(f"\nCEILING GATE (n={len(out)} matched GTEx whole-blood eQTLs):")
        print(f"  signed Spearman(ag_raw, nes) = {sr:+.3f} (p={sp:.3f})")
        print(f"  sign concordance (|nes|>=0.1) = {conc:.2f}  (n={len(clear)}, chance 0.50)")
        ok = sr > 0.3 and sp < 0.05 and conc >= 0.65
        print(f"  -> {'PASS: AlphaGenome reads eQTL direction' if ok else 'INCONCLUSIVE: weak/underpowered; use AlphaGenome documented eQTL benchmark + more variants'}")
    except ImportError:
        print("scipy not available; wrote scores to", OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
