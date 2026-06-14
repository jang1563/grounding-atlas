"""AlphaGenome ceiling: score variants for the regulatory rung pre-check.

Pre-check for docs/ALPHAGENOME_CEILING_DESIGN.md. Given a CSV of variants
(chrom, pos, ref, alt, plus a `set` label = known_eqtl / novel and an optional
`rsid`), score each with AlphaGenome's recommended variant scorers and write a
per-variant scalar ceiling. The goal of the pre-check is to confirm the
web-rich (known eQTL) vs web-poor (novel) split is separable on the AlphaGenome
score BEFORE building the full encode-vs-verbalize rung.

This is the SPECIALIST ceiling arm only. The LLM output / activation arms are
separate (eval/). No Anthropic API and no GPU here.

Setup (one time):
    pip install -U alphagenome
    # get a free non-commercial key at https://deepmind.google.com/science/alphagenome
    export ALPHA_GENOME_API_KEY=...

Run:
    python signal/regulatory/alphagenome_score.py            # uses variants_demo.csv
    AG_VARIANTS=my_variants.csv python signal/regulatory/alphagenome_score.py

AlphaGenome is free for non-commercial use only (DeepMind terms); the scores are
predictions, not measured ground truth, so the rung frames them as a
predicted-structural label with a proxy-gold gap, not empirical.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = os.environ.get("AG_VARIANTS", os.path.join(HERE, "variants_demo.csv"))
OUT = os.path.join(HERE, "variant_scores.csv")
SEQ_LEN = 1_048_576  # AlphaGenome supported context (2^20 bp); interval = pos +/- half
HALF = SEQ_LEN // 2


def _need(msg):
    print(msg)
    sys.exit(1)


def main():
    api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    if not api_key:
        _need(
            "ALPHA_GENOME_API_KEY is not set.\n"
            "  1) pip install -U alphagenome\n"
            "  2) get a key: https://deepmind.google.com/science/alphagenome\n"
            "  3) export ALPHA_GENOME_API_KEY=..."
        )
    try:
        from alphagenome.data import genome
        from alphagenome.models import dna_client, variant_scorers
    except ImportError:
        _need("alphagenome is not installed. Run: pip install -U alphagenome")

    rows = list(csv.DictReader(open(VARIANTS)))
    if not rows:
        _need(f"no variants in {VARIANTS}")

    model = dna_client.create(api_key)
    scorers = list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values())

    out = []
    for i, r in enumerate(rows, 1):
        chrom = r["chrom"] if r["chrom"].startswith("chr") else "chr" + r["chrom"]
        pos = int(r["pos"])
        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=r["ref"],
            alternate_bases=r["alt"],
        )
        interval = genome.Interval(chromosome=chrom, start=pos - HALF, end=pos + HALF)
        scores = model.score_variant(
            interval=interval, variant=variant, variant_scorers=scorers
        )
        df = variant_scorers.tidy_scores(scores)
        # scalar ceiling summaries (downstream picks one)
        abs_raw = df["raw_score"].abs()
        abs_quant = df["quantile_score"].abs()
        rna = df[df["output_type"].astype(str).str.contains("RNA_SEQ", case=False, na=False)]
        out.append(
            {
                "id": r.get("rsid") or f"{chrom}:{pos}:{r['ref']}>{r['alt']}",
                "chrom": chrom,
                "pos": pos,
                "ref": r["ref"],
                "alt": r["alt"],
                "set": r.get("set", ""),
                "max_abs_raw": round(float(abs_raw.max()), 6),
                "max_abs_quantile": round(float(abs_quant.max()), 6),
                "max_abs_raw_rnaseq": round(float(rna["raw_score"].abs().max()), 6)
                if len(rna)
                else "",
                "n_tracks": len(df),
            }
        )
        print(f"  [{i}/{len(rows)}] {out[-1]['id']:30s} set={out[-1]['set']:10s} "
              f"max|raw|={out[-1]['max_abs_raw']}")

    fields = list(out[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out)} variant ceilings -> {OUT}")

    # pre-check: is the known_eqtl vs novel split separable on the score?
    by_set = {}
    for o in out:
        by_set.setdefault(o["set"], []).append(o["max_abs_raw"])
    if len(by_set) >= 2:
        print("\nPRE-CHECK (max_abs_raw by set; expect known_eqtl > novel if split is real):")
        for s, vals in by_set.items():
            print(f"  {s:12s} n={len(vals):3d}  mean={sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
