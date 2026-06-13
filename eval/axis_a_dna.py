"""Axis A (identity resolution), gene/DNA rung: within-entity recognition across notations.

The third modality for axis A (protein via FRT, chem via axis_a_chem.py, gene here), so
the recognition gap is measured on one instrument across three modalities. Uses TRUSTED
ClinVar data (variant_grounding/data/variant_clinvar_full.csv), no curation: each gene
carries a symbol (web-frequent), a UniProt accession (web-rare, the FRT accession analog),
a dbSNP rsID (web-rare), and a protein sequence window (web-rare). The web-exposure law
predicts recognition tracks notation web-frequency: symbol >> accession / rsID / sequence.

Popular genes (most ClinVar variants = most-studied = web-frequent symbols) are sampled so
the symbol ceiling is high and a low accession rate is a notation gap, not ignorance.
Deterministic scoring: a response is recognized if the correct gene symbol is present.

Run: python3 eval/axis_a_dna.py   Env: AXISA_MODEL, AXISA_N (default 40), AXISA_DRY.
No em dashes.
"""
import os
import re
import csv
import json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLINVAR = os.path.join(ROOT, "variant_grounding", "data", "variant_clinvar_full.csv")

PROMPTS = {
    "symbol":   "Do you recognize the human gene '{q}'? If so, name the protein it encodes or its function; if not, reply UNKNOWN.",
    "uniprot":  "Which human gene corresponds to UniProt accession {q}? Reply with the gene symbol only, or UNKNOWN.",
    "rsid":     "Which human gene contains the dbSNP variant {q}? Reply with the gene symbol only, or UNKNOWN.",
    "sequence": "Which human gene or protein is this amino acid sequence from? Reply with the gene symbol only, or UNKNOWN.\nSequence: {q}",
}


def load_popular(n):
    """Top-n genes by ClinVar variant count with a usable uniprot + rsid + window."""
    counts = Counter()
    rep = {}
    with open(CLINVAR) as fh:
        for row in csv.DictReader(fh):
            g = row["gene"]
            counts[g] += 1
            if g not in rep and row.get("uniprot") and row.get("rsid") and row.get("wt_window"):
                rep[g] = {"gene": g, "uniprot": row["uniprot"],
                          "rsid": "rs" + re.sub(r"^rs", "", row["rsid"]),
                          "sequence": row["wt_window"]}
    ranked = [g for g, _ in counts.most_common() if g in rep]
    return [rep[g] for g in ranked[:n]]


def recognized_symbol(resp, gene):
    """Correct gene symbol present as a word, and not an UNKNOWN-only reply."""
    t = resp.lower()
    has = re.search(rf"\b{re.escape(gene.lower())}\b", t) is not None
    if re.search(r"\bunknown\b", t) and not has:
        return False
    return has


def recognized_ceiling(resp):
    """Symbol condition: recognized if it engages (names a protein/function), not UNKNOWN."""
    return re.search(r"\bunknown\b", resp.lower()) is None and len(resp.strip()) > 0


def main():
    n = int(os.environ.get("AXISA_N", "40"))
    genes = load_popular(n)
    dry = os.environ.get("AXISA_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("AXISA_MODEL", "claude-sonnet-4-6")
    print(f"loaded {len(genes)} popular genes (top by ClinVar variant count)\n")

    if dry:
        ex = genes[0]
        print("example gene:", ex["gene"], "uniprot", ex["uniprot"], "rsid", ex["rsid"],
              "seqlen", len(ex["sequence"]))
        for cond, tmpl in PROMPTS.items():
            print(f"[{cond}] {tmpl.format(q=ex[cond] if cond != 'symbol' else ex['gene'])[:150]}")
        print("\nDRY: data + prompts only, no API.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    hits = defaultdict(int)
    per = []
    for g in genes:
        row = {"gene": g["gene"]}
        for cond, tmpl in PROMPTS.items():
            q = g["gene"] if cond == "symbol" else g[cond]
            msg = client.messages.create(model=model, max_tokens=60,
                messages=[{"role": "user", "content": tmpl.format(q=q)}])
            resp = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            ok = recognized_ceiling(resp) if cond == "symbol" else recognized_symbol(resp, g["gene"])
            hits[cond] += ok
            row[cond] = int(ok)
        per.append(row)
        print(f"  {g['gene']:12} symbol={row['symbol']} uniprot={row['uniprot']} rsid={row['rsid']} seq={row['sequence']}")
    N = len(genes)
    rates = {c: round(hits[c] / N, 3) for c in PROMPTS}
    out = {"n": N, "model": model, "recognition_rate": rates,
           "gap_symbol_minus_uniprot": round(rates["symbol"] - rates["uniprot"], 3),
           "gap_symbol_minus_sequence": round(rates["symbol"] - rates["sequence"], 3),
           "per_gene": per}
    with open(os.path.join(ROOT, "results", "axis_a_dna.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nrecognition rate (n={N}): " + "  ".join(f"{c}={rates[c]}" for c in PROMPTS))
    print(f"gap symbol-uniprot={out['gap_symbol_minus_uniprot']}  symbol-sequence={out['gap_symbol_minus_sequence']}")


if __name__ == "__main__":
    main()
