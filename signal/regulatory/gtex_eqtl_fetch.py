"""Fetch GTEx single-tissue eQTLs as the measured ground truth for the AlphaGenome
ceiling gate (docs/ALPHAGENOME_CEILING_DESIGN.md).

The ceiling gate asks: does the AlphaGenome score track a MEASURED regulatory
effect? GTEx eQTL `nes` (normalized effect size of the ALT allele on expression)
is that independent ground truth, so we correlate AlphaGenome's predicted effect
against `nes`, no constructed negatives, no circularity.

This pulls eGenes for one tissue, then variant-level eQTLs per gene, samples a
spread of |nes|, and writes chrom,pos,ref,alt,rsid,gene,nes for scoring. GTEx is
open access; no model calls here.

Run:
    python signal/regulatory/gtex_eqtl_fetch.py
    GTEX_TISSUE=Liver GTEX_N=40 python signal/regulatory/gtex_eqtl_fetch.py
"""
import csv
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://gtexportal.org/api/v2"
TISSUE = os.environ.get("GTEX_TISSUE", "Whole_Blood")
DATASET = os.environ.get("GTEX_DATASET", "gtex_v8")
N_GENES = int(os.environ.get("GTEX_GENES", "90"))
N_TARGET = int(os.environ.get("GTEX_N", "50"))
MAX_PER_GENE = int(os.environ.get("GTEX_MAX_PER_GENE", "3"))
OUT = os.path.join(HERE, f"gtex_eqtl_{TISSUE}.csv")


def get(path, **params):
    params.update(datasetId=DATASET, itemsPerPage=params.get("itemsPerPage", 250))
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("data", [])


def main():
    egenes = get("association/egene", tissueSiteDetailId=TISSUE)[:N_GENES]
    print(f"{len(egenes)} eGenes for {TISSUE}")
    rows = []
    for g in egenes:
        try:
            eqtls = get("association/singleTissueEqtl",
                        gencodeId=g["gencodeId"], tissueSiteDetailId=TISSUE)
        except Exception as e:
            print("  skip", g.get("geneSymbol"), e)
            continue
        for q in eqtls:
            vid = q["variantId"]                       # chr1_64764_C_T_b38
            parts = vid.split("_")
            if len(parts) < 4:
                continue
            chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
            if len(ref) != 1 or len(alt) != 1:         # SNVs only for the pre-check
                continue
            rows.append({
                "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                "rsid": q.get("snpId", ""), "gene": q.get("gencodeId", ""),
                "gene_symbol": q.get("geneSymbol", ""),
                "nes": float(q["nes"]), "pval": float(q.get("pValue", "nan")),
                "ontology": q.get("ontologyId", ""), "tissue": TISSUE, "set": "web_rich",
            })
    # dedup by variant, then sample a spread across |nes|
    seen, uniq = set(), []
    for r in rows:
        k = (r["chrom"], r["pos"], r["ref"], r["alt"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    # cap per gene (one low-expression gene can otherwise dominate the |nes| tail)
    per_gene, capped = {}, []
    for r in sorted(uniq, key=lambda r: abs(r["nes"])):
        per_gene[r["gene"]] = per_gene.get(r["gene"], 0) + 1
        if per_gene[r["gene"]] <= MAX_PER_GENE:
            capped.append(r)
    uniq = capped
    uniq.sort(key=lambda r: abs(r["nes"]))
    if len(uniq) > N_TARGET:
        step = len(uniq) / N_TARGET
        uniq = [uniq[int(i * step)] for i in range(N_TARGET)]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader(); w.writerows(uniq)
    nes = [r["nes"] for r in uniq]
    print(f"wrote {len(uniq)} eQTLs -> {OUT}")
    print(f"|nes| range: {min(abs(x) for x in nes):.3f} to {max(abs(x) for x in nes):.3f}")


if __name__ == "__main__":
    main()
