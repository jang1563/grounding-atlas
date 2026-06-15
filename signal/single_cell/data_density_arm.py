"""Data-density rung, LLM grounding arm (docs/DATA_DENSITY_RUNG_DESIGN.md).

First model-calling measurement: does a frontier model's grounding of a cell type
track functional-data density D(c) or literature naming N(c)? Restricted to
BIOLOGICAL cell types (immortalized lines have no canonical markers and dominate
ENCODE, breaking the comparison; see the design doc construct note). D(c) here is
the GEO single-cell dataset count (a biological-cell data proxy, not ENCODE).

Task: ask the model for a cell type's canonical markers; score recall against a
curated textbook marker set. Because markers are TEXTUAL, the prediction is that
recall tracks N (naming) more than D (functional data): the model grounds cell
identity by how often the type is named, not by how much functional data exists.
This is the cell-state form of the name-vs-content split; the functional-grounding
half (which should track D) is a heavier follow-up.

Curated markers are textbook-canonical (e.g. CD3 for T cells), not author opinion;
the curation is the honest limit (n=12, recall-only, alias misses).

Run:  set -a; source <your-keys-file>; set +a
      python signal/single_cell/data_density_arm.py
"""
import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_density_arm.csv")
MODEL = os.environ.get("DD_MODEL", "claude-opus-4-8")

# cell | PubMed query | GEO single-cell query | curated canonical markers
CELLS = [
    ("T cell",               "T cell",               "T cell single cell",               ["CD3D", "CD3E", "CD8A", "CD4", "IL7R"]),
    ("macrophage",           "macrophage",           "macrophage single cell",           ["CD68", "CD14", "LYZ", "CSF1R", "MRC1"]),
    ("hepatocyte",           "hepatocyte",           "hepatocyte single cell",           ["ALB", "APOA1", "TTR", "CYP3A4", "SERPINA1"]),
    ("microglia",            "microglia",            "microglia single cell",            ["P2RY12", "CX3CR1", "TMEM119", "AIF1", "C1QA"]),
    ("astrocyte",            "astrocyte",            "astrocyte single cell",            ["GFAP", "AQP4", "S100B", "SLC1A3", "ALDH1L1"]),
    ("pancreatic beta cell", "pancreatic beta cell", "pancreatic beta cell single cell", ["INS", "MAFA", "NKX6-1", "PDX1", "IAPP"]),
    ("cardiomyocyte",        "cardiomyocyte",        "cardiomyocyte single cell",        ["TNNT2", "MYH6", "NPPA", "ACTN2", "TNNI3"]),
    ("dopaminergic neuron",  "dopaminergic neuron",  "dopaminergic neuron single cell",  ["TH", "SLC6A3", "NR4A2", "DDC", "SLC18A2"]),
    ("regulatory T cell",    "regulatory T cell",    "regulatory T cell single cell",    ["FOXP3", "IL2RA", "CTLA4", "IKZF2"]),
    ("Paneth cell",          "Paneth cell",          "Paneth cell single cell",          ["LYZ", "DEFA5", "DEFA6", "REG3A"]),
    ("Tuft cell",            "Tuft cell",            "Tuft cell single cell",            ["POU2F3", "DCLK1", "TRPM5", "GFI1B"]),
    ("Kupffer cell",         "Kupffer cell",         "Kupffer cell single cell",         ["CLEC4F", "VSIG4", "CD5L", "TIMD4"]),
]

# Cell STATES (resting vs disease/activated): the salvage. Disease states are named a
# lot (clinical N) but functionally under-measured (low D), and their markers are less
# textbook-canonical, so the task may un-saturate and D may dissociate from N.
STATES = [
    ("naive CD8 T cell",             "naive CD8 T cell",             "naive CD8 T cell single cell",             ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"]),
    ("exhausted CD8 T cell",         "exhausted T cell",             "exhausted T cell single cell",             ["PDCD1", "HAVCR2", "LAG3", "TOX", "TIGIT"]),
    ("M1 macrophage",                "M1 macrophage",                "M1 macrophage single cell",                ["NOS2", "CD80", "IL1B", "TNF", "CXCL9"]),
    ("M2 macrophage",                "M2 macrophage",                "M2 macrophage single cell",                ["MRC1", "ARG1", "CD163", "IL10", "MSR1"]),
    ("homeostatic microglia",        "homeostatic microglia",        "homeostatic microglia single cell",        ["P2RY12", "TMEM119", "CX3CR1", "SALL1"]),
    ("disease-associated microglia", "disease associated microglia", "disease associated microglia single cell", ["TREM2", "APOE", "CST7", "LPL", "ITGAX"]),
    ("reactive astrocyte",           "reactive astrocyte",           "reactive astrocyte single cell",           ["GFAP", "VIM", "C3", "SERPINA3N", "LCN2"]),
    ("cancer-associated fibroblast", "cancer associated fibroblast", "cancer associated fibroblast single cell", ["FAP", "ACTA2", "PDPN", "COL1A1", "PDGFRB"]),
    ("senescent cell",               "cellular senescence",          "senescent cell single cell",               ["CDKN1A", "CDKN2A", "GLB1", "SERPINE1", "LMNB1"]),
    ("plasma cell",                  "plasma cell",                  "plasma cell single cell",                  ["MZB1", "XBP1", "SDC1", "PRDM1", "JCHAIN"]),
    ("tissue-resident memory T cell", "tissue resident memory T cell", "tissue resident memory T cell single cell", ["ITGAE", "CD69", "CXCR6", "ITGA1"]),
]


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def pubmed(q):
    try:
        return int(_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed"
                        f"&term={urllib.parse.quote(q)}&retmode=json")["esearchresult"]["count"])
    except Exception:
        return 0


def geo(q):
    try:
        return int(_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds"
                        f"&term={urllib.parse.quote(q)}&retmode=json")["esearchresult"]["count"])
    except Exception:
        return 0


def ask_markers(client, cell):
    msg = client.messages.create(
        model=MODEL, max_tokens=120,
        messages=[{"role": "user", "content":
                   f"List the top canonical marker genes used to identify {cell} in single-cell "
                   "RNA-seq. Reply with official human gene symbols only, comma-separated, nothing else."}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    toks = {t.strip().upper() for t in text.replace("\n", ",").split(",") if t.strip()}
    return {t for t in toks if t.replace("-", "").isalnum() and len(t) <= 12}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY (source <your-keys-file>)")
    import anthropic
    client = anthropic.Anthropic()

    is_states = os.environ.get("DD_SET") == "states"
    cells = STATES if is_states else CELLS
    out_path = os.path.join(HERE, f"data_density_arm_{'states' if is_states else 'types'}.csv")
    rows = []
    for cell, pq, gq, markers in cells:
        n = pubmed(pq); time.sleep(0.35)
        d = geo(gq); time.sleep(0.35)
        pred = ask_markers(client, cell)
        truth = set(markers)
        recall = len(pred & truth) / len(truth)
        rows.append({"cell": cell, "N_pubmed": n, "D_geo": d,
                     "recall": round(recall, 3), "hit": len(pred & truth), "n_markers": len(truth)})
        print(f"  {cell:22s} N={n:6d} D={d:5d} recall={recall:.2f} ({len(pred & truth)}/{len(truth)})")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from scipy.stats import spearmanr
    rec = [r["recall"] for r in rows]
    logn = [math.log10(r["N_pubmed"] + 1) for r in rows]
    logd = [math.log10(r["D_geo"] + 1) for r in rows]
    rn, pn = spearmanr(rec, logn)
    rd, pd = spearmanr(rec, logd)
    rnd, _ = spearmanr(logn, logd)
    print(f"\nGROUNDING ARM (n={len(rows)} biological cell types, model={MODEL}):")
    print(f"  Spearman(recall, log N_pubmed) = {rn:+.2f} (p={pn:.3f})")
    print(f"  Spearman(recall, log D_geo)    = {rd:+.2f} (p={pd:.3f})")
    print(f"  Spearman(log N, log D)         = {rnd:+.2f}  (covariate collinearity)")
    ceiling = sum(1 for r in rows if r["recall"] >= 0.999) >= 0.8 * len(rows)
    flags = []
    if ceiling:
        flags.append(f"ceiling effect (recall~1.0 for {sum(1 for r in rows if r['recall'] >= 0.999)}/{len(rows)}; task too easy at frontier)")
    if abs(rnd) > 0.7:
        flags.append(f"covariates collinear (Spearman log N,log D = {rnd:+.2f}); D and N inseparable among commensurable cell types")
    print(f"  -> {'INVALID: ' + '; '.join(flags) if flags else 'grounding loads on ' + ('N>D' if abs(rn) > abs(rd) else 'D>N')}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
