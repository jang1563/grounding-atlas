"""Data-density vs naming-frequency probe for the data-density rung
(docs/DATA_DENSITY_RUNG_DESIGN.md).

The rung's precondition: the two web-exposures must dissociate, so we can later
tell whether an LLM's grounding of a cell state tracks functional-DATA density
D(c) (content) or literature NAMING frequency N(c) (recognition). This probe is
model-free: D(c) = public functional-genomics assay count (ENCODE Experiments),
N(c) = literature naming count (PubMed). It tests whether D and N dissociate and
whether high-N / low-D anchors (famous-by-name, data-poor disease states) exist.

D uses the closest ENCODE biosample term (disease states map to their base cell
type, where the functional data actually lives); N uses the specific state name
the model would recognize. The mismatch is the signal.

Run:  python signal/single_cell/data_density.py     # public APIs, no key, no model
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_density.csv")

# name | class | ENCODE biosample term (functional data) | PubMed query (the named state)
CELLS = [
    ("K562",               "line",    "K562",                "K562"),
    ("HepG2",              "line",    "HepG2",               "HepG2"),
    ("GM12878",            "line",    "GM12878",             "GM12878"),
    ("A549",               "line",    "A549",                "A549"),
    ("MCF-7",              "line",    "MCF-7",               "MCF-7"),
    ("hepatocyte",         "primary", "hepatocyte",          "hepatocyte"),
    ("astrocyte",          "primary", "astrocyte",           "astrocyte"),
    ("cardiac fibroblast", "primary", "cardiac fibroblast",  "cardiac fibroblast"),
    ("reactive astrocyte", "disease", "astrocyte",           "reactive astrocyte"),
    ("pancreatic beta cell", "disease", "endocrine pancreas", "pancreatic beta cell"),
    ("dopaminergic neuron", "disease", "dopaminergic neuron", "dopaminergic neuron"),
    ("microglia",          "disease", "microglial cell",     "microglia"),
    ("regulatory T cell",  "disease", "regulatory T cell",   "regulatory T cell"),
]


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def encode_count(term):
    url = ("https://www.encodeproject.org/search/?type=Experiment"
           f"&biosample_ontology.term_name={urllib.parse.quote(term)}&format=json&limit=0")
    try:
        return int(_get(url).get("total", 0))
    except Exception:
        return 0


def pubmed_count(query):
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed"
           f"&term={urllib.parse.quote(query)}&retmode=json")
    try:
        return int(_get(url)["esearchresult"]["count"])
    except Exception:
        return 0


def main():
    rows = []
    for name, cls, eterm, pq in CELLS:
        d = encode_count(eterm)
        n = pubmed_count(pq)
        rows.append({"cell": name, "class": cls, "encode_term": eterm,
                     "D_encode": d, "N_pubmed": n,
                     "N_over_D": round(n / (d + 1), 1)})
        print(f"  {name:22s} D={d:5d}  N={n:6d}  N/D={rows[-1]['N_over_D']:8.1f}  [{cls}]")
        time.sleep(0.4)  # NCBI politeness

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import math

    from scipy.stats import spearmanr
    logd = [math.log10(r["D_encode"] + 1) for r in rows]
    logn = [math.log10(r["N_pubmed"] + 1) for r in rows]
    rho, p = spearmanr(logd, logn)
    anchors = sorted(rows, key=lambda r: -r["N_over_D"])[:4]
    print(f"\nDISSOCIATION PROBE (n={len(rows)} cell states):")
    print(f"  Spearman(log D, log N) = {rho:+.2f} (p={p:.3f})")
    print("  top high-N / low-D anchors (named, data-poor; the testable cells):")
    for a in anchors:
        print(f"    {a['cell']:22s} N/D={a['N_over_D']:8.1f}  (D={a['D_encode']}, N={a['N_pubmed']})")
    dissociable = rho < 0.8 and anchors[0]["N_over_D"] > 5 * anchors[-1]["N_over_D"]
    print(f"  -> {'GO: D and N dissociate; high-N/low-D anchors exist' if dissociable else 'NO-GO: D and N too collinear / no anchors'}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
