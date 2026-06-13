"""anon-gate map: named vs anonymized content-signal, the SIGNAL side of the web-exposure law.

For each representation that has a named and an anonymized rendering of the SAME content, run
the verifiability gate on both and report Delta = named_best_cold - anon_best_cold. A small
Delta with both high = CONTENT (the signal is in the data pattern, anon-invariant; retrieve
closes it). A named that the cheap featurizer cannot even read (best_cold < 0.6) = surface-
blind: the signal is not in the text, it lives only in the LLM's pretrained web-memory of that
name (so the LLM-output arm can be high while the gate is at chance, e.g. PPI 0.95 vs 0.50).

Reuses featurizers + gate() from gate_multimodal. No em dashes.
"""
import csv
import os
import re

import numpy as np
from gate_multimodal import ROOT, feat_formula, feat_geneset, feat_kv, feat_text, gate, make_groups

FEAT = {"kv": feat_kv, "formula": feat_formula, "text": feat_text, "geneset": feat_geneset}


def anon_variant(t):
    s = re.sub(r"^[A-Za-z0-9_.]+\([^)]+\)\s*:\s*", "", t or "").strip()
    s = re.sub(r"^[NX][MR]_[0-9.]+\s*:\s*", "", s).strip()
    return s if s else (t or "")


# (name, file, kind, named_col, anon_col)  anon_col "__anon_variant__" = strip gene from text
PAIRS = [
    ("single_cell", "single_cell/pbmc_Tcell.csv", "geneset", "cell_sentence", "anon"),
    ("methyl_anchor", "methyl/methyl_anchor.csv", "kv", "text_gene", "text_anon"),
    ("glass", "generality/glass.csv", "formula", "formula", "anon"),
    ("metal", "materials/metal.csv", "formula", "formula", "anon"),
    ("variant", "clinvar/variant_text.csv", "text", "text", "__anon_variant__"),
    ("ppi", "ppi/ppi.csv", "text", "text_name", "text_anon"),
]


def run_one(vals, labels, kind):
    X, keep, grp = FEAT[kind](vals)
    keep = np.asarray(keep, dtype=bool)
    y = np.array([int(float(labels[i])) for i in range(len(labels)) if keep[i]])
    if len(set(y)) < 2 or len(y) < 40:
        return None
    groups = make_groups(X, None, len(y))
    g = gate(X, y, groups)
    g["best_cold"] = max(g["cold_auroc"], g["cold_auroc_rf"])
    return g


def verdict(nb, ab, d):
    if nb < 0.60:
        return "surface-blind (not in text; LLM web-memory only)"
    if ab >= 0.65 and abs(d) < 0.10:
        return "CONTENT (anon-invariant)"
    if d >= 0.10:
        return "web-anchor (name-dependent)"
    return "mixed"


def main():
    print(f"{'pair':15s} {'featurizer':10s}  named  anon   Delta  verdict", flush=True)
    print("-" * 80, flush=True)
    out = []
    for name, f, kind, ncol, acol in PAIRS:
        rows = list(csv.DictReader(open(os.path.join(ROOT, f))))
        if len(rows) > 4000:
            rng = np.random.RandomState(0)
            rows = [rows[i] for i in rng.choice(len(rows), 4000, replace=False)]
        labels = [r.get("label", "") for r in rows]
        nvals = [r.get(ncol, "") for r in rows]
        avals = ([anon_variant(r.get(ncol, "")) for r in rows] if acol == "__anon_variant__"
                 else [r.get(acol, "") for r in rows])
        gn = run_one(nvals, labels, kind)
        ga = run_one(avals, labels, kind)
        if gn is None or ga is None:
            print(f"{name:15s} SKIP", flush=True)
            continue
        nb, ab = gn["best_cold"], ga["best_cold"]
        d = round(nb - ab, 3)
        v = verdict(nb, ab, d)
        print(f"{name:15s} {kind:10s}  {nb:.3f} {ab:.3f} {d:+.3f}  {v}", flush=True)
        out.append((name, kind, nb, ab, d, v))
    print("-" * 80, flush=True)
    print("CONTENT = anon-invariant in-data-pattern (retrieve closes); surface-blind = signal "
          "only in LLM web-memory (anon collapses the OUTPUT arm).", flush=True)
    return out


if __name__ == "__main__":
    main()
