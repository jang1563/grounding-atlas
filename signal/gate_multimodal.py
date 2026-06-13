"""WS2 verifiability gate, generalized from ADMET-only to all signal modalities.

Reuses the gate(X, y, groups) logic from generate_signal.py unchanged in spirit:
  content-feature probe under a leakage-controlled (cold) split
  + a shuffled-label selectivity control (Hewitt-Liang)
  PASS = cold AUROC >= 0.65 AND selectivity >= 0.10.

The only modality-specific piece is the featurizer (representation -> X) and the
leakage-control grouper. Natural group columns are used where present (protein cluster,
MSA family, variant gene); otherwise the cold split is a feature-space KMeans cluster,
which is the modality-general analog of the Murcko-scaffold split (separate near-duplicate
representations so the probe cannot win by analog leakage).

This is CONSOLIDATION, not a new measurement: it certifies which of the 16 representations
already carry a content-readable, leakage-controlled signal under a CHEAP local featurizer
(Morgan / k-mer / char-ngram / composition / pixel-stat), i.e. without the SFM embedding the
3-arm sweep used. A PASS = the signal is real even under a cheap encoder. A FAIL = "no
measurable signal under THIS cheap encoder", not "no signal" (an SFM featurizer may still
find it; flagged per row). No em dashes.

Usage: python gate_multimodal.py
"""
import csv
import json
import os
import re
from collections import Counter
from itertools import product

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

ROOT = os.path.dirname(os.path.abspath(__file__))
GATE_COLD_AUROC = 0.65
GATE_SELECTIVITY = 0.10
CAP = 4000
AA = "ACDEFGHIKLMNPQRSTVWY"


# ----------------------------- featurizers ------------------------------------------
def feat_smiles(vals):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, keep, grp = [], [], []
    for s in vals:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            keep.append(False); continue
        X.append(gen.GetFingerprintAsNumPy(m)); keep.append(True)
        try:
            grp.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) or s)
        except Exception:
            grp.append(s)
    return np.asarray(X, dtype=float), np.array(keep), np.array(grp)


def feat_kmer(vals, alpha="ACGT", k=4):
    kmers = ["".join(p) for p in product(alpha, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    aset = set(alpha)
    X, keep = [], []
    for s in vals:
        s = (s or "").upper().replace("U", "T")
        if len(s) < k:
            keep.append(False); continue
        v = np.zeros(len(kmers))
        tot = 0
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            if set(km) <= aset:
                v[idx[km]] += 1; tot += 1
        if tot == 0:
            keep.append(False); continue
        X.append(v / tot); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_protein(vals, k=2):
    kmers = ["".join(p) for p in product(AA, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    aset = set(AA)
    X, keep = [], []
    for s in vals:
        s = (s or "").upper()
        if len(s) < k:
            keep.append(False); continue
        v = np.zeros(len(kmers))
        tot = 0
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            if set(km) <= aset:
                v[idx[km]] += 1; tot += 1
        # add single-aa composition too
        comp = np.array([s.count(a) for a in AA], dtype=float)
        comp = comp / (comp.sum() or 1)
        if tot == 0:
            keep.append(False); continue
        X.append(np.concatenate([v / tot, comp])); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_msa(rows, col):
    # AA composition of the alignment column ONLY (depth/gap/n_distinct are excluded as
    # potential label-leaks; conservation must be read from the residue distribution).
    X, keep = [], []
    aset = set(AA)
    for r in rows:
        toks = [t for t in (r[col] or "").split() if t in aset]
        if not toks:
            keep.append(False); continue
        c = Counter(toks)
        tot = sum(c.values())
        v = np.array([c.get(a, 0) / tot for a in AA])
        # add shannon entropy + max-freq as content-derived summaries
        p = v[v > 0]
        ent = float(-(p * np.log(p)).sum())
        X.append(np.concatenate([v, [ent, float(v.max())]])); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_kv(vals):
    # "cgXXXX:0.078 cgYYYY:0.103 ..." -> aligned numeric vector over the shared key set
    parsed = []
    for t in vals:
        d = {}
        for kv in (t or "").split():
            if ":" in kv:
                k, _, v = kv.partition(":")
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
        parsed.append(d)
    keys = sorted(set().union(*[set(p) for p in parsed])) if parsed else []
    idx = {k: i for i, k in enumerate(keys)}
    X, keep = [], []
    for p in parsed:
        if not p:
            keep.append(False); continue
        v = np.zeros(len(keys))
        for k, val in p.items():
            v[idx[k]] = val
        X.append(v); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_floats(vals):
    arrs, keep = [], []
    for t in vals:
        try:
            a = np.array([float(x) for x in (t or "").split()])
        except ValueError:
            keep.append(False); continue
        if a.size == 0:
            keep.append(False); continue
        arrs.append(a); keep.append(True)
    L = min(len(a) for a in arrs)
    X = np.array([a[:L] for a in arrs])
    return X, np.array(keep), None


def feat_nmr(vals, bins=40, lo=0.0, hi=220.0):
    X, keep = [], []
    for t in vals:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", t or "")]
        if not nums:
            keep.append(False); continue
        h, _ = np.histogram(nums, bins=bins, range=(lo, hi))
        s = h.sum() or 1
        X.append(h / s); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_text(vals):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=3000)
    M = v.fit_transform([t or "" for t in vals]).toarray()
    keep = np.array([bool((t or "").strip()) for t in vals])
    return M[keep], keep, None


def feat_geneset(vals):
    v = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_features=3000)
    M = v.fit_transform([t or "" for t in vals]).toarray()
    keep = np.array([bool((t or "").strip()) for t in vals])
    return M[keep], keep, None


def feat_formula(vals):
    parsed = []
    for t in vals:
        d = {}
        for m in re.finditer(r"([A-Za-z_]+[0-9]*)\s*:\s*([\d.]+)", t or ""):
            try:
                d[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        parsed.append(d)
    els = sorted(set().union(*[set(p) for p in parsed])) if parsed else []
    idx = {e: i for i, e in enumerate(els)}
    X, keep = [], []
    for p in parsed:
        if not p:
            keep.append(False); continue
        v = np.zeros(len(els))
        tot = sum(p.values()) or 1
        for e, val in p.items():
            v[idx[e]] = val / tot
        X.append(v); keep.append(True)
    return np.asarray(X), np.array(keep), None


def feat_image(vals):
    from PIL import Image
    X, keep = [], []
    for p in vals:
        fp = p if os.path.isabs(p) else os.path.join(os.path.dirname(ROOT), p)
        if not os.path.exists(fp):
            fp = os.path.join(ROOT, os.path.relpath(p, "signal")) if p.startswith("signal") else fp
        try:
            im = Image.open(fp).convert("RGB").resize((32, 32))
            a = np.asarray(im).reshape(-1, 3) / 255.0
            hist = np.concatenate([np.histogram(a[:, c], bins=12, range=(0, 1))[0] for c in range(3)])
            hist = hist / (hist.sum() or 1)
            X.append(np.concatenate([hist, a.mean(0), a.std(0)])); keep.append(True)
        except Exception:
            keep.append(False)
    return (np.asarray(X) if X else np.zeros((0, 42))), np.array(keep), None


FEAT = {"smiles": feat_smiles, "kmer": feat_kmer, "protein": feat_protein, "msa": feat_msa,
        "kv": feat_kv, "floats": feat_floats, "nmr": feat_nmr, "text": feat_text,
        "geneset": feat_geneset, "formula": feat_formula, "image": feat_image}


# ----------------------------- the gate ---------------------------------------------
def make_groups(X, natural, n_rows):
    if natural is not None:
        return np.asarray(natural)
    k = max(5, min(25, n_rows // 150))
    if k < 2 or len(X) < k:
        return np.arange(len(X))            # degenerate, every point its own group
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
    return km.labels_


def gate(X, y, groups):
    out = {"n": int(len(y)), "pos": int(y.sum()), "baseline": round(float(y.mean()), 3),
           "n_groups": int(len(set(groups))), "dim": int(X.shape[1])}
    ng = len(set(groups))
    nsplit = max(2, min(5, ng))
    rand = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    if ng >= 2:
        cold = list(GroupKFold(nsplit).split(X, y, groups=groups))
    else:
        cold = rand
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    pr = cross_val_predict(clf, X, y, cv=rand, method="predict_proba", n_jobs=-1)[:, 1]
    pc = cross_val_predict(clf, X, y, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    pc_rf = cross_val_predict(rf, X, y, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    ys = np.random.RandomState(123).permutation(y)
    ps = cross_val_predict(clf, X, ys, cv=cold, method="predict_proba", n_jobs=-1)[:, 1]
    out["random_auroc"] = round(float(roc_auc_score(y, pr)), 3)
    out["cold_auroc"] = round(float(roc_auc_score(y, pc)), 3)
    out["cold_auroc_rf"] = round(float(roc_auc_score(y, pc_rf)), 3)
    out["cold_auprc"] = round(float(average_precision_score(y, pc)), 3)
    out["control_auroc"] = round(float(roc_auc_score(ys, ps)), 3)
    best_cold = max(out["cold_auroc"], out["cold_auroc_rf"])
    out["selectivity"] = round(best_cold - out["control_auroc"], 3)
    out["leakage_drop"] = round(out["random_auroc"] - out["cold_auroc"], 3)
    out["passed"] = bool(best_cold >= GATE_COLD_AUROC and out["selectivity"] >= GATE_SELECTIVITY)
    return out


# ----------------------------- task table -------------------------------------------
TASKS = [
    dict(name="admet_bace", file="admet_tdc/bace.csv", kind="smiles", col="smiles", rep="SMILES"),
    dict(name="admet_bbbp", file="admet_tdc/bbbp.csv", kind="smiles", col="smiles", rep="SMILES"),
    dict(name="admet_hiv", file="admet_tdc/hiv.csv", kind="smiles", col="smiles", rep="SMILES"),
    dict(name="withdrawn", file="withdrawn/withdrawn.csv", kind="smiles", col="smiles", rep="SMILES"),
    dict(name="dna_promoter", file="dna_promoter.csv", kind="kmer", col="smiles", rep="DNA seq", alpha="ACGT", k=4),
    dict(name="rna_coding", file="rna/coding.csv", kind="kmer", col="smiles", rep="RNA seq", alpha="ACGT", k=4),
    dict(name="protein_meltome", file="protein_meltome_named.csv", kind="protein", col="sequence", rep="protein seq", group_col="cluster"),
    dict(name="msa_conservation", file="msa/msa_conservation.csv", kind="msa", col="column", rep="MSA column", group_col="family"),
    dict(name="variant_clinvar", file="clinvar/variant_text.csv", kind="text", col="text", rep="variant HGVS", group="gene"),
    dict(name="ppi_name", file="ppi/ppi.csv", kind="text", col="text_name", rep="PPI (named)"),
    dict(name="ppi_anon", file="ppi/ppi.csv", kind="text", col="text_anon", rep="PPI (anon)"),
    dict(name="methyl_age", file="methyl/methyl_age.csv", kind="kv", col="beta_text", rep="CpG betas"),
    dict(name="ecg5000", file="ecg/ecg5000.csv", kind="floats", col="series", rep="ECG series"),
    dict(name="nmr_herg", file="nmr/herg_nmr.csv", kind="nmr", col="nmr", rep="NMR shifts"),
    dict(name="single_cell_name", file="single_cell/pbmc_Tcell.csv", kind="geneset", col="cell_sentence", rep="cell sentence (named)"),
    dict(name="single_cell_anon", file="single_cell/pbmc_Tcell.csv", kind="geneset", col="anon", rep="cell sentence (anon)"),
    dict(name="materials_metal", file="materials/metal.csv", kind="formula", col="formula", rep="metal formula"),
    dict(name="glass", file="generality/glass.csv", kind="formula", col="formula", rep="glass formula"),
    dict(name="histo_pcam", file="histo/pcam.csv", kind="image", col="img", rep="H&E patch (pixel-stat)"),
]


def gene_from_variant(t):
    m = re.search(r"\(([^)]+)\)", t or "")
    return m.group(1) if m else "NA"


def main():
    report = []
    print(f"{'task':20s} {'rep':22s}  n    base  rand  cold  (drop)  selec  gate", flush=True)
    print("-" * 92, flush=True)
    for t in TASKS:
        try:
            path = os.path.join(ROOT, t["file"])
            rows = list(csv.DictReader(open(path)))
            if len(rows) > CAP:
                rng = np.random.RandomState(0)
                rows = [rows[i] for i in rng.choice(len(rows), CAP, replace=False)]
            col = t["col"]
            vals = [r.get(col, "") for r in rows]
            labels_raw = [r.get("label", "") for r in rows]
            kind = t["kind"]
            if kind == "msa":
                X, keep, grp = feat_msa(rows, col)
            elif kind == "kmer":
                X, keep, grp = feat_kmer(vals, alpha=t.get("alpha", "ACGT"), k=t.get("k", 4))
            else:
                X, keep, grp = FEAT[kind](vals)
            keep = np.asarray(keep, dtype=bool)
            rows_k = [r for r, k in zip(rows, keep) if k]
            y = np.array([int(float(labels_raw[i])) for i in range(len(labels_raw)) if keep[i]])
            if len(set(y)) < 2 or len(y) < 40:
                print(f"{t['name']:20s} SKIP (n={len(y)}, classes={len(set(y))})", flush=True)
                continue
            # groups
            if grp is not None:
                groups = grp
            elif t.get("group_col"):
                groups = np.array([r.get(t["group_col"], "NA") for r in rows_k])
            elif t.get("group") == "gene":
                groups = np.array([gene_from_variant(r.get(col, "")) for r in rows_k])
            else:
                groups = make_groups(X, None, len(y))
            g = gate(X, y, groups)
            g.update({"task": t["name"], "rep": t["rep"], "kind": kind})
            report.append(g)
            print(f"{t['name']:20s} {t['rep']:22s} {g['n']:5d} {g['baseline']:.2f} "
                  f"{g['random_auroc']:.3f} {g['cold_auroc']:.3f} ({g['leakage_drop']:+.3f}) "
                  f"{g['selectivity']:+.3f}  {'PASS' if g['passed'] else 'fail'}", flush=True)
        except Exception as e:
            print(f"{t['name']:20s} ERR {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(ROOT, "verifiability_multimodal.json"), "w") as f:
        json.dump(report, f, indent=2)
    npass = sum(r["passed"] for r in report)
    print("-" * 92, flush=True)
    print(f"{npass}/{len(report)} modalities PASS (best-cold AUROC >= {GATE_COLD_AUROC} AND "
          f"selectivity >= {GATE_SELECTIVITY}). Wrote verifiability_multimodal.json", flush=True)


if __name__ == "__main__":
    main()
