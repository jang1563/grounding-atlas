"""WS1 computable-property row: deterministic descriptors as the encode-vs-verbalize CONTROL.

The 17 measured rungs use EMPIRICAL properties (hERG, Tm, age, pathogenicity): the label
needs an experiment, the structure-probe ceiling is < 1.0, and the verbalization gap is set
by the web-exposure law (PROJECT_DESIGN section 7). This generator adds the orthogonal
column: COMPUTABLE properties (atom count, ring count, molecular weight, TPSA, logP), where

  the label is a closed-form function of the representation (RDKit / Biopython computes it
  exactly), so the structure-probe CEILING IS 1.0 BY CONSTRUCTION.

That changes what the row measures. It is not "is the signal in the content" (it always is);
it is a dissociation: a computable property has ceiling 1.0 and (we predict) HIGH activation
(the info is surface-decodable), yet the OUTPUT arm can still fail, and when it fails the
cause is tokenization / arithmetic execution, NOT web-exposure of a content->property mapping.
Prior art: GPT-4o gets carbon counting right ~4 percent of the time, reasoning models (o3-mini)
70-92 percent (arXiv 2505.07735). So this row marks the BOUNDARY of the web-exposure law: a
place where the verbalization gap is closed by reasoning mode and representation format, not by
web frequency. It is the clean control the law needs.

Schema and conditions match generate_signal.py exactly, so the row slots into the same 3-arm
instrument (eval/activation_arm.py for ceiling+activation+output on the open model;
eval/output_arm_computable.py for the frontier output arm). Two extra fields are stored that
the empirical rungs do not need: "value" (the exact descriptor value) and "threshold" (the
median used to binarize), so the output arm can score BOTH exact-match / MAE on the raw value
AND AUROC on the binarized label (comparable to every other rung).

Local CPU only (rdkit + sklearn; Biopython for sequences). No API, no GPU. No em dashes.

Usage:
  python generate_computable.py                      # SMILES from NegBioDB, all RDKit descriptors
  python generate_computable.py --props n_carbon,n_rings
  python generate_computable.py --modality protein --source path/to/seqs.fasta
  python generate_computable.py --modality dna --source path/to/seqs.csv   # column 'sequence'
"""
import argparse
import json
import os
import sqlite3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ADMET_DB = os.environ.get("NEGBIODB_ADMET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Negative_result_DB", "data", "negbiodb_admet.db"))
OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "computable")
MAX_MOL = 1500          # cap molecules featurized (CPU budget)
VARIANT_SUBSET = 250    # how many also get re_notation + scrambled variants


# ---------------------------------------------------------------------------
# descriptor panels: name -> (fn(parsed_obj) -> value, kind, human clause)
# the clause is duplicated in eval/output_arm_computable.py; keep the two in sync.
# ---------------------------------------------------------------------------
def smiles_panel():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    def n_stereo(m):
        try:
            return len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        except Exception:
            return None

    return {
        "n_carbon":         (lambda m: sum(a.GetSymbol() == "C" for a in m.GetAtoms()), "int"),
        "n_rings":          (lambda m: rdMolDescriptors.CalcNumRings(m), "int"),
        "n_aromatic_rings": (lambda m: rdMolDescriptors.CalcNumAromaticRings(m), "int"),
        "n_hbd":            (lambda m: rdMolDescriptors.CalcNumHBD(m), "int"),
        "n_hba":            (lambda m: rdMolDescriptors.CalcNumHBA(m), "int"),
        "n_rot_bonds":      (lambda m: rdMolDescriptors.CalcNumRotatableBonds(m), "int"),
        "n_stereo":         (n_stereo, "int"),
        "mol_wt":           (lambda m: Descriptors.MolWt(m), "float"),
        "tpsa":             (lambda m: rdMolDescriptors.CalcTPSA(m), "float"),
        "logp":             (lambda m: Crippen.MolLogP(m), "float"),
    }


def protein_panel():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    STD = set("ACDEFGHIKLMNPQRSTVWY")

    def pa(s):
        # ProtParam molecular_weight() / isoelectric_point() raise on non-standard residues
        # (X, B, Z, U, *), so keep only the 20 standard AA for the biochem descriptors. The
        # length descriptor below uses the raw sequence, not pa().
        return ProteinAnalysis("".join(c for c in str(s).upper() if c in STD))

    return {
        "length":            (lambda s: len(str(s).strip()), "int"),
        "mol_wt":            (lambda s: pa(s).molecular_weight(), "float"),
        "aromaticity":       (lambda s: pa(s).aromaticity(), "float"),
        "instability_index": (lambda s: pa(s).instability_index(), "float"),
        "gravy":             (lambda s: pa(s).gravy(), "float"),
        "isoelectric_point": (lambda s: pa(s).isoelectric_point(), "float"),
        "frac_cys":          (lambda s: str(s).upper().count("C") / max(len(str(s)), 1), "float"),
    }


def dna_panel():
    def gc(s):
        s = str(s).upper()
        try:
            from Bio.SeqUtils import gc_fraction
            return gc_fraction(s) * 100.0
        except Exception:
            n = sum(c in "GC" for c in s)
            return 100.0 * n / max(len(s), 1)

    return {
        "length":     (lambda s: len(str(s).strip()), "int"),
        "gc_content": (gc, "float"),
        "n_a":        (lambda s: str(s).upper().count("A"), "int"),
        "n_codons":   (lambda s: len(str(s).strip()) // 3, "int"),
    }


# ---------------------------------------------------------------------------
# sources: return list[(id, representation_string, parsed_object)]
# ---------------------------------------------------------------------------
def load_smiles_source():
    from rdkit import Chem
    con = sqlite3.connect(ADMET_DB)
    rows = con.execute(
        "SELECT DISTINCT canonical_smiles FROM admet_compounds "
        "WHERE canonical_smiles IS NOT NULL AND canonical_smiles != ''"
    ).fetchall()
    con.close()
    smis = [r[0] for r in rows]
    rng = np.random.RandomState(42)
    rng.shuffle(smis)
    out = []
    for i, smi in enumerate(smis):
        if len(out) >= MAX_MOL:
            break
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            out.append((f"smiles_{i}", smi, m))
    return out


def _read_sequences(path, seq_col=None):
    """FASTA (.fa/.fasta) or CSV with a sequence column (seq_col, else 'sequence'/'seq')."""
    seqs = []
    if path.lower().endswith((".fa", ".fasta", ".faa")):
        name, buf = None, []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if name is not None:
                        seqs.append((name, "".join(buf)))
                    name, buf = line[1:].split()[0], []
                elif line:
                    buf.append(line)
            if name is not None:
                seqs.append((name, "".join(buf)))
    else:
        import csv
        with open(path) as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                s = (row.get(seq_col) if seq_col else None) or row.get("sequence") or row.get("seq")
                if s:
                    seqs.append((row.get("id", f"seq_{i}"), s.strip()))
    return seqs


def load_sequence_source(path, seq_col=None):
    out = []
    for i, (name, s) in enumerate(_read_sequences(path, seq_col)):
        if len(out) >= MAX_MOL:
            break
        if s:
            out.append((f"{name}", s, s))   # representation == parsed object for sequences
    return out


# ---------------------------------------------------------------------------
# notation variants (SMILES only; sequences have no canonical re-notation here)
# ---------------------------------------------------------------------------
def renotate_smiles(smi):
    """Same molecule, alternate VALID notation. The computed descriptor is INVARIANT, so the
    label/value are unchanged: a true reader must give the same number (re_notation control)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, doRandom=True, canonical=False)
    except Exception:
        return None


def scramble(s, seed=0):
    """Character shuffle. NOTE the diagnostic asymmetry this exposes: composition-type
    descriptors (n_carbon, mol_wt) are PRESERVED by a shuffle (same letters), so a model that
    counts characters is unchanged; topology-type descriptors (n_rings, tpsa, n_stereo) are
    DESTROYED (no valid parse), so a true structural reader must break. The split is the read."""
    rng = np.random.RandomState(seed)
    ch = list(str(s))
    rng.shuffle(ch)
    return "".join(ch)


# ---------------------------------------------------------------------------
# ceiling (decodability) check: exact ceiling is 1.0; we also report how linearly decodable
# the binarized property is from a standard structural featurizer (Morgan FP for SMILES),
# a proxy for how easily the model's own structural features could carry it.
# ---------------------------------------------------------------------------
def morgan_decodability(items, labels):
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, y, grp = [], [], []
    for (_id, _rep, mol), lab in zip(items, labels):
        X.append(gen.GetFingerprintAsNumPy(mol))
        y.append(lab)
        try:
            grp.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or _id)
        except Exception:
            grp.append(_id)
    X, y, grp = np.asarray(X), np.asarray(y), np.asarray(grp)
    if len(set(y)) < 2 or len(set(grp)) < 5:
        return None
    cv = GroupKFold(5)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=cv, groups=grp, method="predict_proba", n_jobs=-1)[:, 1]
    return round(float(roc_auc_score(y, p)), 3)


# ---------------------------------------------------------------------------
def emit(modality, prop, kind, items, values, threshold, labels, outdir):
    os.makedirs(outdir, exist_ok=True)
    is_smiles = modality == "smiles"
    rng = np.random.RandomState(7)
    order = list(range(len(items)))
    rng.shuffle(order)
    vset = set(order[:VARIANT_SUBSET])
    pairs_path = os.path.join(outdir, "pairs.jsonl")
    n_var = 0
    with open(pairs_path, "w") as f:
        for i in order:
            _id, rep, _obj = items[i]
            rec = {"id": _id, "modality": modality, "property": prop, "condition": "matched",
                   "representation": rep, "label": int(labels[i]),
                   "value": round(float(values[i]), 4), "threshold": round(float(threshold), 4),
                   "kind": kind, "source": "negbiodb_admet" if is_smiles else "sequence_file"}
            f.write(json.dumps(rec) + "\n")
            if i in vset and is_smiles:
                rn = renotate_smiles(rep)
                if rn:
                    f.write(json.dumps({**rec, "condition": "re_notation", "representation": rn}) + "\n")
                    n_var += 1
                # scrambled keeps the ORIGINAL value/label; the test is whether the answer moves
                f.write(json.dumps({**rec, "condition": "scrambled",
                                    "representation": scramble(rep, seed=i)}) + "\n")
    # activation-arm CSV (smiles,label) for SMILES, matched only, exactly eval/activation_arm.py load()
    csv_path = None
    if is_smiles:
        csv_path = os.path.join(outdir, f"{prop}.csv")
        with open(csv_path, "w") as f:
            f.write("smiles,label\n")
            for i in range(len(items)):
                f.write(f"{items[i][1]},{int(labels[i])}\n")
    return pairs_path, csv_path, n_var


def build(modality, props, source_path, seq_col=None):
    if modality == "smiles":
        panel = smiles_panel()
        items = load_smiles_source()
    elif modality == "protein":
        panel = protein_panel()
        items = load_sequence_source(source_path, seq_col)
    elif modality == "dna":
        panel = dna_panel()
        items = load_sequence_source(source_path, seq_col)
    else:
        raise SystemExit(f"unknown modality {modality}")

    if not items:
        raise SystemExit("no input items loaded (check --source / DB path)")
    chosen = list(panel) if (not props or props == ["all"]) else props
    print(f"== computable row :: modality={modality} :: n_items={len(items)} :: props={chosen} ==\n")

    report = []
    for prop in chosen:
        if prop not in panel:
            print(f"[{prop}] not in {modality} panel, skip"); continue
        fn, kind = panel[prop]
        vals, keep = [], []
        for it in items:
            try:
                v = fn(it[2])
            except Exception:
                v = None
            if v is not None and np.isfinite(v):
                vals.append(float(v)); keep.append(it)
        if len(keep) < 50:
            print(f"[{prop}] only {len(keep)} computable, skip"); continue
        vals = np.asarray(vals)
        thr = float(np.median(vals))
        labels = (vals > thr).astype(int)
        if len(set(labels)) < 2:                      # degenerate (e.g. all equal); nudge by mean
            thr = float(vals.mean()); labels = (vals > thr).astype(int)
        if len(set(labels)) < 2:                      # still single-class (constant descriptor): skip
            print(f"[{prop:16s}] degenerate, all values ~equal (n_distinct={len(set(vals.tolist()))}), skip")
            continue
        outdir = os.path.join(OUTROOT, modality, prop)
        ceil = morgan_decodability(keep, labels) if modality == "smiles" else None
        pairs_path, csv_path, n_var = emit(modality, prop, kind, keep, vals, thr, labels, outdir)
        rec = {"property": prop, "modality": modality, "kind": kind,
               "n": len(keep), "threshold_median": round(thr, 4),
               "value_min": round(float(vals.min()), 3), "value_max": round(float(vals.max()), 3),
               "ceiling_exact": 1.0,                  # closed-form: RDKit/Biopython computes it
               "morgan_decodability_auroc": ceil,     # surface-decodability proxy, not the ceiling
               "n_renotation_variants": n_var, "pairs": pairs_path, "act_csv": csv_path}
        with open(os.path.join(outdir, "verifiability.json"), "w") as f:
            json.dump(rec, f, indent=2)
        report.append(rec)
        print(f"[{prop:16s}] n={rec['n']:5d} kind={kind:5s} median={rec['threshold_median']:>9} "
              f"range=[{rec['value_min']},{rec['value_max']}] exact_ceiling=1.0 "
              f"morgan_decode={ceil}  -> {outdir}")

    os.makedirs(OUTROOT, exist_ok=True)
    with open(os.path.join(OUTROOT, "decodability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {len(report)} computable properties under {OUTROOT}")
    print("next: eval/output_arm_computable.py (frontier output arm, dry-run first) and, on GPU,")
    print("  ACT_CSV=<prop>.csv ACT_PARSE=number ACT_PROMPT='...{smiles}...' python eval/activation_arm.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="smiles", choices=["smiles", "protein", "dna"])
    ap.add_argument("--props", default="all", help="comma list of descriptor names, or 'all'")
    ap.add_argument("--source", default=None, help="FASTA or CSV path (required for protein/dna)")
    ap.add_argument("--seq-col", default=None, help="CSV column holding the sequence (default: sequence/seq)")
    a = ap.parse_args()
    props = [p.strip() for p in a.props.split(",")] if a.props else ["all"]
    if a.modality in ("protein", "dna") and not a.source:
        raise SystemExit(f"--source FASTA/CSV is required for modality={a.modality}")
    build(a.modality, props, a.source, a.seq_col)


if __name__ == "__main__":
    main()
