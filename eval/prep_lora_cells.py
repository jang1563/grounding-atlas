"""Prep the two WS3 train-placement candidate cells into the ws3_lora.py jsonl format.

Each line: {"representation": str, "label": 0/1, "group": str} (+ "condition":"matched").
group = the cold-split key (gene for variant, Murcko scaffold for MS). These are the cells where
the cheap specialist is weakest, so the cleanest tests of whether TRAIN (LoRA) can win:
- variant_seq: raw protein-sequence window -> pathogenic. Notation-invariance / the "ready train
  target". Bar to beat: seq-form 0-shot 0.58; orchestrate AlphaMissense 0.96 (novel 0.985).
- spectra_ms: BRICS-fragment m/z peak list -> hERG block. Bar: cheap binned-m/z 0.667, retrieve 0.586.
Local CPU (rdkit). No em dashes.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def prep_variant_seq():
    src = os.path.join(ROOT, "variant_grounding", "data", "variant_clinvar.csv")
    out = os.path.join(ROOT, "signal", "variant_seq")
    os.makedirs(out, exist_ok=True)
    n = 0
    with open(os.path.join(out, "pairs.jsonl"), "w") as fh:
        for r in csv.DictReader(open(src)):
            seq = (r.get("wt_window") or "").strip()
            if not seq or r.get("label") in (None, ""):
                continue
            fh.write(json.dumps({"representation": seq, "label": int(float(r["label"])),
                                 "group": r.get("gene") or r.get("uniprot") or "na",
                                 "condition": "matched",
                                 "post_cutoff": str(r.get("post_cutoff")).lower() in ("1", "true")}) + "\n")
            n += 1
    print(f"variant_seq: {n} -> {out}/pairs.jsonl  (group=gene; representation=wt_window seq)")


def prep_spectra_ms():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    src = os.path.join(ROOT, "data", "herg.csv")
    out = os.path.join(ROOT, "signal", "spectra_ms")
    os.makedirs(out, exist_ok=True)

    def peaks(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None
        masses = {round(Descriptors.ExactMolWt(m), 1)}            # molecular ion
        try:
            for f in BRICS.BRICSDecompose(m):
                fm = Chem.MolFromSmiles(f)
                if fm is not None:
                    masses.add(round(Descriptors.ExactMolWt(fm), 1))
        except Exception:
            pass
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smi
        except Exception:
            scaf = smi
        return ", ".join(str(x) for x in sorted(masses)), scaf

    n = 0
    with open(os.path.join(out, "pairs.jsonl"), "w") as fh:
        for r in csv.DictReader(open(src)):
            pk, scaf = peaks(r["smiles"])
            if pk is None:
                continue
            fh.write(json.dumps({"representation": pk, "label": int(float(r["label"])),
                                 "group": scaf, "condition": "matched"}) + "\n")
            n += 1
    print(f"spectra_ms: {n} -> {out}/pairs.jsonl  (group=Murcko scaffold; representation=BRICS m/z peak list)")


if __name__ == "__main__":
    prep_variant_seq()
    prep_spectra_ms()
