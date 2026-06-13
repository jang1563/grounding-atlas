"""Frontier OUTPUT-arm panel on the two new hard-representation rungs (DNA, spectra).

The 8B output was measured (DNA promoter 0.396 anti-correlated, MS hERG 0.502 chance). This
runs the OUTPUT arm (verbalized property) for the three latest Claude models, to (a) complete
the frontier comparison the new rungs lack, (b) test whether a frontier model reads a DNA
sequence or a mass spectrum better than the 8B, and (c) check the expression gap's vendor/
scale invariance on hard representations. (Activation is open-weight-only; Claude has no
hidden states.) source <your-keys-file>. Env: PANEL_N (default 300), PANEL_WORKERS (8). No em dashes.
"""
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("PANEL_N", "300"))
WORKERS = int(os.environ.get("PANEL_WORKERS", "8"))
MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
QWEN_REF = {"dna": 0.396, "spectra": 0.502, "protein_seq": 0.486, "protein_org": 0.486,
            "sc_cellsentence": None, "sc_anon": None, "structure3d": None,
            "mat_formula": None, "mat_anon": None}
RUNGS = os.environ.get("PANEL_RUNGS", "dna,spectra,protein_seq,protein_org").split(",")

SYSTEM = ("You are a property predictor. Respond with ONLY a single decimal number between 0 and 1 "
          "(for example: 0.42). No words, no explanation. Just the number.")
P_DNA = ("You are given a DNA sequence. Estimate the probability (a single number between 0 and 1) "
         "that it is a PROMOTER (a regulatory region that initiates transcription). Judge only from "
         "the sequence. Reply with ONLY the number.\n\nSequence: {x}\nProbability:")
P_MS = ("The following are the fragment m/z peaks from the mass spectrum of a molecule. Estimate the "
        "probability (a single number between 0 and 1) that the molecule BLOCKS the hERG potassium "
        "channel (cardiotoxicity risk). Reply with ONLY the number.\n\nPeaks (m/z): {x}\nProbability:")
P_PROT = ("Estimate the probability (a single number between 0 and 1) that this protein has HIGH "
          "thermostability (a high melting temperature). Judge only from the amino-acid sequence. "
          "Reply with ONLY the number.\n\nSequence: {x}\nProbability:")
P_PROT_ORG = ("Estimate the probability (a single number between 0 and 1) that this protein, from the "
              "organism {org}, has HIGH thermostability (a high melting temperature). Reply with ONLY "
              "the number.\n\nSequence: {x}\nProbability:")
P_SC = ("Below are the most highly expressed genes of a single human PBMC cell, in descending order of "
        "expression. Estimate the probability (a single number between 0 and 1) that this cell is a "
        "T lymphocyte (T cell). Reply with ONLY the number.\n\nGenes: {x}\nProbability:")
P_3D = ("Below are the 3D atomic coordinates (element, x, y, z in Angstroms) of a molecule. Estimate the "
        "probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel "
        "(cardiotoxicity risk). Reply with ONLY the number.\n\nCoordinates:\n{x}\nProbability:")
P_MAT = ("Below is the chemical composition of an inorganic compound. Estimate the probability (a single "
         "number between 0 and 1) that it is a METAL (a metallic conductor, as opposed to an insulator or "
         "semiconductor). Reply with ONLY the number.\n\nComposition: {x}\nProbability:")


def parse_prob(t):
    for tok in reversed(re.findall(r"\d*\.?\d+", t)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "parsed"
        if 1.0 < v <= 100.0:
            return v / 100.0, "percent"
    return 0.5, "fallback"


def load_dna(n):
    rows = [(r["smiles"], int(r["label"])) for r in csv.DictReader(open(os.path.join(ROOT, "signal", "dna_promoter.csv")))]
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = [(s, l) for s, l in pos[:k]] + [(s, l) for s, l in neg[:k]]
    return [(P_DNA.format(x=s), l) for s, l in data]


def sim_ms(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "none"
    masses = [Descriptors.ExactMolWt(m)]
    try:
        for f in BRICS.BRICSDecompose(m):
            fm = Chem.MolFromSmiles(f)
            if fm is not None:
                masses.append(Descriptors.ExactMolWt(fm))
    except Exception:
        pass
    return ", ".join(f"{x:.1f}" for x in sorted(masses))


def load_spectra(n):
    by = defaultdict(list)
    for l in open(os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl")):
        r = json.loads(l)
        if r["condition"] == "matched":
            by[int(r["label"])].append(r["representation"])
    rng = random.Random(42)
    out = []
    for lab in (1, 0):
        it = by[lab][:]
        rng.shuffle(it)
        for s in it[:n // 2]:
            out.append((P_MS.format(x=sim_ms(s)), lab))
    return out


def load_protein(n, with_org):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", "protein_meltome_named.csv"))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    sel = pos[:k] + neg[:k]
    out = []
    for r in sel:
        if with_org:
            out.append((P_PROT_ORG.format(x=r["sequence"], org=r["organism"]), int(r["label"])))
        else:
            out.append((P_PROT.format(x=r["sequence"]), int(r["label"])))
    return out


# Generality probes: config-driven so adding a (web-rich vs anon) domain is one line.
# Each: (csv under signal/, [web-rich field, anon field], yes/no prompt). Rung name = "{domain}_{field}".
GEN_CONFIGS = {
    "minerals": ("generality/minerals.csv", ["name", "anon"],
                 "Estimate the probability (a single number between 0 and 1) that {x} is HARD (Mohs hardness 5 or above). Reply with ONLY the number.\n\n{x}\nProbability:"),
    "elements": ("generality/elements.csv", ["name", "anon"],
                 "Estimate the probability (a single number between 0 and 1) that {x} is a METAL. Reply with ONLY the number.\n\n{x}\nProbability:"),
    "glass": ("generality/glass.csv", ["formula", "anon"],
              "Below is the chemical composition of an alloy. Estimate the probability (a single number between 0 and 1) that it is a GLASS FORMER (it can form a metallic glass when quenched). Reply with ONLY the number.\n\nComposition: {x}\nProbability:"),
    "aminoacid": ("generality/aminoacid.csv", ["name", "anon"],
                  "Estimate the probability (a single number between 0 and 1) that {x} is HYDROPHOBIC (a nonpolar amino acid). Reply with ONLY the number.\n\n{x}\nProbability:"),
    "drugclass": ("generality/drugclass.csv", ["name", "anon"],
                  "Estimate the probability (a single number between 0 and 1) that {x} acts on the CENTRAL NERVOUS SYSTEM (a CNS / psychoactive drug). Reply with ONLY the number.\n\n{x}\nProbability:"),
    "metabolite": ("generality/metabolite.csv", ["name", "anon"],
                   "Estimate the probability (a single number between 0 and 1) that {x} is a LIPID (a fatty / lipid metabolite, as opposed to a sugar, amino acid, or nucleotide). Reply with ONLY the number.\n\n{x}\nProbability:"),
}


# Molecular-representation rungs (hERG from a non-SMILES rendering of the same molecules).
MOLREP_CONFIGS = {
    "graph": ("graph/herg_graph.csv", "graph",
              "Below is a molecular graph (an atom list and a bond edge list) of a molecule. Estimate the probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel (cardiotoxicity risk). Reply with ONLY the number.\n\n{x}\nProbability:"),
    "nmr": ("nmr/herg_nmr.csv", "nmr",
            "Below is the simulated 13C NMR spectrum (a list of carbon chemical shifts in ppm) of a molecule. Estimate the probability (a single number between 0 and 1) that it BLOCKS the hERG potassium channel (cardiotoxicity risk). Reply with ONLY the number.\n\nShifts (ppm): {x}\nProbability:"),
}


def load_molrep(rung, n):
    path, field, prompt = MOLREP_CONFIGS[rung]
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", path))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return [(prompt.format(x=r[field]), int(r["label"])) for r in (pos[:k] + neg[:k])]


def load_gen(rung, n):
    domain, field = rung.rsplit("_", 1)
    path, _fields, prompt = GEN_CONFIGS[domain]
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", path))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return [(prompt.format(x=r[field]), int(r["label"])) for r in (pos[:k] + neg[:k])]


def load_materials(n, field):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", "materials", "metal.csv"))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return [(P_MAT.format(x=r[field]), int(r["label"])) for r in (pos[:k] + neg[:k])]


def load_3d(n):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", "structure3d", "herg_xyz.csv"))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return [(P_3D.format(x=r["xyz"]), int(r["label"])) for r in (pos[:k] + neg[:k])]


def load_sc(n, field):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "signal", "single_cell", "pbmc_Tcell.csv"))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    sel = pos[:k] + neg[:k]
    return [(P_SC.format(x=r[field]), int(r["label"])) for r in sel]


def boot(y, p, nb=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    a = []
    for _ in range(nb):
        b = rng.choice(idx, len(idx), True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)


def run(client, model, items):
    def call(prompt):
        try:
            m = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
            t = [b.text for b in m.content if getattr(b, "type", None) == "text"]
            return parse_prob(t[0]) if t else (0.5, "empty")
        except Exception as e:
            return (0.5, f"err:{type(e).__name__}")
    res = [None] * len(items)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, p): i for i, (p, _) in enumerate(items)}
        done = 0
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(items)}", flush=True)
    return np.array([r[0] for r in res]), [r[1] for r in res]


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    builders = {"dna": lambda: load_dna(N), "spectra": lambda: load_spectra(N),
                "protein_seq": lambda: load_protein(N, False), "protein_org": lambda: load_protein(N, True),
                "sc_cellsentence": lambda: load_sc(N, "cell_sentence"), "sc_anon": lambda: load_sc(N, "anon"),
                "structure3d": lambda: load_3d(N),
                "mat_formula": lambda: load_materials(N, "formula"), "mat_anon": lambda: load_materials(N, "anon")}
    for _dom, (_p, _fields, _pr) in GEN_CONFIGS.items():
        for _fld in _fields:
            _r = f"{_dom}_{_fld}"
            builders[_r] = (lambda rr: (lambda: load_gen(rr, N)))(_r)
    for _mr in MOLREP_CONFIGS:
        builders[_mr] = (lambda rr: (lambda: load_molrep(rr, N)))(_mr)
    rungs = {r: builders[r]() for r in RUNGS if r in builders}
    out = {}
    outpath = os.path.join(ROOT, "results", "frontier_output_panel.json")
    if os.path.isfile(outpath):
        try:
            out = json.load(open(outpath))  # resume / merge with the DNA+spectra already done
        except Exception:
            out = {}
    for rung, items in rungs.items():
        y = np.array([l for _, l in items])
        out[rung] = {"n": len(y), "qwen3_8b": QWEN_REF.get(rung), "models": {}}
        print(f"\n== {rung} (n={len(y)}, pos={int(y.sum())}) ==", flush=True)
        for model in MODELS:
            print(f"  {model}", flush=True)
            p, kinds = run(client, model, items)
            auc = round(float(roc_auc_score(y, p)), 3)
            lo, hi = boot(y, p)
            out[rung]["models"][model] = {"auroc": auc, "ci95": [lo, hi], "parse": dict(Counter(kinds))}
            print(f"    AUROC={auc} CI={lo,hi} parse={dict(Counter(kinds))}", flush=True)
            json.dump(out, open(outpath, "w"), indent=2)  # kill-safe
    print("\n=== SUMMARY (output AUROC; activation is open-weight-only) ===")
    for rung in rungs:
        r = out[rung]
        qref = f"{r['qwen3_8b']:.3f}" if r['qwen3_8b'] is not None else "n/a"
        print(f"{rung:13s} qwen3-8b {qref} | " +
              " | ".join(f"{m.split('-')[1]}-{m.split('-')[2]} {r['models'][m]['auroc']:.3f}" for m in MODELS))
    json.dump(out, open(outpath, "w"), indent=2)


if __name__ == "__main__":
    main()
