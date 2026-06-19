"""GroundBench task registry: the benchmark's task set, decoupled from data format.

Each task is a (representation, verifiable binary property) classification with:
  - prompt   : a versioned template with a {rep} slot, ending in a numeric anchor;
  - orient   : which label the clause counts as positive ("align" = label 1, "oppose" = label 0);
  - web      : the a-priori WEB-EXPOSURE tag ("rich" = the representation's tokens are
               web-documented, "zero" = anonymized / undocumented, "mixed"), the distinctive
               metadata that lets the benchmark score grounding AND when to defer;
  - ceiling  : a cheap-specialist ceiling, a float or a key into results/benchmark/ceilings.json;
  - kind     : "pairs" (a pairs.jsonl with matched/scrambled conditions, enabling memo_delta) or
               "twocol" (a CSV with two representation columns, a web-rich vs web-zero contrast).

Adding a task is one entry here; the harness (run_grounding_eval.py) loops this registry. Keeping
the registry separate from the runner is what makes the benchmark extensible by others.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIGNAL = os.path.join(ROOT, "signal")


def _mol(clause):
    return ("Estimate the probability (a single number between 0 and 1) that this molecule "
            f"{clause}. Judge only from the representation below.\nsmiles: {{rep}}\nProbability:")


def _cell(pos, neg):
    return ("These are the most distinctively expressed genes of one human immune cell (PBMC), "
            f"highest first:\n{{rep}}\nIs it a {pos} or a {neg}? Reply with ONLY a number between "
            f"0 and 1 = the probability that it is a {pos}.")


_VARIANT = ("Estimate the probability (a single number between 0 and 1) that this human genetic "
            "variant is pathogenic. Judge only from the representation below.\nvariant: {rep}\nProbability:")
_METHYL = ("Estimate the probability (a single number between 0 and 1) that this individual is over "
           "40 years old, from their blood DNA methylation profile (CpG site: beta-value pairs). "
           "Judge only from the values below.\nmethylation: {rep}\nProbability:")
_MSA = ("Estimate the probability (a single number between 0 and 1) that this column of a protein "
        "multiple-sequence alignment is evolutionarily conserved. Judge only from the residues "
        "below.\ncolumn: {rep}\nProbability:")
_METAL = ("Estimate the probability (a single number between 0 and 1) that this chemical compound is "
          "a metal. Judge only from the composition below.\ncomposition: {rep}\nProbability:")
_ESM = ("Below is a 640-dimensional protein embedding from the ESM-2 protein language model (a "
        "scientific foundation model). Estimate the probability (a single number between 0 and 1) "
        "that this protein is thermostable (melting temperature above the dataset median). Judge "
        "only from the embedding.\nembedding: {rep}\nProbability:")


def _herg(repline):
    return ("Estimate the probability (a single number between 0 and 1) that this molecule blocks the "
            "hERG potassium channel (cardiotoxicity risk). Judge only from the representation below.\n"
            f"{repline}: {{rep}}\nProbability:")


_RNA = ("Estimate the probability (a single number between 0 and 1) that this nucleotide sequence is "
        "protein-coding (vs non-coding). Judge only from the sequence below.\nsequence: {rep}\nProbability:")
_HISTO = ("This is an H&E-stained histopathology image patch. Estimate the probability (a single "
          "number between 0 and 1) that it contains tumor tissue (vs normal).\nProbability:")
_NT = ("Below is a 512-dimensional nucleotide-sequence embedding from the Nucleotide Transformer (a "
       "genomic foundation model). Estimate the probability (a single number between 0 and 1) that the "
       "underlying sequence is protein-coding (vs non-coding). Judge only from the embedding.\n"
       "embedding: {rep}\nProbability:")


TASKS = {
    # ADMET: SMILES -> empirical property (pairs.jsonl, matched/scrambled). web-rich (drug/SMILES
    # tokens are web-documented). orientation per the structural-alert audit (ames = oppose).
    "admet/herg":         dict(kind="pairs", data="admet/herg/pairs.jsonl",
                               prompt=_mol("blocks the hERG potassium channel (cardiotoxicity risk)"),
                               orient="align", web="rich", ceiling="admet/herg"),
    "admet/cyp3a4":       dict(kind="pairs", data="admet/cyp3a4/pairs.jsonl",
                               prompt=_mol("inhibits the CYP3A4 enzyme"),
                               orient="align", web="rich", ceiling="admet/cyp3a4"),
    "admet/cyp2d6":       dict(kind="pairs", data="admet/cyp2d6/pairs.jsonl",
                               prompt=_mol("inhibits the CYP2D6 enzyme"),
                               orient="align", web="rich", ceiling="admet/cyp2d6"),
    "admet/ames":         dict(kind="pairs", data="admet/ames/pairs.jsonl",
                               prompt=_mol("is mutagenic in the Ames test"),
                               orient="oppose", web="rich", ceiling="admet/ames"),
    "admet/solubility":   dict(kind="pairs", data="admet/solubility/pairs.jsonl",
                               prompt=_mol("is highly soluble in water"),
                               orient="oppose", web="rich", ceiling="admet/solubility"),
    "admet/permeability": dict(kind="pairs", data="admet/permeability/pairs.jsonl",
                               prompt=_mol("is highly permeable across a cell membrane"),
                               orient="oppose", web="rich", ceiling="admet/permeability"),
    # Single-cell: expression -> cell type, the SAME cells in a web-rich (gene NAMES) and a
    # web-zero (ANONYMIZED ids) form. The name/anon pair is the controlled web-exposure contrast.
    "single_cell/cd8t_nk:name": dict(kind="twocol", data="single_cell/cd8t_nk.csv", col="cell_sentence",
                                     prompt=_cell("CD8+ T cell", "NK cell"), orient="align",
                                     web="rich", ceiling=0.992),
    "single_cell/cd8t_nk:anon": dict(kind="twocol", data="single_cell/cd8t_nk.csv", col="anon",
                                     prompt=_cell("CD8+ T cell", "NK cell"), orient="align",
                                     web="zero", ceiling=0.992),
    "single_cell/mono:name":    dict(kind="twocol", data="single_cell/mono_cd14_fcgr3a.csv", col="cell_sentence",
                                     prompt=_cell("classical CD14+ monocyte", "non-classical CD16+ monocyte"),
                                     orient="align", web="rich", ceiling=0.989),
    "single_cell/mono:anon":    dict(kind="twocol", data="single_cell/mono_cd14_fcgr3a.csv", col="anon",
                                     prompt=_cell("classical CD14+ monocyte", "non-classical CD16+ monocyte"),
                                     orient="align", web="zero", ceiling=0.989),
    # Variant effect: the SAME variants as web-rich HGVS text vs a web-poor protein sequence
    # (within-entity web-exposure pair). label 1 = pathogenic (align; leakage-audited, 0/2400).
    "variant/text":       dict(kind="twocol", data="clinvar/variant_text.csv", col="text",
                               prompt=_VARIANT, orient="align", web="rich", ceiling=0.962),
    "variant/seq":        dict(kind="pairs", data="variant_seq/pairs.jsonl",
                               prompt=_VARIANT, orient="align", web="zero", ceiling=0.962),
    # Methylation beta vector -> age (web-zero numeric) and its web-rich controlled twin, MSA
    # column -> conserved. Same "encoded-to-ceiling" shape, opposite web-documentation.
    "methyl/age":         dict(kind="twocol", data="methyl/methyl_age.csv", col="beta_text",
                               prompt=_METHYL, orient="align", web="zero", ceiling=0.701),
    "msa/conservation":   dict(kind="twocol", data="msa/msa_conservation.csv", col="column",
                               prompt=_MSA, orient="align", web="rich", ceiling=0.999),
    # Materials (generality beyond biology): metal-vs-not from a formula, web-rich element symbols
    # vs anonymized elements (a third name/anon controlled pair). label 1 = metal.
    "materials/metal:formula": dict(kind="twocol", data="materials/metal.csv", col="formula",
                                    prompt=_METAL, orient="align", web="rich", ceiling=0.927),
    "materials/metal:anon":    dict(kind="twocol", data="materials/metal.csv", col="anon",
                                    prompt=_METAL, orient="align", web="zero", ceiling=0.927),
    # SFM leg (the LLM x SFM interface): can the LLM read a scientific foundation model's OUTPUT
    # embedding? ESM-2 (150M, 640-dim) embedding -> thermostability. The ultimate web-zero form (an
    # abstract float vector). ceiling = a read-out HEAD on the same embedding (LogReg, cluster
    # GroupKFold, leakage-controlled) = the "orchestrate via a trained head" baseline; the LLM
    # reading the embedding-as-text is expected at chance (prompt-pasting an SFM output fails).
    "protein/esm2_emb": dict(kind="emb", data="sfm_embedding/meltome_esm2.npz",
                             prompt=_ESM, orient="align", web="zero", ceiling=0.633),
    # hERG in three more representations of the SAME molecules (graph / 13C-NMR shifts / 3D coords).
    # The property is Morgan-predictable (ceiling ~0.89) but these representation->hERG mappings are
    # web-undocumented, so the LLM is expected at chance. The point: the REPRESENTATION's web-exposure
    # governs verbalization, not the property's (hERG verbalizes from SMILES, not from these). align.
    "herg/graph":    dict(kind="twocol", data="graph/herg_graph.csv", col="graph",
                          prompt=_herg("molecular graph"), orient="align", web="zero", ceiling=0.892),
    "herg/nmr":      dict(kind="twocol", data="nmr/herg_nmr.csv", col="nmr",
                          prompt=_herg("carbon-13 NMR chemical shifts"), orient="align", web="zero", ceiling=0.892),
    "herg/struct3d": dict(kind="twocol", data="structure3d/herg_xyz.csv", col="xyz",
                          prompt=_herg("3D atomic coordinates (element x y z)"), orient="align", web="zero", ceiling=0.868),
    # RNA coding-vs-noncoding from the nucleotide sequence. web=mixed: ORF/codon structure is a
    # partially documented heuristic the model may use. label 1 = coding. ceiling = 3-mer LR.
    "rna/coding":    dict(kind="twocol", data="rna/coding.csv", col="smiles",
                          prompt=_RNA, orient="align", web="mixed", ceiling=0.856),
    # Image / VLM arm: H&E histopathology patch -> tumor (PatchCamelyon, 96x96). The flagship
    # expression gap: an open VLM ENCODES tumor (activation probe 0.827) but verbalizes near chance
    # (0.463). web=rich (tumor morphology is heavily documented/imaged online) so by the law it
    # "should" verbalize -- the revealing exception that surfaces a SECOND gate (medical refusal /
    # hedging). ceiling = CONCH pathology FM ~0.90 (cheap color classifier 0.746, VLM activation 0.827).
    "histo/pcam_tumor": dict(kind="image", data="histo/pcam.csv", col="img",
                             prompt=_HISTO, orient="align", web="rich", ceiling=0.90),
    # 2nd SFM leg (genomic): Nucleotide Transformer (v2-50m) embedding of the RNA coding sequences ->
    # coding-vs-noncoding. Generalizes the ESM/protein finding to a genomic FM, and gives a within-RNA
    # contrast: the raw sequence-as-text verbalizes (rna/coding 0.84-0.96, ORF heuristic) but the SFM
    # embedding should not. ceiling = a read-out head on the NT embedding (0.918, the info IS encoded).
    "rna/nt_emb": dict(kind="emb", data="sfm_embedding/rna_nt.npz",
                       prompt=_NT, orient="align", web="zero", ceiling=0.918),
}

# Default benchmark set (the empirical output arm). Computable / reasoning-mode tasks are excluded.
CORE = list(TASKS)

# Backward-compat for eval/routing_arm.py and eval/elicit_confidence.py: the ADMET endpoint clause
# text + orientation (mirrors the admet/* tasks above).
CLAUSES = {
    "herg":         ("blocks the hERG potassium channel (cardiotoxicity risk)", "align"),
    "cyp3a4":       ("inhibits the CYP3A4 enzyme", "align"),
    "cyp2d6":       ("inhibits the CYP2D6 enzyme", "align"),
    "ames":         ("is mutagenic in the Ames test", "oppose"),
    "solubility":   ("is highly soluble in water", "oppose"),
    "permeability": ("is highly permeable across a cell membrane", "oppose"),
}


def task_items(task_id, n, rng):
    """Return (items, scrambled): items = [{id, rep, label}] balanced to n; scrambled only for
    the 'pairs' kind (used for memo_delta)."""
    t = TASKS[task_id]
    if t["kind"] == "pairs":
        rows = [json.loads(line) for line in open(os.path.join(SIGNAL, t["data"])) if line.strip()]
        matched = [r for r in rows if r.get("condition", "matched") == "matched"]
        scr = [r for r in rows if r.get("condition") == "scrambled"]
        pos = [r for r in matched if int(r["label"]) == 1]
        neg = [r for r in matched if int(r["label"]) == 0]
        k = min(n // 2, len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        items = [{"id": r.get("id"), "rep": r["representation"], "label": int(r["label"])}
                 for r in pos[:k] + neg[:k]]
        scrm = [{"id": r.get("id"), "rep": r["representation"], "label": int(r["label"])}
                for r in scr[:2 * k]]
        return items, scrm
    if t["kind"] == "emb":
        d = np.load(os.path.join(SIGNAL, t["data"]))
        emb, y = d["emb"], d["y"]
        pos = [i for i in range(len(y)) if y[i] == 1]
        neg = [i for i in range(len(y)) if y[i] == 0]
        k = min(n // 2, len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        items = [{"id": f"{task_id}:{i}", "rep": " ".join(f"{v:.3f}" for v in emb[i]),
                  "label": int(y[i])} for i in pos[:k] + neg[:k]]
        return items, []
    if t["kind"] == "image":
        rows = list(csv.DictReader(open(os.path.join(SIGNAL, t["data"]))))
        pos = [r for r in rows if r["label"] == "1"]
        neg = [r for r in rows if r["label"] == "0"]
        k = min(n // 2, len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        sel = pos[:k] + neg[:k]
        items = [{"id": f"{task_id}:{i}", "rep": "", "image": os.path.join(ROOT, r[t["col"]]),
                  "label": int(r["label"])} for i, r in enumerate(sel)]
        return items, []
    rows = list(csv.DictReader(open(os.path.join(SIGNAL, t["data"]))))
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    k = min(n // 2, len(pos), len(neg))
    rng.shuffle(pos); rng.shuffle(neg)
    sel = pos[:k] + neg[:k]
    items = [{"id": f"{task_id}:{i}", "rep": r[t["col"]], "label": int(r["label"])}
             for i, r in enumerate(sel)]
    return items, []
