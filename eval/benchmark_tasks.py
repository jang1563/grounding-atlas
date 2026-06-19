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

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNAL = os.path.join(os.path.dirname(HERE), "signal")


def _mol(clause):
    return ("Estimate the probability (a single number between 0 and 1) that this molecule "
            f"{clause}. Judge only from the representation below.\nsmiles: {{rep}}\nProbability:")


def _cell(pos, neg):
    return ("These are the most distinctively expressed genes of one human immune cell (PBMC), "
            f"highest first:\n{{rep}}\nIs it a {pos} or a {neg}? Reply with ONLY a number between "
            f"0 and 1 = the probability that it is a {pos}.")


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
    rows = list(csv.DictReader(open(os.path.join(SIGNAL, t["data"]))))
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    k = min(n // 2, len(pos), len(neg))
    rng.shuffle(pos); rng.shuffle(neg)
    sel = pos[:k] + neg[:k]
    items = [{"id": f"{task_id}:{i}", "rep": r[t["col"]], "label": int(r["label"])}
             for i, r in enumerate(sel)]
    return items, []
