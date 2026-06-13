"""Frontier OUTPUT arm for the computable-property row (companion to signal/generate_computable.py).

Same shape as output_arm_admet.py (balanced seed-42 sample, anchored parse, SYSTEM-forced bare
number, dry-run, incremental save), so the numbers are comparable to the empirical rungs. The
difference is what a miss MEANS:

  empirical rung (hERG): output near chance = the content->property mapping is not web-documented
                         (web-exposure law). Closing it needs retrieval / a specialist.
  computable rung (n_carbon): the answer is a closed-form function of the representation and the
                         exact ceiling is 1.0. Output failure here is tokenization / arithmetic
                         execution, NOT web-exposure. It is expected to close with REASONING mode
                         and representation format, not web frequency (arXiv 2505.07735).

So this arm is scored two ways:
  EXACT  : did the model compute the right value (int exact-match; float within tolerance), plus MAE
           and Spearman on the raw value. This is where the carbon-counting collapse shows up.
  AUROC  : rank the model's raw number against the binarized (median-split) label, so the rung
           lands on the same 0.5-to-ceiling axis as every other rung in results/SYNTHESIS.md.

Conditions (CMP_CONDITION): matched (canonical) | re_notation (same molecule, different valid
SMILES: a true computer is INVARIANT) | scrambled (corrupted string). CMP_PAIRED=1 also queries
matched on the same items and reports mean|delta| + correlation (the invariance / sensitivity read,
like head_to_head.py).

DRY-RUN (CMP_DRY=1 or no ANTHROPIC_API_KEY): loads pairs, balances, prints n / class balance / a
sample prompt per property, no API spend. REAL run needs ANTHROPIC_API_KEY (source ~/.api_keys).

Env: CMP_MODALITY (smiles|protein|dna, default smiles), CMP_PROP (comma list or 'all'),
CMP_N (balanced total, default 200), CMP_MODEL (default claude-sonnet-4-6), CMP_CONDITION,
CMP_PAIRED, CMP_DRY. No em dashes.
"""
import os
import re
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CMP_DIR = os.path.join(ROOT, "signal", "computable")

# property -> (human clause, kind, float tolerance (rel, abs)). Mirrors generate_computable.py.
PANEL = {
    # SMILES / RDKit
    "n_carbon":         ("how many carbon atoms are in this molecule (count every carbon, including aromatic carbons)", "int", None),
    "n_rings":          ("how many rings does this molecule have", "int", None),
    "n_aromatic_rings": ("how many aromatic rings does this molecule have", "int", None),
    "n_hbd":            ("how many hydrogen bond donors does this molecule have", "int", None),
    "n_hba":            ("how many hydrogen bond acceptors does this molecule have", "int", None),
    "n_rot_bonds":      ("how many rotatable bonds does this molecule have", "int", None),
    "n_stereo":         ("how many stereocenters (chiral centers) does this molecule have", "int", None),
    "mol_wt":           ("what is the molecular weight of this molecule in g/mol", "float", (0.02, 1.0)),
    "tpsa":             ("what is the topological polar surface area (TPSA) in square angstroms", "float", (0.10, 5.0)),
    "logp":             ("what is the Crippen logP (octanol-water partition coefficient)", "float", (0.0, 0.5)),
    # protein / Biopython
    "length":           ("how many residues are in this sequence", "int", None),
    "aromaticity":      ("what is the aromaticity (fraction of Phe, Trp, Tyr residues)", "float", (0.0, 0.03)),
    "instability_index": ("what is the instability index of this protein", "float", (0.0, 5.0)),
    "gravy":            ("what is the GRAVY (grand average of hydropathy) of this protein", "float", (0.0, 0.2)),
    "isoelectric_point": ("what is the isoelectric point (pI) of this protein", "float", (0.0, 0.3)),
    "frac_cys":         ("what fraction of residues are cysteine", "float", (0.0, 0.02)),
    # DNA
    "gc_content":       ("what is the GC content percentage of this sequence", "float", (0.0, 2.0)),
    "n_a":              ("how many adenine (A) bases are in this sequence", "int", None),
    "n_codons":         ("how many codons are in this sequence (length divided by 3)", "int", None),
}

# Regime is set by the token budget. Diagnostic finding: sonnet-4-6 will NOT emit a bare
# descriptor value. For "how many carbons / what is logP" it always opens a reasoning preamble
# ("Let me count the carbons systematically...") and, capped at 16 tokens, never reaches a
# number (near-total fallback). That is itself the result: unlike an empirical gestalt (a hERG
# probability), a computable property cannot be verbalized as a snap answer, it has to be
# EXECUTED. So two regimes:
#   snap   (CMP_MAXTOK<=32): the value is unobtainable as a bare number; the FALLBACK RATE is
#          the metric (high = cannot answer without scratch space).
#   reason (CMP_MAXTOK>32, default 512): give it room, parse the FINAL number = "can it compute
#          WITH reasoning". This is the thesis test (reasoning, not web frequency, closes it).
MAXTOK = int(os.environ.get("CMP_MAXTOK", "512"))
SYSTEM_SNAP = (
    "You are a precise calculator for molecular and sequence descriptors. Respond with ONLY a "
    "single number (an integer or a decimal). No words, no units, no explanation."
)
SYSTEM_REASON = (
    "You compute molecular and sequence descriptors from the given representation. Work it out, "
    "reasoning step by step if needed, but END your reply with the final numeric answer on its "
    "own line and write nothing after it."
)
SYSTEM = SYSTEM_SNAP if MAXTOK <= 32 else SYSTEM_REASON


def repr_label(modality):
    return "SMILES" if modality == "smiles" else "Sequence"


def build_prompt(modality, clause, rep):
    return (f"You are given a molecule as a SMILES string. Working only from the structure, {clause}? "
            f"Reply with ONLY the number.\n\n{repr_label(modality)}: {rep}\nAnswer:") if modality == "smiles" else (
            f"You are given a biological sequence. Working only from the sequence, {clause}? "
            f"Reply with ONLY the number.\n\n{repr_label(modality)}: {rep}\nAnswer:")


def parse_number(txt):
    """Anchored: last number in the text, returned as-is (no [0,1] clamp). Handles a leading
    minus (logP, GRAVY can be negative) and scientific-free decimals."""
    for tok in reversed(re.findall(r"-?\d*\.?\d+", txt)):
        try:
            return float(tok), "parsed"
        except ValueError:
            continue
    return float("nan"), "fallback"


def load_pairs(modality, prop, condition, n):
    """Balanced (n/2 above-median + n/2 below), seed 42. Returns list of (rep, value, label, id)."""
    path = os.path.join(CMP_DIR, modality, prop, "pairs.jsonl")
    if not os.path.isfile(path):
        return None, path
    by_label = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("condition") != condition:
            continue
        by_label[int(r["label"])].append((r["representation"], float(r["value"]), int(r["label"]), r["id"]))
    rng = np.random.RandomState(42)
    pos, neg = by_label[1][:], by_label[0][:]
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k], path


def matched_by_id(modality, prop):
    """id -> representation for condition=matched (for the paired delta)."""
    path = os.path.join(CMP_DIR, modality, prop, "pairs.jsonl")
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("condition") == "matched":
            out[r["id"]] = r["representation"]
    return out


def ask(client, model, modality, clause, rep):
    try:
        msg = client.messages.create(
            model=model, max_tokens=MAXTOK, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(modality, clause, rep)}],
        )
    except Exception:
        return float("nan"), "error"   # transient API error: NaN drops the item, run survives
    texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not texts:
        return float("nan"), "empty"
    # CRITICAL (reason regime): a trace cut off at max_tokens ends mid-enumeration, so its LAST
    # number is a partial count, not the answer. Parsing it manufactures a wrong value and
    # masquerades truncation as inability. Mark truncated -> NaN ("truncated") instead. Diagnostic
    # 2026-06-13: n_carbon @400 all truncated (parsed 2/3/4 vs true 22/24); @2000 finished -> 22/23.
    if MAXTOK > 32 and getattr(msg, "stop_reason", None) == "max_tokens":
        return float("nan"), "truncated"
    return parse_number(texts[0])


def is_exact(pred, true, kind, tol):
    if not np.isfinite(pred):
        return False
    if kind == "int":
        return round(pred) == round(true)
    rel, ab = tol if tol else (0.0, 0.0)
    return abs(pred - true) <= max(ab, rel * abs(true))


def score(preds, vals, labels, kind, tol):
    preds, vals, labels = np.asarray(preds), np.asarray(vals), np.asarray(labels)
    fin = np.isfinite(preds)
    out = {"n_scored": int(len(preds)), "n_parsed": int(fin.sum())}
    if fin.sum() < 3:
        return out
    p, v, y = preds[fin], vals[fin], labels[fin]
    out["exact_acc"] = round(float(np.mean([is_exact(pi, vi, kind, tol) for pi, vi in zip(p, v)])), 3)
    out["mae"] = round(float(np.mean(np.abs(p - v))), 4)
    if spearmanr is not None and len(set(v)) > 1 and len(set(p)) > 1:
        out["spearman"] = round(float(spearmanr(p, v).correlation), 3)
    # AUROC: rank the raw prediction against the binarized label (mean-impute fallbacks so they
    # do not distort ranks), comparable to every other rung
    p_full = np.where(fin, preds, np.nanmean(p))
    if len(set(labels)) > 1:
        out["auroc"] = round(float(roc_auc_score(labels, p_full)), 3)
    return out


def run_prop(modality, prop, n, model, client, condition, paired):
    clause, kind, tol = PANEL[prop]
    data, path = load_pairs(modality, prop, condition, n)
    if data is None:
        return {"property": prop, "error": f"no pairs.jsonl at {path} (run generate_computable.py)"}
    if not data:
        return {"property": prop, "error": f"no records for condition={condition}"}
    sample_prompt = build_prompt(modality, clause, data[0][0])

    if client is None:  # dry-run
        ys = [d[2] for d in data]
        return {"property": prop, "dry_run": True, "kind": kind, "condition": condition,
                "n_scored": len(data), "pos": int(sum(ys)), "neg": int(len(ys) - sum(ys)),
                "sample_prompt": sample_prompt[:160]}

    matched_map = matched_by_id(modality, prop) if (paired and condition != "matched") else None
    preds, vals, labels, kinds = [], [], [], []
    dpred = []  # paired matched predictions for delta
    for i, (rep, val, lab, _id) in enumerate(data):
        pr, kd = ask(client, model, modality, clause, rep)
        preds.append(pr); vals.append(val); labels.append(lab); kinds.append(kd)
        if matched_map is not None and _id in matched_map:
            mp, _ = ask(client, model, modality, clause, matched_map[_id])
            dpred.append((mp, pr))
        if (i + 1) % 25 == 0:
            print(f"  [{prop}] {i+1}/{len(data)}", flush=True)

    res = {"property": prop, "model": model, "modality": modality, "kind": kind,
           "condition": condition, "max_tokens": MAXTOK, "parse": dict(Counter(kinds)),
           **score(preds, vals, labels, kind, tol)}
    if dpred:
        a = np.array([d[0] for d in dpred]); b = np.array([d[1] for d in dpred])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 2:
            res["paired_vs_matched"] = {
                "mean_abs_delta": round(float(np.abs(a[m] - b[m]).mean()), 4),
                "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 3),
                "note": "re_notation: delta~0 = invariant (good); scrambled: delta large = structure-sensitive (good)",
            }
    return res


def main():
    modality = os.environ.get("CMP_MODALITY", "smiles")
    prop_env = os.environ.get("CMP_PROP", "all").strip()
    n = int(os.environ.get("CMP_N", "200"))
    model = os.environ.get("CMP_MODEL", "claude-sonnet-4-6")
    condition = os.environ.get("CMP_CONDITION", "matched")
    paired = os.environ.get("CMP_PAIRED", "0") == "1"
    dry = os.environ.get("CMP_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")

    # default props = whatever was generated for this modality
    moddir = os.path.join(CMP_DIR, modality)
    avail = sorted(d for d in os.listdir(moddir)) if os.path.isdir(moddir) else []
    if prop_env == "all":
        props = [p for p in avail if p in PANEL] or [p for p in PANEL]
    else:
        props = [p.strip() for p in prop_env.split(",")]

    client = None
    if not dry:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    mode = "DRY-RUN (no API call)" if dry else f"REAL run, model={model}"
    print(f"== computable output arm :: {mode} :: modality={modality} :: condition={condition} "
          f":: props={props} :: N={n} ==\n")

    regime = "snap" if MAXTOK <= 32 else "reason"
    suffix = "" if condition == "matched" else f"_{condition}"
    outpath = os.path.join(ROOT, "results", f"output_arm_computable_{modality}_{regime}{suffix}.json")
    merged = {}
    if not dry and os.path.isfile(outpath):
        for r in json.load(open(outpath)):
            merged[r["property"]] = r

    for prop in props:
        if prop not in PANEL:
            print(f"[{prop}] not in PANEL, skip"); continue
        if not dry and merged.get(prop, {}).get("n_scored", 0) >= n and os.environ.get("CMP_FORCE") != "1":
            print(f"[{prop}] already done (n={merged[prop]['n_scored']}), skip"); continue
        r = run_prop(modality, prop, n, model, client, condition, paired)
        if r.get("error"):
            print(f"[{prop}] {r['error']}"); continue
        if r.get("dry_run"):
            print(f"[{prop:16s}] n={r['n_scored']} (pos={r['pos']} neg={r['neg']}) kind={r['kind']}")
            print(f"      prompt: {r['sample_prompt']}...\n")
            continue
        line = (f"[{prop:16s}] exact={r.get('exact_acc')} mae={r.get('mae')} "
                f"spearman={r.get('spearman')} AUROC={r.get('auroc')} parse={r['parse']}")
        if r.get("paired_vs_matched"):
            line += f"  delta={r['paired_vs_matched']['mean_abs_delta']} corr={r['paired_vs_matched']['corr']}"
        print(line, flush=True)
        merged[prop] = r
        with open(outpath, "w") as f:
            json.dump([merged[k] for k in sorted(merged)], f, indent=2)
        print(f"      saved -> {outpath} ({len(merged)} props)")


if __name__ == "__main__":
    main()
