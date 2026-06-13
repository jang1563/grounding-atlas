"""Computable reason-regime SCALE sweep = the recovery curve (deferred item c).

Question 1 (recovery): does scale/reasoning close the computable gaps the way it closes web-rich
EMPIRICAL gaps (PROJECT_DESIGN 7.3, frontier_output_panel)? Counting (n_carbon) is already
AUROC 1.0 at sonnet given room; does it hold down to haiku and up to opus.
Question 2 (the seam): does the pI seam (sonnet AUROC 0.79) CLOSE with a stronger model, or is it
constant-recall-bound and therefore scale-invariant (only a tool closes it, cf. bridge_test_pi)?
This is the computable analog of the DNA-closes / spectra-invariant split in 7.3.

Baseline prompt (NO pKa supplied): we already know constants help pI (bridge_test_pi.py); here we
ask whether SCALE ALONE helps. Uniform N + token budget + truncation guard across models, so the
curve is clean. Reuses output_arm_computable helpers. Writes results/computable_scale_sweep.json.
No em dashes. Env: CSW_N (10), CSW_MAXTOK (4000), CSW_MODELS, CSW_PROPS, CSW_DRY.
"""
import json
import os
from collections import Counter

import output_arm_computable as oac  # sibling import (run as: python eval/computable_scale_sweep.py)

MODELS = os.environ.get("CSW_MODELS",
                        "claude-haiku-4-5-20251001,claude-sonnet-4-6,claude-opus-4-8").split(",")
PROPS = os.environ.get("CSW_PROPS", "n_carbon,isoelectric_point").split(",")
N = int(os.environ.get("CSW_N", "10"))
MAXTOK = int(os.environ.get("CSW_MAXTOK", "4000"))
DRY = os.environ.get("CSW_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
PROP_MOD = {"n_carbon": "smiles", "mol_wt": "smiles", "n_hbd": "smiles",
            "isoelectric_point": "protein", "length": "protein",
            "gc_content": "dna", "n_a": "dna"}


def ask(client, model, modality, clause, rep):
    try:
        m = client.messages.create(model=model, max_tokens=MAXTOK, system=oac.SYSTEM_REASON,
                                   messages=[{"role": "user", "content": oac.build_prompt(modality, clause, rep)}])
    except Exception:
        return float("nan"), "error"
    texts = [b.text for b in m.content if getattr(b, "type", None) == "text"]
    if not texts:
        return float("nan"), "empty"
    if getattr(m, "stop_reason", None) == "max_tokens":
        return float("nan"), "truncated"   # same guard as output_arm_computable
    return oac.parse_number(texts[0])


def main():
    print(f"== computable SCALE sweep :: {'DRY' if DRY else 'REAL'} :: models={MODELS} :: "
          f"props={PROPS} :: N={N} :: MAXTOK={MAXTOK} ==")
    out = os.path.join(oac.ROOT, "results", "computable_scale_sweep.json")
    rows = json.load(open(out)) if (not DRY and os.path.isfile(out)) else []
    done = {(r["model"], r["property"]) for r in rows}

    client = None
    if not DRY:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for prop in PROPS:
        mod = PROP_MOD.get(prop)
        clause, kind, tol = oac.PANEL[prop]
        data, path = oac.load_pairs(mod, prop, "matched", N)
        if not data:
            print(f"[{prop}] no pairs at {path}, skip"); continue
        for model in MODELS:
            if (model, prop) in done and os.environ.get("CSW_FORCE") != "1":
                print(f"[{model} / {prop}] already done, skip"); continue
            if DRY:
                print(f"[{model} / {prop}] would score n={len(data)} ({mod}, {kind})"); continue
            preds, vals, labels, kinds = [], [], [], []
            for seq_or_smi, val, lab, _id in data:
                p, k = ask(client, model, mod, clause, seq_or_smi)
                preds.append(p); vals.append(val); labels.append(lab); kinds.append(k)
            rec = {"model": model, "property": prop, "modality": mod, "kind": kind,
                   "max_tokens": MAXTOK, "parse": dict(Counter(kinds)),
                   **oac.score(preds, vals, labels, kind, tol)}
            pa = rec["parse"]; ok = pa.get("parsed", 0); tot = sum(pa.values()) or 1
            print(f"[{model:28s} / {prop:17s}] compl={ok}/{tot} exact={rec.get('exact_acc')} "
                  f"mae={rec.get('mae')} AUROC={rec.get('auroc')}", flush=True)
            rows = [r for r in rows if not (r["model"] == model and r["property"] == prop)] + [rec]
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)

    if not DRY:
        print(f"\nsaved -> {out}")
        print("\n=== recovery curve (AUROC by model x property) ===")
        order = {"claude-haiku-4-5-20251001": 0, "claude-sonnet-4-6": 1, "claude-opus-4-8": 2}
        for prop in PROPS:
            line = "  " + f"{prop:18s}"
            for model in sorted(MODELS, key=lambda m: order.get(m, 9)):
                r = next((x for x in rows if x["model"] == model and x["property"] == prop), None)
                line += f" {model.split('-')[1]}={r.get('auroc') if r else '?'}"
            print(line)


if __name__ == "__main__":
    main()
