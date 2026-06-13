"""T2-SOLVE solo arm, generalized over the 7 WS2 ADMET endpoints.

head_to_head.py is hERG-hardcoded (one prompt, the NegBioDB sqlite). This reads the
modality-general WS2 output instead (signal/admet/<ep>/pairs.jsonl, condition=matched)
and runs the LLM-output arm per endpoint, so the SOLVE-mode coverage in
results/t2_apply.md extends from 2 of 7 endpoints to all 7 at a near-fixed ceiling.

Scoring and parsing are identical to head_to_head.py / activation_arm.py (last-number
anchored parse, balanced sample, seed 42), so the new numbers are comparable to the
hERG anchor.

DRY-RUN (no API key, no spend): ADMET_DRY=1 loads the data, balances, builds prompts and
prints n / class balance / a sample prompt per endpoint, validating everything except
the model call. REAL run needs ANTHROPIC_API_KEY (source <your-keys-file>).

Env: ADMET_EP (comma list or 'all', default the 5 unmeasured), ADMET_N (balanced total,
default 200 = 100/100), ADMET_MODEL (default claude-sonnet-4-6), ADMET_DRY.
No em dashes.
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADMET = os.path.join(ROOT, "signal", "admet")

# endpoint -> (property clause for the prompt, clause_matches_label1)
# label-1 = NegBioDB FAIL (generate_signal.py). Direction RESOLVED against assay-value
# medians in negbiodb_admet.db (outcome vs standard_value): a hERG/CYP/AMES fail is the
# active/positive outcome (clause aligned), but a solubility/permeability fail is the
# LOW-value compound (insoluble fail 1.1 vs pass 313 ug/mL; impermeable fail 71 vs pass
# 579), so the "soluble/permeable" clause OPPOSES label-1 there -> oriented AUROC = 1-raw
# (handled in post-processing; see results/t2_apply.md R3 + results/output_arm_admet.json).
# Clearance pools heterogeneous units (fail 26.6% vs pass 22 mL/min/kg), left unresolved.
PROPERTY = {
    "herg":         ("BLOCKS the hERG potassium channel (cardiotoxicity risk)", True),
    "cyp3a4":       ("INHIBITS the CYP3A4 enzyme", True),
    "cyp2d6":       ("INHIBITS the CYP2D6 enzyme", True),
    "ames":         ("is MUTAGENIC in the Ames test", True),
    "solubility":   ("is highly soluble in water", False),    # clause opposes label-1 (fail=insoluble)
    "permeability": ("is highly permeable across a cell membrane", False),  # clause opposes label-1 (fail=impermeable)
    "clearance":    ("has high metabolic clearance", None),   # heterogeneous units, unresolved
}
DEFAULT_EPS = ["cyp2d6", "ames", "solubility", "permeability", "clearance"]  # the 5 unmeasured

PROMPT = (
    "You are given a molecule as a SMILES string. Estimate the probability "
    "(a single number between 0 and 1) that it {clause}. Judge only from the "
    "structure. Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:"
)
# Without a system constraint, sonnet-4-6 ignores "only the number" and emits a long
# reasoning preamble that never reaches a number within the token budget (fallback rate
# up to 96% on permeability). This system message forces a bare number (validated: ~90%
# compliance vs 4-67% before). Note: this model does not support assistant prefill.
SYSTEM = (
    "You are a molecular property predictor. Respond with ONLY a single decimal number "
    "between 0 and 1 (for example: 0.42). No words, no explanation, no analysis, no units. "
    "Your entire reply must be just the number."
)


def parse_prob(txt):
    """Anchored: last number; [0,1] direct, (1,100] as percent. Same as head_to_head.py."""
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "parsed"
        if 1.0 < v <= 100.0:
            return v / 100.0, "percent"
    return 0.5, "fallback"


def load_matched(ep, n):
    """Balanced (n/2 pos + n/2 neg) sample, seed 42. ADMET_CONDITION (default 'matched';
    set 're_notation' or 'scrambled' for the grounding-vs-memorization control)."""
    cond = os.environ.get("ADMET_CONDITION", "matched")
    path = os.path.join(ADMET, ep, "pairs.jsonl")
    by_label = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("condition") != cond:
                continue
            by_label[int(r["label"])].append(r["representation"])
    rng = np.random.RandomState(42)
    pos, neg = by_label[1][:], by_label[0][:]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = [(s, 1) for s in pos[:k]] + [(s, 0) for s in neg[:k]]
    return data, len(pos), len(neg), k


def run_endpoint(ep, n, model, client):
    clause, verified = PROPERTY[ep]
    data, n_pos, n_neg, k = load_matched(ep, n)
    sample_prompt = PROMPT.format(clause=clause, smiles=data[0][0])

    if client is None:  # dry-run
        return {
            "endpoint": ep, "dry_run": True, "clause": clause,
            "clause_matches_label1": verified,
            "pool_pos": n_pos, "pool_neg": n_neg, "balanced_per_class": k,
            "n_scored": 2 * k, "sample_prompt": sample_prompt,
        }

    y, p, kinds = [], [], []
    for i, (smi, label) in enumerate(data):
        msg = client.messages.create(
            model=model, max_tokens=16, system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT.format(clause=clause, smiles=smi)}],
        )
        texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        val, kind = parse_prob(texts[0]) if texts else (0.5, "empty")
        y.append(label); p.append(val); kinds.append(kind)
        if (i + 1) % 25 == 0:
            print(f"  [{ep}] {i+1}/{len(data)}")
    y, p = np.array(y), np.array(p)
    two_class = len(set(y)) > 1
    auroc = float(roc_auc_score(y, p)) if two_class else None
    # oriented = how well solo reads the TRUE property; flip when the clause opposes label-1
    oriented = None if auroc is None else round(1 - auroc if verified is False else auroc, 3)
    return {
        "endpoint": ep, "model": model, "n_scored": len(y),
        "balanced_per_class": k, "clause_matches_label1": verified, "clause": clause,
        "auroc": round(auroc, 3) if auroc is not None else None,
        "oriented_auroc": oriented,
        "auprc": round(float(average_precision_score(y, p)), 3) if two_class else None,
        "parse": dict(Counter(kinds)),
    }


def main():
    ep_env = os.environ.get("ADMET_EP", "").strip()
    eps = DEFAULT_EPS if not ep_env else (
        list(PROPERTY) if ep_env == "all" else [e.strip() for e in ep_env.split(",")]
    )
    n = int(os.environ.get("ADMET_N", "200"))
    model = os.environ.get("ADMET_MODEL", "claude-sonnet-4-6")
    dry = os.environ.get("ADMET_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")

    client = None
    if not dry:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    mode = "DRY-RUN (no API call)" if dry else f"REAL run, model={model}"
    print(f"== T2-SOLVE ADMET output arm :: {mode} :: endpoints={eps} :: N={n} ==\n")

    force = os.environ.get("ADMET_FORCE", "0") == "1"
    _cond = os.environ.get("ADMET_CONDITION", "matched")
    outpath = os.path.join(ROOT, "results",
                           f"output_arm_admet{'' if _cond == 'matched' else '_' + _cond}.json")
    merged = {}  # endpoint -> result, persisted after EACH endpoint (kill-safe + resumable)
    if not dry and os.path.isfile(outpath):
        with open(outpath) as fh:
            for r in json.load(fh):
                merged[r["endpoint"]] = r

    for ep in eps:
        if not dry and not force and merged.get(ep, {}).get("n_scored", 0) >= n:
            print(f"[{ep}] already done (n={merged[ep]['n_scored']}, AUROC={merged[ep].get('auroc')}), skip")
            continue
        r = run_endpoint(ep, n, model, client)
        flag = "" if r.get("clause_matches_label1") else "  [!] clause opposes/unresolved vs label-1 (use oriented_auroc)"
        if r.get("dry_run"):
            print(f"[{ep}] pool pos={r['pool_pos']} neg={r['pool_neg']} -> balanced "
                  f"{r['balanced_per_class']}/class, n_scored={r['n_scored']}{flag}")
            print(f"      prompt: {r['sample_prompt'][:110]}...\n")
            continue
        print(f"[{ep}] n={r['n_scored']} AUROC={r['auroc']} oriented={r['oriented_auroc']} "
              f"AUPRC={r['auprc']} parse={r['parse']}{flag}")
        merged[ep] = r  # incremental save: write the json now, before the next endpoint
        with open(outpath, "w") as fh:
            json.dump([merged[k] for k in sorted(merged)], fh, indent=2)
        print(f"      saved -> {outpath} ({len(merged)} endpoints)")


if __name__ == "__main__":
    main()
