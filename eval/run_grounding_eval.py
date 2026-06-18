"""grounding-atlas-eval: model-agnostic grounding + calibration + memorization-transparency
harness (docs/BENCHMARK_DESIGN.md).

Dataset / Solver / Scorer (Inspect-style), reproducible per EleutherAI "Lessons from the
Trenches": one versioned prompt, fixed decode, raw outputs + manifest saved, every metric
with a bootstrap CI, no single-number reduction. GPU-free output arm.

Run:
  python eval/run_grounding_eval.py --dry-run                      # no API, validates pipeline
  python eval/run_grounding_eval.py --model claude-opus-4-8 --rungs admet/herg,computable/smiles/n_carbon --n 80
  python eval/run_grounding_eval.py --model gpt-4o --rungs all --n 120
"""
import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIGNAL = os.path.join(ROOT, "signal")
OUT = os.path.join(ROOT, "results", "benchmark")

PROMPT_VERSION = "v3"   # v3: directional clause + a priori orientation (v2 = system + anchor)
DECODE = {"temperature": 0.0, "max_tokens": 16}

# Without a system constraint a reasoning model ignores "only the number" and emits a preamble
# that never reaches a number in the token budget (validated in eval/output_arm_admet.py: up to
# 96% fallback). This forces a bare number (~90% compliance). The model must not use prefill.
SYSTEM = (
    "You are a property predictor. Respond with ONLY a single decimal number between 0 and 1 "
    "(for example: 0.42). No words, no explanation, no analysis, no units. Your entire reply "
    "must be just the number."
)

# A grounding benchmark must anchor directionality: a bare property name ("ames") does not tell
# the model which way is positive, and the AUROC sign is then meaningless. Each empirical
# endpoint gets a directional clause and an a priori orientation (ported from
# eval/output_arm_admet.py). label-1 = NegBioDB FAIL; an hERG/CYP/AMES fail IS the active
# (positive) outcome so the clause "align"s with label-1, but a solubility/permeability fail is
# the LOW-value compound so the natural clause "oppose"s label-1 (oriented label = 1 - label).
# clearance is omitted: heterogeneous units leave its orientation unresolved.
CLAUSES = {
    "herg":         ("blocks the hERG potassium channel (cardiotoxicity risk)", "align"),
    "cyp3a4":       ("inhibits the CYP3A4 enzyme", "align"),
    "cyp2d6":       ("inhibits the CYP2D6 enzyme", "align"),
    "ames":         ("is mutagenic in the Ames test", "align"),
    "solubility":   ("is highly soluble in water", "oppose"),
    "permeability": ("is highly permeable across a cell membrane", "oppose"),
}


def prompt_for(item):
    """Versioned output-arm prompt. Computable rungs ask about a threshold, empirical ones about
    presence of the property. Ends with a `Probability:` anchor so the model emits the number."""
    mod = item.get("modality", "input")
    prop = item.get("property", "the target property")
    rep = item["representation"]
    if item.get("threshold") not in (None, ""):
        q = f"that the {prop} of this {mod} exceeds {item['threshold']}"
    elif prop in CLAUSES:
        q = f"that this molecule {CLAUSES[prop][0]}"
    else:
        q = f"that this {mod} has the property: {prop}"
    return (f"Estimate the probability (a single number between 0 and 1) {q}. "
            f"Judge only from the representation below.\n{mod}: {rep}\nProbability:")


# ---------- Dataset ----------

def rung_paths():
    paths = {}
    for p in glob.glob(os.path.join(SIGNAL, "**", "pairs.jsonl"), recursive=True):
        if "sfm_embedding" in p:
            continue
        rung = os.path.relpath(os.path.dirname(p), SIGNAL)
        paths[rung] = p
    return paths


def load_rung(path, n, rng):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    matched = [r for r in rows if r.get("condition", "matched") == "matched"]
    scr = [r for r in rows if r.get("condition") == "scrambled"]
    # balanced sample of matched items, plus any scrambled (for memo_delta)
    pos = [r for r in matched if int(r["label"]) == 1]
    neg = [r for r in matched if int(r["label"]) == 0]
    k = min(n // 2, len(pos), len(neg))
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:k] + neg[:k], scr[: 2 * k]


# ---------- Solver (model-agnostic) ----------

def complete(model, prompt):
    if model.startswith(("claude", "anthropic")):
        import anthropic
        client = anthropic.Anthropic()
        kw = dict(model=model, max_tokens=DECODE["max_tokens"], system=SYSTEM,
                  messages=[{"role": "user", "content": prompt}])
        try:
            m = client.messages.create(temperature=DECODE["temperature"], **kw)
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            m = client.messages.create(**kw)   # newer models (e.g. opus-4-8) deprecate temperature
        return "".join(b.text for b in m.content if b.type == "text")
    if model.startswith(("gpt", "o1", "o3")):
        import openai
        r = openai.OpenAI().chat.completions.create(
            model=model, max_tokens=DECODE["max_tokens"], temperature=DECODE["temperature"],
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    raise SystemExit(f"unknown model provider for {model}")


def parse_prob(text):
    """Anchored parse: take the LAST number (the answer follows the preamble); a value in [0,1]
    is a probability, (1,100] is read as a percent. Falls back to 0.5 when nothing parses."""
    for tok in reversed(re.findall(r"\d*\.?\d+", text or "")):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def solve(model, items, dry, rng):
    """Returns (probs, raw_texts). The raw text is saved so anyone can re-score."""
    probs, texts = [], []
    for it in items:
        if dry:
            p = min(1.0, max(0.0, 0.30 + 0.40 * int(it["label"]) + rng.normal(0, 0.18)))
            probs.append(p); texts.append(f"{p:.3f} (dry)")
        else:
            t = complete(model, prompt_for(it))
            texts.append(t); probs.append(parse_prob(t))
    return np.array(probs), texts


# ---------- Scorer ----------

def ece(prob, y, bins=10):
    conf = np.maximum(prob, 1 - prob)
    pred = (prob > 0.5).astype(int)
    correct = (pred == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def aurc(prob, y):
    conf = np.maximum(prob, 1 - prob)
    err = ((prob > 0.5).astype(int) != y).astype(float)
    order = np.argsort(-conf)
    risks = np.cumsum(err[order]) / (np.arange(len(err)) + 1)
    return float(risks.mean())


def sel_acc(prob, y, cov=0.5):
    conf = np.maximum(prob, 1 - prob)
    k = max(1, int(len(y) * cov))
    top = np.argsort(-conf)[:k]
    return float(((prob[top] > 0.5).astype(int) == y[top]).mean())


def auroc(prob, y):
    return roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")


def ci(fn, prob, y, rng, b=500):
    vals = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(prob[idx], y[idx]))
        except Exception:
            pass
    lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return round(float(lo), 3), round(float(hi), 3)


def score_rung(prob, y, scr_prob, scr_y, ceiling, rng):
    a = auroc(prob, y)
    rec = {
        "n": len(y),
        "output_auroc": round(a, 3), "output_auroc_ci": ci(auroc, prob, y, rng),
        "ece": round(ece(prob, y), 3), "aurc": round(aurc(prob, y), 3),
        "sel_acc_50": round(sel_acc(prob, y), 3),
        "ceiling": ceiling, "gap": round(ceiling - a, 3) if ceiling is not None else None,
        "memo_delta": None,
    }
    if scr_y is not None and len(scr_y) and len(set(scr_y)) > 1:
        rec["memo_delta"] = round(a - auroc(scr_prob, scr_y), 3)   # matched - scrambled
    return rec


def update_leaderboard():
    """Aggregate every committed (non-dry) scorecard into LEADERBOARD.md. Stratified, never a
    single number."""
    models = []
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        sc = os.path.join(d, "scorecard.json")
        name = os.path.basename(d)
        if name != "dry" and os.path.isfile(sc):
            models.append((name, json.load(open(sc))))
    out = ["# grounding-atlas-eval leaderboard", "",
           "Per-rung output-arm AUROC (matched condition), 95% bootstrap CI. No single-number "
           "reduction: read grounding (AUROC, gap to ceiling), calibration (ECE / AURC), and "
           "memorization-transparency (`memo_delta` = AUROC(matched) - AUROC(scrambled)) "
           "together. Generated by `eval/run_grounding_eval.py`.", ""]
    if not models:
        out.append("_No models scored yet. Run "
                   "`python eval/run_grounding_eval.py --model <id> --rungs all`._")
    else:
        rungs = sorted({r for _, sc in models for r in sc})
        out += ["| rung | " + " | ".join(f"{m} AUROC [CI]" for m, _ in models) + " |",
                "|" + "---|" * (len(models) + 1)]
        for rung in rungs:
            cells = []
            for _, sc in models:
                r = sc.get(rung)
                cells.append(f"{r['output_auroc']} {tuple(r['output_auroc_ci'])}" if r else "-")
            out.append(f"| `{rung}` | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "LEADERBOARD.md"), "w").write("\n".join(out) + "\n")


# ---------- harness ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dry")
    ap.add_argument("--rungs", default="admet/herg,computable/smiles/n_carbon")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    paths = rung_paths()
    want = sorted(paths) if args.rungs == "all" else [r.strip() for r in args.rungs.split(",")]
    ceilings = {}
    cf = os.path.join(OUT, "ceilings.json")
    if os.path.exists(cf):
        # ceilings.json values may be a bare float or a {ceiling, method, n} provenance dict
        ceilings = {k: (v["ceiling"] if isinstance(v, dict) else v)
                    for k, v in json.load(open(cf)).items()}

    model_dir = os.path.join(OUT, args.model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    scorecard, raw = {}, []
    for rung in want:
        if rung not in paths:
            print(f"  skip {rung} (no pairs.jsonl)"); continue
        items, scr = load_rung(paths[rung], args.n, rng)
        prob, texts = solve(args.model, items, args.dry_run, rng)
        prop = items[0].get("property") if items else None
        orient = CLAUSES.get(prop, (None, "align"))[1]   # 'oppose': positive class is label 0
        y = np.array([int(it["label"]) for it in items])
        sprob = solve(args.model, scr, args.dry_run, rng)[0] if scr else None
        sy = np.array([int(it["label"]) for it in scr]) if scr else None
        if orient == "oppose":   # model emits P(clause); orient the label to the clause
            y = 1 - y
            sy = None if sy is None else 1 - sy
        scorecard[rung] = score_rung(prob, y, sprob, sy, ceilings.get(rung), rng)
        scorecard[rung]["orientation"] = orient
        for i, (it, p, t) in enumerate(zip(items, prob, texts)):
            raw.append({"rung": rung, "id": it.get("id", f"{rung}:{i}"),
                        "label": int(it["label"]), "prob": round(float(p), 4), "output": t})
        r = scorecard[rung]
        print(f"  {rung:30s} AUROC={r['output_auroc']} {r['output_auroc_ci']} ECE={r['ece']} "
              f"AURC={r['aurc']} memo_delta={r['memo_delta']}")

    manifest = {
        "model": args.model, "prompt_version": PROMPT_VERSION, "decode": DECODE,
        "seed": args.seed, "n_per_rung": args.n, "dry_run": args.dry_run,
        "data_commit": subprocess.getoutput("git rev-parse --short HEAD"),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "rungs": list(scorecard),
    }
    json.dump(scorecard, open(os.path.join(model_dir, "scorecard.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(model_dir, "manifest.json"), "w"), indent=2)
    with open(os.path.join(model_dir, "raw.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    update_leaderboard()
    print(f"\nwrote {model_dir}/ (scorecard, manifest, raw)  [commit {manifest['data_commit']}, prompt {PROMPT_VERSION}]")
    print("no single-number reduction by design; read per-rung AUROC, calibration, and memo_delta together.")


if __name__ == "__main__":
    main()
