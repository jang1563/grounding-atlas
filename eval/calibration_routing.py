"""Calibration / routing benchmark: does Claude KNOW when it cannot ground, and route?

The grounded-orchestrator decision function. Across rungs spanning the web-exposure spectrum,
Claude is asked, per item, for three things at once: PRED (probability the property holds),
CONF (self-reported confidence that PRED is reliable from the representation), and ROUTE
(SELF = answer myself, DEFER = a specialist tool is more reliable). We then test whether CONF
and DEFER track ACTUAL grounding (the output AUROC), i.e. whether Claude is calibrated about its
own limits. If web-zero rungs draw low CONF + high DEFER and web-rich rungs draw high CONF +
SELF, AND that matches where Claude is actually right, then calibrated routing works and an
orchestrator can be built on it. Over-confidence on a web-zero rung (high CONF, chance AUROC)
is the orchestrator's failure mode. source ~/.api_keys. No em dashes.
"""
import os
import re
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("CR_N", "80"))
WORKERS = int(os.environ.get("CR_WORKERS", "8"))
MODELS = os.environ.get("CR_MODELS", "claude-opus-4-8,claude-haiku-4-5-20251001").split(",")

SYSTEM = ("You are the reasoning core of a tool-augmented scientific analysis system. For every item a "
          "trained SPECIALIST model is available for that data type (a cheminformatics model, a genomics "
          "model, an epigenetic clock, and so on). Your job is to give your best prediction, honestly rate "
          "how reliable it is, and decide whether to answer yourself or defer to the specialist. Be honest: "
          "deferring when you genuinely cannot read the representation is correct behavior, not failure.")

# rung: (csv, repr_field, label_field, property question, ceiling, web tag). label_field age handled below.
HERG_Q = "this molecule BLOCKS the hERG potassium channel (cardiotoxicity)"
SC_Q = "this cell is a T cell (based on its gene expression)"
RUNGS = {
    "dna_promoter":   ("signal/dna_promoter.csv",        "smiles",        "label", "this DNA sequence is a PROMOTER (a regulatory region that initiates transcription)", 0.889, "web-rich"),
    "msa_conserv":    ("signal/msa/msa_conservation.csv", "column",        "label", "this protein multiple-sequence-alignment column is evolutionarily CONSERVED", 0.999, "web-rich"),
    "sc_cellsentence":("signal/single_cell/pbmc_Tcell.csv","cell_sentence","label", SC_Q, 0.989, "web-rich"),
    "sc_anon":        ("signal/single_cell/pbmc_Tcell.csv","anon",         "label", SC_Q, 0.989, "web-zero"),
    "smiles_herg":    ("signal/nmr/herg_nmr.csv",         "smiles",        "label", HERG_Q, 0.825, "web-zero-out"),
    "nmr_herg":       ("signal/nmr/herg_nmr.csv",         "nmr",           "label", HERG_Q, 0.866, "web-zero"),
    "methyl_age":     ("signal/methyl/methyl_age.csv",    "beta_text",     "label", "this blood sample is from a person OLDER than the cohort median age (33 years)", 0.701, "web-zero"),
}


def trunc(field, text):
    if field == "beta_text":
        return " ".join(text.split()[:100])
    return text[:1600]


def parse(t):
    pred = conf = None
    route = "SELF"
    mp = re.search(r"PRED\s*[:=]\s*([01]?\.?\d+)", t, re.I)
    mc = re.search(r"CONF\s*[:=]\s*([01]?\.?\d+)", t, re.I)
    if mp:
        try:
            pred = float(mp.group(1))
        except ValueError:
            pred = None
    if mc:
        try:
            conf = float(mc.group(1))
        except ValueError:
            conf = None
    if re.search(r"ROUTE\s*=\s*DEFER|\bDEFER\b", t, re.I):
        route = "DEFER"
    return pred, conf, route


def load(rung, n):
    csvp, field, labf, q, ceil, tag = RUNGS[rung]
    rows = list(csv.DictReader(open(os.path.join(ROOT, csvp))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = pos[:k] + neg[:k]
    items = [(trunc(field, r[field]), int(r["label"])) for r in data]
    return items, q


def prompt(repr_text, q):
    return (f"DATA:\n{repr_text}\n\n"
            f"QUESTION: Estimate the probability that {q}.\n\n"
            "Do NOT explain or add any preamble. Your entire reply must be EXACTLY this one line:\n"
            "PRED=<number 0-1> CONF=<number 0-1, how reliable your PRED is given you can or cannot read this representation> ROUTE=<SELF or DEFER>\n"
            "Start your reply with 'PRED='.")


def run_model(client, model, items, q):
    def call(rt):
        try:
            m = client.messages.create(model=model, max_tokens=96, system=SYSTEM,
                                       messages=[{"role": "user", "content": prompt(rt, q)}])
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return parse(t)
        except Exception as e:
            return (None, None, f"err:{type(e).__name__}")
    res = [None] * len(items)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, rt): i for i, (rt, _) in enumerate(items)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return res


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = {}
    for model in MODELS:
        print(f"\n########## {model} ##########", flush=True)
        print(f"{'rung':18s} {'tag':12s} {'AUROC':>6s} {'mCONF':>6s} {'defer%':>7s} {'ceil':>6s}", flush=True)
        rows_out = []
        for rung in RUNGS:
            items, q = load(rung, N)
            y = np.array([l for _, l in items])
            res = run_model(client, model, items, q)
            preds = np.array([r[0] if r[0] is not None else 0.5 for r in res])
            confs = [r[1] for r in res if r[1] is not None]
            defers = [r[2] == "DEFER" for r in res]
            auc = roc_auc_score(y, preds)
            mconf = float(np.mean(confs)) if confs else float("nan")
            defr = float(np.mean(defers))
            ceil = RUNGS[rung][4]
            tag = RUNGS[rung][5]
            print(f"{rung:18s} {tag:12s} {auc:6.3f} {mconf:6.2f} {defr*100:6.1f}% {ceil:6.3f}", flush=True)
            rows_out.append({"rung": rung, "tag": tag, "auroc": round(auc, 3), "mean_conf": round(mconf, 3),
                             "defer_rate": round(defr, 3), "ceiling": ceil})
        # cross-rung calibration: does (confidence, 1-defer) track actual AUROC?
        a = np.array([r["auroc"] for r in rows_out])
        c = np.array([r["mean_conf"] for r in rows_out])
        d = np.array([r["defer_rate"] for r in rows_out])
        rc = spearmanr(c, a).correlation
        rd = spearmanr(d, a).correlation
        print(f"\nCALIBRATION (cross-rung): corr(CONF, AUROC)={rc:+.2f}  corr(DEFER, AUROC)={rd:+.2f}", flush=True)
        print("  (CONF should rise with AUROC -> positive; DEFER should fall as AUROC rises -> negative)", flush=True)
        # over-confidence flags: web-poor rung where CONF high but AUROC ~ chance
        oc = [r for r in rows_out if r["tag"] != "web-rich" and r["mean_conf"] > 0.6 and r["auroc"] < 0.6]
        print(f"  OVER-CONFIDENT (web-zero, CONF>0.6, AUROC<0.6): {[r['rung'] for r in oc] or 'none'}", flush=True)
        # routing utility: per-rung, use ceiling when DEFER-majority else Claude
        self_mean = float(np.mean(a))
        defer_mean = float(np.mean([r["ceiling"] for r in rows_out]))
        routed = float(np.mean([r["ceiling"] if r["defer_rate"] > 0.5 else r["auroc"] for r in rows_out]))
        oracle = float(np.mean([max(r["ceiling"], r["auroc"]) for r in rows_out]))
        print(f"  ROUTING UTILITY (mean AUROC): always-self={self_mean:.3f}  Claude-routed={routed:.3f}  always-tool={defer_mean:.3f}  oracle={oracle:.3f}", flush=True)
        out[model] = {"rungs": rows_out, "calib_conf": rc, "calib_defer": rd,
                      "routing": {"self": self_mean, "routed": routed, "tool": defer_mean, "oracle": oracle}}
    json.dump(out, open(os.path.join(ROOT, "results", "calibration_routing.json"), "w"), indent=2)
    print("\n[wrote results/calibration_routing.json]", flush=True)


if __name__ == "__main__":
    main()
