"""Calibration inside discovery: the PER-ITEM selective-prediction benchmark (Move 2).

Promotes the rung-level +0.90 router (results/calibration_routing.md) to the per-item decision an
autonomous scientist makes: given THIS item, answer or abstain. Same elicitation as
eval/calibration_routing.py (PRED/CONF/ROUTE), so results are comparable, but here we SAVE per item
and compute:

  1. risk-coverage frontier (sort by CONF, sweep coverage) -> AURC, E-AURC, selective acc @ 0.5/0.8
  2. behavioral abstention point (model's own SELF/DEFER) -> calibration-ACTION gap vs the frontier
  3. web-exposure as a FREE selective score (item tag, no model conf) -> AURC_webexp
  + baselines: prediction margin |PRED-0.5|, random, oracle (sort by correctness = lower bound)

Prediction: AURC_conf improves with scale but the action gap (behavioral worse than frontier) does
NOT shrink (AbstentionBench); web-exposure ~ conf on in/out strata but FAILS on the novel stratum.

Set ANTHROPIC_API_KEY in the environment for real API runs.
"""
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BRANCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # calibration_discovery/
ROOT = os.path.dirname(BRANCH)                                          # Bio_Grounding_Eval/
RESDIR = os.path.join(BRANCH, "results")

N = int(os.environ.get("SE_N", "80"))
WORKERS = int(os.environ.get("SE_WORKERS", "8"))
MODELS = os.environ.get("SE_MODELS", "claude-opus-4-8").split(",")

# Two framings, single-variable manipulation: SPECIALIST nudges toward deferral (a specialist is available,
# deferring is correct), NEUTRAL does not. v3 robustness check: does the CONF-vs-ROUTE signal-usage gap
# survive removing the defer nudge, or was it an artifact of a defer-happy prompt compressing ROUTE variance.
SYSTEM_SPECIALIST = ("You are the reasoning core of a tool-augmented scientific analysis system. For every item a "
          "trained SPECIALIST model is available for that data type (a cheminformatics model, a genomics "
          "model, an epigenetic clock, and so on). Your job is to give your best prediction, honestly rate "
          "how reliable it is, and decide whether to answer yourself or defer to the specialist. Be honest: "
          "deferring when you genuinely cannot read the representation is correct behavior, not failure.")
SYSTEM_NEUTRAL = ("You are a scientific analysis assistant. For each item, give your best prediction, honestly "
          "rate how reliable it is, and decide whether to answer it yourself (SELF) or decline (DEFER). "
          "Answer whenever you can; decline only when you genuinely cannot determine it from the data provided.")
FRAMING = os.environ.get("SE_FRAMING", "specialist")
SYSTEM = SYSTEM_NEUTRAL if FRAMING == "neutral" else SYSTEM_SPECIALIST

HERG_Q = "this molecule BLOCKS the hERG potassium channel (cardiotoxicity)"
SC_Q = "this cell is a T cell (based on its gene expression)"
VAR_Q = "this genetic variant is PATHOGENIC"

# rung -> dict(csv, field|build, q, ceil, tag, comp). comp in {in, out, novel} = the ground-truth right move.
RUNGS = {
    "sc_cellsentence": dict(csv="signal/single_cell/pbmc_Tcell.csv", field="cell_sentence", q=SC_Q,   ceil=0.989, tag="web-rich", comp="in"),
    "msa_conserv":     dict(csv="signal/msa/msa_conservation.csv",    field="column",        q="this protein multiple-sequence-alignment column is evolutionarily CONSERVED", ceil=0.999, tag="web-rich", comp="in"),
    "dna_promoter":    dict(csv="signal/dna_promoter.csv",            field="smiles",        q="this DNA sequence is a PROMOTER (a regulatory region that initiates transcription)", ceil=0.889, tag="web-rich", comp="in"),
    "sc_anon":         dict(csv="signal/single_cell/pbmc_Tcell.csv",  field="anon",          q=SC_Q,   ceil=0.989, tag="web-zero", comp="out"),
    "smiles_herg":     dict(csv="signal/nmr/herg_nmr.csv",            field="smiles",        q=HERG_Q, ceil=0.825, tag="web-zero-out", comp="out"),
    "nmr_herg":        dict(csv="signal/nmr/herg_nmr.csv",            field="nmr",           q=HERG_Q, ceil=0.866, tag="web-zero", comp="out"),
    "methyl_age":      dict(csv="signal/methyl/methyl_age.csv",       field="beta_text",     q="this blood sample is from a person OLDER than the cohort median age (33 years)", ceil=0.701, tag="web-zero", comp="out"),
    "variant_novel":     dict(csv="variant_grounding/data/variant_clinvar_post2026_01.csv", build="variant",     q=VAR_Q, ceil=0.962, tag="web-rich-novel", comp="novel"),
    "variant_novel_seq": dict(csv="variant_grounding/data/variant_clinvar_post2026_01.csv", build="variant_seq", q=VAR_Q, ceil=0.962, tag="web-poor",       comp="novel"),  # v2: same variants, web-poor seq notation
}
RUNG_FILTER = os.environ["SE_RUNGS"].split(",") if os.environ.get("SE_RUNGS") else None
TAG = os.environ.get("SE_TAG", "")  # output filename suffix, so a variant-only v2 run does not clobber v1 files

# web-exposure free score: higher = more trustworthy a-priori (web-rich), used for AURC_webexp.
WEBEXP_SCORE = {"web-rich": 1.0, "web-rich-novel": 1.0, "web-zero-out": 0.0, "web-zero": 0.0, "web-poor": 0.0}


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
    spec = RUNGS[rung]
    rows = list(csv.DictReader(open(os.path.join(ROOT, spec["csv"]))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    data = pos[:k] + neg[:k]
    if spec.get("build") == "variant":
        items = [(f"{r['gene']} {r['hgvs_p']}", int(r["label"])) for r in data]
    elif spec.get("build") == "variant_seq":
        items = [(f"Human protein sequence fragment (judge only from the sequence and the substitution):\n"
                  f"{r['wt_window']}\nThe residue at position {r['win_pos']} (1-based in this fragment) "
                  f"is mutated from {r['wt']} to {r['mut']}.", int(r["label"])) for r in data]
    else:
        f = spec["field"]
        items = [(trunc(f, r[f]), int(r["label"])) for r in data]
    return items, spec["q"]


def variant_am_auc(n):
    """Real per-item AlphaMissense specialist on the SAME post-cutoff variants load() samples (seed 0)."""
    spec = RUNGS["variant_novel"]
    rows = list(csv.DictReader(open(os.path.join(ROOT, spec["csv"]))))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    y, am = [], []
    for r in pos[:k] + neg[:k]:
        try:
            am.append(float(r.get("am", "")))
            y.append(int(r["label"]))
        except (ValueError, TypeError):
            continue
    if len(set(y)) < 2:
        return float("nan"), 0.0
    return roc_auc_score(y, am), len(y) / (2 * k)


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
            if getattr(m, "stop_reason", None) == "refusal":
                return (0.5, 0.0, "DEFER", "refusal")   # safety-refusal = declined (the seq gotcha); counted separately
            t = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            pred, conf, route = parse(t)
            return (pred, conf, route, "parsed" if pred is not None else "empty")
        except Exception as e:
            return (None, None, "DEFER", f"err:{type(e).__name__}")
    res = [None] * len(items)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, rt): i for i, (rt, _) in enumerate(items)}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    return res


# ---------- selective-prediction metrics ----------

def risk_coverage(err, score, seeds=20):
    """Sort by score desc (random tie-break), return mean AURC over seeds. err: 1=wrong. Lower AURC better."""
    err = np.asarray(err, float)
    n = len(err)
    aurcs, risk_acc = [], None
    for s in range(seeds):
        rng = np.random.RandomState(s)
        order = np.lexsort((rng.random(n), -np.asarray(score, float)))  # primary -score, tie-break random
        e = err[order]
        risk = np.cumsum(e) / np.arange(1, n + 1)
        aurcs.append(risk.mean())
        risk_acc = risk if risk_acc is None else risk_acc + risk
    return float(np.mean(aurcs)), (risk_acc / seeds)  # mean AURC, mean risk curve (len n, coverage k/n)


def risk_at_coverage(risk_curve, cov):
    if cov <= 0:
        return 0.0
    idx = int(round(cov * len(risk_curve))) - 1
    return float(risk_curve[max(0, min(len(risk_curve) - 1, idx))])


def cov_at_risk(risk_curve, target):
    """Max coverage the confidence frontier sustains at risk <= target (the over-defer waste check)."""
    ok = np.where(np.asarray(risk_curve) <= target + 1e-9)[0]
    return float((ok[-1] + 1) / len(risk_curve)) if len(ok) else 0.0


def ece(pred, label, bins=10):
    pred, label = np.asarray(pred, float), np.asarray(label, float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(pred)
    for i in range(bins):
        m = (pred >= edges[i]) & (pred < edges[i + 1] if i < bins - 1 else pred <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(pred[m].mean() - label[m].mean())
    return float(e)


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    os.makedirs(RESDIR, exist_ok=True)
    out, peritem = {}, []

    for model in MODELS:
        print(f"\n########## {model} [{FRAMING}] ##########", flush=True)
        recs = []  # per item: rung, tag, comp, label, pred, conf, route, kind, err
        for rung in [r for r in RUNGS if not RUNG_FILTER or r in RUNG_FILTER]:
            items, q = load(rung, N)
            res = run_model(client, model, items, q)
            spec = RUNGS[rung]
            for (rt, lab), (pred, conf, route, kind) in zip(items, res):
                p = pred if pred is not None else 0.5
                c = conf if conf is not None else 0.0
                err = int((p > 0.5) != (lab == 1))
                rec = dict(model=model, rung=rung, tag=spec["tag"], comp=spec["comp"],
                           label=lab, pred=p, conf=c, route=route, kind=kind, err=err)
                recs.append(rec)
                peritem.append(rec)
            sub = [r for r in recs if r["rung"] == rung]
            auc = roc_auc_score([r["label"] for r in sub], [r["pred"] for r in sub]) if len(set(r["label"] for r in sub)) > 1 else float("nan")
            dfr = np.mean([r["route"] == "DEFER" for r in sub])
            refu = sum(r["kind"] == "refusal" for r in sub)
            print(f"  {rung:18s} {spec['comp']:6s} n={len(sub):3d} AUROC={auc:.3f} acc={1-np.mean([r['err'] for r in sub]):.3f} mCONF={np.mean([r['conf'] for r in sub]):.2f} defer={dfr*100:4.0f}% refuse={refu}", flush=True)

        err = np.array([r["err"] for r in recs])
        conf = np.array([r["conf"] for r in recs])
        pred = np.array([r["pred"] for r in recs])
        lab = np.array([r["label"] for r in recs])
        webx = np.array([WEBEXP_SCORE[r["tag"]] for r in recs])
        self_mask = np.array([r["route"] == "SELF" for r in recs])

        aurc_conf, curve_conf = risk_coverage(err, conf, seeds=25)        # conf is tie-heavy (rounded values + fallback 0) -> average tie-breaks
        aurc_marg, _ = risk_coverage(err, np.abs(pred - 0.5), seeds=25)
        aurc_webx, _ = risk_coverage(err, webx, seeds=30)                  # heavy ties -> average
        aurc_rand, _ = risk_coverage(err, np.zeros_like(err), seeds=50)
        aurc_orac, _ = risk_coverage(err, 1.0 - err, seeds=30)            # oracle lower bound
        eaurc = aurc_conf - aurc_orac

        cov_beh = float(self_mask.mean())
        risk_beh = float(err[self_mask].mean()) if self_mask.any() else 0.0
        risk_front = risk_at_coverage(curve_conf, cov_beh)
        gap = risk_beh - risk_front                          # >0: over-answer (AbstentionBench). over-defer -> ~0
        cov_gap = cov_at_risk(curve_conf, risk_beh) - cov_beh  # >0: over-defer waste (frontier sustains more coverage at the model's own risk)
        # does the binary ROUTE action exploit the graded CONF signal it possesses?
        correct = 1 - err
        twoclass = len(set(correct)) > 1
        auc_conf_sig = roc_auc_score(correct, conf) if twoclass else float("nan")
        auc_route_sig = roc_auc_score(correct, self_mask.astype(float)) if twoclass and self_mask.any() and not self_mask.all() else float("nan")

        sel50 = 1 - risk_at_coverage(curve_conf, 0.5)
        sel80 = 1 - risk_at_coverage(curve_conf, 0.8)
        corr_ca = spearmanr(conf, 1 - err).correlation
        ece_v = ece(pred, lab)

        # per-comp overlay: behavioral defer rate + accuracy + mean conf
        comp_tab = {}
        for comp in ("in", "out", "novel"):
            cm = [r for r in recs if r["comp"] == comp]
            if not cm:
                continue
            comp_tab[comp] = dict(n=len(cm), acc=round(1 - np.mean([r["err"] for r in cm]), 3),
                                  defer=round(np.mean([r["route"] == "DEFER" for r in cm]), 3),
                                  mconf=round(np.mean([r["conf"] for r in cm]), 3))

        print(f"\n  --- selective prediction (pooled, n={len(recs)}) ---", flush=True)
        print(f"  AURC: conf={aurc_conf:.3f}  margin={aurc_marg:.3f}  webexp={aurc_webx:.3f}  random={aurc_rand:.3f}  oracle={aurc_orac:.3f}   E-AURC(conf)={eaurc:.3f}", flush=True)
        print(f"  selective acc @cov0.5={sel50:.3f}  @cov0.8={sel80:.3f}   corr(conf,correct)={corr_ca:+.2f}  ECE={ece_v:.3f}", flush=True)
        print(f"  BEHAVIORAL abstention: coverage(SELF)={cov_beh:.3f}  risk={risk_beh:.3f}  frontier-risk@same-cov={risk_front:.3f}", flush=True)
        print(f"    risk-gap(over-answer)={gap:+.3f}   coverage-gap(over-defer waste)={cov_gap:+.3f}   signal-usage AUROC: CONF={auc_conf_sig:.3f} vs ROUTE={auc_route_sig:.3f}", flush=True)
        print("  per-comp: " + "  ".join(f"{k}(acc{v['acc']},defer{v['defer']},conf{v['mconf']})" for k, v in comp_tab.items()), flush=True)

        out[model] = dict(framing=FRAMING, n=len(recs), aurc_conf=aurc_conf, aurc_margin=aurc_marg, aurc_webexp=aurc_webx,
                          aurc_random=aurc_rand, aurc_oracle=aurc_orac, eaurc_conf=eaurc,
                          sel_acc_50=sel50, sel_acc_80=sel80, corr_conf_correct=corr_ca, ece=ece_v,
                          behavioral=dict(coverage=cov_beh, risk=risk_beh, frontier_risk=risk_front,
                                          risk_gap=gap, coverage_gap=cov_gap,
                                          signal_auc_conf=auc_conf_sig, signal_auc_route=auc_route_sig),
                          per_comp=comp_tab)

    json.dump(out, open(os.path.join(RESDIR, f"selective_eval{TAG}.json"), "w"), indent=2)
    with open(os.path.join(RESDIR, f"per_item{TAG}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "rung", "tag", "comp", "label", "pred", "conf", "route", "kind", "err"])
        w.writeheader()
        w.writerows(peritem)

    # v2: within-entity text-vs-seq calibration contrast on the same post-cutoff variants
    if any(r["rung"] == "variant_novel_seq" for r in peritem) and any(r["rung"] == "variant_novel" for r in peritem):
        am_auc, am_cov = variant_am_auc(N)
        print("\n########## V2: variant text-vs-seq within-entity calibration ##########", flush=True)
        print(f"  AlphaMissense specialist (real per-item, am score), same variants: AUROC={am_auc:.3f} (coverage {am_cov:.0%})", flush=True)
        print(f"  {'model':28s} {'AUROC txt/seq':>13s} {'mCONF txt/seq':>13s} {'defer txt/seq':>13s} {'refuse seq':>10s}", flush=True)
        for m in MODELS:
            rt = [r for r in peritem if r["model"] == m and r["rung"] == "variant_novel"]
            rs = [r for r in peritem if r["model"] == m and r["rung"] == "variant_novel_seq"]
            if not rt or not rs:
                continue
            at = roc_auc_score([r["label"] for r in rt], [r["pred"] for r in rt])
            asq = roc_auc_score([r["label"] for r in rs], [r["pred"] for r in rs])
            ct, cs = np.mean([r["conf"] for r in rt]), np.mean([r["conf"] for r in rs])
            dt, ds = np.mean([r["route"] == "DEFER" for r in rt]), np.mean([r["route"] == "DEFER" for r in rs])
            refu = sum(r["kind"] == "refusal" for r in rs)
            print(f"  {m:28s}  {at:.3f}/{asq:.3f}    {ct:.2f}/{cs:.2f}      {dt*100:3.0f}%/{ds*100:3.0f}%       {refu:3d}", flush=True)
        print("  (within-entity web-exposure on the CALIBRATION axis: CONF_seq should drop below CONF_text in step with", flush=True)
        print("   the AUROC drop; CONF high while seq-AUROC falls = miscalibration on the web-poor notation -> over-trust.)", flush=True)

    if len(MODELS) > 1:
        print("\n########## SCALE AXIS ##########", flush=True)
        print(f"  {'model':28s} {'AURC_conf':>9s} {'E-AURC':>7s} {'sel@0.8':>8s} {'cov-gap':>8s} {'sigAUC c/r':>14s}", flush=True)
        for m in MODELS:
            o = out[m]
            b = o["behavioral"]
            print(f"  {m:28s} {o['aurc_conf']:9.3f} {o['eaurc_conf']:7.3f} {o['sel_acc_80']:8.3f} {b['coverage_gap']:+8.3f} {b['signal_auc_conf']:6.3f}/{b['signal_auc_route']:.3f}", flush=True)
        print("  (P1: AURC_conf falls with scale. P2/AbstentionBench: action does NOT exploit the CONF signal -> sigAUC ROUTE << CONF, cov-gap stays large.)", flush=True)

    print(f"\n[wrote {RESDIR}/selective_eval{TAG}.json + per_item{TAG}.csv]", flush=True)


if __name__ == "__main__":
    main()
