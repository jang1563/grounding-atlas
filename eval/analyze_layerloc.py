"""Section-5 analysis for the layer-localization prereg (docs/LAYER_LOCALIZATION_PREREG.md).

Ingests the task-tagged results/benchmark/layerloc/layer_loc_<task>_<model>.json files (the arms write
them to the cluster's ~/results, pulled here) and emits the H1/H2/H3 verdict table per model, the
cross-architecture H1 check, and the single-cell web-exposure layer-shift. Pre-registered thresholds:
  H1  encoding peak at depth fraction 0.45-0.80, across-fold layer range <= 18 (headline tasks dna/herg)
  H2  EXCESS gap (raw_gap - msa positive-control floor) >= +0.15 AND selectivity >= +0.15
      AND (dna) GC-residualized margin >= +0.10
  H3  probe AURC < output AURC (better routing) AND probe ECE <= 0.15
No em dashes. Usage: python analyze_layerloc.py [results/benchmark/layerloc]
"""
import glob
import json
import os
import sys

H1_LO, H1_HI, H1_RANGE = 0.45, 0.80, 18
H2_EXCESS, H2_SEL, GC_MARGIN = 0.15, 0.15, 0.10
H3_ECE = 0.15
HEADLINE = ("dna/promoter", "admet/herg")


def depth_stats(d):
    """H1 endpoint depth. Prefer the PEAK-SELECTIVITY layer (where the property is COMPUTED); fall
    back to the mean nested-CV raw pick when the selectivity curve is absent (older runs). Also return
    the across-fold raw-pick range as the jumpiness proxy."""
    nl = len(d["layer_auroc"]) - 1
    pk = d["picked_layers"]
    if "peak_sel_layer" in d:
        return d["peak_sel_layer"] / nl, min(pk), max(pk), max(pk) - min(pk), nl, "sel"
    return (sum(pk) / len(pk)) / nl, min(pk), max(pk), max(pk) - min(pk), nl, "raw"


def load(folder):
    by_model = {}
    for f in sorted(glob.glob(os.path.join(folder, "layer_loc_*.json"))):
        d = json.load(open(f))
        by_model.setdefault(d["model"], {})[d["task"]] = d
    return by_model


def analyze(by_model):
    out = []
    for model, tasks in by_model.items():
        floor = next((t["raw_gap"] for k, t in tasks.items() if "msa" in k), None)
        out.append(f"\n## {model}   (msa positive-control floor raw_gap = "
                   f"{floor:+.3f})" if floor is not None else f"\n## {model}   (no msa floor yet)")
        out.append(f"{'task':30s} {'nest':>5s} {'out':>5s} {'gap':>6s} {'excess':>6s} "
                   f"{'depth':>5s} {'range':>5s} {'pECE':>5s} {'dAURC':>6s} {'sel':>5s} {'bias':>5s} verdicts")
        for task, d in sorted(tasks.items()):
            mdep, lo, hi, rng, nl, src = depth_stats(d)
            gap = d["raw_gap"]
            excess = (gap - floor) if floor is not None else float("nan")
            daurc = d.get("probe_aurc", float("nan")) - d.get("output_aurc", float("nan"))
            ece = d.get("probe_ece_10", float("nan"))
            sel = d["selectivity"]
            v = []
            # H1 only adjudicated on the headline expression-gap tasks
            if task in HEADLINE:
                v.append("H1+" if (H1_LO <= mdep <= H1_HI and rng <= H1_RANGE) else "H1-")
            # H2
            h2 = (excess >= H2_EXCESS and sel >= H2_SEL)
            if task == "dna/promoter" and "gc_margin" in d:
                h2 = h2 and d["gc_margin"] >= GC_MARGIN
                v.append(f"GC{d['gc_margin']:+.2f}")
            if "msa" not in task:
                v.append("H2+" if h2 else "H2-")
            # H3
            v.append("H3+" if (daurc < 0 and ece <= H3_ECE) else "H3-")
            out.append(f"{task:30s} {d['nested_auroc']:.3f} {d['output_auroc']:.3f} {gap:+.3f} "
                       f"{excess:+.3f} {mdep:.2f}{src[0]} [{lo:2d}-{hi:2d}] {ece:.3f} {daurc:+.3f} {sel:+.3f} "
                       f"{d['selection_bias']:+.3f}  {' '.join(v)}")

    # cross-architecture H1 (do the headline tasks share a band across models?)
    out.append("\n## Cross-architecture H1 (headline tasks: is the peak-depth band shared?)")
    for task in HEADLINE:
        deps = {m: depth_stats(t[task])[0] for m, t in by_model.items() if task in t}
        if len(deps) >= 2:
            ok = all(H1_LO <= x <= H1_HI for x in deps.values())
            out.append(f"  {task:18s} " + " | ".join(f"{m.split('/')[-1]}={x:.2f}" for m, x in deps.items())
                       + ("  -> CONSISTENT mid-band" if ok else "  -> divergent"))
        elif deps:
            out.append(f"  {task:18s} " + " | ".join(f"{m.split('/')[-1]}={x:.2f}" for m, x in deps.items())
                       + "  (one model so far)")

    # single-cell web-exposure layer-shift (descriptive; power-limited)
    out.append("\n## Single-cell web-exposure layer-shift (DESCRIPTIVE, power-limited n<=470)")
    for m, t in by_model.items():
        nm = t.get("single_cell/cd8t_nk:cell_sentence")
        an = t.get("single_cell/cd8t_nk:anon")
        if nm and an:
            dn, da = depth_stats(nm)[0], depth_stats(an)[0]
            out.append(f"  {m.split('/')[-1]:20s} name(web-rich) depth={dn:.2f} vs anon(web-zero) depth={da:.2f}"
                       f"  shift={dn - da:+.2f} ({'familiar=late, alien=early' if dn > da + 0.1 else 'no clear shift'})")
    return "\n".join(out)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark/layerloc"
    by_model = load(folder)
    if not by_model:
        print(f"no layer_loc_*.json in {folder}")
        return
    report = analyze(by_model)
    print(report)
    md = os.path.join(folder, "SUMMARY.md")
    with open(md, "w") as f:
        f.write("# Layer-localization results (auto-generated by analyze_layerloc.py)\n\n```\n")
        f.write(report)
        f.write("\n```\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
