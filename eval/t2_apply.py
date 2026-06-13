"""T2 (apply): does T1 grounding transfer downstream, solo vs orchestrate.

WS1 tier T2 (eval/README "Bridge to T2"). T1 returned its first numbers (the
3-arm head-to-head, results/head_to_head.md + PROJECT_DESIGN section 7.2), which
unblocks T2. This script does the SOLVE mode of T2 as the orchestrate-vs-solo
contrast the backlog calls for (docs/WS1_BACKLOG.md B.T2), reproducibly and with
no GPU or API: it reads the WS2 specialist ceilings from signal/admet/*/ and the
measured 3-arm anchors, then decomposes each modality's orchestrate headroom into
the part a read-out could train back (expression) and the part the model does not
hold at all (encoding). That decomposition is the T2 routing rule and the seed of
the WS3 decision map.

T2 has three modes (eval/README); only SOLVE is assembled here because its arm
(LLM-output scored vs ground truth) is already measured. PROPOSE and EVALUATE need
fresh runs and are specified, not scored (see results/t2_apply.md).

No em dashes. Deterministic. Run: python eval/t2_apply.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADMET = os.path.join(ROOT, "signal", "admet")

# --- Measured 3-arm anchors (T1). Provenance per row. AUROC, zero-shot output. ---
# ceiling   = structure-probe arm (specialist an orchestrator would call)
# activation = LLM hidden-state linear probe (what the model encodes internally)
# output    = LLM-output arm = T2-SOLVE solo score (property scored vs ground truth)
# Sources: results/head_to_head.md R2 (8B anchor) and PROJECT_DESIGN.md 7.2.
ANCHORS = [
    # modality, label, ceiling, activation, output_8b, ceiling_src, note
    ("smiles",  "hERG",            0.825, 0.787, 0.453, "Morgan r2 (balanced same-set)", "best frontier output 0.581 (sonnet-4-6)"),
    ("smiles",  "CYP3A4",          0.745, 0.684, 0.502, "Morgan r2 (balanced same-set)", ""),
    ("variant", "ClinVar (text)",  0.962, 0.795, 0.599, "AlphaMissense + ESM-1v",        "web-rich text form; gene-prior recall confound"),
    ("variant", "ClinVar (seq)",   0.962, 0.740, 0.494, "AlphaMissense + ESM-1v",        "web-poor seq form; floor rises with scale"),
    ("protein", "meltome Tm",      0.699, 0.609, 0.486, "ESM-2",                          "low ceiling, encoding-weak + organism-name"),
]

# T2-SOLVE solo output measured so far (AUROC). Only 2 of 7 WS2 endpoints have an
# output arm; the rest have a verified ceiling but no solo run yet (light API).
SOLO_OUTPUT = {"herg": 0.453, "cyp3a4": 0.502}


def route(enc_gap, exp_gap, ceiling):
    """T2 placement rule from the gap decomposition."""
    if ceiling < 0.72:
        return "orchestrate (thin signal: low ceiling, little to surface)"
    if exp_gap > enc_gap and enc_gap < 0.10:
        return "WS3-weights: train the read-out (expression-limited)"
    if enc_gap >= 0.10 and exp_gap >= 0.10:
        return "weights + orchestrate (mixed: both gaps large)"
    if enc_gap > exp_gap:
        return "orchestrate specialist (encoding-limited)"
    return "WS3-weights: train the read-out"


def fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def main():
    rows = []
    for modality, label, ceiling, act, out, csrc, note in ANCHORS:
        headroom = round(ceiling - out, 3)     # solo gap an orchestrator/specialist closes
        exp_gap = round(act - out, 3)          # internally present, unsurfaced -> trainable read-out
        enc_gap = round(ceiling - act, 3)      # not internally present -> must orchestrate
        rows.append({
            "modality": modality, "task": label,
            "ceiling": ceiling, "activation": act, "solo_output": out,
            "orchestrate_headroom": headroom,
            "expression_gap": exp_gap, "encoding_gap": enc_gap,
            "t2_route": route(enc_gap, exp_gap, ceiling),
            "ceiling_source": csrc, "note": note,
        })

    # WS2 ADMET ceilings (cold AUROC): orchestrate ceiling exists for all 7; solo
    # output measured for 2. This is the SOLVE-mode coverage map.
    # Frontier solo sweep (sonnet-4-6, direction-oriented AUROC) from output_arm_admet.py.
    frontier = {}
    fp = os.path.join(ROOT, "results", "output_arm_admet.json")
    if os.path.isfile(fp):
        with open(fp) as fh:
            for r in json.load(fh):
                frontier[r["endpoint"]] = r.get("oriented_auroc", r.get("auroc"))

    admet = []
    for ep in sorted(os.listdir(ADMET)):
        p = os.path.join(ADMET, ep, "verifiability.json")
        if not os.path.isfile(p):
            continue
        with open(p) as fh:
            v = json.load(fh)
        # prefer the frontier oriented solo; fall back to the 8B anchor (hERG/CYP3A4)
        if ep in frontier:
            solo, src = frontier[ep], "frontier sonnet-4-6 (oriented)"
        else:
            solo, src = SOLO_OUTPUT.get(ep), "8B anchor"
        admet.append({
            "endpoint": ep, "n": v["n"],
            "orchestrate_ceiling_cold": v["cold_auroc"],
            "solo_auroc": solo, "solo_source": src,
            "orchestrate_headroom": round(v["cold_auroc"] - solo, 3) if solo is not None else None,
            "solo_status": "measured" if solo is not None else "to run (light API)",
        })

    out = {
        "title": "T2 apply: SOLVE mode, orchestrate vs solo",
        "anchors": rows,
        "admet_solve_coverage": admet,
        "headline": (
            "On the 8B 3-arm anchors solo output sits near chance (0.45-0.60) below a "
            "0.70-0.96 specialist ceiling, and the headroom splits into an expression "
            "part (recoverable by training the read-out, WS3-weights) and an encoding "
            "part (must orchestrate the specialist) = the T2 routing rule. But the clean "
            "frontier ADMET sweep (sonnet-4-6, all 7, oriented) shows solo transfer is "
            "endpoint-dependent and mostly ABOVE chance: 5 of 7 at 0.61-0.72 (permeability "
            "0.72, solubility 0.65, hERG 0.63, CYP3A4 0.61, CYP2D6 0.61, all web-exposed "
            "physicochemical), AMES 0.38 (anti, toxicophore SAR), clearance 0.49 (chance). "
            "So the 8B near-chance output is a model-scale expression gap, not a ceiling on "
            "the capability; the frontier closes most of it. Orchestration still wins for "
            "max accuracy (cold ceiling > solo everywhere)."
        ),
    }

    outpath = os.path.join(ROOT, "results", "t2_apply.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)

    # Console table
    print("T2-SOLVE: orchestrate vs solo (AUROC), gap decomposition\n")
    hdr = ("modality/task", "ceiling", "activ", "solo", "headroom", "exp_gap", "enc_gap", "route")
    print(f"{hdr[0]:<20}{hdr[1]:>8}{hdr[2]:>7}{hdr[3]:>7}{hdr[4]:>10}{hdr[5]:>9}{hdr[6]:>9}  {hdr[7]}")
    for r in rows:
        print(f"{r['modality']+'/'+r['task']:<20}{r['ceiling']:>8.3f}{r['activation']:>7.3f}"
              f"{r['solo_output']:>7.3f}{r['orchestrate_headroom']:>10.3f}"
              f"{fmt(r['expression_gap']):>9}{fmt(r['encoding_gap']):>9}  {r['t2_route']}")
    print("\nWS2 ADMET SOLVE coverage (cold ceiling vs frontier oriented solo, all 7):")
    for a in admet:
        so = f"{a['solo_auroc']:.3f}" if a["solo_auroc"] is not None else "  -  "
        hr = f"{a['orchestrate_headroom']:+.3f}" if a["orchestrate_headroom"] is not None else "  -  "
        print(f"  {a['endpoint']:<12} n={a['n']:<6} ceiling={a['orchestrate_ceiling_cold']:.3f}  "
              f"solo={so}  headroom={hr}  [{a['solo_source']}]")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
