"""Layer-resolved contrast of the two expression gaps (no new compute, parses the logs).

hERG (structure gap) vs drug-withdrawal (knowledge gap): the per-layer ACT AUROC profiles
from the activation runs, canonical vs randomized SMILES, on one axis. The mechanistic claim:
- hERG signal peaks EARLY (surface layers) and is randomization-ROBUST -> surface structure.
- withdrawal signal peaks DEEP (semantic layers) and the deep part is randomization-FRAGILE
  -> drug recognition / knowledge that needs deep computation and is canonical-string-keyed.
The killer panel is the per-layer (canonical - randomized) deficit: for withdrawal it should
concentrate in deep layers (recognition lives there); for hERG it should be small and flat.

Input: /tmp/layer_profiles.txt (4 blocks, '### name' then 'layer auroc' lines), extracted
from act_rand_3038486.log (hERG canon+rand), act_wd_3038493.log (wd canon), act_wd_3038495.log
(wd rand). Outputs results/layer_profiles.json + results/layer_profiles.png. No em dashes.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/tmp/layer_profiles.txt"
# structure-probe / output reference lines (from the same runs' SUMMARY)
REF = {"herg": {"morgan": 0.825, "output": 0.453}, "withdrawn": {"morgan": 0.643, "output": 0.469}}


def parse():
    prof, cur = {}, None
    for line in open(SRC):
        line = line.strip()
        if line.startswith("###"):
            cur = line[3:].strip()
            prof[cur] = {}
        elif line and cur and line[0].isdigit():
            L, a = line.split()
            prof[cur][int(L)] = float(a)
    return {k: np.array([prof[k][L] for L in sorted(prof[k])]) for k in prof}


def band(a, lo, hi):
    return round(float(np.mean(a[lo:hi + 1])), 3)


def main():
    p = parse()
    hc, hr = p["herg_canonical"], p["herg_randomized"]
    wc, wr = p["withdrawn_canonical"], p["withdrawn_randomized"]
    nL = len(hc)
    layers = np.arange(nL)

    stats = {}
    for name, c, r in [("herg", hc, hr), ("withdrawn", wc, wr)]:
        stats[name] = {
            "peak_layer_canonical": int(np.argmax(c)), "peak_auroc_canonical": round(float(c.max()), 3),
            "peak_layer_randomized": int(np.argmax(r)), "peak_auroc_randomized": round(float(r.max()), 3),
            "early_band_1_5_canonical": band(c, 1, 5), "late_band_25_36_canonical": band(c, 25, 36),
            "late_minus_early_canonical": round(band(c, 25, 36) - band(c, 1, 5), 3),
            "randomization_deficit_early_1_8": round(band(c, 1, 8) - band(r, 1, 8), 3),
            "randomization_deficit_late_22_36": round(band(c, 22, 36) - band(r, 22, 36), 3),
        }
    json.dump({"profiles": {k: [round(float(x), 3) for x in v] for k, v in p.items()}, "stats": stats},
              open(os.path.join(ROOT, "results", "layer_profiles.json"), "w"), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # panel 1: profiles
    ax1.plot(layers, hc, "-o", ms=3, color="#1f77b4", label="hERG canonical (structure)")
    ax1.plot(layers, hr, "--", color="#1f77b4", alpha=0.5, label="hERG randomized")
    ax1.plot(layers, wc, "-o", ms=3, color="#d62728", label="withdrawal canonical (knowledge)")
    ax1.plot(layers, wr, "--", color="#d62728", alpha=0.5, label="withdrawal randomized")
    ax1.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax1.scatter([np.argmax(hc)], [hc.max()], color="#1f77b4", s=80, zorder=5, edgecolor="k")
    ax1.scatter([np.argmax(wc)], [wc.max()], color="#d62728", s=80, zorder=5, edgecolor="k")
    ax1.annotate(f"peak L{np.argmax(hc)}", (np.argmax(hc), hc.max()), textcoords="offset points",
                 xytext=(6, 6), color="#1f77b4", fontsize=9)
    ax1.annotate(f"peak L{np.argmax(wc)}", (np.argmax(wc), wc.max()), textcoords="offset points",
                 xytext=(-10, -14), color="#d62728", fontsize=9)
    ax1.set_xlabel("layer"); ax1.set_ylabel("activation-probe AUROC")
    ax1.set_title("Signal vs depth: hERG peaks early (L2), withdrawal peaks deep (L27)\n(both profiles flat, so peak location is suggestive not strong)")
    ax1.legend(fontsize=8, loc="lower center"); ax1.set_ylim(0.44, 0.83)
    # panel 2: randomization deficit (canonical - randomized) by layer
    ax2.plot(layers, hc - hr, "-o", ms=3, color="#1f77b4", label="hERG  (mean deficit ~0.05)")
    ax2.plot(layers, wc - wr, "-o", ms=3, color="#d62728", label="withdrawal  (mean deficit ~0.11)")
    ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_xlabel("layer"); ax2.set_ylabel("AUROC lost to re-notation")
    ax2.set_title("Re-notation removes ~2x more from withdrawal at EVERY depth\n(its signal is canonical-string-keyed = recognition, not robust structure)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "layer_profiles.png"), dpi=130)
    print("wrote results/layer_profiles.{json,png}")
    for k, v in stats.items():
        print(f"\n{k}:")
        for kk, vv in v.items():
            print(f"  {kk} = {vv}")


if __name__ == "__main__":
    main()
