"""Risk-coverage figure for the calibration_discovery branch (data-driven, reproducible).

Regenerates the teaching figure from the REAL saved per-item runs (results/per_item.csv for the
specialist framing, results/per_item_neutral.csv for the neutral framing) rather than hand-drawn
coordinates. Pools the 8 rungs for opus and plots:
  - the confidence-ranked risk-coverage curve (the model's selective-prediction frontier)
  - the oracle curve (sort by correctness = perfect ranking, the lower bound)
  - random (no skill, flat at base error)
  - the two behavioral operating points (specialist-framing vs neutral-framing prompt), showing
    that prompt framing only slides the point ALONG the same curve (v3)
  - the selective-accuracy point at 50% coverage

Outputs results/risk_coverage.png (+ _hires) and results/risk_coverage.svg. Mirrors the
eval/make_synthesis_figure.py convention. No API calls. No em dashes.

Run: PYTHONPATH=calibration_discovery/eval python calibration_discovery/eval/plot_risk_coverage.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selective_eval import risk_coverage   # the exact metric used in the eval

BRANCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BRANCH, "results")
MODEL = os.environ.get("RC_MODEL", "claude-opus-4-8")

BLUE, GRAY, DARK, MARK = "#2563eb", "#9ca3af", "#374151", "#b45309"


def load(path, model):
    rows = [r for r in csv.DictReader(open(path)) if r["model"] == model]
    err = np.array([int(r["err"]) for r in rows], float)
    conf = np.array([float(r["conf"]) for r in rows], float)
    selfm = np.array([r["route"] == "SELF" for r in rows])
    return err, conf, selfm


def behavioral(err, selfm):
    cov = float(selfm.mean())
    risk = float(err[selfm].mean()) if selfm.any() else 0.0
    return cov, risk


def main():
    err, conf, selfm = load(os.path.join(RES, "per_item.csv"), MODEL)         # specialist framing
    n = len(err)
    cov = np.arange(1, n + 1) / n

    aurc_c, curve_c = risk_coverage(err, conf, seeds=25)
    aurc_o, curve_o = risk_coverage(err, 1.0 - err, seeds=30)                  # oracle
    aurc_r, curve_r = risk_coverage(err, np.zeros_like(err), seeds=50)         # random
    base_err = float(err.mean())

    cov_spec, risk_spec = behavioral(err, selfm)
    try:
        errn, _, selfn = load(os.path.join(RES, "per_item_neutral.csv"), MODEL)
        cov_neu, risk_neu = behavioral(errn, selfn)
    except FileNotFoundError:
        cov_neu = risk_neu = None

    # selective accuracy at 50% coverage
    i50 = int(round(0.5 * n)) - 1
    risk50 = float(curve_c[i50])

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
    ax.fill_between(cov, curve_c, base_err, color=BLUE, alpha=0.07, zorder=1)
    ax.plot(cov, curve_r, color=GRAY, lw=1.4, ls=":", label=f"random, no skill (AURC {aurc_r:.3f})", zorder=2)
    ax.plot(cov, curve_o, color=GRAY, lw=1.6, ls="--", label=f"oracle, perfect ranking (AURC {aurc_o:.3f})", zorder=3)
    ax.plot(cov, curve_c, color=BLUE, lw=2.6, label=f"confidence-ranked (AURC {aurc_c:.3f})", zorder=4)

    # selective accuracy at 50%
    ax.scatter([0.5], [risk50], s=42, color=DARK, zorder=6)
    ax.annotate(f"answer the most-confident 50%\n-> {1 - risk50:.0%} accuracy (risk {risk50:.2f})",
                xy=(0.5, risk50), xytext=(0.55, risk50 + 0.085), fontsize=9.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1))

    # behavioral operating points (framing)
    ax.scatter([cov_spec], [risk_spec], s=70, color=MARK, zorder=7, edgecolor="white", lw=1.2)
    lab_spec = f"specialist-framing prompt\nanswers {cov_spec:.0%} (defers {1 - cov_spec:.0%})"
    ax.annotate(lab_spec, xy=(cov_spec, risk_spec), xytext=(0.06, 0.30), fontsize=9.5, color=MARK,
                arrowprops=dict(arrowstyle="->", color=MARK, lw=1))
    if cov_neu is not None:
        ax.scatter([cov_neu], [risk_neu], s=70, color=MARK, zorder=7, edgecolor="white", lw=1.2)
        ax.annotate(f"neutral prompt\nanswers {cov_neu:.0%}", xy=(cov_neu, risk_neu),
                    xytext=(0.12, 0.20), fontsize=9.5, color=MARK,
                    arrowprops=dict(arrowstyle="->", color=MARK, lw=1))
        # arrow showing framing slides the point along the curve
        ax.annotate("", xy=(cov_neu, risk_neu), xytext=(cov_spec, risk_spec),
                    arrowprops=dict(arrowstyle="->", color=MARK, lw=1.3, ls=(0, (4, 3)),
                                    connectionstyle="arc3,rad=-0.25"), zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(base_err * 1.18, 0.42))
    ax.set_xlabel("coverage  (fraction of items the model answers)", fontsize=11)
    ax.set_ylabel("risk  (error rate among answered)", fontsize=11)
    ax.set_title(f"Risk-coverage: selective prediction on biology grounding ({MODEL.split('-')[1]})",
                 fontsize=12.5, pad=10)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, bbox_to_anchor=(0.0, 1.0))
    ax.text(0.99, 0.02, "framing slides the operating point along the SAME curve (v3);\n"
                        "the curve itself (calibration quality) is what scale improves",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#6b7280")
    ax.grid(True, lw=0.4, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    png = os.path.join(RES, "risk_coverage.png")
    fig.savefig(png)
    fig.savefig(os.path.join(RES, "risk_coverage_hires.png"), dpi=300)
    fig.savefig(os.path.join(RES, "risk_coverage.svg"))
    print(f"[wrote {png} (+ _hires.png, .svg)]  n={n} base_err={base_err:.3f} "
          f"spec=({cov_spec:.2f},{risk_spec:.2f}) neutral=({cov_neu},{risk_neu})", flush=True)


if __name__ == "__main__":
    main()
