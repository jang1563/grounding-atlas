"""P1 figure: the confidence frontier sharpens with scale (data-driven, reproducible).

Overlays the three models' risk-coverage curves (haiku, sonnet, opus) from results/per_item.csv.
As scale grows the curve moves toward the lower-left (lower risk at every coverage = better
selective prediction); the legend reports AURC and E-AURC (AURC minus the model's own oracle, which
removes its base-accuracy advantage so only calibration QUALITY is compared). Mirrors the
make_synthesis_figure.py convention. No API calls. No em dashes.

Run: PYTHONPATH=calibration_discovery/eval python calibration_discovery/eval/plot_scale_frontier.py
"""
import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from selective_eval import risk_coverage

BRANCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BRANCH, "results")
# scale order light -> dark
MODELS = [("claude-haiku-4-5-20251001", "haiku-4.5", "#93c5fd"),
          ("claude-sonnet-4-6", "sonnet-4.6", "#3b82f6"),
          ("claude-opus-4-8", "opus-4.8", "#1e3a8a")]


def load(model):
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "per_item.csv"))) if r["model"] == model]
    return (np.array([int(r["err"]) for r in rows], float),
            np.array([float(r["conf"]) for r in rows], float))


def main():
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
    n = None
    for model, label, color in MODELS:
        err, conf = load(model)
        n = len(err)
        cov = np.arange(1, n + 1) / n
        aurc_c, curve_c = risk_coverage(err, conf, seeds=25)
        aurc_o, _ = risk_coverage(err, 1.0 - err, seeds=30)
        ax.plot(cov, curve_c, color=color, lw=2.6, label=f"{label}  (AURC {aurc_c:.3f}, E-AURC {aurc_c - aurc_o:.3f})")

    ax.annotate("scale moves the curve\ntoward the lower-left\n(better selective prediction)",
                xy=(0.30, 0.10), xytext=(0.40, 0.30), fontsize=10, color="#374151", ha="left",
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2, connectionstyle="arc3,rad=0.2"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("coverage  (fraction of items the model answers)", fontsize=11)
    ax.set_ylabel("risk  (error rate among answered)", fontsize=11)
    ax.set_title("The confidence frontier sharpens with scale (P1)", fontsize=12.5, pad=10)
    ax.legend(loc="upper left", fontsize=10, frameon=False, title="confidence-ranked curve  (lower = better)")
    ax.text(0.99, 0.02, "E-AURC removes each model's base-accuracy advantage,\nso only calibration quality is compared",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#6b7280")
    ax.grid(True, lw=0.4, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "scale_frontier.png"))
    fig.savefig(os.path.join(RES, "scale_frontier_hires.png"), dpi=300)
    fig.savefig(os.path.join(RES, "scale_frontier.svg"))
    print(f"[wrote {RES}/scale_frontier.png (+ _hires.png, .svg)]  n={n}", flush=True)


if __name__ == "__main__":
    main()
