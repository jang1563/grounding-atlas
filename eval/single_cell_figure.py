"""Plot the capability x web-exposure interaction across BOTH cell-pair substrates, with bootstrap
CIs, computed from the per-item raw probabilities.

Two panels (CD8-T vs NK; CD14+ vs CD16+ monocyte), each: gene-NAME AUROC (rises with capability)
and ANON AUROC (flat at chance), over the within-Claude-4 ladder + GPT-4o reference, with 95%
bootstrap error bars. Writes results/benchmark/single_cell/interaction.png.

Run:  python eval/single_cell_figure.py
"""
import json
import os

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(os.path.dirname(HERE), "results", "benchmark", "single_cell")
LADDER = [("claude-haiku-4-5-20251001", "Haiku 4.5"), ("claude-sonnet-4-6", "Sonnet 4.6"),
          ("claude-opus-4-8", "Opus 4.8")]
CROSS = ("gpt-4o", "GPT-4o")
PAIRS = [("", "CD8-T vs NK"), ("mono", "CD14+ vs CD16+ monocyte")]


def auroc_ci(model, sub, cond, rng, b=1000):
    rows = [json.loads(line) for line
            in open(os.path.join(SC, sub, model.replace("/", "_") + "_raw.jsonl"))]
    rows = [r for r in rows if r["condition"] == cond]
    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob"] for r in rows])
    a = roc_auc_score(y, p)
    boots = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) > 1:
            boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, a - lo, hi - a


def series(sub, cond, rng):
    a, lo, hi = zip(*[auroc_ci(m, sub, cond, rng) for m, _ in LADDER])
    return np.array(a), np.array([lo, hi])


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (sub, title) in zip(axes, PAIRS):
        xs = np.arange(len(LADDER))
        na, ne = series(sub, "name", rng)
        an, ae = series(sub, "anon", rng)
        ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
        ax.errorbar(xs, na, yerr=ne, fmt="o-", c="#1b6", lw=2.4, ms=8, capsize=3, label="gene NAMES")
        ax.errorbar(xs, an, yerr=ae, fmt="s--", c="#c33", lw=2.4, ms=8, capsize=3, label="ANON ids")
        xc = len(LADDER) + 0.4
        cn = auroc_ci(CROSS[0], sub, "name", rng)
        ca = auroc_ci(CROSS[0], sub, "anon", rng)
        ax.errorbar([xc], [cn[0]], yerr=[[cn[1]], [cn[2]]], fmt="o", c="#1b6", ms=8, mfc="white", mew=2, capsize=3)
        ax.errorbar([xc], [ca[0]], yerr=[[ca[1]], [ca[2]]], fmt="s", c="#c33", ms=8, mfc="white", mew=2, capsize=3)
        ax.set_xticks(list(xs) + [xc])
        ax.set_xticklabels([lbl for _, lbl in LADDER] + [CROSS[1]], rotation=12)
        ax.set_title(title)
        ax.set_ylim(0.40, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("output-arm AUROC")
    axes[0].legend(loc="center left", fontsize=9, framealpha=0.9)
    fig.suptitle("Capability x web-exposure generalizes: names rise with scale, anon stays at chance "
                 "(n=200/model, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(SC, "interaction.png"), dpi=150)
    print(f"wrote {SC}/interaction.png")
    for sub, title in PAIRS:
        print(f"\n{title}")
        for cond in ("name", "anon"):
            a, _ = series(sub, cond, rng)
            print(f"  {cond:5s} " + " ".join(f"{lbl.split()[0]}={v:.3f}" for (_, lbl), v in zip(LADDER, a)))


if __name__ == "__main__":
    main()
