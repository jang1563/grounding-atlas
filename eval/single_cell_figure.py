"""Plot the capability x web-exposure interaction (CD8-T vs NK) from the per-model results.

The within-Claude-4 ladder (Haiku -> Sonnet -> Opus) is the clean capacity axis; GPT-4o is a
cross-provider reference. Two lines: gene-NAME AUROC (should rise with capability) and ANON
AUROC (should stay flat at chance). Reads results/benchmark/single_cell/<model>.json, writes
interaction.png + a summary table.

Run:  python eval/single_cell_figure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(os.path.dirname(HERE), "results", "benchmark", "single_cell")
LADDER = [("claude-haiku-4-5-20251001", "Haiku 4.5"), ("claude-sonnet-4-6", "Sonnet 4.6"),
          ("claude-opus-4-8", "Opus 4.8")]
CROSS = ("gpt-4o", "GPT-4o\n(cross-prov.)")


def get(model):
    return json.load(open(os.path.join(SC, model.replace("/", "_") + ".json")))


def main():
    lad = [(lbl, get(m)) for m, lbl in LADDER]
    xs = list(range(len(lad)))
    name = [d["name"] for _, d in lad]
    anon = [d["anon"] for _, d in lad]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax.plot(xs, name, "o-", c="#1b6", lw=2.4, ms=9, label="gene NAMES (web-documented)")
    ax.plot(xs, anon, "s--", c="#c33", lw=2.4, ms=9, label="ANON ids (same vector, names removed)")
    for x, n, a in zip(xs, name, anon):
        ax.annotate(f"{n:.2f}", (x, n), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        ax.annotate(f"{a:.2f}", (x, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)

    xc = len(lad) + 0.4
    cd = get(CROSS[0])
    ax.plot([xc], [cd["name"]], "o", c="#1b6", ms=9, mfc="white", mew=2)
    ax.plot([xc], [cd["anon"]], "s", c="#c33", ms=9, mfc="white", mew=2)
    ax.annotate(f"{cd['name']:.2f}", (xc, cd["name"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.annotate(f"{cd['anon']:.2f}", (xc, cd["anon"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)

    ax.set_xticks(xs + [xc])
    ax.set_xticklabels([lbl for lbl, _ in lad] + [CROSS[1]])
    ax.set_ylabel("output-arm AUROC (CD8-T vs NK)")
    ax.set_ylim(0.40, 1.0)
    ax.set_title("Capability x web-exposure: scale lifts name-grounding,\nanon stays at chance "
                 f"(n={lad[0][1]['n']}/model)")
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(SC, "interaction.png"), dpi=150)
    print(f"wrote {SC}/interaction.png")
    print(f"\n{'model':16s} {'name':>6} {'anon':>6} {'gap':>6}")
    for lbl, d in lad + [(CROSS[1].split(chr(10))[0], cd)]:
        print(f"{lbl:16s} {d['name']:6.3f} {d['anon']:6.3f} {d['name'] - d['anon']:6.3f}")


if __name__ == "__main__":
    main()
