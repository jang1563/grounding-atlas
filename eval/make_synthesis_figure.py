"""Core synthesis figure (premium, intuitive). Every rung on (encoding gap, verbalization gap),
colored by what CLOSES the gap. Auto-placed labels (adjustText) remove overlap; leader lines stop
before the text. Legend sits in the empty upper-right. Plain-language axes and regions for a
non-specialist reader. ASCII arrows (font-safe). Standard + hi-res PNG. No em dashes."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, encoding_gap, verbalization_gap, category)
D = [
    ("DNA -> promoter",          0.009, 0.484, "closes"),
    ("single-cell -> T cell (gene names)", 0.006, 0.486, "closes"),
    ("variant -> pathogenic (text)", 0.167, 0.196, "closes"),
    ("variant -> pathogenic (sequence)", 0.222, 0.246, "closes"),
    ("SMILES -> hERG block",      0.038, 0.334, "partial"),
    ("histopathology -> tumor",   0.073, 0.364, "partial"),
    ("single-cell -> T cell (anonymized)", 0.025, 0.467, "specialist"),
    ("mass spectrum -> hERG",     0.096, 0.227, "specialist"),
    ("methylation -> age",        0.017, 0.198, "specialist"),
    ("NMR -> hERG",               0.119, 0.313, "specialist"),
    ("SMILES -> CYP3A4",          0.061, 0.182, "specialist"),
    ("molecule image -> hERG",    0.096, 0.298, "specialist"),
    ("molecular graph -> hERG",   0.157, 0.250, "structure"),
    ("3-D coordinates -> hERG",   0.156, 0.180, "structure"),
    ("MSA column -> conserved",   0.000, 0.205, "grounds"),
    ("RNA -> coding",             0.064, 0.146, "grounds"),
    ("protein -> stability",      0.090, 0.123, "grounds"),
]
COL = {"grounds": "#1e88e5", "closes": "#2e9e4f", "partial": "#ef8c0c",
       "specialist": "#e0392f", "structure": "#7c4dff"}
ORDER = ["grounds", "closes", "partial", "specialist", "structure"]
LEG = {
    "grounds":   "already handled well (says what it knows)",
    "closes":    "a bigger model will say it (web-documented)",
    "partial":   "a bigger model helps, but only part-way",
    "specialist": "no model will say it - call a specialist tool",
    "structure": "needs a 3-D / structure model to even grasp it",
}

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "axes.edgecolor": "#cccccc"})

fig, ax = plt.subplots(figsize=(15.5, 10.6))
fig.subplots_adjust(top=0.82, left=0.078, right=0.975, bottom=0.115)

# region shading
ax.add_patch(Rectangle((-0.02, 0.295), 0.15, 0.315, color="#f6c343", alpha=0.10, zorder=0))    # top-left, the story
ax.add_patch(Rectangle((0.13, 0.05), 0.185, 0.56, color="#7c4dff", alpha=0.055, zorder=0))      # right, can't encode
ax.axvline(0.13, ls=(0, (5, 5)), color="#7c4dff", lw=1.1, alpha=0.4, zorder=1)

# region captions (plain language)
ax.text(0.045, 0.585, "knows it inside, but won't say it out loud", fontsize=15,
        color="#9a6a00", style="italic", ha="center", zorder=5, fontweight="medium")
ax.text(0.138, 0.55, "can't even work it out", fontsize=14,
        color="#5e35b1", style="italic", ha="left", zorder=5, fontweight="medium")
ax.text(0.045, 0.082, "knows it, and says it", fontsize=13,
        color="#666", style="italic", ha="center", zorder=5)

# points + labels (leader lines stop before the glyphs via shrinkB)
texts = []
xs = [ex for _, ex, _, _ in D]
ys = [vy for _, _, vy, _ in D]
for label, ex, vy, cat in D:
    ax.scatter(ex, vy, s=150, color=COL[cat], edgecolor="white", linewidth=1.6, zorder=3, alpha=0.96)
    texts.append(ax.text(ex, vy, label, fontsize=9.5, color="#1a1a1a", zorder=4))
# adjustText 1.4 API (the old expand_*/force_points/only_move kwargs are silently ignored on 1.x).
# Pass the point coords so labels repel from glyphs; pin the RNG so placement is reproducible.
np.random.seed(7)
adjust_text(texts, x=xs, y=ys, ax=ax,
            force_text=(0.6, 1.15), force_static=(0.5, 0.85), force_pull=(0.004, 0.008),
            expand=(1.6, 2.0), min_arrow_len=10, max_move=90, ensure_inside_axes=True,
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6))

ax.set_xlim(-0.02, 0.315)
ax.set_ylim(0.05, 0.615)
ax.set_xlabel("how hard it is for the model to work the answer out at all   (to the right = harder)\n"
              "encoding gap  =  specialist accuracy  minus  what the model internally separates",
              fontsize=12.5, labelpad=12, color="#333")
ax.set_ylabel("how much it works out inside but cannot put into words   (up = more)\n"
              "verbalization gap  =  internal  minus  spoken",
              fontsize=12.5, labelpad=12, color="#333")
ax.grid(True, alpha=0.16, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# title + subtitle
fig.text(0.078, 0.95, "What an AI works out, versus what it will say",
         fontsize=22, fontweight="bold", color="#111", ha="left")
fig.text(0.078, 0.90,
         "17 ways to hand a biology question to a language model. Almost every one lands in the top-left:\n"
         "the model works the answer out inside its activations, but cannot put it into words. Color shows what fixes that.",
         fontsize=12.5, color="#555", ha="left", linespacing=1.5)

# legend in the empty upper-right
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[k], markeredgecolor="white",
                  markersize=13, label=LEG[k]) for k in ORDER]
leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.999, 0.995), fontsize=10.8,
                title="what closes the gap", title_fontsize=11.5, framealpha=0.97, edgecolor="#dddddd",
                borderpad=1, labelspacing=0.75)
leg.get_title().set_fontweight("bold")

out = os.path.join(ROOT, "results", "synthesis_figure.png")
outhi = os.path.join(ROOT, "results", "synthesis_figure_hires.png")
fig.savefig(out, dpi=200, facecolor="white")
fig.savefig(outhi, dpi=300, facecolor="white")
print(f"wrote {out} (dpi 200) and {outhi} (dpi 300)")
