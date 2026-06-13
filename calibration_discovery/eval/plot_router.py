"""v4 figure: per-item placement, the model-vs-specialist router (data-driven, reproducible).

Reads results/router_results.json (written by per_item_router.py) and draws, per model, a track from
always-model (floor) to the per-item oracle (ceiling), with the policies marked as dots: always-model,
the model's own binary DEFER, CONF-route (recommended), and always-specialist. The story: cheap
specialists dominate the solo LLM, CONF-routing reduces to almost-always-call-the-specialist, and a
real gap to the oracle remains because confidence cannot flag the items where the LLM beats the
specialist. Lollipop/dot layout (non-zero x-axis is honest here, no bars-from-zero). No em dashes.

Run: PYTHONPATH=calibration_discovery/eval python calibration_discovery/eval/plot_router.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BRANCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BRANCH, "results")
ORDER = [("claude-haiku-4-5-20251001", "haiku-4.5"),
         ("claude-sonnet-4-6", "sonnet-4.6"),
         ("claude-opus-4-8", "opus-4.8")]
# (key, label, color, marker)
POLICIES = [("always_model", "always-model (solo LLM)", "#9ca3af", "o"),
            ("own_defer", "model's own DEFER", "#b45309", "o"),
            ("conf_route", "CONF-route (recommended)", "#2563eb", "o"),
            ("always_spec", "always-specialist", "#0d9488", "o")]


def main():
    d = json.load(open(os.path.join(RES, "router_results.json")))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)

    for i, (model, label) in enumerate(ORDER):
        m = d["models"][model]
        y = i
        ax.plot([m["always_model"], m["oracle"]], [y, y], color="#d1d5db", lw=2, zorder=1)
        for key, _, color, mark in POLICIES:
            ax.scatter([m[key]], [y], s=130 if key == "conf_route" else 95, color=color,
                       marker=mark, zorder=3, edgecolor="white", lw=1.2)
        ax.scatter([m["oracle"]], [y], s=150, color="#111827", marker="D", zorder=3,
                   edgecolor="white", lw=1.2)
        # gap annotation conf-route -> oracle
        ax.annotate("", xy=(m["oracle"], y + 0.16), xytext=(m["conf_route"], y + 0.16),
                    arrowprops=dict(arrowstyle="<->", color="#6b7280", lw=1))
        ax.text((m["conf_route"] + m["oracle"]) / 2, y + 0.24,
                f"+{m['oracle'] - m['conf_route']:.2f}", ha="center", fontsize=8.5, color="#6b7280")
        ax.text(m["always_model"] - 0.004, y, f"{m['always_model']:.2f}", ha="right", va="center", fontsize=8.5, color="#6b7280")
        ax.text(m["oracle"] + 0.004, y, f"{m['oracle']:.2f}", ha="left", va="center", fontsize=8.5, color="#374151")

    ax.text(0.5 * (d["models"]["claude-opus-4-8"]["conf_route"] + d["models"]["claude-opus-4-8"]["oracle"]),
            2.34, "the LLM beats the specialist here, but CONF cannot flag which items",
            ha="center", fontsize=8.5, color="#6b7280")

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([l for _, l in ORDER], fontsize=10)
    ax.set_ylim(-0.5, 2.7)
    ax.set_xlim(0.55, 0.95)
    ax.set_xlabel("accuracy  (pooled over 8 rungs, per-item)", fontsize=11)
    ax.set_title("Per-item placement: cheap specialists dominate; routing nears them but not the oracle (v4)",
                 fontsize=12, pad=10)
    handles = [Line2D([0], [0], marker=mk, color="white", markerfacecolor=c, markersize=10, label=lab)
               for _, lab, c, mk in POLICIES]
    handles.append(Line2D([0], [0], marker="D", color="white", markerfacecolor="#111827", markersize=10,
                          label="per-item oracle (ceiling)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              fontsize=9, frameon=False)
    ax.grid(True, axis="x", lw=0.4, alpha=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "router_placement.png"), bbox_inches="tight")
    fig.savefig(os.path.join(RES, "router_placement_hires.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, "router_placement.svg"), bbox_inches="tight")
    print(f"[wrote {RES}/router_placement.png (+ _hires.png, .svg)]", flush=True)


if __name__ == "__main__":
    main()
