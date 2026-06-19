"""Permissioning half: does the A-PRIORI web-exposure tag beat the model's SELF-CONFIDENCE for
deciding when to trust it?

Selective prediction (abstain) on the pooled name+anon items per model. Two ways to choose
which items to answer vs abstain on:
  - a-priori tag: answer name items (web-documented), abstain on anon -- known before any call.
  - self-confidence: answer the most-confident items by |P-0.5|.
Lower area-under-risk-coverage (AURC) is better. If the model is confidently wrong on anon, the
self-confidence ranking keeps those, while the a-priori tag defers them. Also reports whether the
model even self-flags anon (mean confidence on name vs anon).

Reads results/benchmark/single_cell/<model>_raw.jsonl. Writes deferral.png + prints a table.
Run:  python eval/single_cell_permission.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(os.path.dirname(HERE), "results", "benchmark", "single_cell")
MODELS = [("claude-haiku-4-5-20251001", "Haiku 4.5"), ("claude-sonnet-4-6", "Sonnet 4.6"),
          ("claude-opus-4-8", "Opus 4.8"), ("gpt-4o", "GPT-4o")]


def load(model):
    rows = [json.loads(line) for line in open(os.path.join(SC, model.replace("/", "_") + "_raw.jsonl"))]
    prob = np.array([r["prob"] for r in rows])
    y = np.array([r["label"] for r in rows])
    name = np.array([r["condition"] == "name" for r in rows])
    correct = ((prob > 0.5).astype(int) == y).astype(float)
    conf = np.abs(prob - 0.5)
    return prob, y, name, correct, conf


def risk_curve(order, correct):
    err = 1 - correct[order]
    return np.cumsum(err) / (np.arange(len(err)) + 1)


def main():
    recs = []
    print(f"{'model':12s} {'conf_name':>9} {'conf_anon':>9} | {'AURC_self':>9} {'AURC_tag':>8} | "
          f"{'acc@.5_self':>11} {'acc@.5_tag':>10}")
    for model, lbl in MODELS:
        prob, y, name, correct, conf = load(model)
        n = len(y)
        self_order = np.argsort(-conf)                       # most confident first
        tag_order = np.lexsort((-conf, ~name))               # name first (then by conf), anon last
        rc_self, rc_tag = risk_curve(self_order, correct), risk_curve(tag_order, correct)
        half = n // 2
        rec = {"conf_name": float(conf[name].mean()), "conf_anon": float(conf[~name].mean()),
               "aurc_self": float(rc_self.mean()), "aurc_tag": float(rc_tag.mean()),
               "acc_self_50": float(correct[self_order[:half]].mean()),
               "acc_tag_50": float(correct[tag_order[:half]].mean())}
        print(f"{lbl:12s} {rec['conf_name']:9.3f} {rec['conf_anon']:9.3f} | {rec['aurc_self']:9.3f} "
              f"{rec['aurc_tag']:8.3f} | {rec['acc_self_50']:11.3f} {rec['acc_tag_50']:10.3f}")
        json.dump(rec, open(os.path.join(SC, model.replace("/", "_") + "_perm.json"), "w"), indent=2)
        recs.append((lbl, rec))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = [lbl for lbl, _ in recs]
    x = np.arange(len(recs))
    w = 0.38
    ax1.bar(x - w / 2, [r["conf_name"] for _, r in recs], w, label="gene names", color="#1b6")
    ax1.bar(x + w / 2, [r["conf_anon"] for _, r in recs], w, label="anon (web-zero)", color="#c33")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=12)
    ax1.set_ylabel("mean self-confidence  |P-0.5|")
    ax1.set_title("Does the model self-flag anon?\nlow anon bar = it knows it is guessing")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax2.bar(x - w / 2, [r["aurc_self"] for _, r in recs], w, label="defer by self-confidence", color="#c33")
    ax2.bar(x + w / 2, [r["aurc_tag"] for _, r in recs], w, label="defer by a-priori web-exposure tag", color="#1b6")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=12)
    ax2.set_ylabel("AURC  (lower = better)")
    ax2.set_title("Deferral quality:\na-priori tag robust, self-confidence is not")
    ax2.legend(fontsize=9, framealpha=0.9)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(SC, "deferral.png"), dpi=150)
    print(f"\nwrote {SC}/deferral.png")


if __name__ == "__main__":
    main()
