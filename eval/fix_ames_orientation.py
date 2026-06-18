"""One-shot correction: re-score the ames rung under the corrected `oppose` orientation.

A structural-alert audit (eval/analyze_ames.py) showed the ames label direction was inverted
(label-0 is the nitroaromatic-rich, mutagenic class), so the committed scorecards scored ames
against the wrong label and read as anti-grounding (~0.32). This re-scores ames from each
model's *already emitted* probabilities (results/benchmark/<model>/raw.jsonl) with the oriented
label (positive = label-0) -- exact, no new API calls -- and regenerates the leaderboard. The
harness CLAUSES entry is already fixed, so future runs produce the correct orientation natively.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_grounding_eval import OUT, aurc, auroc, ci, ece, sel_acc, update_leaderboard  # noqa: E402

MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]


def main():
    rng = np.random.default_rng(0)
    for m in MODELS:
        scp = os.path.join(OUT, m, "scorecard.json")
        sc = json.load(open(scp))
        rec = sc["admet/ames"]
        raw = [json.loads(line) for line in open(os.path.join(OUT, m, "raw.jsonl"))
               if '"admet/ames"' in line]
        prob = np.array([r["prob"] for r in raw])
        y = 1 - np.array([r["label"] for r in raw])   # oriented oppose: positive = label-0
        a = auroc(prob, y)
        old = rec["output_auroc"]
        rec["output_auroc"] = round(a, 3)
        rec["output_auroc_ci"] = ci(auroc, prob, y, rng)
        rec["ece"] = round(ece(prob, y), 3)
        rec["aurc"] = round(aurc(prob, y), 3)
        rec["sel_acc_50"] = round(sel_acc(prob, y), 3)
        rec["gap"] = round(rec["ceiling"] - a, 3) if rec["ceiling"] is not None else None
        if rec["memo_delta"] is not None:
            rec["memo_delta"] = round(-rec["memo_delta"], 3)   # orientation flip negates memo_delta
        rec["orientation"] = "oppose"
        json.dump(sc, open(scp, "w"), indent=2)
        print(f"{m}: ames {old} -> {rec['output_auroc']} {rec['output_auroc_ci']} "
              f"ECE {rec['ece']} AURC {rec['aurc']} gap {rec['gap']} memoΔ {rec['memo_delta']}")
    update_leaderboard()
    print("leaderboard regenerated")


if __name__ == "__main__":
    main()
