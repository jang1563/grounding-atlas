"""Capability trend: does the token-familiarity SHARE of the single-cell grounding gap rise with model
capability? Tests the decomposition in token_familiarity.md across a capability ladder by adding weaker
models (Haiku, gpt-4o-mini) on the name/obscure/anon triple, without touching the leaderboard.

  share = (obscure - anon) / (name - anon)

= the fraction of the name/anon gap explained by token-familiarity/reasoning (vs mapping-documentation).
Prediction: the share rises with capability (opus ~0.80 > sonnet ~0.50 > gpt-4o ~0.33 so far), so within
each family Haiku < Sonnet < Opus and gpt-4o-mini < gpt-4o.

Run: python eval/capability_trend.py     (writes results/benchmark/capability_trend.{json,md})
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_grounding_eval as rge  # noqa: E402
from benchmark_tasks import TASKS, task_items  # noqa: E402

EXISTING = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]
NEW = ["claude-haiku-4-5-20251001", "gpt-4o-mini"]
CONDS = {"name": "single_cell/cd8t_nk:name",
         "obscure": "single_cell/cd8t_nk:obscure",
         "anon": "single_cell/cd8t_nk:anon"}
ORDER = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8", "gpt-4o-mini", "gpt-4o"]
N = 200


def run_model(m):
    rng = np.random.default_rng(0)
    out = {}
    for k, tid in CONDS.items():
        t = TASKS[tid]
        items, _ = task_items(tid, N, rng)
        y = np.array([it["label"] for it in items])
        if t["orient"] == "oppose":
            y = 1 - y
        p = np.array([rge.parse_prob(rge.complete(m, t["prompt"].format(rep=it.get("rep", "")),
                                                  image=it.get("image"))) for it in items])
        out[k] = round(float(rge.auroc(p, y)), 3)
        print(f"  {m:28s} {k:8s} AUROC={out[k]}", flush=True)
    return out


def from_scorecard(m):
    sc = json.load(open(os.path.join(rge.OUT, m, "scorecard.json")))
    return {k: sc[tid]["output_auroc"] for k, tid in CONDS.items()}


def main():
    res = {m: from_scorecard(m) for m in EXISTING}
    for m in NEW:
        res[m] = run_model(m)
    for v in res.values():
        gap = v["name"] - v["anon"]
        v["familiarity_share"] = round((v["obscure"] - v["anon"]) / gap, 2) if gap > 0 else None
    json.dump(res, open(os.path.join(rge.OUT, "capability_trend.json"), "w"), indent=2)
    lines = ["# Capability trend: token-familiarity share vs model capability", "",
             "share = (obscure - anon) / (name - anon), the fraction of the single-cell name/anon gap due",
             "to token-familiarity + reasoning rather than mapping-documentation. Prediction "
             "(token_familiarity.md): rises with capability.", "",
             "| model | name | obscure | anon | familiarity share |", "|---|---|---|---|---|"]
    for m in ORDER:
        if m in res:
            v = res[m]
            lines.append(f"| {m} | {v['name']} | {v['obscure']} | {v['anon']} | {v['familiarity_share']} |")
    open(os.path.join(rge.OUT, "capability_trend.md"), "w").write("\n".join(lines) + "\n")
    print("\nwrote results/benchmark/capability_trend.{json,md}")


if __name__ == "__main__":
    main()
