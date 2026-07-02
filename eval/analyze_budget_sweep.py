"""Experiment-3 budget sweep: arm A (internalized RL) vs arm B (external guidance) at MATCHED
reward-query budget Q, across a Q sweep, to show route-don't-train holds across budgets (and to
tighten via a larger M). Reads {endpoint}_armA_ppo_s0_Q{Q}.json (one per budget) + the single
{endpoint}_armB_guidance.json (its delivered_idx has the top-M set per budget). Reuses the Murcko
clustering + two-sample cluster bootstrap from compare_rl_orchestrate. No em dashes.
Usage: RL_ENDPOINT=herg RL_BUDGETS=2500,5000,10000 python eval/analyze_budget_sweep.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_rl_orchestrate import ENDPOINT, OUT, ROOT, murcko, two_sample_cluster_boot  # noqa: E402

BUDGETS = [int(x) for x in os.environ.get("RL_BUDGETS", "2500,5000,10000").split(",")]


def main():
    b = json.load(open(os.path.join(OUT, f"{ENDPOINT}_armB_guidance.json")))
    print(f"=== budget sweep: A(Q) vs B(Q), matched reward-query budget ({ENDPOINT}) ===", flush=True)
    rows = []
    for q in BUDGETS:
        af = os.path.join(OUT, f"{ENDPOINT}_armA_ppo_s0_Q{q}.json")
        if not os.path.isfile(af) or str(q) not in b["delivered_idx"]:
            print(f"  Q={q}: missing (armA={os.path.isfile(af)} armB={str(q) in b['delivered_idx']})", flush=True)
            continue
        a = json.load(open(af))
        a_pass = np.array(a["oracle_pass_vec"])
        idx = b["delivered_idx"][str(q)]
        b_pass = np.array([b["oracle_pass"][i] for i in idx])
        pt, ci, na, nb = two_sample_cluster_boot(a_pass, [murcko(s) for s in a["designs"]],
                                                 b_pass, [murcko(b["designs"][i]) for i in idx])
        ra, rb = float(a_pass.mean()), float(b_pass.mean())
        print(f"  Q={q:6d}:  A {int(a_pass.sum())}/{len(a_pass)}={ra:.4f}   B {int(b_pass.sum())}/{len(b_pass)}={rb:.4f}"
              f"   (A-B)={pt:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
        rows.append({"Q": q, "rate_A_rl": round(ra, 4), "rate_B_guidance": round(rb, 4),
                     "diff_A_minus_B": round(pt, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]})
    tie = all(r["ci95"][0] <= 0 <= r["ci95"][1] or abs(r["diff_A_minus_B"]) < 0.03 for r in rows)
    print(f"  -> {'ALL budgets CONFIRM: A ties B across the sweep (route-dont-train)' if tie else 'a budget SEPARATES'}",
          flush=True)
    out = os.path.join(OUT, f"{ENDPOINT}_budget_sweep.json")
    json.dump({"endpoint": ENDPOINT, "sweep": rows, "all_confirm": tie}, open(out, "w"), indent=1)
    print(f"  saved -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
