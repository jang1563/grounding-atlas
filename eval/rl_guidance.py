"""Experiment-3 (RL_ENV_PREREG) arm B: EXTERNAL inference-time guidance (Best-of-N) of the FROZEN
generator. This is the 'orchestrate' analogue (frozen FM + external reward steering at inference),
our route-don't-train winner's generative sibling.

At reward-query budget Q: draw Q valid-novel-unique designs from the FROZEN generator, rank by the
block-R reward, keep the top-M; the held-out block-O oracle judges the delivered M. Sweeps Q to
trace the oracle-success-vs-budget curve that arm A (internalized RL) must beat at MATCHED Q. Dumps
per-design (SMILES + reward + oracle) so compare_rl_orchestrate.py can run the scaffold-clustered
two-sample bootstrap. Generator is never updated (drift guard: this is route, not train).

Usage: sbatch --export=ALL,E3_SCRIPT=rl_guidance.py,RL_ENDPOINT=herg,RL_M=500 eval/cayuga_rl.sbatch
No em dashes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_common import (  # noqa: E402
    OUT,
    ROOT,
    build_oracle,
    build_reward,
    load_generator,
    oracle_scores,
    reward_scores,
    sample_designs,
    valid_novel_unique,
)

ENDPOINT = os.environ.get("RL_ENDPOINT", "herg")
M = int(os.environ.get("RL_M", "500"))                 # designs delivered per budget
BUDGETS = [M, 2 * M, 5 * M, 10 * M]                     # Q = reward-query budget sweep (one decade)


def main():
    model, vocab, temp = load_generator()
    members = build_reward(ENDPOINT)
    rf, bar = build_oracle(ENDPOINT)
    qmax = max(BUDGETS)
    print(f"[armB] endpoint={ENDPOINT} M={M} budgets={BUDGETS} temp={temp} bar={bar:.3f}", flush=True)

    pool = []
    while len(pool) < qmax:
        pool = list(dict.fromkeys(pool + valid_novel_unique(sample_designs(model, vocab, qmax, temp))))
        print(f"[armB] pool {len(pool)}/{qmax}", flush=True)
    pool = pool[:qmax]
    rew = reward_scores(members, pool)
    orac = oracle_scores(rf, pool)
    opass = (orac > bar).astype(int)
    print(f"[armB] pool oracle-pass base rate={opass.mean():.4f}", flush=True)

    rows, delivered = [], {}
    for q in BUDGETS:
        order = np.argsort(-rew[:q])[:M]                # top-M by reward within the budget-Q draw
        passes = int(opass[order].sum())
        rows.append({"budget_Q": q, "delivered_M": int(min(M, q)), "oracle_pass": passes,
                     "oracle_pass_rate": round(passes / min(M, q), 4),
                     "mean_reward_topM": round(float(rew[order].mean()), 4),
                     "mean_oracle_topM": round(float(orac[order].mean()), 4)})
        delivered[str(q)] = order.tolist()
        print(f"  Q={q}: top-{M} oracle-pass={passes} ({passes / min(M, q):.3f}) "
              f"meanReward={rew[order].mean():.3f}", flush=True)

    dump = {"arm": "B_guidance", "endpoint": ENDPOINT, "temp": temp, "bar": bar, "M": M,
            "budgets": rows, "delivered_idx": delivered, "designs": pool,
            "reward": [round(float(x), 4) for x in rew], "oracle": [round(float(x), 4) for x in orac],
            "oracle_pass": opass.tolist()}
    npos = os.environ.get("REWARD_NPOS", "")
    out = os.path.join(OUT, f"{ENDPOINT}_armB_guidance{('_np' + npos) if npos else ''}.json")
    json.dump(dump, open(out, "w"))
    print(f"[armB] saved -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
