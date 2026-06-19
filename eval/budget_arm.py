"""Budget arm: disentangle the web-exposure story from a compute-budget confound.

The leaderboard uses a 16-token SNAP decode, which for the NUMERIC web-zero task (methylation betas ->
age) conflates two things: "the representation-to-property mapping is web-undocumented" (web-exposure)
vs "integrating a numeric vector is impossible in 16 tokens" (compute budget). This arm re-runs a few
tasks with a REASONING budget (high max_tokens + step-by-step) and compares to the snap AUROC.

Pre-registered predictions:
  - methyl/age (numeric, web-zero): if the snap gap was a compute limit, AUROC RISES with budget;
    if it is web-exposure (an empirical clock you cannot derive without the documented coefficients),
    it STAYS near chance.
  - single_cell/cd8t_nk:anon (symbolic, web-zero): reasoning should NOT help (the anon ids carry no
    web knowledge regardless of budget) -> stays near chance either way.
  - msa/conservation (web-rich control): already grounds; budget should not hurt it.

Run:  python eval/budget_arm.py     (writes results/benchmark/budget_arm.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_grounding_eval as rge  # noqa: E402
from benchmark_tasks import TASKS, task_items  # noqa: E402

MODELS = os.environ.get("PROBE_MODELS", "claude-opus-4-8,gpt-4o").split(",")
PROBE = os.environ.get("PROBE_TASKS", "methyl/age,single_cell/cd8t_nk:anon,msa/conservation").split(",")
N = int(os.environ.get("PROBE_N", "40"))
TAG = os.environ.get("OUT_TAG", "")
REASON_SUFFIX = ("\n\nReason step by step about the values above, then end your reply with a final "
                 "line exactly: 'Probability: <p>' where <p> is a single number between 0 and 1.")


def main():
    rge.DECODE["max_tokens"] = 1024            # give the model room to reason (snap arm uses 16)
    out = {}
    for tid in PROBE:
        t = TASKS[tid]
        rng = np.random.default_rng(0)
        items, _ = task_items(tid, N, rng)
        y = np.array([it["label"] for it in items])
        if t["orient"] == "oppose":
            y = 1 - y
        out[tid] = {"web": t["web"], "n": int(len(y))}
        for m in MODELS:
            probs = []
            for it in items:
                prompt = t["prompt"].format(rep=it.get("rep", "")) + REASON_SUFFIX
                probs.append(rge.parse_prob(rge.complete(m, prompt, image=it.get("image"))))
            p = np.array(probs)
            a = rge.auroc(p, y)
            lo, hi = rge.ci(rge.auroc, p, y, rng)
            out[tid][m] = {"budget_auroc": round(float(a), 3), "ci": [lo, hi]}
            print(f"  {tid:28s} web={t['web']:5s} {m:20s} budget AUROC={a:.3f} ({lo}, {hi})", flush=True)
    os.makedirs(os.path.join(rge.OUT), exist_ok=True)
    json.dump(out, open(os.path.join(rge.OUT, f"budget_arm{TAG}.json"), "w"), indent=2)
    print(f"\nwrote results/benchmark/budget_arm{TAG}.json")


if __name__ == "__main__":
    main()
