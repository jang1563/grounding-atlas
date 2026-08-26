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
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_grounding_eval as rge  # noqa: E402
from benchmark_tasks import TASKS, GroundBenchSampler  # noqa: E402

MODELS = os.environ.get("PROBE_MODELS", "claude-opus-4-8,gpt-4o").split(",")
PROBE = os.environ.get("PROBE_TASKS", "methyl/age,single_cell/cd8t_nk:anon,msa/conservation").split(",")
N = int(os.environ.get("PROBE_N", "40"))
TAG = os.environ.get("OUT_TAG", "")
REASON_SUFFIX = ("\n\nReason step by step about the values above, then end your reply with a final "
                 "line exactly: 'Probability: <p>' where <p> is a single number between 0 and 1.")
REASON_SYSTEM = (
    "Reason about the provided representation and property. You may include intermediate analysis. "
    "Your final non-empty line must be exactly 'Probability: <p>', where <p> is one decimal number "
    "between 0 and 1."
)


def parse_final_probability(text):
    """Parse only the declared final-line answer while allowing reasoning above it."""
    match = re.search(
        r"(?:^|\n)Probability:\s*((?:0(?:\.\d*)?|1(?:\.0*)?|\.\d+))\s*$",
        text or "",
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)), True
    return 0.5, False


def complete_reasoning(model, prompt, image=None):
    """Call the provider with the reasoning arm's non-conflicting output contract."""
    return rge.complete(model, prompt, image=image, system=REASON_SYSTEM)


def main():
    rge.DECODE["max_tokens"] = 1024            # give the model room to reason (snap arm uses 16)
    out = {}
    sampler = GroundBenchSampler(seed=0)
    for tid in PROBE:
        t = TASKS[tid]
        items, _ = sampler.task_items(tid, N)
        y = np.array([it["label"] for it in items])
        if t["orient"] == "oppose":
            y = 1 - y
        out[tid] = {"web": t["web"], "n": int(len(y))}
        for m in MODELS:
            probs, outputs, parsed = [], [], []
            for it in items:
                base_prompt = t["prompt"].format(rep=it.get("rep", "")).rstrip()
                if base_prompt.endswith("Probability:"):
                    base_prompt = base_prompt.removesuffix("Probability:").rstrip()
                prompt = base_prompt + REASON_SUFFIX
                output = complete_reasoning(m, prompt, image=it.get("image"))
                probability, valid = parse_final_probability(output)
                probs.append(probability)
                outputs.append(output)
                parsed.append(valid)
            p = np.array(probs)
            a = rge.auroc(p, y)
            lo, hi = rge.ci(rge.auroc, p, y, rge._task_rng(0, tid, f"budget-{m}"))
            out[tid][m] = {
                "budget_auroc": round(float(a), 3),
                "ci": [lo, hi],
                "valid_response_rate": round(float(np.mean(parsed)), 3),
                "n_invalid": int(np.sum(~np.asarray(parsed, dtype=bool))),
                "raw": [
                    {
                        "id": item["id"],
                        "entity_id": item["entity_id"],
                        "target_label": int(target),
                        "prob": float(probability),
                        "parse_valid": bool(valid),
                        "output": output,
                    }
                    for item, target, probability, valid, output
                    in zip(items, y, probs, parsed, outputs)
                ],
            }
            print(f"  {tid:28s} web={t['web']:5s} {m:20s} budget AUROC={a:.3f} ({lo}, {hi})", flush=True)
    os.makedirs(os.path.join(rge.OUT), exist_ok=True)
    json.dump(out, open(os.path.join(rge.OUT, f"budget_arm{TAG}.json"), "w"), indent=2)
    print(f"\nwrote results/benchmark/budget_arm{TAG}.json")


if __name__ == "__main__":
    main()
