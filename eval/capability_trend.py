"""Descriptive capability trend for the paired single-cell name/obscure/anon contrast.

  contrast ratio = (obscure - anon) / (name - anon)

The ratio describes how the obscure-token condition lies between named and anonymous
representations on identical entities. It does not identify token familiarity,
documentation, memory, or any other mechanism. Existing results must come from the
current prompt and score schema; legacy unversioned scorecards are rejected.

Run: python eval/capability_trend.py     (writes results/benchmark/capability_trend.{json,md})
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_grounding_eval as rge  # noqa: E402
from benchmark_tasks import TASKS, GroundBenchSampler  # noqa: E402

EXISTING = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]
NEW = ["claude-haiku-4-5-20251001", "gpt-4o-mini"]
CONDS = {"name": "single_cell/cd8t_nk:name",
         "obscure": "single_cell/cd8t_nk:obscure",
         "anon": "single_cell/cd8t_nk:anon"}
ORDER = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8", "gpt-4o-mini", "gpt-4o"]
N = 200


def run_model(m):
    sampler = GroundBenchSampler(seed=0)
    out = {}
    for k, tid in CONDS.items():
        t = TASKS[tid]
        items = sampler.task_condition_items(tid, N)["matched"]
        y = np.array([it["label"] for it in items])
        if t["orient"] == "oppose":
            y = 1 - y
        p = np.array([rge.parse_prob(rge.complete(m, t["prompt"].format(rep=it.get("rep", "")),
                                                  image=it.get("image"))) for it in items])
        out[k] = round(float(rge.auroc(p, y)), 3)
        print(f"  {m:28s} {k:8s} AUROC={out[k]}", flush=True)
    return out


def from_scorecard(m):
    model_dir = os.path.join(rge.OUT, f"{m.replace('/', '_')}__{rge.PROMPT_VERSION}")
    manifest = json.load(open(os.path.join(model_dir, "manifest.json")))
    if (
        manifest.get("prompt_version") != rge.PROMPT_VERSION
        or manifest.get("score_schema") != rge.SCORE_SCHEMA
    ):
        raise ValueError(f"{m}: legacy or incompatible GroundBench contract")
    sc = json.load(open(os.path.join(model_dir, "scorecard.json")))
    pair_artifact = json.load(open(os.path.join(model_dir, "pair_group_comparisons.json")))
    compared_pairs = {
        frozenset((row["task_a"], row["task_b"]))
        for row in pair_artifact.get("comparisons", [])
    }
    required_pairs = {
        frozenset((left, right))
        for left in CONDS.values()
        for right in CONDS.values()
        if left != right
    }
    if not required_pairs.issubset(compared_pairs):
        raise ValueError(f"{m}: missing current entity-paired single-cell comparisons")
    return {k: sc[tid]["output_auroc"] for k, tid in CONDS.items()}


def main():
    res = {m: from_scorecard(m) for m in EXISTING}
    for m in NEW:
        res[m] = run_model(m)
    for v in res.values():
        gap = v["name"] - v["anon"]
        v["contrast_ratio"] = round((v["obscure"] - v["anon"]) / gap, 2) if gap > 0 else None
    payload = {
        "contract": {
            "prompt_version": rge.PROMPT_VERSION,
            "score_schema": rge.SCORE_SCHEMA,
            "interpretation": "descriptive paired-representation contrast; mechanism not identified",
        },
        "models": res,
    }
    json.dump(payload, open(os.path.join(rge.OUT, "capability_trend.json"), "w"), indent=2)
    lines = ["# Capability trend: paired representation contrast", "",
             "ratio = (obscure - anon) / (name - anon). This is a descriptive location of the",
             "obscure-token condition between named and anonymous conditions on shared entities;",
             "it does not decompose a causal mechanism.", "",
             "| model | name | obscure | anon | contrast ratio |", "|---|---|---|---|---|"]
    for m in ORDER:
        if m in res:
            v = res[m]
            lines.append(
                f"| {m} | {v['name']} | {v['obscure']} | {v['anon']} | "
                f"{v['contrast_ratio']} |"
            )
    open(os.path.join(rge.OUT, "capability_trend.md"), "w").write("\n".join(lines) + "\n")
    print("\nwrote results/benchmark/capability_trend.{json,md}")


if __name__ == "__main__":
    main()
