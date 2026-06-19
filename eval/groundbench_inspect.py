"""Run GroundBench inside Inspect (UK AISI's inspect_ai).

This reuses the SAME task registry, prompts, decode, and probability parser as
eval/run_grounding_eval.py, so an Inspect run is the same benchmark, just driven by Inspect's model
providers, logging, and viewer. The reported metric is per-task AUROC (with the registry's a-priori
label orientation) -- the same grounding number the standalone harness reports.

Examples:
  inspect eval eval/groundbench_inspect.py@groundbench -T task_id=single_cell/cd8t_nk:name \\
      --model anthropic/claude-opus-4-8
  inspect eval eval/groundbench_inspect.py@groundbench_all --model openai/gpt-4o --limit 50

Use the standalone harness (eval/run_grounding_eval.py) for the leaderboard scorecard with calibration,
gap-to-ceiling, memo_delta, and bootstrap CIs; use this adapter to run GroundBench from Inspect.
"""
import os
import sys

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText, GenerateConfig
from inspect_ai.scorer import SampleScore, Score, grouped, metric, scorer
from inspect_ai.solver import generate, system_message
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS, task_items  # noqa: E402
from run_grounding_eval import DECODE, SYSTEM, parse_prob  # noqa: E402


def _dataset(task_ids, n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for tid in task_ids:
        t = TASKS[tid]
        items, _ = task_items(tid, n, rng)
        for i, it in enumerate(items):
            text = t["prompt"].format(rep=it.get("rep", ""))
            if it.get("image"):   # multimodal task: image + the text prompt
                inp = [ChatMessageUser(content=[ContentImage(image=it["image"]), ContentText(text=text)])]
            else:
                inp = text
            md = {"task": tid, "label": int(it["label"]), "orient": t["orient"], "web": t["web"]}
            samples.append(Sample(input=inp, target=str(int(it["label"])), id=f"{tid}#{i}", metadata=md))
    rng.shuffle(samples)   # mix classes so a user's --limit still gets both labels
    return MemoryDataset(samples)


@metric
def auroc():
    """Per-task grounding AUROC over (oriented label, parsed probability). Robust to the metric being
    handed Score or SampleScore objects; label/orientation are carried in the Score metadata."""
    def compute(scores: list[SampleScore]) -> float:
        ys, ps = [], []
        for s in scores:
            sc = getattr(s, "score", s)      # SampleScore.score, or a bare Score
            md = sc.metadata or {}
            y = int(md.get("label", 0))
            if md.get("orient") == "oppose":
                y = 1 - y
            ys.append(y)
            ps.append(float(sc.value))
        if len(set(ys)) < 2:
            return float("nan")
        return float(roc_auc_score(ys, ps))
    return compute


@scorer(metrics=[auroc()])
def prob_scorer():
    async def score(state, target):  # noqa: ARG001 (label carried via sample metadata)
        out = state.output.completion or ""
        md = state.metadata or {}
        return Score(value=parse_prob(out), answer=out,
                     metadata={"label": md.get("label"), "orient": md.get("orient"),
                               "task": md.get("task"), "web": md.get("web")})
    return score


def _config():
    return GenerateConfig(temperature=DECODE["temperature"], max_tokens=DECODE["max_tokens"])


@task
def groundbench(task_id="admet/herg", n=100, seed=0):
    """One GroundBench task; metric = AUROC."""
    if task_id not in TASKS:
        raise ValueError(f"unknown task_id {task_id!r}; choose from {', '.join(CORE)}")
    return Task(
        dataset=_dataset([task_id], n, seed),
        solver=[system_message(SYSTEM), generate()],
        scorer=prob_scorer(),
        config=_config(),
        name=f"groundbench/{task_id}",
    )


@task
def groundbench_all(n=100, seed=0):
    """All CORE tasks in one run; metric = AUROC grouped per task."""
    return Task(
        dataset=_dataset(list(CORE), n, seed),
        solver=[system_message(SYSTEM), generate()],
        scorer=prob_scorer(),
        metrics=[grouped(auroc(), "task")],
        config=_config(),
        name="groundbench/all",
    )
