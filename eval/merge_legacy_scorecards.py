"""One-off: consolidate the pre-registry results into unified GroundBench scorecards.

The ADMET tasks were scored by the v3 harness (results/benchmark/<model>/scorecard.json) and the
single-cell tasks by eval/single_cell_arm.py (results/benchmark/single_cell[/mono]/<model>_raw.jsonl)
with identical SYSTEM prompt, decode, and parser, so the numbers are consistent. This stitches them
into one scorecard.json per model (the registry's task ids) and regenerates the leaderboard, with no
new API calls. Future models are produced directly by run_grounding_eval.evaluate(); this is only
for the three models measured before the registry existed.

Run:  python eval/merge_legacy_scorecards.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import TASKS  # noqa: E402
from run_grounding_eval import OUT, score_task, update_leaderboard  # noqa: E402

MODELS = ["claude-sonnet-4-6", "gpt-4o", "claude-opus-4-8"]
SC_TASKS = {  # task id -> (raw file relative to OUT, condition)
    "single_cell/cd8t_nk:name": ("single_cell/{m}_raw.jsonl", "name"),
    "single_cell/cd8t_nk:anon": ("single_cell/{m}_raw.jsonl", "anon"),
    "single_cell/mono:name":    ("single_cell/mono/{m}_raw.jsonl", "name"),
    "single_cell/mono:anon":    ("single_cell/mono/{m}_raw.jsonl", "anon"),
}


def main():
    rng = np.random.default_rng(0)
    for m in MODELS:
        sc = json.load(open(os.path.join(OUT, m, "scorecard.json")))
        for t in list(sc):                      # tag the ADMET tasks web-rich
            sc[t].setdefault("web_exposure", "rich")
        for task, (ft, cond) in SC_TASKS.items():
            path = os.path.join(OUT, ft.format(m=m))
            if not os.path.exists(path):
                continue
            rows = [r for r in (json.loads(line) for line in open(path)) if r["condition"] == cond]
            y = np.array([r["label"] for r in rows])
            p = np.array([r["prob"] for r in rows])
            rec = score_task(p, y, None, None, TASKS[task]["ceiling"], rng)
            rec["orientation"], rec["web_exposure"] = "align", TASKS[task]["web"]
            sc[task] = rec
        json.dump(sc, open(os.path.join(OUT, m, "scorecard.json"), "w"), indent=2)
        print(f"{m}: unified scorecard -> {len(sc)} tasks "
              f"( admet/herg {sc['admet/herg']['output_auroc']}, "
              f"single_cell/cd8t_nk:name {sc.get('single_cell/cd8t_nk:name', {}).get('output_auroc')})")
    update_leaderboard()
    print(f"wrote unified LEADERBOARD.md ({len(MODELS)} models)")


if __name__ == "__main__":
    main()
