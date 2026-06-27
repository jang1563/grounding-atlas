"""Experiment-2 paired analysis (the prereg's score_arm + paired_cluster_boot). Reads the per-item dumps
(bridge: test_p + bypass_p; orchestrate: in_property_p + transfer_p), aligns by the shared test_id, and
computes the formal H1/H2 paired-difference CIs on scaffold groups:
  H1b  bridge vs its LLM-bypass head (does routing through the frozen LLM add anything),
  H1   bridge vs orchestrate (placement ordering),
plus score_arm (AUROC / AURC / temperature-scaled ECE) per arm. No em dashes.
Usage: python analyze_bridge3way.py   (reads results/benchmark/bridge3way + results/orchestrate_arm.json)
"""
import json
import os

import numpy as np
from probe_common import paired_cluster_boot, score_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B3 = os.path.join(ROOT, "results", "benchmark", "bridge3way")
MODEL = os.environ.get("B3_MODEL", "Qwen3-8B")
ENDPOINT = os.environ.get("B3_ENDPOINT", "herg")


def main():
    orch = json.load(open(os.path.join(ROOT, "results", "orchestrate_arm.json")))
    ot = orch["transfer"][ENDPOINT]
    for mode, okey in [("within", "in_property_p"), ("transfer", "transfer_p")]:
        bp = os.path.join(B3, f"bridge_arm_{MODEL}_{ENDPOINT}_{mode}.json")
        if not os.path.isfile(bp):
            print(f"[{mode}] no bridge dump yet ({os.path.basename(bp)})")
            continue
        bd = json.load(open(bp))
        if "test_p" not in bd:
            print(f"[{mode}] bridge json has no per-item dump (re-run needed)")
            continue
        ids = bd["test_ids"]
        y = np.array(bd["test_y"])
        g = np.array(bd["test_groups"])
        bridge_p = np.array(bd["test_p"])
        bypass_p = np.array(bd["bypass_p"])
        o_byid = dict(zip(ot["test_ids"], ot[okey]))
        orch_p = np.array([o_byid[i] for i in ids])   # orchestrate, aligned by id
        floor = ot["floor_morgan"] if mode == "transfer" else None

        print(f"\n## {mode.upper()} ({ENDPOINT}, n={len(y)}, scaffold groups={len(set(g))})")
        for name, p in [("orchestrate(head)", orch_p), ("bridge(->LLM)", bridge_p), ("bypass(head)", bypass_p)]:
            s = score_arm(y, p, groups=g)
            print(f"  {name:18s} AUROC={s['auroc']:.3f} AURC={s['aurc']:.3f} ECE10={s['ece10']:.3f} T={s['T']}")
        if floor is not None:
            print(f"  {'Morgan floor':18s} AUROC={floor:.3f}")
        h1b = paired_cluster_boot(y, bridge_p, bypass_p, groups=g, n_boot=2000)
        h1 = paired_cluster_boot(y, bridge_p, orch_p, groups=g, n_boot=2000)
        print(f"  H1b  bridge - bypass      = {h1b['diff']:+.3f}  CI[{h1b['ci'][0]:+.3f},{h1b['ci'][1]:+.3f}]  "
              f"{'excludes 0' if h1b['excludes_zero'] else 'includes 0'} -> "
              f"{'bridge reads THROUGH the LLM' if h1b['ci'][0] > 0 else 'LLM adds nothing (dead weight)'}")
        print(f"  H1   bridge - orchestrate = {h1['diff']:+.3f}  CI[{h1['ci'][0]:+.3f},{h1['ci'][1]:+.3f}]  "
              f"{'excludes 0' if h1['excludes_zero'] else 'includes 0'} -> "
              f"{'bridge beats head (H1b surprise)' if h1['ci'][0] > 0 else 'head >= bridge'}")


if __name__ == "__main__":
    main()
