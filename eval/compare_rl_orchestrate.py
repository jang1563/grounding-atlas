"""Experiment-3 (RL_ENV_PREREG) the compare: the formal H1 read. Loads arm A (internalized RL) and
arm B (external guidance) delivered designs + their held-out-oracle pass indicators, clusters each
arm's designs by Murcko scaffold, and runs the pre-registered scaffold-clustered TWO-SAMPLE
bootstrap on the oracle-pass-rate difference (A - B). probe_common.paired_cluster_boot is
inapplicable here (the arms emit DIFFERENT molecules, no shared item vector), so this is a new
two-sample cluster bootstrap; the Murcko-clustering and the percentile-CI idea are the reused parts.

Decision rule (RL_ENV_PREREG Section 7):
  CONFIRM route-don't-train  if the (A-B) CI includes 0, or favors B, or |point| < 0.03 (tie band);
  OVERTURN                   only if the CI LOWER bound > 0.03 in A's favor (and no novelty/validity
                             collapse, and docking co-primary agrees).
Light analysis (rdkit scaffolds + bootstrap), runs local. No em dashes.
Usage: python eval/compare_rl_orchestrate.py    (RL_ENDPOINT=herg RL_Q=5000)
"""
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "signal", "reward")
ENDPOINT = os.environ.get("RL_ENDPOINT", "herg")
Q = int(os.environ.get("RL_Q", "5000"))
TIE = 0.03


def murcko(s):
    m = Chem.MolFromSmiles(s)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else s
    except Exception:
        return s


def two_sample_cluster_boot(pass_a, scaf_a, pass_b, scaf_b, nboot=4000):
    def groups(passv, scaf):
        g = {}
        for p, s in zip(passv, scaf):
            g.setdefault(s, []).append(p)
        return [np.array(v) for v in g.values()]

    ga, gb = groups(pass_a, scaf_a), groups(pass_b, scaf_b)
    rng = np.random.RandomState(0)

    def boot_rate(gs):
        idx = rng.randint(0, len(gs), len(gs))
        flat = np.concatenate([gs[i] for i in idx])
        return float(flat.mean())

    diffs = np.array([boot_rate(ga) - boot_rate(gb) for _ in range(nboot)])
    point = float(np.mean(pass_a) - np.mean(pass_b))
    return point, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), len(ga), len(gb)


def main():
    a = json.load(open(os.path.join(OUT, f"{ENDPOINT}_armA_ppo.json")))
    b = json.load(open(os.path.join(OUT, f"{ENDPOINT}_armB_guidance.json")))
    a_smis, a_pass = a["designs"], np.array(a["oracle_pass_vec"])
    idx = b["delivered_idx"][str(Q)]
    b_smis = [b["designs"][i] for i in idx]
    b_pass = np.array([b["oracle_pass"][i] for i in idx])
    a_scaf = [murcko(s) for s in a_smis]
    b_scaf = [murcko(s) for s in b_smis]

    point, ci, na, nb = two_sample_cluster_boot(a_pass, a_scaf, b_pass, b_scaf)
    rate_a, rate_b = float(a_pass.mean()), float(b_pass.mean())
    print(f"=== H1: route-vs-train in the generative regime ({ENDPOINT}, matched budget Q={Q}) ===", flush=True)
    print(f"  arm A internalized RL : oracle-pass {int(a_pass.sum())}/{len(a_pass)} = {rate_a:.4f} "
          f"({na} scaffold clusters)", flush=True)
    print(f"  arm B external guidance: oracle-pass {int(b_pass.sum())}/{len(b_pass)} = {rate_b:.4f} "
          f"({nb} scaffold clusters)", flush=True)
    print(f"  (A - B) = {point:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  (tie band |Δ|<{TIE})", flush=True)

    # drift-guard + base context
    extra = {}
    sp = os.path.join(OUT, f"{ENDPOINT}_armA_ppo_shuffle.json")
    if os.path.isfile(sp):
        s = json.load(open(sp))
        extra["shuffle_oracle_pass_rate"] = s["oracle_pass_rate"]
        print(f"  drift guard: arm A SHUFFLE oracle-pass rate = {s['oracle_pass_rate']:.4f} "
              f"(must be ~base, not {rate_a:.3f})", flush=True)

    overturn = ci[0] > TIE
    confirm = (ci[0] <= 0 <= ci[1]) or point <= 0 or abs(point) < TIE
    verdict = ("OVERTURN: internalized RL beats guidance (pending docking + no-collapse)" if overturn
               else "CONFIRM: route-don't-train EXTENDS to generation (train ties/loses to route)"
               if confirm else "INDETERMINATE")
    print(f"  -> {verdict}", flush=True)

    res = {"endpoint": ENDPOINT, "budget_Q": Q, "rate_A_rl": round(rate_a, 4),
           "rate_B_guidance": round(rate_b, 4), "diff_A_minus_B": round(point, 4),
           "ci95": [round(ci[0], 4), round(ci[1], 4)], "n_clusters_A": na, "n_clusters_B": nb,
           "tie_band": TIE, "verdict": verdict, **extra}
    json.dump(res, open(os.path.join(OUT, f"{ENDPOINT}_H1_compare.json"), "w"), indent=1)
    print(f"  saved -> {os.path.relpath(os.path.join(OUT, f'{ENDPOINT}_H1_compare.json'), ROOT)}", flush=True)


if __name__ == "__main__":
    main()
