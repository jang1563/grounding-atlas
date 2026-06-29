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
import glob
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


def _load_armA(pattern):
    """Pool arm-A files matching the glob (seeds), reporting per-seed rates; skip shuffle files."""
    keep_np = "_np" in pattern                                   # low-data files only when asked
    files = [f for f in sorted(glob.glob(os.path.join(OUT, pattern)))
             if "_shuffle" not in f and (keep_np or "_np" not in f)]
    smis, passv, per_seed = [], [], []
    for f in files:
        j = json.load(open(f))
        smis += j["designs"]
        passv += j["oracle_pass_vec"]
        per_seed.append({"seed": j.get("seed"), "rate": round(float(np.mean(j["oracle_pass_vec"])), 4),
                         "kl": j.get("final_kl")})
    return smis, np.array(passv), per_seed, files


def _load_armB(name):
    j = json.load(open(os.path.join(OUT, name)))
    idx = j["delivered_idx"][str(Q)]
    return [j["designs"][i] for i in idx], np.array([j["oracle_pass"][i] for i in idx])


def _cell(a_smis, a_pass, b_smis, b_pass, label):
    point, ci, na, nb = two_sample_cluster_boot(a_pass, [murcko(s) for s in a_smis],
                                                b_pass, [murcko(s) for s in b_smis])
    ra, rb = float(a_pass.mean()), float(b_pass.mean())
    overturn = ci[0] > TIE
    confirm = (ci[0] <= 0 <= ci[1]) or point <= 0 or abs(point) < TIE
    verdict = ("OVERTURN (pending docking + no-collapse)" if overturn
               else "CONFIRM route-don't-train extends" if confirm else "INDETERMINATE")
    print(f"\n[{label}] arm A RL {int(a_pass.sum())}/{len(a_pass)}={ra:.4f} ({na} clusters)  vs  "
          f"arm B guidance {int(b_pass.sum())}/{len(b_pass)}={rb:.4f} ({nb} clusters)", flush=True)
    print(f"  (A-B) = {point:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  -> {verdict}", flush=True)
    return {"label": label, "rate_A_rl": round(ra, 4), "rate_B_guidance": round(rb, 4),
            "diff_A_minus_B": round(point, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "n_A": len(a_pass), "n_B": len(b_pass), "verdict": verdict}


def main():
    print(f"=== H1: route-vs-train in the GENERATIVE regime ({ENDPOINT}, matched budget Q={Q}) ===", flush=True)
    cells = []

    a_smis, a_pass, per_seed, files = _load_armA(f"{ENDPOINT}_armA_ppo_s*.json")
    if not files:                                                # v1 fallback (no-seed single file)
        a_smis, a_pass, per_seed, files = _load_armA(f"{ENDPOINT}_armA_ppo.json")
    b_smis, b_pass = _load_armB(f"{ENDPOINT}_armB_guidance.json")
    if per_seed:
        print("  arm A per-seed oracle-pass rate: "
              + ", ".join(f"s{p['seed']}={p['rate']}(KL{p['kl']})" for p in per_seed), flush=True)
    full = _cell(a_smis, a_pass, b_smis, b_pass, "full reward (seeds pooled)")
    full["per_seed"] = per_seed
    cells.append(full)

    sp = os.path.join(OUT, f"{ENDPOINT}_armA_ppo_shuffle.json")
    if os.path.isfile(sp):
        sr = json.load(open(sp))["oracle_pass_rate"]
        print(f"  drift guard: arm A SHUFFLE oracle-pass rate = {sr:.4f} (must be ~base)", flush=True)
        full["shuffle_rate"] = sr

    la, lb = f"{ENDPOINT}_armA_ppo_s0_np150.json", f"{ENDPOINT}_armB_guidance_np150.json"
    if os.path.isfile(os.path.join(OUT, la)) and os.path.isfile(os.path.join(OUT, lb)):
        la_smis, la_pass, _, _ = _load_armA(la)
        lb_smis, lb_pass = _load_armB(lb)
        cells.append(_cell(la_smis, la_pass, lb_smis, lb_pass, "low-data degraded reward (npos=150)"))

    out = os.path.join(OUT, f"{ENDPOINT}_H1_compare.json")
    json.dump({"endpoint": ENDPOINT, "budget_Q": Q, "tie_band": TIE, "cells": cells}, open(out, "w"), indent=1)
    print(f"\n  saved -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
