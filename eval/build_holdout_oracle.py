"""Experiment-3 (RL_ENV_PREREG) step 2: the contamination-safe 4-way scaffold partition +
the FEASIBILITY GATE (runs FIRST, before any generator / RL build).

Partitions the endpoint's Murcko scaffolds into mutually-disjoint blocks:
  block-R  reward-head training scaffolds
  block-G  generator finetune scaffolds (EMPTY for an external disclosed-corpus generator,
           the locked v1 choice -> a 3-way R/O/E carve)
  block-O  oracle (Morgan-RF) training scaffolds
  block-E  held-out 'novel success' eval scaffolds
Whole scaffolds go to one block (no scaffold leakage), so blocks are disjoint by construction.
Counts molecules + POSITIVES per block and checks pre-registered minima. If the small hERG
corpus starves a block, the locked fallback (RL_ENV_PREREG review lock-in, 2026-06-28) is to
pool ames as a second strong-reward endpoint and/or relax to a 2-way reward-vs-oracle split
with an external oracle.

This step is the GO/NO-GO gate and is rdkit-free (scaffolds are precomputed in groups).
The actual RF-on-Morgan oracle TRAINING on block-O is a later step (needs rdkit/Morgan).

Usage: python eval/build_holdout_oracle.py        # feasibility gate on herg (+ ames fallback)
       FEAS_ENDPOINT=ames python eval/build_holdout_oracle.py
No em dashes.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(ROOT, "signal", "sfm_embedding")
OUT_DIR = os.path.join(ROOT, "signal", "reward")
ENDPOINT = os.environ.get("FEAS_ENDPOINT", "herg")

# pre-registered minima (RL_ENV_PREREG Section 6/7), COMMITTED before the count
MIN_REWARD_POS = 100   # block-R: enough positives to fit the reward head
MIN_ORACLE_POS = 100   # block-O: enough positives to train RF-on-Morgan
MIN_EVAL_POS = 50      # block-E: enough held-out actives to set the percentile bar + a non-degenerate denominator


def load(endpoint):
    d = np.load(os.path.join(EMB_DIR, f"chemberta_{endpoint}.npz"), allow_pickle=True)
    return d["y"].astype(int), d["groups"].astype(str)


def partition_scaffolds(scaffolds, y, blocks=("R", "O", "E")):
    """Two-pass greedy assignment of WHOLE scaffolds (deterministic). Pass 1 balances
    POSITIVES (the scarce, lumpy resource: 285 positive scaffolds, top-10 hold 33%) by
    sending each positive-bearing scaffold to the lightest-by-positives block. Pass 2
    balances MOLECULES (negatives) by sending each zero-positive scaffold to the lightest-
    by-molecules block, so block-O is a representative oracle training set, not positive-
    heavy. block-G is empty (external generator); a self-trained generator would add 'G'."""
    sc = {}
    for s, yy in zip(scaffolds, y):
        p, n = sc.get(s, (0, 0))
        sc[s] = (p + int(yy), n + 1)
    assign, loadp, loadn = {}, {b: 0 for b in blocks}, {b: 0 for b in blocks}
    pos_sc = sorted([s for s in sc if sc[s][0] > 0], key=lambda s: (-sc[s][0], -sc[s][1], s))
    zero_sc = sorted([s for s in sc if sc[s][0] == 0], key=lambda s: (-sc[s][1], s))
    for s in pos_sc:                                              # pass 1: balance POSITIVES
        b = min(blocks, key=lambda b: (loadp[b], loadn[b], b))
        assign[s] = b
        loadp[b] += sc[s][0]
        loadn[b] += sc[s][1]
    for s in zero_sc:                                            # pass 2: balance MOLECULES
        b = min(blocks, key=lambda b: (loadn[b], b))
        assign[s] = b
        loadn[b] += sc[s][1]
    counts = {b: {"pos": loadp[b], "mol": loadn[b],
                  "scaffold": sum(1 for v in assign.values() if v == b)} for b in blocks}
    return assign, counts


def feasibility(endpoint, blocks=("R", "O", "E")):
    y, sc = load(endpoint)
    n, pos, nsc = len(y), int(y.sum()), len(set(sc))
    assign, counts = partition_scaffolds(sc, y, blocks)
    # disjointness is by construction (scaffold -> single block); assert no scaffold double-counted
    assert sum(c["scaffold"] for c in counts.values()) == nsc, "scaffold partition not exhaustive/disjoint"
    floors = {"R": MIN_REWARD_POS, "O": MIN_ORACLE_POS, "E": MIN_EVAL_POS}
    ok = {b: counts[b]["pos"] >= floors.get(b, 0) for b in blocks}
    return {"endpoint": endpoint, "n": n, "pos": pos, "scaffold": nsc,
            "counts": counts, "floors": floors, "ok": ok,
            "pass": all(ok.values()), "assign": assign}


def _load_smiles(endpoint):
    d = np.load(os.path.join(EMB_DIR, f"chemberta_{endpoint}.npz"), allow_pickle=True)
    return d["y"].astype(int), d["groups"].astype(str), d["smiles"].astype(str)


def _morgan(smiles):
    """Morgan-2 2048-bit ECFP4 (rdkit). DIFFERENT featurization than the ChemBERTa reward."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    X = np.zeros((len(smiles), 2048), dtype=np.int8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), X[i])
    return X


def train_oracle(endpoint=ENDPOINT, bar_pct=90):
    """The held-out JUDGE (RL_ENV_PREREG Section 6): RF-on-Morgan trained on block-O. Different
    model class (RF vs LR) AND featurization (Morgan vs ChemBERTa embedding) than the reward, and
    scaffold-disjoint from it = reduced-leakage. Evaluated on block-E (novel scaffolds). The
    fitness bar is a PERCENTILE of the oracle's score distribution on block-E, not an absolute P,
    so success-rate is non-degenerate. Docking (QuickVina2) is the deferred orthogonal co-primary
    required only if arm A appears to win (overturn credibility)."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    y, g, smi = _load_smiles(endpoint)
    s2b = json.load(open(os.path.join(OUT_DIR, f"{endpoint}_partition.json")))["scaffold_to_block"]
    blk = np.array([s2b.get(s, "?") for s in g])
    Xo, yo = _morgan(smi[blk == "O"]), y[blk == "O"]
    Xe, ye = _morgan(smi[blk == "E"]), y[blk == "E"]
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(Xo, yo)
    pe = rf.predict_proba(Xe)[:, 1]
    auc_e = float(roc_auc_score(ye, pe))
    bar = float(np.percentile(pe, bar_pct))
    print(f"\n[oracle] RF-on-Morgan trained on block-O (n={len(yo)} pos={int(yo.sum())}); "
          f"held-out block-E AUROC={auc_e:.3f} (n={len(ye)} pos={int(ye.sum())})", flush=True)
    print(f"[oracle] fitness bar = oracle-P > {bar:.3f} ({bar_pct}th pct of block-E); "
          f"block-E fraction clearing = {float((pe > bar).mean()):.3f}", flush=True)
    out = os.path.join(OUT_DIR, f"{endpoint}_oracle_rf.pkl")
    joblib.dump({"rf": rf, "bar": bar, "bar_pct": bar_pct, "endpoint": endpoint,
                 "block_e_auroc": auc_e}, out)
    print(f"[oracle] saved -> {os.path.relpath(out, ROOT)}", flush=True)
    return {"block_e_auroc": auc_e, "bar": bar, "bar_pct": bar_pct,
            "block_o_n": len(yo), "block_o_pos": int(yo.sum())}


def report(f):
    print(f"\n=== {f['endpoint']}: n={f['n']} pos={f['pos']} scaffolds={f['scaffold']} ===", flush=True)
    for b in ("R", "O", "E"):
        c = f["counts"][b]
        flo = f["floors"].get(b, 0)
        print(f"  block-{b}: pos={c['pos']:4d} (min {flo})  mol={c['mol']:4d}  scaffold={c['scaffold']:4d}  "
              f"{'OK' if f['ok'][b] else 'STARVED'}", flush=True)
    print("  block-G: EMPTY (external disclosed-corpus generator, locked v1)", flush=True)
    print(f"  -> {'PASS (proceed on this endpoint)' if f['pass'] else 'UNDERPOWERED (trigger fallback)'}", flush=True)


def main():
    print("[oracle] 4-way scaffold-partition FEASIBILITY GATE (RL_ENV_PREREG step 2)", flush=True)
    print(f"[oracle] pre-registered minima: block-R>={MIN_REWARD_POS} block-O>={MIN_ORACLE_POS} "
          f"block-E>={MIN_EVAL_POS} positives", flush=True)

    primary = feasibility(ENDPOINT)
    report(primary)

    # fallback feasibility (locked): ames as a second strong-reward endpoint
    fallback = None
    if ENDPOINT == "herg":
        ames = feasibility("ames")
        report(ames)
        fallback = ames
        if not primary["pass"]:
            print("\n[oracle] hERG underpowered -> per the lock-in, pool/swap ames as the reward endpoint "
                  "and/or relax to 2-way (reward vs external oracle).", flush=True)
        else:
            print("\n[oracle] hERG PASSES the gate; ames stays a v2 second-reward option, not needed for v1.",
                  flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{ENDPOINT}_partition.json")
    dump = {k: v for k, v in primary.items() if k != "assign"}
    dump["scaffold_to_block"] = primary["assign"]
    if fallback is not None:
        dump["ames_fallback"] = {k: v for k, v in fallback.items() if k != "assign"}
    json.dump(dump, open(out, "w"), indent=1)
    print(f"\n[oracle] partition + counts -> {os.path.relpath(out, ROOT)}", flush=True)
    print(f"[oracle] VERDICT: {'PROCEED on ' + ENDPOINT if primary['pass'] else 'TRIGGER FALLBACK (ames/2-way)'}",
          flush=True)

    if os.environ.get("ORACLE_TRAIN") and primary["pass"]:
        train_oracle(ENDPOINT)


if __name__ == "__main__":
    main()
