"""Experiment-3 bias-arbiter (docs/DOCKING_PROTOCOL.md), leg 2a: re-judge the high-budget
(A-B)=+0.064 edge with an INDEPENDENT-FEATURIZATION hERG oracle - RDKit physicochemical descriptors +
MACCS keys -> RF - which shares NEITHER the Morgan fingerprint (oracle) NOR the ChemBERTa embedding
(reward). It breaks the 2D-Morgan feature surface the reward+oracle share, but NOT the internal label
lineage (block-R/E from the same hERG set), so it is NECESSARY-NOT-SUFFICIENT: it can FLAG feature-
specific gaming (edge collapses beyond discrimination-loss) but cannot alone license "genuine" (that
needs the external-label leg 2b). One-directional per the protocol.

Gates (block-E controls, held-out from block-R/O): (A) competence AUROC>=0.70 + EF1%; (B) is deferred
to the power note; (C) independence is the 2b leg. Guards: discrimination-loss calibration (a weaker
arbiter regresses any gap toward 0) and arbiter-native applicability-domain (k-NN in feature space).
Primary estimand = FULL-DRAW success-rate over draws (arm A 3000 pooled seeds vs guidance 1000), the
same scale as +0.064. Local CPU.
Usage: python eval/bias_arbiter.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, MACCSkeys  # noqa: E402
from rl_common import OUT, ROOT, load_blocks  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ENDPOINT = "herg"
BAR_PCT = 90
# curated, robust physicochemical descriptors (independent of Morgan / ChemBERTa)
PHYSCHEM = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
            "NumAromaticRings", "RingCount", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
            "LabuteASA", "BalabanJ", "BertzCT", "qed"]
_FN = {n: getattr(Descriptors, n) for n in PHYSCHEM if hasattr(Descriptors, n)}


def feats(smiles):
    """physchem + MACCS(167). Returns (X, ok_mask)."""
    X, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            X.append(np.zeros(len(_FN) + 167)); ok.append(False); continue
        pc = []
        for n, f in _FN.items():
            try:
                pc.append(float(f(m)))
            except Exception:
                pc.append(0.0)
        mk = list(MACCSkeys.GenMACCSKeys(m))
        X.append(np.array(pc + mk, dtype=float)); ok.append(True)
    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(ok)


def ef(y, score, frac=0.01):
    n = max(1, int(len(y) * frac))
    top = np.argsort(-score)[:n]
    return float((y[top].mean()) / (y.mean() + 1e-9))


def main():
    emb, y, g, smi, blk = load_blocks(ENDPOINT)
    Xr, okr = feats(smi[blk == "R"]); yr = y[blk == "R"]
    Xe, oke = feats(smi[blk == "E"]); ye = y[blk == "E"]
    sc = StandardScaler().fit(Xr[okr])
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(sc.transform(Xr[okr]), yr[okr])
    pe = rf.predict_proba(sc.transform(Xe[oke]))[:, 1]
    yeo = ye[oke]
    auc = float(roc_auc_score(yeo, pe))
    ef1 = ef(yeo, pe, 0.01)
    bar = float(np.percentile(pe, BAR_PCT))
    print("[arbiter] independent-featurization (physchem+MACCS) RF on block-R, scored block-E:")
    print(f"  GATE A competence: AUROC={auc:.3f} (floor 0.70)  EF1%={ef1:.2f}x (floor 3x)  "
          f"-> {'PASS' if auc >= 0.70 and ef1 >= 3 else 'FAIL (arbiter uninformative)'}", flush=True)

    # discrimination-loss calibration: how much does THIS arbiter shrink a KNOWN +0.064 gap vs truth?
    rng = np.random.RandomState(0)
    acts = np.where(yeo == 1)[0]; inacts = np.where(yeo == 0)[0]
    rates = []
    for _ in range(400):
        # pseudo-A: 20% actives, pseudo-B: 13.6% actives (true gap +0.064), N=500 each
        def draw(pa):
            na = int(round(pa * 500))
            idx = np.concatenate([rng.choice(acts, na, replace=True), rng.choice(inacts, 500 - na, replace=True)])
            return (pe[idx] > bar).mean()
        rates.append(draw(0.20) - draw(0.136))
    retain = float(np.mean(rates)) / 0.064   # arbiter-measured gap / true gap
    print(f"  discrimination-loss: arbiter measures a true +0.064 gap as {np.mean(rates):+.4f} "
          f"(retention {retain:.2f}x). A genuine +0.064 edge should appear as ~{0.064*retain:+.4f}.", flush=True)

    # re-judge the FULL-DRAW edge under the arbiter
    aS = []
    for f in sorted(glob.glob(os.path.join(OUT, "herg_armA_ppo_s*_Q10000.json"))):
        if "_np" in f or "_shuffle" in f:
            continue
        aS += json.load(open(f))["designs"]
    b = json.load(open(os.path.join(OUT, "herg_armB_guidance.json")))
    bS = [b["designs"][i] for i in b["delivered_idx"]["10000"]]
    Xa, oka = feats(aS); Xb, okb = feats(bS)
    pa = rf.predict_proba(sc.transform(Xa[oka]))[:, 1]
    pb = rf.predict_proba(sc.transform(Xb[okb]))[:, 1]
    ra, rb = float((pa > bar).mean()), float((pb > bar).mean())
    print(f"\n[arbiter] FULL-DRAW re-judge under the independent arbiter (bar=block-E {BAR_PCT}th pct):")
    print(f"  arm A {int((pa>bar).sum())}/{len(pa)}={ra:.4f}   arm B {int((pb>bar).sum())}/{len(pb)}={rb:.4f}"
          f"   (A-B)_arbiter={ra-rb:+.4f}", flush=True)
    print(f"  incumbent-oracle (A-B) was +0.064; discrimination-loss-expected under arbiter ~{0.064*retain:+.4f}", flush=True)

    # arbiter-native applicability domain: k-NN distance to arbiter training set
    Zr = sc.transform(Xr[okr]); Za = sc.transform(Xa[oka])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5).fit(Zr)
    thr = np.percentile(nn.kneighbors(Zr)[0].mean(1), 95)   # AD threshold from training distances
    ad_out = float((nn.kneighbors(Za)[0].mean(1) > thr).mean())
    print(f"  applicability-domain: {ad_out:.1%} of arm-A designs OUTSIDE the arbiter AD (5-NN, 95th-pct thr)", flush=True)

    # one-directional verdict
    survives = (ra - rb) >= 0.5 * 0.064 * retain    # edge appears at >= half the discrimination-loss-expected
    collapses = (ra - rb) <= 0.25 * 0.064 * retain
    verdict = ("EDGE SURVIVES the featurization change (shared-Morgan-bias NOT the explanation; "
               "leg 2a cannot license 'genuine' - needs external-label 2b)" if survives
               else "EDGE COLLAPSES beyond discrimination-loss -> feature-specific (shared-Morgan-bias) GAMING flagged"
               if collapses else "INDETERMINATE (edge shrinks within the discrimination-loss band)")
    print(f"\n[arbiter] leg-2a verdict: {verdict}", flush=True)
    res = {"arbiter": "physchem+MACCS RF (block-R)", "block_e_auroc": round(auc, 3), "ef1": round(ef1, 2),
           "gateA_pass": bool(auc >= 0.70 and ef1 >= 3), "discrimination_retention": round(retain, 3),
           "AmB_incumbent_oracle": 0.064, "AmB_arbiter": round(ra - rb, 4),
           "expected_under_disc_loss": round(0.064 * retain, 4), "rate_A": round(ra, 4), "rate_B": round(rb, 4),
           "armA_frac_out_of_AD": round(ad_out, 3), "verdict_2a": verdict}
    json.dump(res, open(os.path.join(OUT, "herg_bias_arbiter.json"), "w"), indent=1)
    print(f"[arbiter] saved -> {os.path.relpath(os.path.join(OUT, 'herg_bias_arbiter.json'), ROOT)}", flush=True)


if __name__ == "__main__":
    main()
