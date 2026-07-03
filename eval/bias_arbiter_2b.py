"""Experiment-3 bias-arbiter leg 2b (docs/DOCKING_PROTOCOL.md): the DEEP independence test. Trains a
physchem+MACCS RF on the EXTERNAL hERG_Karim patch-clamp dataset (TDC) - breaking BOTH the shared 2D-Morgan
feature surface (physchem+MACCS, not Morgan/ChemBERTa) AND the internal hERG LABEL lineage (external assay,
not the block-R/O labels that co-trained the reward + oracle). This is the only arbiter independent of the
full shared failure mode. InChIKey-dedup vs the internal set is a hard INDEPENDENCE GATE (>5% overlap = the
external set is not independent -> leg 2b inconclusive). One-directional: can rule out shared-bias gaming,
cannot alone license "genuine binding". Local CPU. No em dashes.
Usage: python eval/bias_arbiter_2b.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_arbiter import feats  # noqa: E402  (physchem+MACCS featurizer, reused)
from rdkit import Chem, RDLogger  # noqa: E402
from rl_common import OUT, ROOT, load_blocks  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RDLogger.DisableLog("rdApp.*")
BAR_PCT = 90


def inchikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m) if m is not None else None


def main():
    from tdc.single_pred import Tox
    df = Tox(name="hERG_Karim").get_data()
    ext_smi = df["Drug"].astype(str).tolist()
    ext_y = df["Y"].values.astype(int)
    print(f"[2b] external hERG_Karim: n={len(ext_smi)} pos={int(ext_y.sum())}", flush=True)

    # INDEPENDENCE GATE: InChIKey overlap with the internal hERG set (block-R/O/E label lineage)
    _, y, g, smi, blk = load_blocks("herg")
    int_ik = {inchikey(s) for s in smi} - {None}
    ext_ik = [inchikey(s) for s in ext_smi]
    overlap = float(np.mean([ik in int_ik for ik in ext_ik if ik]))
    gate = overlap <= 0.05
    print(f"[2b] independence gate: InChIKey overlap with internal set = {overlap:.1%} "
          f"(floor <=5%) -> {'PASS' if gate else 'FAIL (Karim not independent; leg 2b inconclusive)'}", flush=True)
    if not gate:
        json.dump({"leg": "2b", "overlap": round(overlap, 3), "independent": False,
                   "verdict_2b": "INCONCLUSIVE (external set overlaps internal >5%; no independent labels)"},
                  open(os.path.join(OUT, "herg_bias_arbiter_2b.json"), "w"), indent=1)
        print("[2b] STOP: cannot break the label lineage with this external set.", flush=True)
        return

    # train the external arbiter (physchem+MACCS on Karim labels), held-out self-validation
    Xe, oke = feats(ext_smi)
    Xe, yeo = Xe[oke], ext_y[oke]
    Xtr, Xte, ytr, yte = train_test_split(Xe, yeo, test_size=0.25, random_state=0, stratify=yeo)
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(sc.transform(Xtr), ytr)
    auc = float(roc_auc_score(yte, rf.predict_proba(sc.transform(Xte))[:, 1]))
    print(f"[2b] GATE A competence (held-out Karim): AUROC={auc:.3f} (floor 0.70) -> "
          f"{'PASS' if auc >= 0.70 else 'FAIL (arbiter uninformative)'}", flush=True)

    # bar from the internal block-E molecules scored by the EXTERNAL arbiter (fixed reference)
    Xbe, okbe = feats(smi[blk == "E"])
    pbe = rf.predict_proba(sc.transform(Xbe[okbe]))[:, 1]
    bar = float(np.percentile(pbe, BAR_PCT))

    # discrimination-loss on the internal block-E labels (how this external arbiter shrinks a +0.064)
    ye = y[blk == "E"][okbe]
    rng = np.random.RandomState(0)
    acts, inacts = np.where(ye == 1)[0], np.where(ye == 0)[0]
    d = []
    for _ in range(400):
        def draw(pa):
            na = int(round(pa * 500))
            idx = np.concatenate([rng.choice(acts, na, replace=True), rng.choice(inacts, 500 - na, replace=True)])
            return (pbe[idx] > bar).mean()
        d.append(draw(0.20) - draw(0.136))
    retain = float(np.mean(d)) / 0.064

    # re-judge the full-draw edge under the external arbiter
    aS = []
    for f in sorted(glob.glob(os.path.join(OUT, "herg_armA_ppo_s*_Q10000.json"))):
        if "_np" not in f and "_shuffle" not in f:
            aS += json.load(open(f))["designs"]
    b = json.load(open(os.path.join(OUT, "herg_armB_guidance.json")))
    bS = [b["designs"][i] for i in b["delivered_idx"]["10000"]]
    Xa, oka = feats(aS)
    Xb, okb = feats(bS)
    ra = float((rf.predict_proba(sc.transform(Xa[oka]))[:, 1] > bar).mean())
    rb = float((rf.predict_proba(sc.transform(Xb[okb]))[:, 1] > bar).mean())
    print(f"\n[2b] FULL-DRAW re-judge under the EXTERNAL-label arbiter (bar=block-E {BAR_PCT}th pct):", flush=True)
    print(f"  arm A {ra:.4f}   arm B {rb:.4f}   (A-B)_2b={ra-rb:+.4f}   "
          f"(disc-loss-expected ~{0.064*retain:+.4f}, retention {retain:.2f}x)", flush=True)

    survives = (ra - rb) >= 0.5 * 0.064 * retain
    collapses = (ra - rb) <= 0.25 * 0.064 * retain
    verdict = ("EDGE SURVIVES external labels -> shared reward-oracle bias (feature AND label lineage) is "
               "NOT the explanation; the high-budget RL edge is most consistent with a REAL (sub-threshold) gain"
               if survives
               else "EDGE COLLAPSES under external labels -> shared-label-lineage GAMING flagged"
               if collapses else "INDETERMINATE (shrinks within the discrimination-loss band)")
    print(f"\n[2b] leg-2b verdict: {verdict}", flush=True)
    json.dump({"leg": "2b", "overlap": round(overlap, 3), "independent": True, "karim_holdout_auroc": round(auc, 3),
               "retention": round(retain, 3), "AmB_2b": round(ra - rb, 4), "rate_A": round(ra, 4),
               "rate_B": round(rb, 4), "expected_under_disc_loss": round(0.064 * retain, 4), "verdict_2b": verdict},
              open(os.path.join(OUT, "herg_bias_arbiter_2b.json"), "w"), indent=1)
    print(f"[2b] saved -> {os.path.relpath(os.path.join(OUT, 'herg_bias_arbiter_2b.json'), ROOT)}", flush=True)


if __name__ == "__main__":
    main()
