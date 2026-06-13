"""Per-drug agreement: does the 8B ACTIVATION encode the same withdrawal knowledge the
frontier model verbalizes from the NAME? (Step 2 of deepening the knowledge-gap finding.)

If the 8B's per-drug activation-probe scores correlate with the frontier name-route scores
MORE than with the frontier SMILES-route scores (and more than two structure readers agree),
then the small model's hidden states carry the same drug knowledge the big model states, which
it cannot itself verbalize. That ties the encoded-knowledge-unspoken claim down per drug.

Inputs (matched by exact SMILES, both read from the same withdrawn.csv):
  results/withdrawn_peritem.json   frontier: name, label, llm_name, llm_smiles, llm_fake, morgan, knn
  results/wd_act_peritem.json      8B: act (best-layer OOF probe), struct, output  (scp from cluster)
Outputs results/peritem_agreement.json. No em dashes.
"""
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT = os.path.join(ROOT, "results", "withdrawn_peritem.json")
ACT = os.path.join(ROOT, "results", "wd_act_peritem.json")


def main():
    front = {r["smiles"]: r for r in json.load(open(FRONT))}
    actj = json.load(open(ACT))
    act = {r["smiles"]: r for r in actj["items"]}
    common = sorted(set(front) & set(act))
    print(f"frontier n={len(front)}  8B n={len(act)}  intersection={len(common)}", flush=True)
    if len(common) < 30:
        print("WARNING: small intersection; the two arms sampled different subsets.")

    lab = np.array([front[s]["label"] for s in common])
    cols = {"front_name": [front[s]["llm_name"] for s in common],
            "front_smiles": [front[s]["llm_smiles"] for s in common],
            "front_morgan": [front[s]["morgan"] for s in common],
            "act_8b": [act[s]["act"] for s in common],
            "struct_8b": [act[s]["struct"] for s in common],
            "out_8b": [act[s]["output"] for s in common]}
    cols = {k: np.array(v) for k, v in cols.items()}

    def corr(a, b):
        return (round(float(spearmanr(cols[a], cols[b]).statistic), 3),
                round(float(pearsonr(cols[a], cols[b])[0]), 3))

    pairs = {
        "act8B_vs_frontierNAME": corr("act_8b", "front_name"),     # the key: same knowledge?
        "act8B_vs_frontierSMILES": corr("act_8b", "front_smiles"),  # reference (weaker expected)
        "act8B_vs_frontierMORGAN": corr("act_8b", "front_morgan"),  # structure-structure baseline
        "struct8B_vs_frontierMORGAN": corr("struct_8b", "front_morgan"),  # two fingerprint readers
        "out8B_vs_frontierNAME": corr("out_8b", "front_name"),      # 8B can NOT verbalize it
    }
    aurocs = {k: round(float(roc_auc_score(lab, cols[k])), 3) for k in cols}

    out = {"intersection_n": len(common), "pos": int(lab.sum()),
           "correlations_spearman_pearson": pairs, "auroc_on_intersection": aurocs}
    json.dump(out, open(os.path.join(ROOT, "results", "peritem_agreement.json"), "w"), indent=2)

    print("\n=== AUROC on the shared drugs ===")
    for k, v in aurocs.items():
        print(f"  {k:14s} {v}")
    print("\n=== per-drug correlation (Spearman, Pearson) ===")
    for k, v in pairs.items():
        print(f"  {k:28s} {v}")
    sn = pairs["act8B_vs_frontierNAME"][0]
    ss = pairs["act8B_vs_frontierSMILES"][0]
    print(f"\nVERDICT: 8B activation correlates with frontier NAME at rho={sn} vs frontier SMILES at rho={ss}. "
          f"{'NAME > SMILES -> the 8B encodes the same knowledge the frontier verbalizes.' if sn > ss + 0.05 else 'no clear name-over-smiles edge.'}")


if __name__ == "__main__":
    main()
