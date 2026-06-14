"""SFM-embedding rung, stage 3: the ACTIVATION arm. Does an open LLM ENCODE the property in its
hidden states when handed the SFM embedding AS TEXT, even though its OUTPUT is at chance?

Completes the 3-arm for the SFM-embedding rung (ceiling 0.81-0.85; output zero-shot 0.47, ICL 0.56,
`results/sfm_embedding_output.json`). The question: is this an EXPRESSION gap (encoded, not
verbalized, like the methylation / single-cell-anon numeric-vector rungs where activation ~ ceiling)
or an ENCODING gap (the LLM does not even carry the embedding-text signal)?

The comparable run is Qwen3-8B on Cayuga (ACT_MODEL=Qwen/Qwen3-8B, GPU); 26GB-RAM local can only
host a small proxy (Qwen2.5-0.5B), which is directional: if even 0.5B encodes it, 8B almost
certainly does; if 0.5B is at chance, it is inconclusive (run the 8B). Same tercile-extreme Tm
label, cluster GroupKFold, shuffled-label selectivity as the other arms. No em dashes.
Env: ACT_MODEL (Qwen/Qwen2.5-0.5B-Instruct), SFM_BATCH (4).
"""
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(ROOT, "signal", "sfm_embedding", "meltome_esm2.npz")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BATCH = int(os.environ.get("SFM_BATCH", "4"))


def vec_str(v):
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"


def prompt(v):
    return ("A protein is represented by its 640-dimensional ESM-2 embedding (a protein language "
            f"model representation).\nEmbedding: {vec_str(v)}\n"
            "Does it have high thermostability? Answer:")


def heldout_layer_auroc(H, y, groups, clf_factory, n_splits=5):
    """Unbiased best-layer AUROC: pick the layer on TRAIN folds only, score the held-out fold
    (Cawley-Talbot JMLR 2010). best-layer minus held-out-layer = the selection bias."""
    oof = np.zeros(len(y)); picked = []
    for tr, te in GroupKFold(n_splits).split(H[0], y, groups):
        g_tr = groups[tr]
        inner = GroupKFold(min(n_splits, len(np.unique(g_tr))))
        bL, ba = 0, -1.0
        for L in range(len(H)):
            p = cross_val_predict(clf_factory(), H[L][tr], y[tr], cv=inner, groups=g_tr,
                                  method="predict_proba", n_jobs=-1)[:, 1]
            a = roc_auc_score(y[tr], p)
            if a > ba:
                ba, bL = a, L
        c = clf_factory().fit(H[bL][tr], y[tr])
        oof[te] = c.predict_proba(H[bL][te])[:, 1]
        picked.append(bL)
    return roc_auc_score(y, oof), picked


def main():
    d = np.load(NPZ, allow_pickle=True)
    X, tm, grp = d["emb"], d["tm"], d["groups"]
    q33, q67 = np.quantile(tm, [1 / 3, 2 / 3])
    m = (tm <= q33) | (tm >= q67)
    X, tm, grp = X[m], tm[m], grp[m]
    y = (tm >= q67).astype(int)
    texts = [prompt(v) for v in X]
    print(f"== SFM-embedding ACTIVATION arm :: {MODEL} :: n={len(texts)} pos={int(y.sum())} "
          f"clusters={len(set(grp))} ==", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", output_hidden_states=True).to(dev).eval()
    print(f"  device={dev}", flush=True)

    H = None
    for b in range(0, len(texts), BATCH):
        enc = tok(texts[b:b + BATCH], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            hs = model(**enc).hidden_states           # tuple(L+1) of (B, T, D)
        last = enc["attention_mask"].sum(1) - 1        # last real token index per row
        vecs = [h[torch.arange(h.shape[0]), last].float().cpu().numpy() for h in hs]
        if H is None:
            H = [[] for _ in hs]
        for L in range(len(hs)):
            H[L].append(vecs[L])
        if (b + BATCH) % 40 < BATCH:
            print(f"  {min(b + BATCH, len(texts))}/{len(texts)}", flush=True)
    H = [np.concatenate(h, 0) for h in H]

    cv = GroupKFold(min(5, len(set(grp))))
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    best, bestL = 0.0, -1
    for L in range(len(H)):
        p = cross_val_predict(clf(), H[L], y, cv=cv, groups=grp, method="predict_proba", n_jobs=-1)[:, 1]
        a = roc_auc_score(y, p)
        if a > best:
            best, bestL = a, L
    ys = np.random.RandomState(123).permutation(y)
    pc = cross_val_predict(clf(), H[bestL], ys, cv=cv, groups=grp, method="predict_proba", n_jobs=-1)[:, 1]
    ctrl = roc_auc_score(ys, pc)
    ho, picked = heldout_layer_auroc(H, y, grp, clf)   # selection-bias-corrected (nested CV)
    print(f"\nACTIVATION best-layer (max over {len(H)} layers): AUROC={best:.3f} (layer {bestL}, "
          f"shuffled-label control={ctrl:.3f}, selectivity={best - ctrl:+.3f})", flush=True)
    print(f"ACTIVATION held-out-layer (nested GroupKFold, UNBIASED): AUROC={ho:.3f} "
          f"| selection bias = {best - ho:+.3f} (layers picked {picked})", flush=True)
    print("  vs ceiling (embedding probe) 0.81-0.85 | output zero-shot 0.47 / ICL 0.56", flush=True)
    gap_enc = 0.83 - ho
    if best - ctrl < 0.10:
        print("  -> probe not selective; inconclusive")
    elif gap_enc <= 0.15:
        print(f"  -> EXPRESSION gap: the LLM ENCODES Tm from the embedding-text (unbiased act {ho:.2f}, "
              f"selective) but says it at chance. Same pattern as the numeric-vector rungs.")
    else:
        print(f"  -> partial/ENCODING gap: unbiased activation {ho:.2f} below ceiling 0.83 (enc gap {gap_enc:.2f}); "
              f"this small 0.5B proxy may not fully carry it, run the 8B on Cayuga before concluding.")

    import json
    tag = MODEL.split("/")[-1]
    json.dump({"model": MODEL, "n": int(len(y)), "best_layer_auroc": round(float(best), 3),
               "best_layer": int(bestL), "heldout_layer_auroc": round(float(ho), 3),
               "selection_bias": round(float(best - ho), 3), "control": round(float(ctrl), 3),
               "selectivity": round(float(best - ctrl), 3), "layers_picked": [int(x) for x in picked],
               "ceiling": 0.83, "output_zeroshot": 0.47, "output_icl": 0.56},
              open(os.path.join(ROOT, "results", f"sfm_embedding_activation_{tag}.json"), "w"), indent=2)
    print(f"saved -> results/sfm_embedding_activation_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
