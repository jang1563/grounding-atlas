"""SFM-embedding rung, stage 2: can the LLM read a property out of an SFM EMBEDDING?

Capability-neutral: ESM-2 (150M, 640-dim) embeddings of Meltome proteins -> thermostability,
tercile-extreme label (high vs low Tm), ceiling 0.754 under cluster GroupKFold (stage 1).
Two conditions, the orchestrate-condition input the plan calls "widest-open, no behavioral
baseline" (PROJECT_DESIGN 7.4):

- zeroshot_full: hand the LLM the raw 640-dim embedding as text, ask P(high Tm). The literal
  "feed the SFM output to the LLM" baseline. Prediction: chance (abstract floats, no anchor).
- icl_pca: reduce to PCA-D (fit on the example pool), give K labeled (vector -> label) examples
  in context plus the query vector. Tests whether the LLM can DECODE the embedding space
  IN-CONTEXT (orchestrate-via-ICL) rather than needing a trained head. The open question.

Compared against: the PCA-D probe ceiling (what ICL could reach) and the raw-sequence output arm
(existing protein rung, 8B 0.486 / opus 0.585). Deterministic AUROC scoring. No em dashes.
Env: SFM_COND (both), SFM_NQ (50), SFM_K (24), SFM_PCA (16), SFM_MODEL_LLM (claude-sonnet-4-6),
SFM_DRY.
"""
import json
import os
import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(ROOT, "signal", "sfm_embedding", "meltome_esm2.npz")
COND = os.environ.get("SFM_COND", "both")
NQ = int(os.environ.get("SFM_NQ", "50"))
K = int(os.environ.get("SFM_K", "24"))
PCA_D = int(os.environ.get("SFM_PCA", "16"))
MODEL = os.environ.get("SFM_MODEL_LLM", "claude-sonnet-4-6")
DRY = os.environ.get("SFM_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
SYSTEM = ("You are a careful few-shot classifier over numeric feature vectors. Reason briefly if "
          "needed but END your reply with ONLY a probability between 0 and 1 on its own line.")


def parse01(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def load_split():
    d = np.load(NPZ, allow_pickle=True)
    X, tm, grp = d["emb"], d["tm"], d["groups"]
    q33, q67 = np.quantile(tm, [1 / 3, 2 / 3])
    m = (tm <= q33) | (tm >= q67)
    X, tm, grp = X[m], tm[m], grp[m]
    y = (tm >= q67).astype(int)
    rng = np.random.RandomState(42)
    idx = np.arange(len(X)); rng.shuffle(idx)
    # balanced query set, rest is the pool (clusters are ~singletons here, so pool/query disjoint)
    pos = [i for i in idx if y[i] == 1]; neg = [i for i in idx if y[i] == 0]
    nq = min(NQ // 2, len(pos) // 2, len(neg) // 2)
    q = pos[:nq] + neg[:nq]
    pool = [i for i in idx if i not in set(q)]
    return X, y, q, pool


def vec_str(v):
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"


def main():
    X, y, q, pool = load_split()
    Xq, yq = X[q], y[q]
    print(f"== SFM-embedding output arm :: {'DRY' if DRY else MODEL} :: cond={COND} :: "
          f"query={len(q)} pool={len(pool)} K={K} PCA={PCA_D} ==")

    # PCA fit on pool; ceiling that ICL could reach (pool-train logistic -> query)
    sc = StandardScaler().fit(X[pool])
    pca = PCA(n_components=PCA_D, random_state=0).fit(sc.transform(X[pool]))
    Zpool, Zq = pca.transform(sc.transform(X[pool])), pca.transform(sc.transform(Xq))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    probe.fit(Zpool, y[pool])
    pca_ceiling = roc_auc_score(yq, probe.predict_proba(Zq)[:, 1])
    full_probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[pool], y[pool])
    full_ceiling = roc_auc_score(yq, full_probe.predict_proba(Xq)[:, 1])
    print(f"  ceilings (pool-train -> query): full-640={full_ceiling:.3f}  PCA-{PCA_D}={pca_ceiling:.3f}")

    if DRY:
        print("  [dry] sample icl prompt head:")
        ex = pool[:2]
        body = "\n".join(f"{vec_str(Zpool[pool.index(i)])} -> {y[i]}" for i in ex)
        print("   " + (body + f"\nQuery: {vec_str(Zq[0])}\nP(label=1):")[:300])
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def ask(prompt):
        try:
            m = client.messages.create(model=MODEL, max_tokens=400, system=SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
        except Exception:
            return 0.5
        t = [b.text for b in m.content if getattr(b, "type", None) == "text"]
        if getattr(m, "stop_reason", None) == "max_tokens":
            return 0.5
        return parse01(t[0]) if t else 0.5

    rng = np.random.RandomState(7)
    posp = [i for i in pool if y[i] == 1]; negp = [i for i in pool if y[i] == 0]
    out = {"model": MODEL, "n_query": len(q), "K": K, "pca_d": PCA_D,
           "ceiling_full": round(float(full_ceiling), 3), "ceiling_pca": round(float(pca_ceiling), 3),
           "raw_seq_output_prior": {"8B": 0.486, "opus": 0.585}}

    if COND in ("zeroshot_full", "both"):
        preds = []
        for j, i in enumerate(q):
            p = ask(f"A protein is represented by its 640-dimensional ESM-2 embedding (a learned "
                    f"protein language model representation). Estimate the probability that it has "
                    f"HIGH thermostability (high melting temperature).\nEmbedding: {vec_str(X[i])}\n"
                    f"P(high thermostability):")
            preds.append(p)
            if (j + 1) % 20 == 0:
                print(f"  [zeroshot_full] {j+1}/{len(q)}", flush=True)
        out["zeroshot_full_auroc"] = round(float(roc_auc_score(yq, preds)), 3)
        print(f"  zeroshot_full AUROC={out['zeroshot_full_auroc']} (ceiling full {full_ceiling:.3f})", flush=True)

    if COND in ("icl_pca", "both"):
        preds = []
        for j, i in enumerate(q):
            ex_pos = list(rng.choice(posp, K // 2, replace=False))
            ex_neg = list(rng.choice(negp, K // 2, replace=False))
            ex = ex_pos + ex_neg; rng.shuffle(ex)
            lines = "\n".join(f"{vec_str(Zpool[pool.index(e)])} -> {int(y[e])}" for e in ex)
            p = ask(f"Each protein is a {PCA_D}-dim feature vector (PCA of an ESM-2 embedding). "
                    f"Label 1 = HIGH thermostability, 0 = LOW. Learn the pattern from these labeled "
                    f"examples, then classify the query.\n\n{lines}\n\nQuery: {vec_str(Zq[j])}\n"
                    f"P(label=1):")
            preds.append(p)
            if (j + 1) % 20 == 0:
                print(f"  [icl_pca] {j+1}/{len(q)}", flush=True)
        out["icl_pca_auroc"] = round(float(roc_auc_score(yq, preds)), 3)
        print(f"  icl_pca AUROC={out['icl_pca_auroc']} (ceiling PCA-{PCA_D} {pca_ceiling:.3f})", flush=True)

    json.dump(out, open(os.path.join(ROOT, "results", "sfm_embedding_output.json"), "w"), indent=2)
    print("\nRead: zeroshot_full ~ chance = raw SFM embedding is not LLM-readable as text; "
          "icl_pca near its ceiling = the LLM can decode the embedding space in-context "
          "(orchestrate-via-ICL); icl_pca ~ chance = orchestrate needs a trained head.")
    print("saved -> results/sfm_embedding_output.json")


if __name__ == "__main__":
    main()
