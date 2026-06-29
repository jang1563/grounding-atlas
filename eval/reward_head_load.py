"""Experiment-3 (RL_ENV_PREREG) step 1: the REWARD head.

Builds the property-predictor reward that drives arms A/B: a balanced-LR head over frozen
ChemBERTa-77M-MTR 384-dim embeddings, as a 5-member bootstrap ENSEMBLE with a pessimistic
penalty reward = mean(P) - lambda*std(P) (per the reward critic, so off-manifold over-
confidence is penalized rather than rewarded). NOT unpickled from orchestrate_arm.json
(that file is metrics-only); the head is RE-FIT here and serialized to a real .pkl.

Drift-guard ties (RL_ENV_PREREG Section 5): this is a reward(x)->scalar, never a headline
metric; arm A's loss must depend on reward(sample); the held-out ORACLE (build_holdout_oracle.py)
is a DIFFERENT model and is what judges designs, not this reward.

Reproduces the orchestrate within-AUROC (GroupKFold(5) OOF balanced_lr) as a recipe sanity
assert before fitting the production head, so a data/recipe drift fails loudly.

Usage:
  python eval/reward_head_load.py                 # herg: reproduce AUROC, fit ensemble, save, self-test
  REWARD_ENDPOINT=clearance python eval/reward_head_load.py
Importable: load_emb, fit_reward_ensemble, reward_from_emb, ChemBertaEmbedder, reward_fn,
            save_reward, load_reward.  No em dashes.
"""
import os
import sys

import numpy as np
from probe_common import balanced_lr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(ROOT, "signal", "sfm_embedding")
OUT_DIR = os.path.join(ROOT, "signal", "reward")
ENDPOINT = os.environ.get("REWARD_ENDPOINT", "herg")
LAMBDA = float(os.environ.get("REWARD_LAMBDA", "1.0"))   # pessimistic penalty weight
N_ENSEMBLE = int(os.environ.get("REWARD_ENSEMBLE", "5"))
# orchestrate within-AUROC targets (results/orchestrate_arm.json) for the recipe sanity assert
AUROC_TARGET = {"herg": 0.8667, "clearance": 0.6070}


def load_emb(endpoint):
    """emb (N,384), y (N,), groups (Murcko scaffold), smiles, ids from the frozen npz."""
    d = np.load(os.path.join(EMB_DIR, f"chemberta_{endpoint}.npz"), allow_pickle=True)
    return d["emb"], d["y"], d["groups"], d["smiles"], d["ids"]


def reproduce_within_auroc(emb, y, groups, tol=0.004):
    """GroupKFold(5) OOF balanced_lr AUROC = the orchestrate within number. Sanity gate."""
    p = cross_val_predict(balanced_lr(), emb, y, cv=GroupKFold(5), groups=groups,
                          method="predict_proba", n_jobs=5)[:, 1]
    auc = float(roc_auc_score(y, p))
    tgt = AUROC_TARGET.get(ENDPOINT)
    ok = tgt is None or abs(auc - tgt) <= tol
    return auc, tgt, ok


def fit_reward_ensemble(emb, y, train_idx=None, n_ensemble=N_ENSEMBLE, seed=0):
    """n_ensemble balanced_lr heads on bootstrap resamples of the reward-training rows.
    train_idx = the block-R scaffold rows (RL_ENV_PREREG Section 6); None = all rows (the
    step-1 characterization head; step 2 / build_holdout_oracle.py passes block-R)."""
    idx = np.arange(len(y)) if train_idx is None else np.asarray(train_idx)
    rng = np.random.RandomState(seed)
    members = []
    for k in range(n_ensemble):
        bs = rng.choice(idx, size=len(idx), replace=True)
        clf = balanced_lr().fit(emb[bs], y[bs])
        members.append(clf)
    return members


def reward_from_emb(members, emb, lam=LAMBDA):
    """Pessimistic ensemble reward in [approx 0,1]: mean(P[y=1]) - lam*std(P[y=1]).
    Penalizes disagreement (a proxy for off-manifold uncertainty) so the policy cannot
    farm over-confident OOD points (reward critic, RL_ENV_PREREG guard 8)."""
    P = np.stack([m.predict_proba(emb)[:, 1] for m in members], 0)   # (M, N)
    return P.mean(0) - lam * P.std(0)


class ChemBertaEmbedder:
    """Lazy ChemBERTa-77M-MTR mean-pool embedder for live reward(smiles). Mirrors
    sfm_embed_chemberta.embed exactly (mean over attention_mask). CPU/MPS-capable; the
    model is cached locally. trust_remote_code stays OFF for ChemBERTa (a BERT)."""

    def __init__(self, model="DeepChem/ChemBERTa-77M-MTR", batch=64):
        self.model_name, self.batch = model, batch
        self._tok = self._model = self._dev = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._dev = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._dev).eval()

    def __call__(self, smiles):
        self._ensure()
        import torch
        out = []
        for i in range(0, len(smiles), self.batch):
            enc = self._tok(list(smiles[i:i + self.batch]), return_tensors="pt",
                            padding=True, truncation=True, max_length=256).to(self._dev)
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, 0)


def reward_fn(members, embedder, smiles, lam=LAMBDA):
    """Live reward over generated SMILES: embed -> pessimistic ensemble reward."""
    return reward_from_emb(members, embedder(smiles), lam=lam)


def save_reward(members, path, meta=None):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"members": members, "lambda": LAMBDA, "endpoint": ENDPOINT,
                 "model": "DeepChem/ChemBERTa-77M-MTR", "meta": meta or {}}, path)


def load_reward(path):
    import joblib
    return joblib.load(path)


def main():
    emb, y, g, smiles, ids = load_emb(ENDPOINT)
    print(f"[reward] endpoint={ENDPOINT} n={len(y)} pos={int(y.sum())} "
          f"emb={emb.shape} lambda={LAMBDA} ensemble={N_ENSEMBLE}", flush=True)

    auc, tgt, ok = reproduce_within_auroc(emb, y, g)
    print(f"[reward] within-AUROC reproduce: {auc:.4f}  target={tgt}  "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        print("[reward] FAIL: recipe/data drift, AUROC does not match orchestrate.", file=sys.stderr)
        sys.exit(1)

    # step-1 characterization head: full-set ensemble (block-R refit happens in step 2)
    members = fit_reward_ensemble(emb, y)
    r = reward_from_emb(members, emb)
    # the pessimistic reward must still separate the classes (sanity, not the headline)
    sep = float(roc_auc_score(y, r))
    print(f"[reward] pessimistic ensemble reward AUROC (full-fit, in-sample sanity)={sep:.4f}  "
          f"mean(pos)={r[y == 1].mean():.3f} mean(neg)={r[y == 0].mean():.3f}", flush=True)

    out = os.path.join(OUT_DIR, f"{ENDPOINT}_reward_ensemble.pkl")
    save_reward(members, out, meta={"within_auroc": auc, "n": len(y), "pos": int(y.sum())})
    print(f"[reward] saved -> {os.path.relpath(out, ROOT)}", flush=True)

    # live reward(smiles) self-test on a few real molecules (proves the RL-loop interface works)
    try:
        emb_te = ChemBertaEmbedder()
        probe = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                 "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)N(=O)=O"]
        rv = reward_fn(members, emb_te, probe)
        print("[reward] live reward(smiles) self-test:", flush=True)
        for s, v in zip(probe, rv):
            print(f"    {v:+.3f}  {s}", flush=True)
        print("[reward] OK: reward(smiles) callable end-to-end.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[reward] NOTE: live embed self-test skipped/failed ({exc}); "
              f"npz-based reward path is verified.", flush=True)


if __name__ == "__main__":
    main()
