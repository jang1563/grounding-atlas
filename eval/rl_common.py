"""Experiment-3 (RL_ENV_PREREG) shared arm machinery. Builds the block-R REWARD and the block-O
ORACLE FRESH (so every arm uses one sklearn version, no cross-version pickle), loads the frozen
generator, and scores designs. Imported by rl_guidance.py (arm B), rl_ppo.py (arm A),
rl_arm_d_sft.py (arm D), compare_rl_orchestrate.py. No em dashes.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_holdout_oracle import _morgan  # noqa: E402
from reward_head_load import ChemBertaEmbedder, fit_reward_ensemble, load_emb, reward_from_emb  # noqa: E402
from smiles_generator_init import CharRNN, build_corpus, canon  # noqa: E402
from smiles_generator_init import sample as gen_sample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "signal", "reward")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_EMB = None        # lazy shared ChemBERTa embedder
_TRAIN_CANON = None


class V:
    """Vocab shim rebuilt from the generator's saved itos."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def load_blocks(endpoint):
    emb, y, g, smi, ids = load_emb(endpoint)
    s2b = json.load(open(os.path.join(OUT, f"{endpoint}_partition.json")))["scaffold_to_block"]
    blk = np.array([s2b.get(str(s), "?") for s in g])
    return emb, y, g, smi, blk


def build_reward(endpoint, n_ensemble=5):
    """Block-R reward ensemble (the steering signal). Fit fresh. REWARD_NPOS subsamples block-R to
    that many positives (+ proportional negatives) = the low-data DEGRADED-reward regime."""
    emb, y, g, smi, blk = load_blocks(endpoint)
    ridx = np.where(blk == "R")[0]
    npos = int(os.environ.get("REWARD_NPOS", "0"))
    if npos:
        rng = np.random.RandomState(0)
        pos, neg = ridx[y[ridx] == 1], ridx[y[ridx] == 0]
        kneg = min(len(neg), int(round(npos * len(neg) / max(1, len(pos)))))
        ridx = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                               rng.choice(neg, kneg, replace=False)])
    return fit_reward_ensemble(emb, y, train_idx=ridx, n_ensemble=n_ensemble)


def build_oracle(endpoint, bar_pct=90):
    """Block-O Morgan-RF held-out judge + the block-E percentile fitness bar. Fit fresh."""
    from sklearn.ensemble import RandomForestClassifier
    emb, y, g, smi, blk = load_blocks(endpoint)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(_morgan(smi[blk == "O"]), y[blk == "O"])
    bar = float(np.percentile(rf.predict_proba(_morgan(smi[blk == "E"]))[:, 1], bar_pct))
    return rf, bar


def load_generator(path=None):
    if path is None:                                             # endpoint-specific generator, hERG fallback
        ep = os.environ.get("RL_ENDPOINT", "herg")
        cand = os.path.join(OUT, f"generator_{ep}_charrnn.pt")
        path = cand if os.path.isfile(cand) else os.path.join(OUT, "generator_charrnn.pt")
    ck = torch.load(path, map_location=DEV)
    vocab = V(ck["vocab"])
    model = CharRNN(len(vocab), hidden=ck["hidden"], layers=ck["layers"]).to(DEV).eval()
    model.load_state_dict(ck["state_dict"])
    return model, vocab, float(ck.get("operating_temp") or 0.9)


def train_canon_set():
    """Canonical SMILES the generator was trained on (for novelty); cached."""
    global _TRAIN_CANON
    if _TRAIN_CANON is None:
        corpus, _ = build_corpus()
        _TRAIN_CANON = {c for c in (canon(s) for s in corpus) if c}
    return _TRAIN_CANON


def valid_novel_unique(smiles):
    """Canonicalize, drop invalid, drop training-set members, dedup. Returns canonical list."""
    tc, seen, out = train_canon_set(), set(), []
    for s in smiles:
        c = canon(s)
        if c and c not in tc and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def reward_scores(members, smiles, lam=1.0):
    global _EMB
    if _EMB is None:
        _EMB = ChemBertaEmbedder()
    return reward_from_emb(members, _EMB(list(smiles)), lam=lam)


def oracle_scores(rf, smiles):
    return rf.predict_proba(_morgan(list(smiles)))[:, 1]


def sample_designs(model, vocab, n, temp):
    return gen_sample(model, vocab, n, temp=temp)
