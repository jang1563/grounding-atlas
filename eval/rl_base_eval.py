"""Experiment-3 (RL_ENV_PREREG): arm C (base generator) + the reward-oracle AGREEMENT viability
check, the pivotal measurement before arms A/B.

Samples the FROZEN base generator (no steering), scores every design with the block-R REWARD
(the steering signal arms A/B optimize) and the block-O held-out ORACLE (the independent judge),
and reports whether the reward even AGREES with the oracle on GENERATED molecules. Interpretation
(RL_ENV_PREREG H3 / Ferruz reward-reliability):
  - agreement high  -> the reward carries real signal on novel chemistry; an A-vs-B contrast is
                       informative (guidance and RL can both raise oracle success).
  - agreement ~0    -> the reward is uninformative off-distribution; ANY arm that optimizes it
                       reward-hacks (train-reward up, oracle flat) -> route-don't-train trivially,
                       and the binding constraint is reward reliability, not train-vs-route.
Also emits arm C's base oracle-pass rate = the floor every other arm must beat.

Runs on Cayuga (generator sampling + ChemBERTa embedding = GPU). No em dashes.
Usage: sbatch --export=ALL,E3_SCRIPT=rl_base_eval.py,RL_N=4000 eval/cayuga_rl.sbatch
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_common import build_oracle, build_reward, oracle_scores, reward_scores  # noqa: E402
from smiles_generator_init import CharRNN, build_corpus, canon  # noqa: E402
from smiles_generator_init import sample as gen_sample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "signal", "reward")
ENDPOINT = os.environ.get("RL_ENDPOINT", "herg")
N = int(os.environ.get("RL_N", "4000"))


class V:
    """Vocab shim rebuilt from the saved itos (smiles_generator_init.sample needs .stoi/.itos)."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load_generator(dev):
    cand = os.path.join(OUT, f"generator_{ENDPOINT}_charrnn.pt")   # endpoint-specific, hERG fallback
    path = cand if os.path.isfile(cand) else os.path.join(OUT, "generator_charrnn.pt")
    ck = torch.load(path, map_location=dev)
    vocab = V(ck["vocab"])
    model = CharRNN(len(vocab), hidden=ck["hidden"], layers=ck["layers"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    return model, vocab, (ck.get("operating_temp") or 0.9)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, temp = load_generator(dev)
    print(f"[baseC] device={dev} N={N} temp={temp}", flush=True)

    smis = gen_sample(model, vocab, N, temp=temp)
    corpus, _ = build_corpus()
    train_canon = {c for c in (canon(s) for s in corpus) if c}
    seen, uniq = set(), []
    for s in smis:
        c = canon(s)
        if c and c not in train_canon and c not in seen:
            seen.add(c)
            uniq.append(c)
    print(f"[baseC] sampled={N} valid+novel+unique={len(uniq)}", flush=True)

    rscore = reward_scores(build_reward(ENDPOINT), uniq)        # block-R reward, fit fresh (rl_common)
    rf, bar = build_oracle(ENDPOINT)                            # block-O oracle, fit fresh
    oscore = oracle_scores(rf, uniq)
    opass = (oscore > bar).astype(int)

    sp = spearman(rscore, oscore)
    from sklearn.metrics import roc_auc_score
    auc = (float(roc_auc_score(opass, rscore)) if 0 < opass.sum() < len(opass) else float("nan"))
    base_pass = float(opass.mean())
    # the top-reward designs: do they enrich for oracle-pass? (this is exactly what guidance exploits)
    top = np.argsort(-rscore)[:max(10, len(uniq) // 20)]   # top 5%
    top_pass = float(opass[top].mean())

    res = {"endpoint": ENDPOINT, "n_sampled": N, "n_valid_novel_unique": len(uniq),
           "base_oracle_pass_rate": round(base_pass, 4), "fitness_bar": round(bar, 4),
           "reward_oracle_spearman": round(sp, 4), "reward_predicts_oraclepass_auroc": round(auc, 4),
           "top5pct_reward_oracle_pass_rate": round(top_pass, 4),
           "enrichment_top5pct_over_base": round(top_pass / base_pass, 2) if base_pass > 0 else None,
           "mean_reward": round(float(rscore.mean()), 4), "mean_oracle": round(float(oscore.mean()), 4)}
    print("\n[baseC] reward-oracle AGREEMENT on generated molecules:", flush=True)
    for k, v in res.items():
        print(f"    {k}: {v}", flush=True)
    # the right metric for a needle-in-haystack task (low base oracle-pass rate) is the reward's
    # ability to ENRICH oracle-actives (AUROC for oracle-pass + top-k enrichment), NOT the
    # whole-blob Spearman (depressed by the uniformly-inactive bulk that dominates the ranking).
    enrich = (top_pass / base_pass) if base_pass > 0 else float("inf")
    informative = (not np.isnan(auc) and auc >= 0.70) or enrich >= 3.0
    verdict = ("INFORMATIVE reward (top-reward enriches oracle-actives; A-vs-B is a real contest)"
               if informative
               else "WEAK reward (binding constraint = reward reliability; arms likely reward-hack)")
    print(f"  -> {verdict}", flush=True)

    json.dump(res, open(os.path.join(OUT, f"{ENDPOINT}_baseC_agreement.json"), "w"), indent=1)
    print(f"[baseC] saved -> {os.path.relpath(os.path.join(OUT, f'{ENDPOINT}_baseC_agreement.json'), ROOT)}",
          flush=True)


if __name__ == "__main__":
    main()
