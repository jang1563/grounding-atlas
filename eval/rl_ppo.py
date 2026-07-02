"""Experiment-3 (RL_ENV_PREREG) arm A: INTERNALIZED reinforcement learning of the generator.

On-policy policy gradient fine-tuning of a trainable COPY of the frozen generator toward the
block-R reward (RLXF-style: two model copies, on-policy rollouts, reward-driven update, base
anchor). Uses the REINVENT augmented-likelihood loss (regress policy log-prob toward
base_lp + sigma*reward); PPO-clipped diverged on this char-RNN (KL blew to ~10), so this is the
canonical STABLE SMILES-RL loss with a built-in base anchor. The 'train' lever. After training on a
reward-query budget Q = steps x batch, sample M designs; the held-out block-O oracle judges. The
verdict (H1) compares this to arm B (external guidance) at MATCHED Q.

Drift guards (RL_ENV_PREREG Section 5):
  - reward is IN the loss (advantage-weighted log-prob), gradient depends on reward(sample);
  - RL_SHUFFLE=1 permutes the batch rewards -> the advantage is uninformative -> the run MUST
    collapse to base (the verdict-voiding ablation; reward-driven gain must vanish);
  - KL-to-base reported every step; an A-win achievable only at near-zero KL is drift, not a lever win.

Needs a GPU for the policy training. Usage:
  sbatch --export=ALL,E3_SCRIPT=rl_ppo.py,RL_ENDPOINT=herg,RL_BATCH=50,RL_STEPS=100,RL_BETA=0.1 eval/cayuga_rl.sbatch
No em dashes.
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_common import (  # noqa: E402
    DEV,
    OUT,
    ROOT,
    build_oracle,
    build_reward,
    load_generator,
    oracle_scores,
    reward_scores,
    valid_novel_unique,
)
from smiles_generator_init import BOS, EOS, MAX_LEN, PAD  # noqa: E402

ENDPOINT = os.environ.get("RL_ENDPOINT", "herg")
BATCH = int(os.environ.get("RL_BATCH", "50"))
STEPS = int(os.environ.get("RL_STEPS", "100"))           # budget Q = STEPS x BATCH reward queries
LR = float(os.environ.get("RL_LR", "1e-4"))
BETA = float(os.environ.get("RL_BETA", "0.1"))           # (legacy; recorded in the dump)
SIGMA = float(os.environ.get("RL_SIGMA", "20"))          # REINVENT reward scaling in the augmented likelihood
M = int(os.environ.get("RL_M", "500"))
SHUFFLE = bool(os.environ.get("RL_SHUFFLE", ""))
SEED = int(os.environ.get("RL_SEED", "0"))
NPOS = os.environ.get("REWARD_NPOS", "")              # low-data degraded reward (block-R subsample)


@torch.no_grad()
def rollout(model, vocab, n, temp):
    bos, eos, pad = vocab.stoi[BOS], vocab.stoi[EOS], vocab.stoi[PAD]
    x = torch.full((n, 1), bos, dtype=torch.long, device=DEV)
    h = None
    seqs = [[bos] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool, device=DEV)
    for _ in range(MAX_LEN):
        logits, h = model(x, h)
        nxt = torch.multinomial(torch.softmax(logits[:, -1, :] / temp, -1), 1)
        for j in range(n):
            if not done[j]:
                t = int(nxt[j])
                seqs[j].append(t)
                if t == eos:
                    done[j] = True
        x = nxt
        if done.all():
            break
    length = max(len(s) for s in seqs)
    toks = torch.full((n, length), pad, dtype=torch.long, device=DEV)
    for j, s in enumerate(seqs):
        toks[j, :len(s)] = torch.tensor(s, device=DEV)
    smis = ["".join(vocab.itos[t] for t in s if t not in (bos, eos, pad)) for s in seqs]
    return toks, (toks != pad).float(), smis


def seq_logprob(model, toks, mask):
    logits, _ = model(toks[:, :-1])
    lp = F.log_softmax(logits, -1).gather(2, toks[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * mask[:, 1:]).sum(1)


def main():
    base, vocab, temp = load_generator()
    for p in base.parameters():
        p.requires_grad_(False)
    policy = copy.deepcopy(base)
    policy.train()                # cuDNN RNN backward requires train mode (the loader left it in eval)
    policy.gru.dropout = 0.0      # but disable dropout so the PPO log-probs stay deterministic
    for p in policy.parameters():
        p.requires_grad_(True)
    members = build_reward(ENDPOINT)
    rf, bar = build_oracle(ENDPOINT)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    print(f"[armA] endpoint={ENDPOINT} budgetQ={STEPS * BATCH} (steps={STEPS} x batch={BATCH}) "
          f"sigma={SIGMA} shuffle={SHUFFLE} temp={temp} dev={DEV}", flush=True)
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    hist = []
    for step in range(STEPS):
        toks, mask, smis = rollout(policy, vocab, BATCH, temp)
        r = reward_scores(members, smis)
        if SHUFFLE:
            r = rng.permutation(r)                       # drift-guard ablation
        rt = torch.tensor(r, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            base_lp = seq_logprob(base, toks, mask)
        policy_lp = seq_logprob(policy, toks, mask)
        # REINVENT augmented likelihood: regress the policy's log-prob of its own rollouts toward
        # base_lp + sigma*reward. The base term is a BUILT-IN anchor that holds the policy near the
        # prior (PPO-clipped diverged on this char-RNN, KL blew to ~10), so this is the canonical
        # STABLE SMILES-RL loss (REINVENT / ProtRL). Reward is in the loss; under RL_SHUFFLE the
        # target carries no consistent signal -> the policy must collapse to base (drift guard).
        target = base_lp + SIGMA * rt
        loss = ((policy_lp - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()
        kl_val = float((policy_lp.detach() - base_lp).mean())
        if step % 10 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}  meanReward={float(rt.mean()):.3f}  KL={kl_val:+.3f}", flush=True)
        hist.append({"step": step, "mean_reward": round(float(rt.mean()), 4), "kl": round(kl_val, 4)})

    # deliver M designs from the tuned policy; the held-out oracle judges
    from rl_common import sample_designs
    out_smis = []
    while len(out_smis) < M:
        out_smis = list(dict.fromkeys(out_smis + valid_novel_unique(sample_designs(policy, vocab, M, temp))))
    out_smis = out_smis[:M]
    rew = reward_scores(members, out_smis)
    orac = oracle_scores(rf, out_smis)
    opass = (orac > bar).astype(int)
    print(f"\n[armA] delivered M={len(out_smis)} oracle-pass={int(opass.sum())} ({opass.mean():.3f}) "
          f"meanReward={rew.mean():.3f} finalKL={kl_val:+.3f}", flush=True)

    tag = "A_ppo_shuffle" if SHUFFLE else "A_ppo"
    dump = {"arm": tag, "endpoint": ENDPOINT, "budget_Q": STEPS * BATCH, "sigma": SIGMA, "seed": SEED,
            "npos": NPOS, "temp": temp,
            "bar": bar, "M": len(out_smis), "oracle_pass": int(opass.sum()),
            "oracle_pass_rate": round(float(opass.mean()), 4), "final_kl": round(kl_val, 4),
            "designs": out_smis, "reward": [round(float(x), 4) for x in rew],
            "oracle": [round(float(x), 4) for x in orac], "oracle_pass_vec": opass.tolist(),
            "train_history": hist}
    suffix = f"_s{SEED}_Q{STEPS * BATCH}" + (f"_np{NPOS}" if NPOS else "")
    out = os.path.join(OUT, f"{ENDPOINT}_arm{tag}{suffix}.json")
    json.dump(dump, open(out, "w"))
    print(f"[armA] saved -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
