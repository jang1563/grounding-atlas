"""Experiment-3 (RL_ENV_PREREG) arm A: INTERNALIZED reinforcement learning of the generator.

On-policy policy-gradient (PPO-clipped) fine-tuning of a trainable COPY of the frozen generator
toward the block-R reward, with a KL leash to the frozen base (RLXF-style: two model copies, on-
policy rollouts, reward-driven update, KL-to-reference). The 'train' lever. After training on a
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
BETA = float(os.environ.get("RL_BETA", "0.1"))           # KL-to-base coefficient (prereg swept {.01,.05,.1,.5})
CLIP = 0.2
EPOCHS = 4
M = int(os.environ.get("RL_M", "500"))
SHUFFLE = bool(os.environ.get("RL_SHUFFLE", ""))


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
    for p in policy.parameters():
        p.requires_grad_(True)
    members = build_reward(ENDPOINT)
    rf, bar = build_oracle(ENDPOINT)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    print(f"[armA] endpoint={ENDPOINT} budgetQ={STEPS * BATCH} (steps={STEPS} x batch={BATCH}) "
          f"beta={BETA} shuffle={SHUFFLE} temp={temp} dev={DEV}", flush=True)
    rng = np.random.RandomState(0)
    baseline, hist = 0.0, []
    for step in range(STEPS):
        toks, mask, smis = rollout(policy, vocab, BATCH, temp)
        r = reward_scores(members, smis)
        if SHUFFLE:
            r = rng.permutation(r)                       # drift-guard ablation
        rt = torch.tensor(r, dtype=torch.float32, device=DEV)
        baseline = 0.9 * baseline + 0.1 * float(rt.mean())
        adv = rt - baseline
        with torch.no_grad():
            old_lp = seq_logprob(policy, toks, mask)
            base_lp = seq_logprob(base, toks, mask)
        kl_val = 0.0
        for _ in range(EPOCHS):
            new_lp = seq_logprob(policy, toks, mask)
            ratio = torch.exp(new_lp - old_lp)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
            kl = (new_lp - base_lp).mean()               # forward KL(policy||base) on policy samples
            loss = pg + BETA * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            kl_val = float(kl)
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
    dump = {"arm": tag, "endpoint": ENDPOINT, "budget_Q": STEPS * BATCH, "beta": BETA, "temp": temp,
            "bar": bar, "M": len(out_smis), "oracle_pass": int(opass.sum()),
            "oracle_pass_rate": round(float(opass.mean()), 4), "final_kl": round(kl_val, 4),
            "designs": out_smis, "reward": [round(float(x), 4) for x in rew],
            "oracle": [round(float(x), 4) for x in orac], "oracle_pass_vec": opass.tolist(),
            "train_history": hist}
    out = os.path.join(OUT, f"{ENDPOINT}_arm{tag}.json")
    json.dump(dump, open(out, "w"))
    print(f"[armA] saved -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
