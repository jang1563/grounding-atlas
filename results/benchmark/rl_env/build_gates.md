# Experiment-3 build gates (RL_ENV_PREREG steps 1-3)

The pre-registered de-risking gates, run in order before the RL arms. Date 2026-06-28.
Compute split (per the cluster rule): light CPU local (steps 1-2), GPU on Cayuga (step 3+).

## Step 1: reward head (eval/reward_head_load.py) - PASS
The property-predictor reward for arms A/B: a balanced-LR head over frozen ChemBERTa-77M-MTR
384-dim embeddings, as a 5-member bootstrap ENSEMBLE with a pessimistic reward = mean(P) -
lambda*std(P) (lambda=1.0). Re-fit + serialized to a real .pkl (orchestrate_arm.json is
metrics-only).
- Reproduces the orchestrate within-AUROC: GroupKFold(5) OOF = 0.8667 (target 0.8667). PASS.
- live reward(smiles) embed matches the stored npz EXACTLY (cos=1.0, max|diff|=0); reward varies
  correctly on real hERG molecules (neg ~0.09 -> pos ~0.98); the pessimistic penalty zeros
  uncertain OOD molecules. CPU-callable end-to-end.

## Step 2: 4-way scaffold-partition feasibility gate (eval/build_holdout_oracle.py) - PASS
Murcko-scaffold partition into block-R (reward) / block-G (generator, EMPTY for the external
generator) / block-O (oracle) / block-E (eval); whole scaffolds per block (disjoint by
construction). Two-pass greedy balances positives (285 positive-bearing scaffolds, top-10 hold
33% of the 625) then molecules. hERG (3963 mol / 625 pos / 1773 scaffolds):

| block | pos (floor) | mol | scaffold |
|---|---|---|---|
| R reward | 208 (>=100) | 1321 | 591 |
| O oracle | 209 (>=100) | 1321 | 593 |
| E eval   | 208 (>=50)  | 1321 | 589 |

PASS -> proceed on hERG; no pivot to the ames fallback (ames also passes, 138 pos/block, kept as
the v2 second-reward option).

## Step 3: in-repo SMILES generator + power gate (eval/smiles_generator_init.py) - PASS
A 4.18M REINVENT-style char-RNN, self-trained on Cayuga A100 on 14,104 disclosed in-repo SMILES
(the 7 ADMET endpoints minus any molecule whose hERG-Murcko scaffold is in block-O/E, so block-G
stays empty and samples are novel w.r.t. the eval). PRE-RUN POWER GATE temperature sweep:

| temp | validity | unique-novel/1k | diversity | |
|---|---|---|---|---|
| 0.7  | 0.831 | 521 | 0.872 | PASS |
| 0.8  | 0.783 | 562 | 0.876 | PASS |
| 0.85 | 0.752 | 557 | 0.876 | PASS |
| 0.9  | 0.730 | 569 | 0.879 | PASS (operating) |
| 1.0  | 0.673 | 558 | 0.881 | fail (validity) |

Floors: validity>=0.70, unique-novel/1k>=200, diversity>=0.70. Operating temp = 0.9 (the most
exploratory temp clearing all floors; all arms sample at this temp so it cancels in A-vs-B). The
in-repo generator powers the cell; no ChEMBL escalation needed.

## Next (all on Cayuga via eval/cayuga_rl.sbatch)
- Oracle RF-on-Morgan trained on block-O (the held-out, scaffold-disjoint judge) + QuickVina2
  docking co-primary; the independent fitness that judges all arms (NOT the training reward).
- Reward re-fit on block-R (the production reward; step 1 built the full-set characterization head).
- Arm B (rl_guidance.py: Best-of-N / rerank), Arm A (rl_ppo.py: on-policy PPO + KL), Arm D
  (CE control), compare_rl_orchestrate.py (scoring + the new scaffold-clustered two-sample bootstrap).
- The 4-cell grid: reward {hERG-strong, clearance-degraded} x data-size {low ~150, full}.
