"""Experiment 2, Arm A (LEARNED BRIDGE): the molecule enters the frozen LLM as a SOFT-PROMPT projected
from its frozen molecular-FM embedding (NOT as SMILES text); the LLM then verbalizes yes/no. Only the
projection trains. Per docs/BRIDGE_3WAY_PREREG.md Section 5 (Arm A).

- Projection: e in R^384 -> W2 GELU(W1 e) -> (k, d_model) soft-prompt token vectors (default k=4,
  bottleneck 512, init std 0.02). PRIMARY injection = prepend the k vectors as the first sequence
  positions via inputs_embeds at layer 0 (standard prompt-tuning; positions consistent for RoPE/KV).
- Prompt skeleton (shared with the LoRA arm, only the soft injection differs):
  "SMILES: <soft>\nDoes this molecule {QUESTION}? Answer yes or no.\nAnswer:" -> next token " yes"/" no".
- Training: LLM FROZEN, Stage-B next-token CE on the " yes"/" no" target (target id = ws3_lora build).
- Score test_ids: P(" yes") via the two-token renorm -> AUROC.
- LLM-BYPASS control (the parity check): the IDENTICAL projection -> a fixed linear read-out -> 2 logits,
  NO transformer. If the bridge does not beat this same-param-budget head, H1b is a capacity artifact.

Reuses the shared fold signal/admet/folds/<fold>.json for test_ids (parity with orchestrate/LoRA).
Env: BRIDGE_MODEL (Qwen/Qwen3-8B), BRIDGE_ENDPOINT (herg), BRIDGE_MODE (within|transfer),
BRIDGE_K (4), BRIDGE_LR (5e-4), BRIDGE_EPOCHS (10), BRIDGE_BATCH (16), BRIDGE_N (cap train, 0=all),
BRIDGE_FOLD (lpo_herg_clearance). GPU (Cayuga a40); local 0.5B smoke is directional. No em dashes.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "signal", "sfm_embedding")
MODEL = os.environ.get("BRIDGE_MODEL", "Qwen/Qwen3-8B")
ENDPOINT = os.environ.get("BRIDGE_ENDPOINT", "herg")
MODE = os.environ.get("BRIDGE_MODE", "within")            # within | transfer
K = int(os.environ.get("BRIDGE_K", "4"))
LR = float(os.environ.get("BRIDGE_LR", "5e-4"))
EPOCHS = int(os.environ.get("BRIDGE_EPOCHS", "10"))
BATCH = int(os.environ.get("BRIDGE_BATCH", "16"))
NCAP = int(os.environ.get("BRIDGE_N", "0"))
FOLD = os.environ.get("BRIDGE_FOLD", "lpo_herg_clearance")
TRAIN_EP = ["ames", "cyp3a4", "cyp2d6", "solubility", "permeability"]
QUESTION = {"herg": "block the hERG potassium channel (cardiotoxicity risk)",
            "clearance": "have high metabolic clearance", "ames": "test positive for Ames mutagenicity",
            "cyp3a4": "inhibit CYP3A4", "cyp2d6": "inhibit CYP2D6",
            "solubility": "have high aqueous solubility", "permeability": "have high membrane permeability"}


class Projection(nn.Module):
    def __init__(self, in_dim, d_model, k, bottleneck=512):
        super().__init__()
        self.k, self.d = k, d_model
        self.w1 = nn.Linear(in_dim, bottleneck)
        self.w2 = nn.Linear(bottleneck, k * d_model)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.zeros_(self.w2.bias)

    def forward(self, e):
        return self.w2(torch.nn.functional.gelu(self.w1(e))).view(-1, self.k, self.d)


def load(e):
    d = np.load(os.path.join(EMB, f"chemberta_{e}.npz"), allow_pickle=True)
    return d["emb"].astype(np.float32), d["y"].astype(int), d["groups"], np.array([str(x) for x in d["ids"]])


def train_test():
    """Returns (Xtr, ytr, Xte, yte, gte). within = endpoint train/test split; transfer = pooled-train
    -> the held-out endpoint's frozen test split (test_ids from the shared fold)."""
    fold = json.load(open(os.path.join(ROOT, "signal", "admet", "folds", f"{FOLD}.json")))
    test_ids = set(fold["held_out"][ENDPOINT]["test_ids"])
    emb, y, g, ids = load(ENDPOINT)
    te = np.array([i for i, x in enumerate(ids) if x in test_ids])
    if MODE == "within":
        tr = np.array([i for i in range(len(ids)) if i not in set(te)])
        Xtr, ytr = emb[tr], y[tr]
    else:  # transfer: train on the pooled 5 train endpoints
        Xs, Ys = [], []
        for ep in TRAIN_EP:
            e2, y2, _, _ = load(ep)
            Xs.append(e2); Ys.append(y2)
        Xtr, ytr = np.concatenate(Xs, 0), np.concatenate(Ys, 0)
    if NCAP and len(ytr) > NCAP:
        rng = np.random.RandomState(42); idx = rng.permutation(len(ytr))[:NCAP]
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, ytr, emb[te], y[te], g[te], np.array([str(x) for x in ids[te]])


def main():
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").to(dev).eval()
    model.requires_grad_(False)
    d_model = model.config.hidden_size
    emb_layer = model.get_input_embeddings()
    yes_id = tok(" yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" no", add_special_tokens=False).input_ids[0]

    pre = tok("SMILES: ", add_special_tokens=True, return_tensors="pt").input_ids.to(dev)
    post = tok(f"\nDoes this molecule {QUESTION[ENDPOINT]}? Answer yes or no.\nAnswer:",
               add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        pre_e = emb_layer(pre)                                  # (1, Tpre, d)
        post_e = emb_layer(post)                                # (1, Tpost, d)

    Xtr, ytr, Xte, yte, gte, ids_te = train_test()
    print(f"MODEL={MODEL} endpoint={ENDPOINT} mode={MODE} d_model={d_model} k={K} | "
          f"train n={len(ytr)} pos={int(ytr.sum())} | test n={len(yte)} pos={int(yte.sum())} dev={dev}", flush=True)

    proj = Projection(Xtr.shape[1], d_model, K).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=LR)
    tgt = torch.tensor([yes_id if v == 1 else no_id for v in ytr], device=dev)
    Xtr_t = torch.tensor(Xtr, device=dev)

    def forward_logits(e_batch):
        soft = proj(e_batch).to(pre_e.dtype)                    # (B, k, d); match LLM embed dtype
        B = soft.shape[0]
        seq = torch.cat([pre_e.expand(B, -1, -1), soft, post_e.expand(B, -1, -1)], dim=1)
        return model(inputs_embeds=seq).logits[:, -1, :]        # (B, V)

    print("== train projection (LLM frozen) ==", flush=True)
    for ep in range(EPOCHS):
        proj.train()
        perm = torch.randperm(len(ytr), device=dev)
        tot = 0.0
        for i in range(0, len(ytr), BATCH):
            b = perm[i:i + BATCH]
            logits = forward_logits(Xtr_t[b])
            loss = torch.nn.functional.cross_entropy(logits, tgt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        # quick train AUROC
        proj.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(ytr), 64):
                lg = forward_logits(Xtr_t[i:i + 64])
                pv.append(torch.softmax(lg[:, [yes_id, no_id]], -1)[:, 0].float().cpu().numpy())
            tr_auc = roc_auc_score(ytr, np.concatenate(pv))
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(ytr):.4f} train_AUROC={tr_auc:.3f}", flush=True)

    # test
    proj.eval()
    Xte_t = torch.tensor(Xte, device=dev)
    with torch.no_grad():
        pv = []
        for i in range(0, len(yte), 64):
            lg = forward_logits(Xte_t[i:i + 64])
            pv.append(torch.softmax(lg[:, [yes_id, no_id]], -1)[:, 0].float().cpu().numpy())
    p_test = np.concatenate(pv)
    bridge_auc = roc_auc_score(yte, p_test)

    # LLM-bypass control: same projection, a fixed linear read-out, NO transformer
    byp = Projection(Xtr.shape[1], d_model, K).to(dev)
    head = nn.Linear(K * d_model, 2).to(dev)
    bopt = torch.optim.Adam(list(byp.parameters()) + list(head.parameters()), lr=LR)
    yt = torch.tensor(ytr, device=dev)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), BATCH):
            b = perm[i:i + BATCH]
            lg = head(byp(Xtr_t[b]).flatten(1))
            loss = torch.nn.functional.cross_entropy(lg, yt[b])
            bopt.zero_grad(); loss.backward(); bopt.step()
    with torch.no_grad():
        bp = torch.softmax(head(byp(Xte_t).flatten(1)), -1)[:, 1].float().cpu().numpy()
    bypass_auc = roc_auc_score(yte, bp)

    pcount = sum(p.numel() for p in proj.parameters())
    res = {"arm": "bridge", "model": MODEL, "endpoint": ENDPOINT, "mode": MODE, "k": K,
           "bridge_auroc": round(float(bridge_auc), 4), "bypass_auroc": round(float(bypass_auc), 4),
           "bridge_minus_bypass": round(float(bridge_auc - bypass_auc), 4),
           "param_count": int(pcount), "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "lr": LR, "epochs": EPOCHS,
           # per-item scores for the shared score_arm + paired_cluster_boot (parity comparison)
           "test_ids": [str(x) for x in ids_te], "test_groups": [str(x) for x in gte],
           "test_y": [int(v) for v in yte], "test_p": [round(float(v), 5) for v in p_test],
           "bypass_p": [round(float(v), 5) for v in bp]}
    tag = MODEL.split("/")[-1]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"bridge_arm_{tag}_{ENDPOINT}_{MODE}.json"), "w"), indent=2)
    print(f"\nBRIDGE test AUROC={bridge_auc:.3f} | BYPASS={bypass_auc:.3f} | "
          f"bridge-bypass={bridge_auc - bypass_auc:+.3f} ({'reads THROUGH the LLM' if bridge_auc - bypass_auc > 0.03 else 'NOT beating bypass = capacity artifact'})", flush=True)
    print(f"saved -> results/bridge_arm_{tag}_{ENDPOINT}_{MODE}.json", flush=True)


if __name__ == "__main__":
    main()
