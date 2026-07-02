"""Experiment-3 (RL_ENV_PREREG) step 3: the frozen UNCONDITIONAL SMILES generator (arm C)
+ the PRE-RUN POWER GATE.

A REINVENT-style char/atom-level GRU language model over SMILES, self-trained on an in-repo,
fully-disclosed corpus (the 7 ADMET endpoints' SMILES MINUS any molecule whose Murcko scaffold
is in the hERG held-out blocks O/E). The generator therefore never sees the oracle/eval regions
(block-G stays empty), so its samples are genuinely novel w.r.t. the eval. Frozen after training,
this is the shared base policy for arms A (internalized RL) / B (external guidance) / C (base).

PRE-RUN POWER GATE (RL_ENV_PREREG step 3 / Section 5 guard, pre-registered minima below): the
generator must emit enough VALID, UNIQUE, NOVEL, DIVERSE molecules or the cell is "underpowered,
no verdict" BEFORE any A-vs-B read. The oracle-pass-count half of the gate is checked later, once
build_holdout_oracle.py trains the RF on block-O.

Usage: python eval/smiles_generator_init.py             # build corpus, train, sample, power gate, save
Env: GEN_EPOCHS (default 40), GEN_SAMPLE (default 2000), GEN_HIDDEN (512), GEN_LAYERS (3).
No em dashes.
"""
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(ROOT, "signal", "sfm_embedding")
OUT_DIR = os.path.join(ROOT, "signal", "reward")
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = int(os.environ.get("GEN_EPOCHS", "40"))
N_SAMPLE = int(os.environ.get("GEN_SAMPLE", "2000"))
HIDDEN = int(os.environ.get("GEN_HIDDEN", "512"))
LAYERS = int(os.environ.get("GEN_LAYERS", "3"))
GEN_LOAD = os.environ.get("GEN_LOAD", "")    # if set, load the saved .pt instead of retraining
GEN_ENDPOINT = os.environ.get("GEN_ENDPOINT", "herg")  # which endpoint's block-O/E to exclude
MAX_LEN = 120
TEMPS = [0.7, 0.8, 0.85, 0.9, 1.0]           # sampling-temperature sweep for the power gate

# pre-registered PRE-RUN POWER GATE minima (committed before sampling)
MIN_VALIDITY = 0.70          # fraction of samples that are RDKit-parseable
MIN_UNIQUE_NOVEL_PER_1K = 200  # unique + novel (not in train corpus) per 1000 samples
MIN_DIVERSITY = 0.70         # mean pairwise Tanimoto DISTANCE among valid samples (1 - similarity)

SMI_TOKEN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
BOS, EOS, PAD = "^", "$", " "


def build_corpus():
    """In-repo SMILES pool minus any molecule whose hERG-style Murcko scaffold is in block-O/E.
    block-R / block-G molecules and all non-hERG molecules are allowed (RL_ENV_PREREG Section 6)."""
    part = json.load(open(os.path.join(OUT_DIR, f"{GEN_ENDPOINT}_partition.json")))
    oe = {s for s, b in part["scaffold_to_block"].items() if b in ("O", "E")}
    pool = {}
    for p in sorted(glob.glob(os.path.join(EMB_DIR, "chemberta_*.npz"))):
        d = np.load(p, allow_pickle=True)
        for s, g in zip(d["smiles"], d["groups"]):
            pool.setdefault(str(s), str(g))
    corpus = [s for s, g in pool.items() if g not in oe]
    return corpus, set(pool)   # corpus (allowed), full pool (for novelty)


def tokenize(smi):
    return SMI_TOKEN.findall(smi)


class Vocab:
    def __init__(self, corpus):
        toks = set()
        for s in corpus:
            toks |= set(tokenize(s))
        self.itos = [PAD, BOS, EOS] + sorted(toks)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, smi):
        return [self.stoi[BOS]] + [self.stoi[t] for t in tokenize(smi) if t in self.stoi] + [self.stoi[EOS]]

    def __len__(self):
        return len(self.itos)


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, layers, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.emb(x)
        o, h = self.gru(e, h)
        return self.out(o), h


def batches(seqs, vocab, bs=128, shuffle=True):
    order = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        chunk = [seqs[j] for j in order[i:i + bs]]
        m = max(len(s) for s in chunk)
        arr = np.zeros((len(chunk), m), dtype=np.int64)
        for k, s in enumerate(chunk):
            arr[k, :len(s)] = s
        yield torch.from_numpy(arr).to(DEV)


def train(model, seqs, vocab, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for ep in range(epochs):
        tot = nb = 0.0
        for b in batches(seqs, vocab):
            logits, _ = model(b[:, :-1])
            loss = lossf(logits.reshape(-1, len(vocab)), b[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:2d}  nll={tot / nb:.3f}", flush=True)


@torch.no_grad()
def sample(model, vocab, n, temp=1.0, bs=500):
    model.eval()
    out = []
    for i in range(0, n, bs):
        k = min(bs, n - i)
        x = torch.full((k, 1), vocab.stoi[BOS], dtype=torch.long, device=DEV)
        h = None
        toks = [[] for _ in range(k)]
        done = torch.zeros(k, dtype=torch.bool, device=DEV)
        for _ in range(MAX_LEN):
            logits, h = model(x, h)
            p = torch.softmax(logits[:, -1, :] / temp, -1)
            nxt = torch.multinomial(p, 1)
            for j in range(k):
                if not done[j]:
                    t = nxt[j].item()
                    if t == vocab.stoi[EOS]:
                        done[j] = True
                    elif t != vocab.stoi[PAD]:
                        toks[j].append(vocab.itos[t])
            x = nxt
            if done.all():
                break
        out.extend("".join(t) for t in toks)
    return out


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def diversity(valid_canon, k=500):
    """Mean pairwise Tanimoto DISTANCE on Morgan-2048 over up to k molecules."""
    sel = valid_canon[:k]
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in sel]
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims)) if sims else 0.0


def power_gate(samples, train_canon):
    valid = [c for c in (canon(s) for s in samples) if c]
    n = len(samples)
    validity = len(valid) / n
    uniq = set(valid)
    novel = uniq - train_canon
    unp1k = 1000 * len(novel) / n
    div = diversity(list(novel) if novel else list(uniq))
    g = {"n": n, "validity": round(validity, 3), "unique": len(uniq),
         "unique_novel": len(novel), "unique_novel_per_1k": round(unp1k, 1),
         "diversity": round(div, 3)}
    g["pass"] = bool(validity >= MIN_VALIDITY and unp1k >= MIN_UNIQUE_NOVEL_PER_1K and div >= MIN_DIVERSITY)
    return g, valid


def main():
    print(f"[gen] device={DEV} epochs={EPOCHS} hidden={HIDDEN}x{LAYERS} sample={N_SAMPLE}", flush=True)
    corpus, pool = build_corpus()
    print(f"[gen] corpus={len(corpus)} SMILES (block-O/E excluded, block-G empty)", flush=True)
    train_canon = {c for c in (canon(s) for s in corpus) if c}
    vocab = Vocab(corpus)
    seqs = [vocab.encode(s) for s in corpus if len(vocab.encode(s)) <= MAX_LEN]
    print(f"[gen] vocab={len(vocab)} usable_seqs={len(seqs)}", flush=True)

    pt_path = os.path.join(OUT_DIR, f"generator_{GEN_ENDPOINT}_charrnn.pt")
    torch.manual_seed(0)
    model = CharRNN(len(vocab)).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    if GEN_LOAD and os.path.isfile(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=DEV)["state_dict"])
        print(f"[gen] LOADED {nparam / 1e6:.2f}M params from {os.path.relpath(pt_path, ROOT)} (no retrain)", flush=True)
    else:
        print(f"[gen] CharRNN params={nparam / 1e6:.2f}M; training...", flush=True)
        train(model, seqs, vocab)

    # temperature sweep: a char-RNN trades validity for diversity via temp; the operating
    # temp is the MOST exploratory (highest) temp that still clears ALL pre-registered floors.
    print(f"\n[gen] PRE-RUN POWER GATE temp-sweep (floors: validity>={MIN_VALIDITY} "
          f"unique_novel/1k>={MIN_UNIQUE_NOVEL_PER_1K} diversity>={MIN_DIVERSITY}):", flush=True)
    sweep, valid_by_t = [], {}
    for t in TEMPS:
        g, valid = power_gate(sample(model, vocab, N_SAMPLE, temp=t), train_canon)
        g["temp"] = t
        sweep.append(g)
        valid_by_t[t] = valid
        print(f"    temp={t}: validity={g['validity']} unique_novel/1k={g['unique_novel_per_1k']} "
              f"diversity={g['diversity']}  {'PASS' if g['pass'] else 'fail'}", flush=True)
    passing = [g for g in sweep if g["pass"]]
    op = max(passing, key=lambda g: g["temp"]) if passing else None
    if op is not None:
        print(f"  -> PASS at operating temp={op['temp']} (validity={op['validity']} "
              f"novel/1k={op['unique_novel_per_1k']} diversity={op['diversity']})", flush=True)
        for s in valid_by_t[op["temp"]][:6]:
            print("    ", s, flush=True)
    else:
        print("  -> UNDERPOWERED at all temps (escalate corpus to ChEMBL)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos, "hidden": HIDDEN,
                "layers": LAYERS, "max_len": MAX_LEN, "corpus_size": len(corpus),
                "operating_temp": (op or {}).get("temp"), "sweep": sweep}, pt_path)
    json.dump({"sweep": sweep, "operating_temp": (op or {}).get("temp"), "pass": op is not None,
               "corpus_size": len(corpus), "vocab": len(vocab), "params_M": nparam / 1e6},
              open(os.path.join(OUT_DIR, "generator_power_gate.json"), "w"), indent=1)
    print(f"[gen] saved -> {os.path.relpath(pt_path, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
