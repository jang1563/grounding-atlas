"""Protein branch, axis-B: three arms on ONE protein set under ONE cluster split (Cayuga GPU).

The protein analog of ../../eval/activation_arm.py. Same instrument, swapped modality:
- ceiling: ESM2 (facebook/esm2_t33_650M_UR50D) mean-pooled over residues + LogisticRegression
  (the structure-probe analog of the Morgan fingerprint). Is the property in the content?
- LLM-activation: linear probe on the general LLM's hidden states (per layer) over the raw
  amino-acid sequence. Does the LLM ENCODE it internally?
- LLM-output: the LLM generates a single probability from the sequence. Does it VERBALIZE it?

Same leakage control as the SMILES branch, modality-appropriate: the two trained probes use
the SAME MMseqs2 sequence-identity cluster GroupKFold (the protein analog of scaffold split).
The output arm is instrumented (parsed / percent / fallback) with the SAME anchored parser as
activation_arm.py + bootstrap CI; best-layer activation is max-over-layers (selection-biased)
with a bootstrap CI.

Hypothesis under test: SMILES appear in web text as "structure -> property" (chemistry
literature), so the LLM encoded hERG near the structural ceiling (expression-dominant gap).
A melting temperature essentially never appears as "sequence -> Tm", so the LLM's internal
ENCODING may be weaker here -> a LARGER encoding gap than SMILES.

Env: ACT_CSV, ACT_MODEL, ACT_N, ACT_THINK, ESM_MODEL, ACT_4BIT, ESM_BATCH.
"""
import os
import re
import csv
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

CSV = os.environ.get("ACT_CSV", "protein_meltome.csv")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
ESM_MODEL = os.environ.get("ESM_MODEL", "facebook/esm2_t33_650M_UR50D")
N = int(os.environ.get("ACT_N", "2000"))
THINK = os.environ.get("ACT_THINK", "0") == "1"   # Qwen3 thinking mode (longer generate)
GEN_TOKENS = 512 if THINK else 12
ESM_BATCH = int(os.environ.get("ESM_BATCH", "8"))

# Threshold is filled at runtime from the loaded set's median Tm so the prompt's concrete
# temperature matches the label definition exactly.
PROMPT = (
    "You are given a protein as an amino-acid sequence. Estimate the probability "
    "(a single number between 0 and 1) that its melting temperature is ABOVE {thresh} "
    "degrees Celsius (i.e. that it is relatively thermostable). Judge only from the "
    "sequence. Reply with ONLY the number.\n\nSequence: {seq}\nProbability:"
)


def load(csv_path, n):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["sequence"], int(row["label"]), float(row["tm"]), row["cluster"]))
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def chat_input(tok, seq, thresh):
    msgs = [{"role": "user", "content": PROMPT.format(seq=seq, thresh=thresh)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_prob(txt):
    """Anchored: take the LAST number; map [0,1] directly, (1,100] as percent.
    Identical to activation_arm.py so the output arm is parsed the same way."""
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "parsed"
        if 1.0 < v <= 100.0:
            return v / 100.0, "percent"
    return 0.5, "fallback"


def bootstrap_ci(y, proba, n_boot=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], proba[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def esm_embed(seqs, model_name, batch_size):
    """Mean-pool ESM2 last_hidden_state over true residues (cls/eos/pad masked out).
    Frees the model + CUDA cache before returning so the LLM can load on the same GPU."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    specials = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id}
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        res = torch.ones_like(enc["input_ids"], dtype=torch.bool)
        for sp in specials:
            if sp is not None:
                res &= enc["input_ids"] != sp
        m = res.unsqueeze(-1).float()
        vec = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(vec.float().cpu().numpy())
        if (i + batch_size) % 200 == 0:
            print(f"  esm embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    quant = None
    if os.environ.get("ACT_4BIT") == "1":
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    elif "gpt-oss" in MODEL.lower():
        from transformers import Mxfp4Config
        quant = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype="auto", device_map="auto", quantization_config=quant, trust_remote_code=True
    )
    model.eval()
    return tok, model


def heldout_layer_auroc(H, y, groups, clf_factory, n_splits=5):
    """Unbiased best-layer AUROC by nested GroupKFold (defends the encoding claim
    against max-over-layers selection bias). Inner folds pick the layer on TRAIN rows
    only; the held-out outer fold is scored at that layer. See ../../results/selection_bias.md.
    """
    Harr = [np.asarray(h) for h in H]
    layers = len(Harr)
    oof = np.zeros(len(y), dtype=float)
    picked = []
    for tr, te in GroupKFold(n_splits).split(Harr[0], y, groups):
        g_tr = groups[tr]
        inner = GroupKFold(min(n_splits, len(np.unique(g_tr))))
        best_L, best_a = -1, -1.0
        for L in range(layers):
            p = cross_val_predict(clf_factory(), Harr[L][tr], y[tr], cv=inner,
                                  groups=g_tr, method="predict_proba", n_jobs=5)[:, 1]
            a = roc_auc_score(y[tr], p)
            if a > best_a:
                best_a, best_L = a, L
        clf = clf_factory().fit(Harr[best_L][tr], y[tr])
        oof[te] = clf.predict_proba(Harr[best_L][te])[:, 1]
        picked.append(best_L)
    return roc_auc_score(y, oof), picked


def main():
    data = load(CSV, N)
    seqs = [s for s, _, _, _ in data]
    y = np.array([l for _, l, _, _ in data])
    groups = np.array([g for _, _, _, g in data])
    tms = np.array([t for _, _, t, _ in data])
    thresh = int(round(float(np.median(tms))))

    # ceiling features: ESM2 embeddings (the structure-probe analog). Computed first, then
    # ESM2 is freed so Qwen3 loads on the same GPU.
    FP = esm_embed(seqs, ESM_MODEL, ESM_BATCH)
    print(f"ESM2 embeddings: {FP.shape}  (model={ESM_MODEL})", flush=True)

    tok, model = load_model()
    device = next(model.parameters()).device

    layers, H, outp, ptypes = None, None, [], []
    for i, s in enumerate(seqs):
        inp = tok(chat_input(tok, s, thresh), return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            fwd = model(**inp, output_hidden_states=True)
        vec = [h[0, -1].float().cpu().numpy() for h in fwd.hidden_states]
        if H is None:
            layers = len(vec)
            H = [[] for _ in range(layers)]
        for L in range(layers):
            H[L].append(vec[L])
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        p, kind = parse_prob(txt)
        outp.append(p)
        ptypes.append(kind)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(seqs)}", flush=True)
    outp = np.array(outp)

    print(f"MODEL={MODEL}  think={THINK}  gen_tokens={GEN_TOKENS}  n={len(y)}  pos={int(y.sum())}  clusters={len(set(groups))}  layers={layers}  Tm_thresh={thresh}C", flush=True)

    cv = GroupKFold(5)

    # structure-probe (ESM2), same cluster split
    lr_fp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    sp = cross_val_predict(lr_fp, FP, y, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    sp_auc = roc_auc_score(y, sp)
    print(f"STRUCTURE-PROBE (ESM2, cluster)  AUROC={sp_auc:.3f}  AUPRC={average_precision_score(y, sp):.3f}", flush=True)

    # output (zero-shot, no training -> direct AUROC) + parse instrumentation
    pc = Counter(ptypes)
    o_auc = roc_auc_score(y, outp)
    o_lo, o_hi = bootstrap_ci(y, outp)
    print(f"OUTPUT  AUROC={o_auc:.3f} [95% {o_lo:.3f},{o_hi:.3f}]  parsed={pc['parsed']} percent={pc['percent']} fallback={pc['fallback']}", flush=True)

    # activation (hidden states), same cluster split, per layer
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, best_L, best_proba = 0.0, -1, None
    for L in range(layers):
        proba = cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, proba)
        if a > best:
            best, best_L, best_proba = a, L, proba
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    a_lo, a_hi = bootstrap_ci(y, best_proba)
    print(f"\nbest ACTIVATION layer {best_L}: AUROC={best:.3f} [95% {a_lo:.3f},{a_hi:.3f}] (MAX over {layers} layers, selection-biased)", flush=True)
    ho_auc, ho_layers = heldout_layer_auroc(
        H, y, groups, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    print(f"HELD-OUT-LAYER ACTIVATION: AUROC={ho_auc:.3f} (nested GroupKFold, layers picked {ho_layers}) | selection bias = {best - ho_auc:+.3f}", flush=True)
    print(f"SUMMARY (same {len(y)} proteins, cluster split):  structure-probe={sp_auc:.3f} | activation(max)={best:.3f} | activation(held-out-layer)={ho_auc:.3f} | output={o_auc:.3f}", flush=True)
    print("gaps: encoding = probe - activation | expression = activation - output  (all one set, one split)", flush=True)

    # control task (Hewitt-Liang 1909.03368): shuffled labels -> selectivity defends the encoding claim.
    # protein was the one rung missing this; see ../../docs/WS1_BACKLOG.md item E.
    y_shuf = np.random.RandomState(123).permutation(y)
    sp_ctrl = cross_val_predict(lr_fp, FP, y_shuf, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    sp_ctrl_auc = roc_auc_score(y_shuf, sp_ctrl)
    ac_ctrl = cross_val_predict(clf, np.asarray(H[best_L]), y_shuf, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    ac_ctrl_auc = roc_auc_score(y_shuf, ac_ctrl)
    print(f"CONTROL (shuffled labels): structure-probe={sp_ctrl_auc:.3f} | activation@L{best_L}={ac_ctrl_auc:.3f}", flush=True)
    print(f"SELECTIVITY: structure-probe={sp_auc - sp_ctrl_auc:+.3f} | activation={best - ac_ctrl_auc:+.3f}  (high = probe reads real signal, near-0 = probe just fits)", flush=True)


if __name__ == "__main__":
    main()
