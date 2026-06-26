"""WS1 axis-B: three arms on ONE molecule set under ONE scaffold split (Cayuga GPU).

Fixes the review's blocking issues:
- C1/C2: structure-probe (Morgan FP), LLM-activation (hidden states), and LLM-output
  (generate) are measured on the SAME balanced hERG sample; the two trained probes use
  the SAME scaffold GroupKFold (leakage-controlled, like ceiling_gate.py).
- C3: the output arm is instrumented (parsed / percent / fallback counts) with an
  anchored parser (last number in range; percents mapped) + bootstrap CI.
- M1: best-layer activation reports a bootstrap CI and is flagged as max-over-layers.

Env: ACT_MODEL, ACT_N, ACT_CSV, ACT_4BIT.
"""
import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from probe_common import dump_layerloc, nested_layer_auroc, results_path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

CSV = os.environ.get("ACT_CSV", "herg.csv")
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("ACT_N", "2000"))
THINK = os.environ.get("ACT_THINK", "0") == "1"   # Qwen3 thinking mode (longer generate)
GEN_TOKENS = 512 if THINK else 12
# ACT_RANDOMIZE=1: feed a deterministic NON-canonical re-notation of each SMILES to the
# model (hidden states + output), while keeping the scaffold split and the Morgan
# structure-probe on the CANONICAL form. Decisive surface-vs-chemistry test: if the
# activation AUROC holds vs the canonical run it survives re-notation (like the char-n-gram,
# `lipophilicity_control.md`); if it drops toward chance it was canonical-string orthography.
RANDOMIZE = os.environ.get("ACT_RANDOMIZE", "0") == "1"

PROMPT = os.environ.get("ACT_PROMPT") or (
    "You are given a molecule as a SMILES string. Estimate the probability "
    "(a single number between 0 and 1) that it BLOCKS the hERG potassium "
    "channel (cardiotoxicity risk). Judge only from the structure. "
    "Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:"
)
# ACT_PROMPT lets the computable-property row (signal/generate_computable.py +
# eval/output_arm_computable.py) reuse this GPU activation arm unchanged: same hidden-state
# extraction and structure-probe, a different question. The prompt must contain a {smiles}
# field. Pair it with ACT_PARSE=number so a raw count/value output is not clamped to [0,1].
PARSE_MODE = os.environ.get("ACT_PARSE", "prob")  # "prob" = hERG default; "number" = raw value


def load(csv_path, n):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["smiles"], int(row["label"])))
    pos = [x for x in rows if x[1] == 1]
    neg = [x for x in rows if x[1] == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def scaffold_of(smi):
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smi)
        return s or smi
    except Exception:
        return smi


def renotate(smi, seed):
    """Deterministic NON-canonical re-notation (same molecule, atoms renumbered by a
    seeded permutation). Reproducible given seed, unlike RDKit doRandom."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    idx = list(range(m.GetNumAtoms()))
    np.random.RandomState(seed).shuffle(idx)
    try:
        return Chem.MolToSmiles(Chem.RenumberAtoms(m, idx), canonical=False)
    except Exception:
        return smi


def chat_input(tok, smi):
    msgs = [{"role": "user", "content": PROMPT.format(smiles=smi)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_prob(txt):
    """Anchored: take the LAST number. prob mode (default, hERG): [0,1] direct, (1,100] as
    percent. number mode (ACT_PARSE=number): return the raw value for a computable descriptor
    (atom/ring count, MW) with no clamp; fallback is NaN (mean-imputed in main)."""
    if PARSE_MODE == "number":
        for tok in reversed(re.findall(r"-?\d*\.?\d+", txt)):
            try:
                return float(tok), "parsed"
            except ValueError:
                continue
        return float("nan"), "fallback"
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


def main():
    data = load(CSV, N)
    smis = [s for s, _ in data]
    y = np.array([l for _, l in data])
    groups = np.array([scaffold_of(s) for s in smis])     # scaffold split: ALWAYS canonical
    # the model sees re-notated strings when RANDOMIZE; scaffold split + Morgan stay canonical
    smis_in = [renotate(s, i) for i, s in enumerate(smis)] if RANDOMIZE else smis

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    FP = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        FP.append(gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(2048, dtype=np.int8))
    FP = np.asarray(FP)

    tok, model = load_model()
    device = next(model.parameters()).device

    layers, H, outp, ptypes = None, None, [], []
    for i, s in enumerate(smis_in):
        inp = tok(chat_input(tok, s), return_tensors="pt", truncation=True, max_length=512).to(device)
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
            print(f"  processed {i+1}/{len(smis)}", flush=True)
    outp = np.array(outp)
    if PARSE_MODE == "number":  # number-mode fallbacks are NaN; mean-impute so OUTPUT AUROC ranks
        fin = np.isfinite(outp)
        outp = np.where(fin, outp, (outp[fin].mean() if fin.any() else 0.0))

    print(f"MODEL={MODEL}  think={THINK}  gen_tokens={GEN_TOKENS}  randomize={RANDOMIZE}  n={len(y)}  pos={int(y.sum())}  scaffolds={len(set(groups))}  layers={layers}", flush=True)
    if RANDOMIZE:
        print("  [RANDOMIZE] model sees deterministic non-canonical re-notations; scaffold split + Morgan on canonical", flush=True)

    cv = GroupKFold(5)

    # structure-probe (Morgan FP), same scaffold split
    lr_fp = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    sp = cross_val_predict(lr_fp, FP, y, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    sp_auc = roc_auc_score(y, sp)
    print(f"STRUCTURE-PROBE (Morgan FP, scaffold)  AUROC={sp_auc:.3f}  AUPRC={average_precision_score(y, sp):.3f}", flush=True)

    # output (zero-shot, no training -> direct AUROC) + parse instrumentation
    pc = Counter(ptypes)
    o_auc = roc_auc_score(y, outp)
    o_lo, o_hi = bootstrap_ci(y, outp)
    print(f"OUTPUT  AUROC={o_auc:.3f} [95% {o_lo:.3f},{o_hi:.3f}]  parsed={pc['parsed']} percent={pc['percent']} fallback={pc['fallback']}", flush=True)

    # activation (hidden states), same scaffold split, per layer
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, best_L, best_proba, aucs = 0.0, -1, None, []
    for L in range(layers):
        proba = cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, proba)
        aucs.append(float(a))
        if a > best:
            best, best_L, best_proba = a, L, proba
        print(f"  layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    a_lo, a_hi = bootstrap_ci(y, best_proba)
    print(f"\nbest ACTIVATION layer {best_L}: AUROC={best:.3f} [95% {a_lo:.3f},{a_hi:.3f}] (MAX over {layers} layers, selection-biased)", flush=True)

    # ACT_DUMP=path: per-item out-of-fold best-layer probe predictions, keyed by SMILES, for
    # the per-drug agreement test (does the 8B activation flag the same drugs as the frontier
    # name route). Also dumps the structure-probe and output predictions for reference.
    if os.environ.get("ACT_DUMP"):
        import json as _json
        rows = [{"smiles": smis[i], "label": int(y[i]),
                 "act": round(float(best_proba[i]), 4), "struct": round(float(sp[i]), 4),
                 "output": round(float(outp[i]), 4)} for i in range(len(y))]
        _json.dump({"best_layer": int(best_L), "model": MODEL, "n": len(y), "items": rows},
                   open(os.environ["ACT_DUMP"], "w"))
        print(f"  [ACT_DUMP] wrote {len(rows)} per-item predictions to {os.environ['ACT_DUMP']}", flush=True)

    # unbiased best-layer estimate (nested CV, shared probe_common); best - heldout = selection bias
    nb = nested_layer_auroc(
        H, y, groups=groups,
        clf_factory=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    ho_auc, ho_layers = nb["auroc"], nb["picked"]
    print(f"HELD-OUT-LAYER ACTIVATION: AUROC={ho_auc:.3f} (nested GroupKFold, layers picked {ho_layers}) | selection bias = {best - ho_auc:+.3f}", flush=True)

    print(f"SUMMARY (same {len(y)} molecules, scaffold split):  structure-probe={sp_auc:.3f} | activation(max)={best:.3f} | activation(held-out-layer)={ho_auc:.3f} | output={o_auc:.3f}", flush=True)
    print("gaps: encoding = probe - activation | expression = activation - output  (all one set, one split)", flush=True)

    # control task (Hewitt-Liang 1909.03368): shuffled labels -> selectivity defends the encoding claim
    y_shuf = np.random.RandomState(123).permutation(y)
    sp_ctrl = cross_val_predict(lr_fp, FP, y_shuf, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    sp_ctrl_auc = roc_auc_score(y_shuf, sp_ctrl)
    ac_ctrl = cross_val_predict(clf, np.asarray(H[best_L]), y_shuf, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    ac_ctrl_auc = roc_auc_score(y_shuf, ac_ctrl)
    print(f"CONTROL (shuffled labels): structure-probe={sp_ctrl_auc:.3f} | activation@L{best_L}={ac_ctrl_auc:.3f}", flush=True)
    print(f"SELECTIVITY: structure-probe={sp_auc - sp_ctrl_auc:.3f} | activation={best - ac_ctrl_auc:.3f}  (high = probe reads real signal, near-0 = probe just fits)", flush=True)

    # task-tagged layer-loc JSON (per-layer curve, nested headline + OOF for H3, output for H2/H3)
    tag = MODEL.split("/")[-1]
    dump_layerloc(results_path(f"layer_loc_herg_{tag}.json"), "admet/herg", MODEL, y, aucs, nb,
                  best - ac_ctrl_auc, output=outp, ceiling=sp_auc)
    print(f"  [layer-loc] wrote results/layer_loc_herg_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
