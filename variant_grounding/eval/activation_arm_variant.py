"""OPTIONAL 3rd arm (GPU, Cayuga): does the LLM ENCODE pathogenicity, per surface form?

The variant copy of ../../eval/activation_arm.py. Three arms on ONE variant set, with the same
anchored parser + parsed/percent/fallback instrumentation and a leakage-controlled split:

  ceiling  : AlphaMissense precomputed score AUROC (+ ESM-1v LLR if VG_ESM1V_COL set). Zero-shot
             specialist, no training split needed (the headline ceiling, see ceiling_gate_variant).
  activation: LogReg linear probe on the LLM hidden states (last token), per layer, run for BOTH
             surface forms (text = gene+HGVS; seq = window, no gene). Does the LLM encode the
             property from each form internally? Best layer = max over layers (selection-biased),
             bootstrap CI.
  output   : the LLM generates a probability for BOTH forms (zero-shot). Does it verbalize it?

LEAKAGE CONTROL (the within-modality analog of the SMILES scaffold split): the activation probe
uses GroupKFold grouped by GENE, so no variant of a gene seen in training appears in test. Without
it the probe trivially learns the memorized gene->pathogenicity prior instead of reading the
variant. A random-label control (Hewitt-Liang) bounds selectivity (VG_CONTROL=1).

Gaps per form: encoding = ceiling - activation; expression = activation - output. The text vs seq
contrast on activation is the encoding-side analog of the output-arm web-exposure test.

Env: VG_CSV, ACT_MODEL, VG_N, VG_FORMS (default "text,seq"), ACT_4BIT, VG_CONTROL.
"""
import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

CSV = os.environ.get("VG_CSV", os.path.join(os.path.dirname(__file__), "..", "data", "variant_clinvar.csv"))
MODEL = os.environ.get("ACT_MODEL", "Qwen/Qwen3-8B")
N = int(os.environ.get("VG_N", "1500"))
FORMS = os.environ.get("VG_FORMS", "text,seq").split(",")
THINK = os.environ.get("ACT_THINK", "0") == "1"
GEN_TOKENS = 512 if THINK else 12
CONTROL = os.environ.get("VG_CONTROL", "0") == "1"

TEXT_PROMPT = ("Is the human variant {gene} {hgvs} pathogenic or benign? Estimate the probability "
               "(a single number between 0 and 1) that it is PATHOGENIC. Reply with ONLY the number.\nProbability:")
SEQ_PROMPT = ("You are given a fragment of a human protein sequence. The residue at position {win_pos} "
              "(1-based) of this fragment is mutated from {wt} to {mut}. Estimate the probability (a single "
              "number between 0 and 1) that this missense change is PATHOGENIC. Judge only from the sequence "
              "and the substitution. Reply with ONLY the number.\n\nSequence: {seq}\nProbability:")


def prompt_for(form, r):
    if form == "text":
        return TEXT_PROMPT.format(gene=r["gene"], hgvs=r["hgvs_p"])
    return SEQ_PROMPT.format(win_pos=r["win_pos"], wt=r["wt"], mut=r["mut"], seq=r["wt_window"])


def parse_prob(txt):
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


def bootstrap_ci(y, p, n_boot=1000):
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    a = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def chat_input(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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
        MODEL, dtype="auto", device_map="auto", quantization_config=quant, trust_remote_code=True)
    model.eval()
    return tok, model


def load(csv_path, n):
    rows = list(csv.DictReader(open(csv_path)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def probe_per_layer(H, y, groups, cv):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    best, best_L, best_p = 0.0, -1, None
    for L in range(len(H)):
        proba = cross_val_predict(clf, np.asarray(H[L]), y, cv=cv, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
        a = roc_auc_score(y, proba)
        if a > best:
            best, best_L, best_p = a, L, proba
        print(f"    layer {L:2d}: ACT AUROC={a:.3f}", flush=True)
    return best, best_L, best_p


def heldout_layer_auroc(H, y, groups, n_splits=5):
    """Unbiased best-layer AUROC by nested GroupKFold (defends the encoding claim
    against max-over-layers selection bias). Inner folds pick the layer on TRAIN rows
    only; the held-out outer fold is scored at that layer. See ../../results/selection_bias.md.
    """
    fac = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    Harr = [np.asarray(h) for h in H]
    layers = len(Harr)
    oof = np.zeros(len(y), dtype=float)
    picked = []
    for tr, te in GroupKFold(n_splits).split(Harr[0], y, groups):
        g_tr = groups[tr]
        inner = GroupKFold(min(n_splits, len(np.unique(g_tr))))
        best_L, best_a = -1, -1.0
        for L in range(layers):
            p = cross_val_predict(fac(), Harr[L][tr], y[tr], cv=inner,
                                  groups=g_tr, method="predict_proba", n_jobs=5)[:, 1]
            a = roc_auc_score(y[tr], p)
            if a > best_a:
                best_a, best_L = a, L
        clf = fac().fit(Harr[best_L][tr], y[tr])
        oof[te] = clf.predict_proba(Harr[best_L][te])[:, 1]
        picked.append(best_L)
    return roc_auc_score(y, oof), picked


def main():
    rows = load(CSV, N)
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["gene"] for r in rows])
    am = np.array([float(r["am"]) if r.get("am") not in (None, "") else np.nan for r in rows])
    print(f"csv={os.path.basename(CSV)} model={MODEL} n={len(y)} pos={int(y.sum())} "
          f"genes={len(set(groups))} forms={FORMS} think={THINK}", flush=True)

    # ceiling (zero-shot specialist, no split)
    ok = ~np.isnan(am)
    if ok.sum() > 0 and len(set(y[ok])) > 1:
        print(f"CEILING AlphaMissense AUROC={roc_auc_score(y[ok], am[ok]):.3f} (n={int(ok.sum())})", flush=True)

    tok, model = load_model()
    device = next(model.parameters()).device
    cv = GroupKFold(5)

    summary = {}
    for form in FORMS:
        H, outp, kinds = None, [], []
        for i, r in enumerate(rows):
            inp = tok(chat_input(tok, prompt_for(form, r)), return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                fwd = model(**inp, output_hidden_states=True)
            vec = [h[0, -1].float().cpu().numpy() for h in fwd.hidden_states]
            if H is None:
                H = [[] for _ in range(len(vec))]
            for L in range(len(vec)):
                H[L].append(vec[L])
            with torch.no_grad():
                gtxt = model.generate(**inp, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
            p, kind = parse_prob(tok.decode(gtxt[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
            outp.append(p); kinds.append(kind)
            if (i + 1) % 100 == 0:
                print(f"  [{form}] processed {i+1}/{len(rows)}", flush=True)
        outp = np.array(outp)
        o_auc = roc_auc_score(y, outp)
        o_lo, o_hi = bootstrap_ci(y, outp)
        print(f"\n[{form}] OUTPUT AUROC={o_auc:.3f} [{o_lo:.3f},{o_hi:.3f}]  parse={dict(Counter(kinds))}", flush=True)
        print(f"[{form}] activation per-layer (GroupKFold by gene):", flush=True)
        best, best_L, best_p = probe_per_layer(H, y, groups, cv)
        a_lo, a_hi = bootstrap_ci(y, best_p)
        print(f"[{form}] best ACTIVATION layer {best_L}: AUROC={best:.3f} [{a_lo:.3f},{a_hi:.3f}] "
              f"(MAX over {len(H)} layers, selection-biased)", flush=True)
        ho_auc, ho_layers = heldout_layer_auroc(H, y, groups)
        print(f"[{form}] HELD-OUT-LAYER ACTIVATION: AUROC={ho_auc:.3f} (nested GroupKFold, layers {ho_layers}) | selection bias = {best - ho_auc:+.3f}", flush=True)
        if CONTROL:
            yr = np.random.RandomState(0).permutation(y)
            bc, _, _ = probe_per_layer(H, yr, groups, cv)
            print(f"[{form}] random-label control best AUROC={bc:.3f}  selectivity={best-bc:+.3f}", flush=True)
        summary[form] = (best, ho_auc, o_auc)

    print("\nSUMMARY (one set, gene GroupKFold):", flush=True)
    for form in FORMS:
        best, ho_auc, o_auc = summary[form]
        print(f"  {form:5s}: activation(max)={best:.3f}  activation(held-out-layer)={ho_auc:.3f}  output={o_auc:.3f}  "
              f"expression_gap={best-o_auc:+.3f}  selection_bias={best-ho_auc:+.3f}", flush=True)
    print("gaps: encoding = AlphaMissense_ceiling - activation | expression = activation - output", flush=True)
    print("Read: text activation/output > seq activation/output => web-exposure raises encoding AND "
          "recall in the symbolic form; the gene GroupKFold removes the trivial gene-prior shortcut.", flush=True)


if __name__ == "__main__":
    main()
