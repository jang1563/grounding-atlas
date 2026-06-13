"""Variant-branch LLM-output arm: the dual-form (text vs sequence) within-modality test.

The headline of the variant branch. The SAME variant is shown to a general LLM in two surface
forms with opposite web-exposure, and we ask the identical question (probability of
pathogenicity), parsed with the SAME anchored parser + parsed/percent/fallback instrumentation
as ../../eval/head_to_head.py:

  text  : "{gene} {HGVS}"  (web-rich symbolic form; co-occurs with pathogenic/benign in ClinVar,
          OMIM, abstracts). Prediction: AUROC ABOVE chance (Hu et al, GPT-4o ~0.73, 2025).
  seq   : a wild-type protein window + the AA substitution, NO gene name (web-poor raw form).
          Prediction: AUROC near chance (SMILES-like).

Leakage controls (run as conditions / strata, see ../README.md):
  text_scramble : the gene symbol is character-shuffled to a pseudonym, HGVS unchanged. The
                  AUROC drop from `text` quantifies reliance on the memorized gene-disease prior.
  text_nogene   : HGVS only, no gene symbol (intermediate between text and seq).
  + every condition's AUROC is also reported stratified by review stars (1 vs 2+) and by the
    temporal first_seen bin (post-2026-01 = strict holdout). The text-minus-seq gap and the
    star/temporal collapse decompose web-exposure from memorization.

Compare the SMILES branch: SMILES has no web-rich symbolic form, so its output sits at chance
(0.45) with no text arm above it. Variants are the predicted opposite extreme.

Env: VG_CSV, VG_MODEL, VG_PROVIDER (anthropic|openai), VG_N (cap, balanced), VG_COND
     (comma list, default "text,seq,text_scramble"), VG_WORKERS, VG_MAXTOK.
Keys: `set -a; source ~/.api_keys; set +a` first. Never prints key values.
"""
import csv
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = os.environ.get("VG_CSV", os.path.join(os.path.dirname(__file__), "..", "data", "variant_clinvar.csv"))
MODEL = os.environ.get("VG_MODEL", "claude-sonnet-4-5-20250929")
PROVIDER = os.environ.get("VG_PROVIDER", "anthropic")
N = int(os.environ.get("VG_N", "400"))
CONDS = os.environ.get("VG_COND", "text,seq,text_scramble").split(",")
WORKERS = int(os.environ.get("VG_WORKERS", "8"))
MAXTOK = int(os.environ.get("VG_MAXTOK", "16"))
# task wording: ClinVar = pathogenicity (default); DMS = function (VG_TASK_Q / VG_TASK_POS).
TASK_Q = os.environ.get("VG_TASK_Q", "pathogenic or benign")
TASK_POS = os.environ.get("VG_TASK_POS", "PATHOGENIC")

TEXT_PROMPT = (
    "Is the human variant {{gene}} {{hgvs}} {q}? Estimate the probability "
    "(a single number between 0 and 1) that it is {pos}. Reply with ONLY the number.\n"
    "Probability:"
).format(q=TASK_Q, pos=TASK_POS)
NOGENE_PROMPT = (
    "Is the human protein variant {{hgvs}} {q}? Estimate the probability "
    "(a single number between 0 and 1) that it is {pos}. Reply with ONLY the number.\n"
    "Probability:"
).format(q=TASK_Q, pos=TASK_POS)
SEQ_PROMPT = (
    "You are given a fragment of a human protein sequence. The residue at position {{win_pos}} "
    "(1-based) of this fragment is mutated from {{wt}} to {{mut}}. Estimate the probability (a "
    "single number between 0 and 1) that this missense change is {pos}. Judge only from "
    "the sequence and the substitution. Reply with ONLY the number.\n\n"
    "Sequence: {{seq}}\nProbability:"
).format(pos=TASK_POS)


def scramble_gene(gene, seed=42):
    """Deterministic char-shuffle of the gene symbol -> a pseudonym (mirrors the SMILES scramble
    arm). Keeps it gene-symbol-shaped but breaks the memorized symbol, isolating the prior."""
    ch = list(gene)
    r = random.Random(f"{seed}:{gene}")
    for _ in range(8):
        r.shuffle(ch)
        if "".join(ch) != gene:
            break
    return "".join(ch)


def build_prompt(cond, row):
    if cond == "text":
        return TEXT_PROMPT.format(gene=row["gene"], hgvs=row["hgvs_p"])
    if cond == "text_scramble":
        return TEXT_PROMPT.format(gene=scramble_gene(row["gene"]), hgvs=row["hgvs_p"])
    if cond == "text_nogene":
        return NOGENE_PROMPT.format(hgvs=row["hgvs_p"])
    if cond == "seq":
        return SEQ_PROMPT.format(win_pos=row["win_pos"], wt=row["wt"], mut=row["mut"], seq=row["wt_window"])
    raise ValueError(f"unknown condition {cond}")


def parse_prob(txt):
    """Anchored: take the LAST number; map [0,1] directly, (1,100] as percent. Identical to
    ../../eval/head_to_head.py so the output arm is parsed the same way across branches."""
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


def make_client():
    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if PROVIDER == "openai":
        # OpenAI-compatible: set VG_BASE_URL + VG_KEY_ENV for Together/DeepSeek/etc.
        from openai import OpenAI
        key = os.environ[os.environ.get("VG_KEY_ENV", "OPENAI_API_KEY")]
        base = os.environ.get("VG_BASE_URL")
        return OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    raise ValueError(f"unknown provider {PROVIDER}")


def ask(client, prompt, tries=3):
    for attempt in range(tries):
        try:
            if PROVIDER == "anthropic":
                msg = client.messages.create(
                    model=MODEL, max_tokens=MAXTOK,
                    messages=[{"role": "user", "content": prompt}])
                if getattr(msg, "stop_reason", None) == "refusal":
                    return 0.5, "refusal"
                texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
                return parse_prob(texts[0]) if texts else (0.5, "empty")
            else:  # openai
                resp = client.chat.completions.create(
                    model=MODEL, max_tokens=MAXTOK,
                    messages=[{"role": "user", "content": prompt}])
                m = resp.choices[0].message
                if getattr(m, "refusal", None):
                    return 0.5, "refusal"
                txt = m.content or ""
                return parse_prob(txt) if txt.strip() else (0.5, "empty")
        except Exception:
            if attempt == tries - 1:
                return 0.5, "error"
            time.sleep(1.5 * (attempt + 1))


def run_condition(client, rows, cond):
    results = [None] * len(rows)
    def work(i):
        results[i] = ask(client, build_prompt(cond, rows[i]))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, range(len(rows))))
    probs = np.array([r[0] for r in results])
    kinds = [r[1] for r in results]
    return probs, kinds


def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")


def bootstrap_ci(y, p, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    a = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"), float("nan"))


def strata(rows):
    return [
        ("ALL", lambda r: True),
        ("star1", lambda r: int(r["stars"]) == 1),
        ("star2+", lambda r: int(r["stars"]) >= 2),
        ("le_2025_06", lambda r: r["first_seen"] == "le_2025_06"),
        ("post_2026_01", lambda r: int(r["post_cutoff"]) == 1),
    ]


def load_rows(path, n):
    rows = list(csv.DictReader(open(path)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def main():
    rows = load_rows(CSV, N)
    y = np.array([int(r["label"]) for r in rows])
    print(f"csv={os.path.basename(CSV)}  model={MODEL} ({PROVIDER})  n={len(rows)}  "
          f"P={int(y.sum())} B={int((1-y).sum())}  conds={CONDS}\n", flush=True)
    client = make_client()

    cond_probs = {}
    for cond in CONDS:
        t0 = time.time()
        probs, kinds = run_condition(client, rows, cond)
        cond_probs[cond] = probs
        pc = Counter(kinds)
        lo, hi = bootstrap_ci(y, probs)
        print(f"=== {cond}  ({time.time()-t0:.0f}s) ===", flush=True)
        print(f"  parse: {dict(pc)}  (refusal/empty/fallback/error -> 0.5)", flush=True)
        # parsed-only AUROC: when the model DOES answer, how good is it? (separates refusal
        # rate from grounding; refusal is a finding, not a 0.5-diluted AUROC artifact)
        ans = np.array([k in ("parsed", "percent") for k in kinds])
        declined = pc.get("refusal", 0) + pc.get("empty", 0)
        if ans.sum() >= 20 and len(set(y[ans])) > 1:
            print(f"  answered={int(ans.sum())}/{len(y)} (declined={declined}); "
                  f"parsed-only AUROC={roc_auc_score(y[ans], probs[ans]):.3f}", flush=True)
        for name, pred in strata(rows):
            mask = np.array([pred(r) for r in rows])
            ys, ps = y[mask], probs[mask]
            if mask.sum() == 0 or len(set(ys)) < 2:
                print(f"    {name:14s} n={int(mask.sum()):4d}  (insufficient)", flush=True)
                continue
            a = roc_auc_score(ys, ps)
            extra = f" [{lo:.3f},{hi:.3f}]" if name == "ALL" else ""
            print(f"    {name:14s} n={int(mask.sum()):4d}  AUROC={a:.3f}{extra}  "
                  f"AUPRC={average_precision_score(ys, ps):.3f}  mean_p={ps.mean():.2f}", flush=True)
        print(flush=True)

    # decomposition summary
    print("=== decomposition ===", flush=True)
    if "text" in cond_probs and "seq" in cond_probs:
        ta, sa = auc(y, cond_probs["text"]), auc(y, cond_probs["seq"])
        print(f"  text AUROC={ta:.3f}  seq AUROC={sa:.3f}  "
              f"text-minus-seq (web-exposure within modality) = {ta-sa:+.3f}", flush=True)
    if "text" in cond_probs and "text_scramble" in cond_probs:
        ta, sca = auc(y, cond_probs["text"]), auc(y, cond_probs["text_scramble"])
        print(f"  text AUROC={ta:.3f}  text_scramble AUROC={sca:.3f}  "
              f"gene-prior reliance (text-minus-scramble) = {ta-sca:+.3f}", flush=True)
    print("\nRead: text > seq and text > scramble => the above-chance signal is web-rich symbol "
          "recall, not grounding from the variant. A flat seq arm (~0.5) is the SMILES-like "
          "floor. Compare ceiling_gate_variant.py (AlphaMissense holds on the same variants) and "
          "the DMS track (prepare_dms.py: recall has no labels to lean on).", flush=True)


if __name__ == "__main__":
    main()
