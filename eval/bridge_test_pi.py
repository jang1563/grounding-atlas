"""Bridge test for isoelectric_point: is its PARTIAL closure (AUROC 0.79, mae 1.28 pH at 100%
completion) a CONSTANT-RECALL limit or an ALGORITHM limit?

The reason-regime trace showed the model does not run rigorous Henderson-Hasselbalch; it counts
charged residues and ESTIMATES pI from charge balance (true 5.40 -> 6.49, 9.14 -> 9.5). Claim:
pI/logp sit on the seam between computable (the algorithm) and empirical (the pKa / Crippen
constants the model only approximately recalls). Test by SUPPLYING the pKa table and instructing
the exact method, i.e. simulating an "orchestrate with a constants tool" condition. If pI then
closes toward AUROC ~1.0, the bottleneck was constant-recall, and the bridge claim is confirmed
(decision-map: pI -> orchestrate lane). If it does not move, the bottleneck is execution.

Reuses output_arm_computable helpers (same seed-42 balanced sample, parse, truncation guard,
scoring), so the +pKa number is directly comparable to the baseline reason-regime pI. No em dashes.
Env: BPI_N (default 12), BPI_MODEL (default claude-sonnet-4-6), BPI_MAXTOK (default 4000), BPI_DRY.
"""
import json
import os
from collections import Counter

import output_arm_computable as oac  # sibling module (run as: python eval/bridge_test_pi.py)

PKA = ("N-terminus 9.0, C-terminus 2.0, Asp(D) 3.9, Glu(E) 4.1, His(H) 6.0, "
       "Cys(C) 8.3, Tyr(Y) 10.5, Lys(K) 10.5, Arg(R) 12.5")
SYSTEM = ("You compute protein descriptors. Work step by step, but END your reply with the "
          "final numeric answer on its own line and write nothing after it.")
N = int(os.environ.get("BPI_N", "12"))
MODEL = os.environ.get("BPI_MODEL", "claude-sonnet-4-6")
MAXTOK = int(os.environ.get("BPI_MAXTOK", "4000"))
DRY = os.environ.get("BPI_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")


def prompt(seq):
    return (f"Compute the isoelectric point (pI) of this protein: the pH at which the net charge "
            f"is zero. Use the Henderson-Hasselbalch method with these standard pKa values: {PKA}. "
            f"Count each ionizable group (D, E, C, Y, H, K, R, and the two termini), then solve for "
            f"the pH at which the net charge equals zero. End with the final pI on its own line."
            f"\n\nSequence: {seq}\nAnswer:")


def ask(client, seq):
    try:
        m = client.messages.create(model=MODEL, max_tokens=MAXTOK, system=SYSTEM,
                                    messages=[{"role": "user", "content": prompt(seq)}])
    except Exception:
        return float("nan"), "error"
    texts = [b.text for b in m.content if getattr(b, "type", None) == "text"]
    if not texts:
        return float("nan"), "empty"
    if getattr(m, "stop_reason", None) == "max_tokens":
        return float("nan"), "truncated"   # same guard as output_arm_computable
    return oac.parse_number(texts[0])


def main():
    data, path = oac.load_pairs("protein", "isoelectric_point", "matched", N)
    if not data:
        raise SystemExit(f"no pI pairs at {path}")
    tol = oac.PANEL["isoelectric_point"][2]
    print(f"== pI BRIDGE TEST (+pKa supplied, rigorous HH instructed) :: "
          f"{'DRY' if DRY else MODEL} :: N={len(data)} :: MAXTOK={MAXTOK} ==")
    print("   baseline (no constants, charge-balance estimate): AUROC 0.79, exact 0.25, mae 1.28")
    if DRY:
        print("   sample prompt:\n   " + prompt(data[0][0])[:300])
        return
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    preds, vals, labels, kinds = [], [], [], []
    for i, (seq, val, lab, _id) in enumerate(data):
        p, k = ask(client, seq)
        preds.append(p); vals.append(val); labels.append(lab); kinds.append(k)
        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(data)}", flush=True)
    res = {"property": "isoelectric_point", "condition": "+pKa_supplied", "model": MODEL,
           "max_tokens": MAXTOK, "parse": dict(Counter(kinds)),
           **oac.score(preds, vals, labels, "float", tol)}
    res["baseline_reason"] = {"auroc": 0.792, "exact": 0.25, "mae": 1.278}
    pa = res["parse"]; ok = pa.get("parsed", 0); tot = sum(pa.values()) or 1
    print(f"\n   +pKa: completion={ok}/{tot} ({ok/tot:.0%}) exact={res.get('exact_acc')} "
          f"mae={res.get('mae')} AUROC={res.get('auroc')} (baseline AUROC 0.79)")
    out = os.path.join(oac.ROOT, "results", "bridge_test_pi.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"   saved -> {out}")
    verdict = ("CONFIRMED: constant-recall was the bottleneck (pI closes when constants supplied)"
               if (res.get("auroc") or 0) >= 0.9 else
               "NOT confirmed: closure is partial even with constants (execution-limited)")
    print(f"   verdict: {verdict}")


if __name__ == "__main__":
    main()
