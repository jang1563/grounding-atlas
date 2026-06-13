"""Axis D inter-rater agreement: per-dimension kappa for the human-rater pass.

Lifts axis D off exploratory (the automated scores sit at kappa ~0.36; the gate is a
human pass clearing ~0.60, docs/D_rater_protocol.md). Reads the long-format rating sheet
(d_rating_sheet_template.csv: one row per item x rater), computes Cohen kappa (exactly 2
raters) or Fleiss kappa (>2) per binary dimension, lists split items for adjudication,
and reports each dimension against the 0.60 substantial-agreement target.

Stdlib only (no numpy/sklearn), so it runs under any python3. Blank cells = 'unsure' and
are excluded from that dimension's kappa (decision rule 4). With no CSV argument it runs a
synthetic self-test so the tool is verifiable without real ratings.

Usage: python3 eval/d_agreement.py [ratings.csv]   No em dashes.
"""
import csv
import sys
from collections import defaultdict

DIMS = ["appropriate_rejection", "tracks_hazard", "over_trusts_corrupted"]
TARGET = 0.60


def cohen_kappa(pairs):
    """pairs: list of (a, b) binary labels from 2 raters on the same items."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def fleiss_kappa(items):
    """items: list of label-lists (one per item), each with the SAME number of raters, binary."""
    items = [it for it in items if len(it) == len(items[0])]
    if not items:
        return None
    n = len(items[0])
    if n < 2:
        return None
    N = len(items)
    # per-item agreement
    P = []
    cat_tot = {0: 0, 1: 0}
    for it in items:
        c = {0: it.count(0), 1: it.count(1)}
        cat_tot[0] += c[0]
        cat_tot[1] += c[1]
        P.append((c[0] ** 2 + c[1] ** 2 - n) / (n * (n - 1)))
    Pbar = sum(P) / N
    pj = {k: cat_tot[k] / (N * n) for k in (0, 1)}
    Pe = pj[0] ** 2 + pj[1] ** 2
    return 1.0 if Pe == 1 else round((Pbar - Pe) / (1 - Pe), 3)


def analyze(rows):
    """rows: list of dicts with item_id, rater_id, and the DIMS columns."""
    out = {}
    for dim in DIMS:
        by_item = defaultdict(dict)  # item_id -> {rater_id: label}
        for r in rows:
            v = r.get(dim, "")
            if v == "" or v is None:
                continue
            by_item[r["item_id"]][r["rater_id"]] = int(v)
        raters = sorted({rid for d in by_item.values() for rid in d})
        items_full = {it: d for it, d in by_item.items() if len(d) == len(raters)}
        splits = [it for it, d in items_full.items() if len(set(d.values())) > 1]
        if not items_full:
            out[dim] = {"kappa": None, "n_items": 0, "n_raters": len(raters), "splits": []}
            continue
        if len(raters) == 2:
            pairs = [(d[raters[0]], d[raters[1]]) for d in items_full.values()]
            k = cohen_kappa(pairs)
        else:
            k = fleiss_kappa([list(d.values()) for d in items_full.values()])
        out[dim] = {"kappa": k, "n_items": len(items_full), "n_raters": len(raters),
                    "splits": sorted(splits)}
    return out


def report(res):
    print(f"{'dimension':24}{'kappa':>8}{'n_items':>9}{'raters':>8}  verdict")
    for dim in DIMS:
        r = res[dim]
        k = r["kappa"]
        if k is None:
            verdict = "no data"
        elif k >= TARGET:
            verdict = f"PASS (>= {TARGET})"
        else:
            verdict = f"below {TARGET}, adjudicate"
        ks = f"{k:.3f}" if k is not None else "  -  "
        print(f"{dim:24}{ks:>8}{r['n_items']:>9}{r['n_raters']:>8}  {verdict}")
        if r["splits"]:
            print(f"    split items for adjudication: {r['splits']}")


def selftest():
    print("SELF-TEST (synthetic ratings, no real data)\n")
    # 2 raters, mostly agree on tracks_hazard (the core dim), split on a couple
    rows = []
    truth = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # tracks_hazard truth, near-zero conflation pattern
    flips = {2, 7}  # rater B disagrees on these item indices
    for i, t in enumerate(truth):
        rows.append({"item_id": f"d{i}", "rater_id": "A",
                     "appropriate_rejection": 1, "tracks_hazard": t, "over_trusts_corrupted": ""})
        b = 1 - t if i in flips else t
        rows.append({"item_id": f"d{i}", "rater_id": "B",
                     "appropriate_rejection": 1, "tracks_hazard": b, "over_trusts_corrupted": ""})
    report(analyze(rows))
    print("\n(over_trusts_corrupted has no data = D-v2 not rated in this synthetic set)")


def main():
    if len(sys.argv) < 2:
        selftest()
        return
    with open(sys.argv[1]) as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} rating rows from {sys.argv[1]}\n")
    report(analyze(rows))


if __name__ == "__main__":
    main()
