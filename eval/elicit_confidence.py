"""Calibration/routing arm Part B: elicit explicit per-item self-confidence.

Asks a model how reliable its OWN prediction is on each ADMET item (the same items it already
scored in results/benchmark/<model>/raw.jsonl) -- a self-assessment of competence, separate
from the prediction itself -- and saves results/benchmark/<model>/confidence.jsonl. Then
routing_arm.py compares this explicit signal against the implicit |P-0.5| signal: does asking
"how sure are you?" recover more of the oracle headroom than the prediction's own decisiveness?

Run:  python eval/elicit_confidence.py --model claude-opus-4-8
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compute_ceilings import SIGNAL, matched  # noqa: E402
from run_grounding_eval import CLAUSES, OUT, complete, parse_prob  # noqa: E402

RUNGS = ["ames", "cyp2d6", "cyp3a4", "herg", "permeability", "solubility"]


def conf_prompt(clause, smi):
    return (f"How reliable is your own prediction of whether this molecule {clause}, judging only "
            f"from the structure below? Reply with ONLY a number between 0 and 1, where 1 means you "
            f"can predict it confidently and 0 means you would be guessing.\n"
            f"molecule: {smi}\nReliability:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()

    smiles = {}
    for rung in RUNGS:
        for r in matched(os.path.join(SIGNAL, "admet", rung, "pairs.jsonl")):
            smiles.setdefault(rung, {})[r["id"]] = r["representation"]
    scored = {}
    for line in open(os.path.join(OUT, args.model, "raw.jsonl")):
        r = json.loads(line)
        rung = r["rung"].split("/")[-1]
        if rung in RUNGS:
            scored.setdefault(rung, []).append(r["id"])

    out = []
    for rung in RUNGS:
        clause = CLAUSES[rung][0]
        for iid in scored.get(rung, []):
            smi = smiles[rung].get(iid)
            if smi is not None:
                out.append({"rung": rung, "id": iid,
                            "confidence": parse_prob(complete(args.model, conf_prompt(clause, smi)))})
        print(f"  {rung}: {sum(1 for o in out if o['rung'] == rung)} done", flush=True)
    with open(os.path.join(OUT, args.model, "confidence.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {OUT}/{args.model}/confidence.jsonl ({len(out)} items)", flush=True)


if __name__ == "__main__":
    main()
