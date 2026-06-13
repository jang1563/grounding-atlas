"""Property-specificity control: is the T2-solve reading specific to the NAMED property,
or a generic "this molecule is bioactive" prior?

The notation control confirmed the CYP3A4 reading is structure-dependent (scrambled to
chance) and notation-invariant. The last objection: maybe the model outputs a generic
structure-based bioactivity score and pins it to whatever property is named, so the 0.58
is not CYP3A4-specific grounding. Test it: on the CYP3A4 molecules, ask the REAL property
(CYP3A4 inhibition) and an ORTHOGONAL control property (aqueous solubility), and read:

  self_auroc  = AUROC(P_cyp3a4, cyp3a4_label)   ~0.58 expected (the reading)
  ctrl_auroc  = AUROC(P_solubility, cyp3a4_label)
  corr        = Pearson(P_cyp3a4, P_solubility)

CORRECTED interpretation (a review caught the original rule as wrong): property-specificity
requires the orthogonal control to be UNINFORMATIVE (ctrl AUROC near 0.5) and the predictions
decorrelated. If ctrl is far below 0.5 (anti-informative), a SHARED structural axis
(lipophilicity raises CYP3A4 inhibition AND lowers solubility) reproduces self>>ctrl plus a
strong-negative corr, which is the generic-axis hypothesis, NOT its refutation. The
discriminator is |ctrl-0.5| small, not self-ctrl large. The observed 0.60 / 0.38 / -0.48 is
CONFOUNDED, so property-specificity is NOT established; a control orthogonal to lipophilicity
is needed. Same SYSTEM bare-number protocol as output_arm_admet.py;
3 conditions concurrent; bootstrap CI on the AUROCs. No em dashes.
Run on Cayuga: NOTE_N (default 300), PS_MODEL. Needs anthropic + sklearn + scipy(optional).
"""
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMET = os.path.join(ROOT, "signal", "admet")
SYSTEM = ("You are a molecular property predictor. Respond with ONLY a single decimal "
          "number between 0 and 1 (for example: 0.42). No words, no explanation.")
PROMPT = ("You are given a molecule as a SMILES string. Estimate the probability "
          "(a single number between 0 and 1) that it {clause}. Judge only from the "
          "structure. Reply with ONLY the number.\n\nSMILES: {smiles}\nProbability:")
REAL = "INHIBITS the CYP3A4 enzyme"
CTRL = "is highly soluble in water"
EP = "cyp3a4"


def parse_prob(txt):
    for tok in reversed(re.findall(r"\d*\.?\d+", txt)):
        v = float(tok)
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
    return 0.5


def load_balanced(n):
    by = defaultdict(list)
    for line in open(os.path.join(ADMET, EP, "pairs.jsonl")):
        r = json.loads(line)
        if r["condition"] == "matched":
            by[int(r["label"])].append(r["representation"])
    rng = np.random.RandomState(42)
    out = []
    for lab in (0, 1):
        it = by[lab][:]
        rng.shuffle(it)
        out += [(s, lab) for s in it[:n // 2]]
    return out


def boot_ci(y, p, n=2000):
    y, p = np.array(y), np.array(p)
    rng = np.random.RandomState(0)
    a = [roc_auc_score(y[b], p[b]) for b in (rng.randint(0, len(y), len(y)) for _ in range(n))
         if len(set(y[b])) > 1]
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)] if a else None


def main():
    n = int(os.environ.get("NOTE_N", "300"))
    model = os.environ.get("PS_MODEL", "claude-sonnet-4-6")
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def ask(clause, smi):
        m = client.messages.create(model=model, max_tokens=16, system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT.format(clause=clause, smiles=smi)}])
        return parse_prob("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))

    data = load_balanced(n)
    y, p_real, p_ctrl = [], [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, (smi, lab) in enumerate(data):
            fr = ex.submit(ask, REAL, smi)
            fc = ex.submit(ask, CTRL, smi)
            y.append(lab); p_real.append(fr.result()); p_ctrl.append(fc.result())
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(data)}", flush=True)
    y = np.array(y)
    self_auc = round(float(roc_auc_score(y, p_real)), 3)
    ctrl_auc = round(float(roc_auc_score(y, p_ctrl)), 3)
    corr = round(float(np.corrcoef(p_real, p_ctrl)[0, 1]), 3)
    # CORRECTED verdict logic (an adversarial review caught the original as wrong):
    # property-specificity requires the control to be UNINFORMATIVE (AUROC near 0.5) and
    # the two predictions decorrelated. An ANTI-correlated control (AUROC far below 0.5,
    # |AUROC-0.5| large) is INFORMATIVE in the opposite direction = a shared structural
    # axis (e.g. lipophilicity raises CYP3A4 inhibition and lowers solubility) reproduces
    # self>>ctrl + strong-negative corr, so that pattern is NOT property-specificity.
    # The discriminator is |ctrl-0.5| small, NOT self-ctrl large.
    if self_auc > 0.55 and abs(ctrl_auc - 0.5) < 0.06 and abs(corr) < 0.3:
        verdict = "property-specific (orthogonal control is uninformative)"
    elif abs(ctrl_auc - 0.5) >= 0.06:
        verdict = ("CONFOUNDED: control AUROC is informative (anti); a shared structural "
                   "axis reproduces this, so property-specificity is NOT established")
    else:
        verdict = "inconclusive"
    out = {"endpoint": EP, "model": model, "n": len(y),
           "self_auroc_cyp3a4_vs_cyp3a4label": self_auc, "self_ci": boot_ci(y, p_real),
           "control_auroc_solubility_vs_cyp3a4label": ctrl_auc, "control_ci": boot_ci(y, p_ctrl),
           "pearson_corr_Preal_Pctrl": corr, "verdict": verdict}
    with open(os.path.join(ROOT, "results", "property_specificity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nself (P_cyp3a4 vs cyp3a4) = {self_auc} {out['self_ci']}")
    print(f"control (P_solubility vs cyp3a4) = {ctrl_auc} {out['control_ci']}")
    print(f"corr(P_cyp3a4, P_solubility) = {corr}  -> {out['verdict']}")


if __name__ == "__main__":
    main()
