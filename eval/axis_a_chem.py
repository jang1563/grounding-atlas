"""Axis A (identity resolution) for chemistry: the within-entity recognition gap.

The protein A-axis anchor is name ~100% vs database-accession ~2-28% (FRT,
docs/FAILURE_MODES.md). This is its chemistry, within-entity form: the SAME famous
drug presented as a web-frequent notation (name), a moderate one (canonical SMILES),
and a web-rare accession-analog (InChIKey, a hash). The web-exposure law predicts
recognition tracks notation web-frequency, so name >> SMILES >> InChIKey on one set
of molecules the model demonstrably knows by name.

Scoring is deterministic (no LLM judge): the model is asked for the common drug name
(or UNKNOWN), and a response counts as recognized if a known name/synonym is present.
The name condition asks for recognition (not the name it was handed), establishing the
ceiling that these ARE name-known drugs, so a low InChIKey rate is a notation gap, not
ignorance.

Curation caveat: the drug SMILES are hand-curated from knowledge and self-validated by
RDKit molecular-formula match against the known formula (catches recall errors in atom
count); a connectivity error that preserves formula would survive, so this is a
validated-but-not-authoritative set. Rows that fail the formula check are dropped.

DRY (ADMET_DRY=1 or no key): validate the set and print prompts, no API.
Env: AXISA_MODEL (default claude-sonnet-4-6), AXISA_DRY. No em dashes.
"""
import json
import os
import re
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# name, synonyms (incl. brand), SMILES, known molecular formula (Hill)
DRUGS = [
    ("aspirin",          ["acetylsalicylic acid"],        "CC(=O)Oc1ccccc1C(=O)O",              "C9H8O4"),
    ("caffeine",         [],                              "Cn1cnc2c1c(=O)n(C)c(=O)n2C",         "C8H10N4O2"),
    ("ibuprofen",        [],                              "CC(C)Cc1ccc(cc1)C(C)C(=O)O",         "C13H18O2"),
    ("paracetamol",      ["acetaminophen", "tylenol"],    "CC(=O)Nc1ccc(O)cc1",                 "C8H9NO2"),
    ("nicotine",         [],                              "CN1CCC[C@H]1c1cccnc1",               "C10H14N2"),
    ("metformin",        [],                              "CN(C)C(=N)NC(N)=N",                  "C4H11N5"),
    ("dopamine",         [],                              "NCCc1ccc(O)c(O)c1",                  "C8H11NO2"),
    ("serotonin",        ["5-hydroxytryptamine"],         "NCCc1c[nH]c2ccc(O)cc12",             "C10H12N2O"),
    ("diazepam",         ["valium"],                      "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21","C16H13ClN2O"),
    ("naproxen",         [],                              "COc1ccc2cc(ccc2c1)C(C)C(=O)O",       "C14H14O3"),
    ("warfarin",         ["coumadin"],                    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O","C19H16O4"),
    ("theophylline",     [],                              "Cn1c2[nH]cnc2c(=O)n(C)c1=O",         "C7H8N4O2"),
    ("amphetamine",      [],                              "CC(N)Cc1ccccc1",                     "C9H13N"),
    ("ascorbic acid",    ["vitamin c"],                   "OC[C@@H](O)[C@H]1OC(=O)C(O)=C1O",    "C6H8O6"),
    ("penicillin g",     ["benzylpenicillin"],            "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O","C16H18N2O4S"),
    ("phenobarbital",    [],                              "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",     "C12H12N2O3"),
    ("lidocaine",        [],                              "CCN(CC)CC(=O)Nc1c(C)cccc1C",         "C14H22N2O"),
    ("salbutamol",       ["albuterol"],                   "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",      "C13H21NO3"),
    ("omeprazole",       ["prilosec"],                    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1","C17H19N3O3S"),
    ("fluoxetine",       ["prozac"],                      "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "C17H18F3NO"),
    ("diphenhydramine",  ["benadryl"],                    "CN(C)CCOC(c1ccccc1)c1ccccc1",        "C17H21NO"),
    ("methamphetamine",  [],                              "CN[C@@H](C)Cc1ccccc1",               "C10H15N"),
    ("histamine",        [],                              "NCCc1c[nH]cn1",                      "C5H9N3"),
    ("levodopa",         ["l-dopa"],                      "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",     "C9H11NO4"),
    ("epinephrine",      ["adrenaline"],                  "CNC[C@H](O)c1ccc(O)c(O)c1",          "C9H13NO3"),
    ("ketamine",         [],                              "CNC1(c2ccccc2Cl)CCCCC1=O",           "C13H16ClNO"),
    ("propofol",         [],                              "CC(C)c1cccc(C(C)C)c1O",              "C12H18O"),
    ("valproic acid",    ["valproate"],                   "CCCC(CCC)C(=O)O",                    "C8H16O2"),
    ("chlorpromazine",   ["thorazine"],                   "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",   "C17H19ClN2S"),
    ("atropine",         [],                              "CN1[C@H]2CC[C@@H]1C[C@@H](C2)OC(=O)C(CO)c1ccccc1","C17H23NO3"),
    ("gabapentin",       ["neurontin"],                   "NCC1(CC(=O)O)CCCCC1",                "C9H17NO2"),
    ("ranitidine",       ["zantac"],                      "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1","C13H22N4O3S"),
    ("acetazolamide",    [],                              "CC(=O)Nc1nnc(S(N)(=O)=O)s1",         "C4H6N4O3S2"),
    ("amoxicillin",      [],                              "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O","C16H19N3O5S"),
    ("ciprofloxacin",    ["cipro"],                       "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O","C17H18FN3O3"),
    ("ranolazine-skip",  [],                              "INVALID_TEST",                        "C0"),
]

PROMPTS = {
    # name condition: ceiling, does the model recognize the drug by name
    "name":     "Do you recognize the drug named '{q}'? If you do, reply with its name; if not, reply UNKNOWN.",
    "smiles":   "What is the common name of the drug with this SMILES? Reply with the drug name only, or UNKNOWN.\nSMILES: {q}",
    "inchikey": "What is the common name of the drug with this InChIKey? Reply with the drug name only, or UNKNOWN.\nInChIKey: {q}",
}


def validate(d):
    name, syns, smi, formula = d
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "unparseable SMILES"
    got = rdMolDescriptors.CalcMolFormula(m)
    if got != formula:
        return None, f"formula {got} != {formula}"
    return {
        "name": name, "synonyms": syns,
        "canonical_smiles": Chem.MolToSmiles(m),
        "inchikey": Chem.MolToInchiKey(m),
        "formula": got,
    }, "ok"


def recognized(resp, name, syns):
    """Deterministic: a known name/synonym present as a word, and not an UNKNOWN-only reply."""
    t = resp.lower()
    if re.search(r"\bunknown\b", t) and not any(re.search(rf"\b{re.escape(n)}", t) for n in [name] + syns):
        return False
    for n in [name] + syns:
        if re.search(rf"\b{re.escape(n.lower())}", t):
            return True
    return False


def main():
    drugs, dropped = [], []
    for d in DRUGS:
        rec, why = validate(d)
        (drugs.append(rec) if rec else dropped.append((d[0], why)))
    print(f"validated {len(drugs)} drugs; dropped {len(dropped)}: {dropped}\n")

    dry = os.environ.get("AXISA_DRY", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("AXISA_MODEL", "claude-sonnet-4-6")
    if dry:
        ex = drugs[0]
        for cond, tmpl in PROMPTS.items():
            q = {"name": ex["name"], "smiles": ex["canonical_smiles"], "inchikey": ex["inchikey"]}[cond]
            print(f"[{cond}] {tmpl.format(q=q)[:160]}")
        print("\nDRY: validated set + prompts only, no API.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    hits = defaultdict(int)
    per = []
    for dr in drugs:
        row = {"drug": dr["name"]}
        for cond, tmpl in PROMPTS.items():
            q = {"name": dr["name"], "smiles": dr["canonical_smiles"], "inchikey": dr["inchikey"]}[cond]
            msg = client.messages.create(model=model, max_tokens=40,
                messages=[{"role": "user", "content": tmpl.format(q=q)}])
            resp = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            ok = recognized(resp, dr["name"], dr["synonyms"])
            hits[cond] += ok
            row[cond] = int(ok)
        per.append(row)
        print(f"  {dr['name']:18} name={row['name']} smiles={row['smiles']} inchikey={row['inchikey']}")
    n = len(drugs)
    rates = {c: round(hits[c] / n, 3) for c in PROMPTS}
    out = {"n": n, "model": model, "recognition_rate": rates, "per_drug": per,
           "gap_name_minus_inchikey": round(rates["name"] - rates["inchikey"], 3),
           "gap_smiles_minus_inchikey": round(rates["smiles"] - rates["inchikey"], 3)}
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "axis_a_chem.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nrecognition rate (n={n}): name={rates['name']}  smiles={rates['smiles']}  inchikey={rates['inchikey']}")
    print(f"gap name-inchikey={out['gap_name_minus_inchikey']}  smiles-inchikey={out['gap_smiles_minus_inchikey']}")


if __name__ == "__main__":
    main()
