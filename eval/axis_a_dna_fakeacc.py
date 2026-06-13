"""Fake-accession control for axis A (gene recognition): is the UniProt 0.60 genuine?

axis_a_dna.py found UniProt-accession to gene recognition at 0.60 for 40 popular genes.
Objection: maybe the model just GUESSES famous genes (BRCA1/TP53) for any accession-looking
string, so 0.60 is high-prior guessing, not real recognition. This control feeds FAKE,
non-existent accessions (the real accession's alphanumerics scrambled) and measures how
often the model confabulates one of the 40 real gene symbols. If recognition is real, a
fake accession should yield ~0 confabulation (UNKNOWN, "not a protein", or an out-of-set
gene); if it is guessing, fakes confabulate famous genes at ~the real rate.

Result (claude-sonnet-4-6, n=40): 0/40 confabulation = the 0.60 is genuine recognition of
real accession-to-gene mappings, not blind famous-gene guessing. (It remains "memorization
of famous accessions," which IS recognition; this does not by itself test a frequency LAW.)

Run: python3 eval/axis_a_dna_fakeacc.py   Env: AXISA_MODEL. No em dashes.
"""
import json
import os
import random
import re

from axis_a_dna import load_popular  # same 40 popular ClinVar genes + UniProt accessions


def fake(acc):
    head, tail = acc[0], list(acc[1:])
    random.Random(hash(acc) % 9999).shuffle(tail)
    f = head + "".join(tail)
    return f if f != acc else acc[::-1]


def main():
    model = os.environ.get("AXISA_MODEL", "claude-sonnet-4-6")
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    genes = load_popular(40)
    syms = [g["gene"] for g in genes]
    confab, unknown, per = 0, 0, []
    for g in genes:
        facc = fake(g["uniprot"])
        m = client.messages.create(model=model, max_tokens=40, messages=[{"role": "user",
            "content": f"Which human gene corresponds to UniProt accession {facc}? "
                       "Reply with the gene symbol only, or UNKNOWN."}])
        resp = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
        c = any(re.search(rf"\b{re.escape(s.lower())}\b", resp.lower()) for s in syms)
        u = re.search(r"\bunknown\b", resp.lower()) is not None
        confab += c; unknown += u
        per.append({"gene": g["gene"], "real_acc": g["uniprot"], "fake_acc": facc,
                    "resp": resp.strip()[:60], "confabulated_real_gene": int(c)})
        print(f"  {g['gene']:10} fake={facc:8} -> {resp.strip()[:34]!r} confab={int(c)}")
    out = {"control": "fake_accession", "model": model, "n": len(genes),
           "confabulation_rate": round(confab / len(genes), 3),
           "unknown_rate": round(unknown / len(genes), 3),
           "real_uniprot_recognition_ref": 0.60, "per_gene": per}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "results", "axis_a_dna_fakeacc.json"), "w"), indent=2)
    print(f"\nFAKE-accession confabulation rate = {out['confabulation_rate']} "
          f"(real-accession recognition was 0.60; << means genuine recognition, not guessing)")


if __name__ == "__main__":
    main()
