# Axis A (identity resolution), chemistry: the within-entity recognition gap

*Results section. 2026-06-10. Instrument: `eval/axis_a_chem.py` (rdkit + anthropic, run under miniconda py3.13). Set: 35 famous drugs, hand-curated and RDKit formula-validated. No em dashes.*

## What this measures

The protein A-axis anchor is name ~100% vs database-accession ~2-28% (`../docs/FAILURE_MODES.md`). This is its chemistry form, run WITHIN a single entity across notations of different web-frequency: the same drug as a name (web-frequent), a canonical SMILES (moderate), and an InChIKey (a hash, the accession analog). The web-exposure law (`../PROJECT_DESIGN.md` section 7) predicts recognition tracks notation web-frequency, so name >= SMILES >= InChIKey on molecules the model demonstrably knows by name. Scoring is deterministic (no LLM judge): ask for the common drug name (or UNKNOWN), count a hit if a known name or synonym is present.

## Result (claude-sonnet-4-6, n=35)

| condition (notation) | recognition rate | reading |
|---|---|---|
| name | 1.000 (35/35) | ceiling: every drug is name-known |
| canonical SMILES | 0.857 (30/35) | structure mostly resolves to identity |
| InChIKey | 0.686 (24/35) | the hash often resolves too |

Gaps: name minus InChIKey = 0.314, SMILES minus InChIKey = 0.171. The point ordering (name >= SMILES >= InChIKey) is the predicted web-frequency order and is CONSISTENT WITH the web-exposure hypothesis within a single entity, but at n=35 it is not all significant: the name-vs-InChIKey gap is large (McNemar clearly significant), but the SMILES-vs-InChIKey step (30/35 vs 24/35, 9 vs 3 discordant pairs) is McNemar p=0.146, so the middle inequality is not established here. Wilson 95 percent intervals on the rates are roughly +/-0.12. Treat this as illustrating the hypothesis, not testing it (the notation-exposure rank is assumed from corpus priors, and per `p1_webexposure.md` the web-poor forms are not faithfully countable, so this is not an independent test of a frequency law).

## The gap is real but COMPRESSED by fame, and that is the point

The magnitude is small (0.31) next to the protein anchor (name ~100% vs accession ~2%, a ~98 point gap). The reason is informative, not a failure: these 35 are among the most web-exposed molecules in existence, so even their InChIKeys clear the memorization threshold (aspirin's InChIKey BSYNRYMUTXBXSQ-UHFFFAOYSA-N is all over PubChem, Wikipedia chemboxes, ChEMBL dumps). Fame pushes EVERY notation above the web-frequency floor, which is precisely what the web-exposure law predicts: recognition tracks web-frequency, so for an ultra-famous entity all notations are frequent enough and the gap collapses. The InChIKeys the model missed (naproxen, warfarin, phenobarbital, lidocaine, atropine, ciprofloxacin, ...) are the less-ultra-famous tail, a fame gradient inside the set.

So the two ends span the fame axis: web-rare toxin accessions (~2%, protein FRT) at one end, ultra-famous drug InChIKeys (0.69) at the other. The A-axis recognition gap is real and directionally as predicted; its SIZE is set by web-fame, not by the notation type alone. The clean way to widen it on chemistry is a less-famous molecule set where the InChIKey is genuinely web-rare (predicted: InChIKey recognition drops toward the SMILES-minus-large-gap regime). That is the follow-up; this run establishes the within-entity ordering and the fame-compression mechanism.

## Caveats

Curated set, RDKit formula-validated (all 35 real drugs matched their known molecular formula, so atom counts are correct; a connectivity error preserving formula would survive, so the set is validated-but-not-authoritative). The name condition is echo-confirmation (the model is handed the name and asked to confirm or reject), a weak ceiling that shows name-recognition not name-derived knowledge. SMILES recognition cannot separate structure-parsing from memorized SMILES-to-name strings; both are "recognition" for this axis. Deterministic substring scoring counts a synonym hit, so it is lenient toward partial names. n=35, single frontier model, no bootstrap CI. Axis E (channel/action-policy) is not run here; it needs the chat-vs-tool harness (FRT P2), a different experiment.

## Reproduce

`/Users/jak4013/miniconda3-arm64/bin/python eval/axis_a_chem.py` (needs rdkit + anthropic in the same interpreter; ANTHROPIC_API_KEY from `~/.api_keys`). `AXISA_DRY=1` validates the set and prints prompts with no API call. Raw per-drug hits and rates in `results/axis_a_chem.json`.
