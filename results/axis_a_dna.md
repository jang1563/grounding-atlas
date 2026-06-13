# Axis A (identity resolution), gene/DNA: the within-entity recognition gap

*Results section. 2026-06-10. Instrument: `eval/axis_a_dna.py`. Data: 40 most-studied ClinVar genes (`variant_grounding/data/variant_clinvar_full.csv`), trusted, no curation. claude-sonnet-4-6. No em dashes.*

## What this measures

The third modality for axis A on one instrument (protein via FRT, chemistry via `axis_a_chem.md`, gene here). Each popular gene is presented as four notations of decreasing web-frequency, and the model is asked to resolve the gene identity: symbol (web-frequent), UniProt accession (web-rare, the FRT accession analog), dbSNP rsID (web-rare), and a 60-residue protein sequence window (web-rare). The web-exposure hypothesis predicts recognition tracks notation web-frequency. Deterministic scoring: recognized if the correct gene symbol is present. Genes are the 40 with the most ClinVar variants (most-studied = web-frequent symbols), so the symbol ceiling is high and a low accession rate is a notation gap, not ignorance.

## Result (n=40)

| condition (notation) | recognition rate | reading |
|---|---|---|
| gene symbol | 1.000 (40/40) | ceiling: every gene known by symbol |
| UniProt accession | 0.600 (24/40) | famous-gene accessions partly memorized |
| dbSNP rsID | 0.025 (1/40) | the rsID hash does not resolve |
| protein sequence window | 0.025 (1/40) | a 60-residue window does not identify the gene |

Gaps: symbol minus sequence = 0.975, symbol minus rsID = 0.975, symbol minus accession = 0.400.

## This is the full FRT recognition gap, and it confirms the chem prediction

The web-rare ends (rsID, sequence) are the SAME ORDER OF MAGNITUDE as the protein FRT anchor (name ~100% vs database-accession ~2%), now on a broad 40-gene panel. But each is a SINGLE successful gene (rsID: BRCA2; sequence: TP53), so 0.025 = 1/40 with a Wilson 95 percent interval of roughly 0.4 to 13 percent; read it as near-floor, not a precise rate, and do not over-read "0.975 gap" to three digits. The UniProt accession sits intermediate (0.60), plausibly because famous-gene accessions are in training text (BRCA1 = P38398, TP53 = P04637, FBN1 = P35555), the same memorization that lifted famous-drug InChIKeys.

This is CONSISTENT WITH the prediction the chem rung could not test. `axis_a_chem.md` found InChIKey recognition compressed to 0.69 because the 35 drugs were so ultra-famous that even their InChIKeys cleared the memorization threshold, and predicted the gap would widen on web-rarer identifiers. Here the rsID and the raw sequence are genuinely web-rare even for popular genes, and the gap is wide (1/40 at each). So across the two modalities the SIZE of the recognition gap tracks the specific identifier's web-rarity (chem InChIKey 0.69 compressed, gene rsID/sequence near-floor), which is consistent with the web-exposure HYPOTHESIS as a statement about notations. It is not an independent test: the notation-exposure rank is assumed from corpus priors (the web-poor forms are not faithfully countable, `p1_webexposure.md`), so this illustrates the hypothesis, not confirms a law.

## Axis A across three modalities, one instrument

| modality | web-frequent | web-rare | rare-notation recognition |
|---|---|---|---|
| protein (FRT) | name | accession | ~0.02 to 0.28 |
| chemistry | name 1.00 | InChIKey | 0.69 (fame-compressed) |
| gene | symbol 1.00 | rsID / sequence | 0.025 |

The instrument now spans protein, chemistry, and gene, and the recognition gap is real in all three with a magnitude set by how web-rare the rare notation actually is. This is the capability framing of the FRT safety finding (`../docs/FAILURE_MODES.md` axis A) generalized cross-modality.

## Caveats

The symbol condition is engagement-scored (recognized if not UNKNOWN) which is a DIFFERENT, easier rule than the strict correct-symbol-substring used for accession/rsID/sequence, so the 1.000 ceiling is "said something," not "named the right gene"; do not compare the 1.000 to the other rates as if same-scored. The substring matcher is lenient (a hedge "could be naproxen or..." counts). Popular genes only, so the symbol ceiling is high by construction. The UniProt 0.60 is confounded with fame, not cleanly a frequency effect: the hits are the famous/collagen genes (BRCA1/2, TP53, all seven COL genes), the misses are obscure large genes (KMT2C, NSD1, NEB), and because the popular-gene selection holds fame roughly constant, only notation TYPE varies, so 0.60 is "famous-accession memorization," indistinguishable here from a web-frequency law. A fake-accession control closes the blind-guessing confound (`../eval/axis_a_dna_fakeacc.py`, `axis_a_dna_fakeacc.json`): feeding 40 scrambled, non-existent accessions yields a confabulation rate of ~0.025 (1 of 40 names one of the 40 real symbols; the rest say UNKNOWN, "not a human protein", or an out-of-set gene). That is far below the 0.60 real-accession recognition, so the 0.60 is genuine recognition of real accession-to-gene mappings for famous genes, not high-prior guessing of BRCA1/TP53. (The scramble uses an unseeded hash so the exact fake strings vary run to run, but the near-zero confabulation is robust.) It remains "memorization of famous accessions" (which IS recognition), so it still does not by itself test a frequency LAW (fame held constant); but it is not a scoring artifact. The sequence is the ClinVar 60-residue wt_window (a protein window), not a full gene. n=40, single frontier model, no bootstrap CI.

## Reproduce

`python3 eval/axis_a_dna.py` (anthropic only, no rdkit; ANTHROPIC_API_KEY from `~/.api_keys`; `AXISA_DRY=1` validates data and prints prompts). Per-gene hits and rates in `results/axis_a_dna.json`.
