# Generality panel: the web-exposure law across seven science domains

*Results. 2026-06-12. `eval/frontier_output_panel.py` (GEN_CONFIGS), data under `signal/generality/` and `signal/materials/`. Frontier output (opus-4.8 / sonnet-4.6 / haiku-4.5); activation is open-weight-only. Tests whether the web-exposure law is biology-specific or a general property of LLM grounding, with the same web-rich-vs-anonymized design as the single-cell rung. No em dashes.*

## The design (one per domain)

Each domain has web-known BUILDING-BLOCK tokens (mineral names, element symbols, drug names, amino-acid names, metabolite names) and a verifiable binary property. The entity is rendered two ways with the SAME information: web-rich (the real name/formula) vs web-zero (the same entity with its name/symbols replaced by fixed anonymized IDs). A non-LLM probe on the underlying features is the ceiling; only token web-exposure varies.

## Result (frontier output AUROC, opus-4.8)

| domain | property | web-rich | anonymized | clean law? |
|---|---|---|---|---|
| minerals (geology) | Mohs hardness >= 5 | 0.999 | 0.609 | YES |
| elements (chemistry) | is a metal | 1.000 | 0.500 | YES |
| amino acids (biochem) | hydrophobic | 0.988 | 0.556 | YES |
| metabolites (biochem) | is a lipid | 1.000 | 0.505 | YES |
| drugs (pharmacology) | acts on the CNS | 1.000 | 0.573 | YES |
| materials (composition) | is a metal | 0.837 | 0.536 | YES |
| materials (composition) | glass-forming ability | 0.652 | 0.603 | NO (exception) |

Six of seven domains show the law cleanly: from the web-rich name the model grounds the property at 0.84 to 1.00 (often at ceiling), and from anonymized tokens carrying the identical information it is at chance. The swing reproduces across geology, chemistry, biochemistry, pharmacology, and materials. So the web-exposure law is NOT biology-specific; it is a general property of LLM grounding.

## The glass exception refines the law: TWO factors

Glass-forming ability is the one domain where the web-rich form does NOT ground (formula 0.652, barely above the anonymized 0.603). The reason is informative: unlike metallicity or hardness, whether an alloy forms a metallic glass is a NICHE property whose composition-to-outcome mapping is sparsely described in text. So the law is two-factor: a model grounds a representation-to-property mapping only when BOTH the surface tokens are web-known AND the property's associations with those tokens are web-documented. Web-known tokens alone (real formulas) are not enough if the property itself is web-poor. This matches the biology rungs (a gene symbol grounds cell type because CD3-marks-T-cells is web-documented; it would not ground an obscure unpublished property).

## Lessons

- Anonymization must cover the FULL entity. A first build anonymized only the last token of multi-word names ("palmitic acid" -> "palmitic acid_NNNN"), leaking "palmitic" and inflating metabolite-anon to 0.806; fixing it (anonymize the whole entity) dropped it to chance 0.505. Always verify the anonymized form carries no descriptive substring.
- The clean web-rich-vs-web-zero contrast requires WEB-KNOWN BUILDING-BLOCK tokens. Domains whose raw representation is continuous measurements with no such vocabulary (mass spectra, 3D coordinates, methylation beta values) have only the web-zero form, which is why they sit at scale-invariant chance with no web-rich counterpart to lift.

## Scope

Seven domains, binary properties, frontier output only (activation open-weight-only, not run for the generality probes). A generality CONTROL for the project's central web-exposure law, establishing it as a general property of LLM grounding (two-factor: token web-exposure AND property web-documentation), not a biology artifact. Small n for several domains (minerals 54, elements 50, amino acids 20, metabolites 32, drugs 44); the swings are large enough to clear the wide CIs, but treat the exact numbers as a pilot.

## Reproduce

`PANEL_RUNGS=minerals_name,minerals_anon,elements_name,elements_anon,glass_formula,glass_anon,aminoacid_name,aminoacid_anon,drugclass_name,drugclass_anon,metabolite_name,metabolite_anon PANEL_N=400 python eval/frontier_output_panel.py`. Data built in `signal/generality/`. Raw in `results/frontier_output_panel.json`.
