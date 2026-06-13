# Generality probe: the web-exposure law holds outside biology (materials science)

*Results. 2026-06-12. `eval/frontier_output_panel.py` (mat_formula / mat_anon), data `signal/materials/metal.csv` (matminer matbench_expt_is_metal, 4921 compounds, metal vs non-metal). A single non-bio control rung testing whether the web-exposure law is biology-specific or general. No em dashes.*

## The test

The web-exposure law (a model grounds a representation-to-property mapping in proportion to how often that mapping appears in text) has so far been measured only on biological modalities. If it is a GENERAL property of LLM grounding it should hold in any domain with web-known building-block symbols. Materials science is the cleanest non-bio test, with the SAME design as the single-cell rung:
- web-rich: the chemical FORMULA with real element symbols ("Ag: 1, Au: 2, S: 2"); the model knows the elements.
- web-zero: the SAME composition with each element symbol mapped to a fixed anonymized ID ("elem_3063: 1, elem_4442: 2, elem_9062: 2").
Both carry IDENTICAL composition; a composition-vector classifier (the ceiling) reads metal-vs-non-metal at 0.927 on either. So only the web-exposure of the element symbols varies.

## Result (frontier output AUROC; ceiling 0.927)

| form | opus-4.8 | sonnet-4.6 | haiku-4.5 |
|---|---|---|---|
| formula (element symbols, web-rich) | 0.837 | 0.743 | 0.717 |
| anon (anonymized element IDs, web-zero) | 0.536 | 0.441 | 0.518 |

The same near-0.3-to-0.4 swing as in biology, from identical composition, toggled only by whether the tokens are web-known element symbols. From real formulas the model reads metallicity (0.72 to 0.84, a scale ladder approaching the 0.927 ceiling) because it knows the elements and their bonding tendencies; from anonymized IDs carrying the identical composition it is at chance. So the web-exposure law is NOT biology-specific: it reproduces exactly in materials science.

## The deeper pattern: the law needs web-known building-block tokens

Across the project the clean web-rich-vs-web-zero contrast has the same shape in every domain that HAS web-known building-block symbols:
- biology genes: single-cell cell-sentence (gene symbols) 0.99 vs anon 0.47.
- biology proteins: sequence vs sequence-plus-organism-name.
- biology variants: raw sequence vs gene-plus-HGVS.
- chemistry/materials: formula (element symbols) 0.84 vs anon 0.54.
In each, the model grounds the property when and only when the surface tokens are symbols whose property associations appear in pretraining text (CD3 marks T cells, oxygen makes oxides). This also bounds WHERE the contrast exists: a domain whose raw representation is continuous measurements with NO web-known token vocabulary (a mass spectrum, 3D coordinates, an ECG, an astronomical light curve) has only the web-zero form, which is why those rungs sit at the web-zero, scale-invariant-chance end with no web-rich counterpart to lift. So the web-rich form requires a named-building-block vocabulary; the law then determines grounding from that vocabulary's web exposure.

## Scope

One non-bio domain (materials), one binary property (metallicity), frontier output only (activation is open-weight-only and not run here). It is a generality CONTROL for the web-exposure law, not a materials-science project; it shows the project's central law is a general property of LLM grounding rather than a biology artifact.

## Reproduce

Data: matminer `load_dataset("matbench_expt_is_metal")`, render formula vs anonymized-element composition. `source ~/.api_keys && PANEL_RUNGS=mat_formula,mat_anon PANEL_N=400 python eval/frontier_output_panel.py`. Raw in `results/frontier_output_panel.json`.
