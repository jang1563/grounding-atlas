# Data-density rung: making the web-exposure law a measured within-modality regression

*Design doc, 2026-06-14. Turns the web-exposure law from a qualitative cross-modality gradient (which `results/p1_webexposure.md` showed is mis-specified) into a MEASURED regression inside one modality (single-cell / regulatory genomics), using a quantifiable external covariate: the volume of public functional-genomics data per cell state. Motivated by the "AlphaGenome-era data war" framing (public data concentrates on K562/ENCODE/GTEx; disease and developmental cell states are under-measured). No em dashes.*

## The question

P1 (the web-exposure law) is currently a within-entity qualitative claim, not a fitted regression, because a cross-modality text web-count proxy did not fit (ceilings span 0.70 to 0.96; `results/p1_webexposure.md`). The open question: is there a covariate that DOES fit, inside a single modality at a fixed ceiling?

The data-war literature supplies one. For regulatory and single-cell biology, "web exposure" of a representation-to-property mapping is not a text count, it is **how much public functional-genomics data exists for that cell state**. K562 has ENCODE, Roadmap, FANTOM, and genome-scale Perturb-seq; a patient-derived disease astrocyte or a developing fetal neuron has almost none. The prediction "verbalization tracks web-exposure" becomes testable and quantitative: does the model verbalize a cell state's regulatory program in proportion to the public data density of that state, at a fixed task and ceiling?

## The construct

Pick N cell states spanning data density, holding the TASK fixed (cell-state identification from an expression vector / cell sentence, the existing `single_cell` rung task, extended across states):

- data-rich: K562, HepG2, GM12878, PBMC T cell (the existing rung anchor)
- mid: common primary tissues from GTEx-style atlases
- data-poor: disease / developmental / stress states (reactive astrocyte, fetal cortical neuron, post-MI cardiac fibroblast, T2D beta cell), the states the data-war essays name as under-measured

**Covariate** D(c) for cell state c = log(count of public functional-genomics assays for c), summed over ENCODE biosamples + scPerturb / Perturb-seq datasets + MPRA/STARR-seq + GEO RNA-seq series. Countable from the ENCODE portal API, the scPerturb catalog (Peidli et al), and IGVF. This is an objective, pre-registerable number, decided before any model call.

**Fixed ceiling.** For each cell state, the ceiling is a supervised classifier (logistic regression or scANVI) trained on the SAME expression data for the task, so the ceiling reflects task difficulty, not the cell state's fame. Holding the ceiling band roughly constant across states is what makes the regression interpretable (the P1 fix: vary web-exposure at a fixed ceiling, not across ceilings).

## Arms

Per cell state: ceiling (supervised), activation (open-weight probe on Qwen3-8B hidden states), output (verbalized probability, 8B and frontier). The existing `single_cell` gene-name vs anonymized-id contrast is crossed in as a second axis.

## The two-covariate dissociation (the sharp control)

Two different "web exposures" must be separated, and this rung is built to do it:

- **name frequency** N(c): how often the cell state is NAMED in text ("K562" mentions). This is axis-A recognition.
- **data density** D(c): how much functional-genomics DATA exists for c. This is axis-B content.

The project's name-vs-content thesis predicts they dissociate: the model's recognition tracks N(c), but its content-grounding (does it know the cell's regulatory program) tracks D(c). A cell state can be famous-by-name yet data-poor, or data-rich yet rarely named. Crossing N(c) and D(c) as two regressors is the test that the law is about the functional MAPPING (content), not entity nameability.

## Prediction (falsifiable)

- output AUROC rises with D(c) at roughly fixed ceiling: a positive, measured slope = the web-exposure law as a regression, the thing P1 could not fit cross-modality.
- encoding (activation) stays high and roughly flat across D(c), consistent with the project's finding that encoding is surface-governed and near-universal; the verbalization gap is what D(c) moves.
- in the two-covariate fit, the content task loads on D(c), not N(c); a name-only model that is data-poor verbalizes at chance. If output tracks N(c) and not D(c), the "law" was recognition all along (a clean negative, equally publishable).
- the anonymized-id arm collapses regardless of D(c) (web-zero ids carry no mapping), bounding the effect to the named-but-data-graded regime.

## Why it matters

1. It resolves the documented P1 weakness: a measured within-modality regression at a fixed ceiling, with an objective covariate.
2. It gives the decision map a quantitative routing prior for genomics: the data-density map predicts, a priori, which cell states the model can verbalize and which need a specialist or new measurement. This is exactly the bridge to the data-war framing: grounding-atlas is the calibration map of where public-data models fail, i.e. where the expensive new Perturb-seq / MPRA data must be produced.
3. It connects to the causal sibling: the data-poor cell states are precisely the discovery regime where the model collapses and an intervention (Perturb-seq) is the only ground truth.

## Honest limits

- D(c) is correlated with cell-line age and popularity (K562), so D(c) and N(c) are partly collinear; the dissociation needs cell states chosen to break the correlation (famous-but-data-poor and data-rich-but-rarely-named anchors), which may be hard to populate.
- ceiling drift across very different cell states is real; treat the ceiling as a band and report the gap relative to it, not an absolute AUROC.
- pilot scale; the slope is the claim, not any single cell state's number.
- activation is open-weight-only (the frontier exposes no hidden states), so the encoding axis is an open-model property as elsewhere in the project.

## Finding (2026-06-14): NO-GO as a clean dissociation experiment

Two probes, reported honestly (a model-free precondition, then a one-model arm):

1. The precondition probe (`signal/single_cell/data_density.py`) appeared to show
   D(ENCODE) and N(PubMed) dissociating (Spearman log D vs log N = -0.23). But that
   dissociation was an ARTIFACT of mixing entity types: immortalized lines have lots
   of ENCODE data but are named less, biological cell types the reverse, so the
   apparent independence lived entirely in the line-vs-primary contrast.

2. The grounding arm (`signal/single_cell/data_density_arm.py`; opus, 12 BIOLOGICAL
   cell types; D = GEO single-cell dataset count) exposes the problem:
   - among commensurable biological cell types, D(GEO) and N(PubMed) are COLLINEAR
     (Spearman +0.92): both just track how studied a cell type is, so the rung's core
     D-vs-N dissociation is impossible there;
   - the marker-grounding task SATURATES at the frontier (recall ~1.0 for 11/12,
     including rare Paneth / Tuft / Kupffer cells), so it resolves no gradient.

Conclusion: the data-density rung does not yield a clean dissociation experiment.
Where D and N separate (cell lines) the entities are incommensurable for a biological
grounding task; where the entities are commensurable (biological cell types) D and N
are collinear and the marker task saturates. The missing ingredient is a dissociating
AND commensurable entity set (a well-named but functionally data-poor biological cell
type), which may not exist cleanly. This mirrors the project's own P1 result (the
web-exposure covariate is confounded; only the within-entity contrast is valid). The
China-bio data-war connection stays a strong framing (grounding-atlas as the map of
where public-data models fail), but not a runnable rung as designed.

A cell-STATE salvage (resting vs disease/activated states; `DD_SET=states`) was tried
and confirms the NO-GO. States de-collinearize D and N somewhat (Spearman log N, log D
+0.61 vs +0.92 for types), but the marker-recall task still saturates at the frontier
(recall ~1.0 for 10/11 states, including disease-associated microglia, exhausted T,
M1/M2, CAF, senescence): cell-identity markers are memorized text the frontier holds in
full, independent of D or N. Marker-grounding is recognition (memorized, uniform), not
the D-gated content signal the rung needed; a non-memorized functional-grounding task
would be a different, heavier construct.

## Implementation entry points

Reuse `signal/single_cell/` (the cell-sentence vs anon generator) extended to N cell states; `eval/activation_arm_sc_cayuga.sh` for the probe; an output arm mirroring `eval/frontier_output_panel.py`. New: a `signal/single_cell/data_density.py` that pulls D(c) and N(c) from the public portals and writes a per-cell-state covariate table. First executable step: build the covariate table for ~10 cell states and confirm D(c) and N(c) are separable before any model run.
