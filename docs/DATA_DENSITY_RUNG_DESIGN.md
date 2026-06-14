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

## Probe finding (2026-06-14): GO

The model-free precondition holds (`signal/single_cell/data_density.py`). Across 13
cell states, D(c) = ENCODE Experiment count and N(c) = PubMed count **dissociate**:
Spearman(log D, log N) = -0.23 (n.s.). Immortalized lines are functionally data-rich
but named less (K562 D=2571), while primary and disease states are heavily named but
ENCODE-sparse (microglia N=64k D=0; dopaminergic neuron N=34k D=0; pancreatic beta
cell N=43k D=22; reactive astrocyte N=12k D=45; regulatory T cell N=66k D=9). The
high-N / low-D anchors the rung needs (famous-by-name, functionally under-measured)
exist abundantly, so an LLM's grounding can be regressed on D vs N to test which
drives it.

Caveats for the full rung (do not affect the go/no-go): D = ENCODE-only is a narrow
regulatory-data proxy, so broaden it with scPerturb / GEO counts and fix the term
mismatch where D=0 reflects an absent ENCODE term (e.g. microglia) rather than no
data; N = PubMed conflates biological attention with naming; n=13 is a probe.

## Implementation entry points

Reuse `signal/single_cell/` (the cell-sentence vs anon generator) extended to N cell states; `eval/activation_arm_sc_cayuga.sh` for the probe; an output arm mirroring `eval/frontier_output_panel.py`. New: a `signal/single_cell/data_density.py` that pulls D(c) and N(c) from the public portals and writes a per-cell-state covariate table. First executable step: build the covariate table for ~10 cell states and confirm D(c) and N(c) are separable before any model run.
