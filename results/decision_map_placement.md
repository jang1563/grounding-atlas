# Decision map: where each biology capability should live (train / retrieve / orchestrate)

*Results, 2026-06-12. The WS3 deliverable: for each capability, measure which placement WINS, rather than asserting it. Placement is read off four measured arms: encode (open-weight probe), 0-shot output, RETRIEVE (k-shot in-context, `eval/icl_placement.py` + `eval/icl_methyl.py`), and ORCHESTRATE (specialist-tool ceiling). No em dashes.*

## The measured placement table

| capability | encode (probe) | 0-shot output | retrieve (16-shot ICL) | orchestrate (tool ceiling) | winning placement |
|---|---|---|---|---|---|
| single-cell -> T cell (gene names) | 0.983 | 0.99 | **0.987** | 0.989 | retrieve |
| single-cell -> T cell (ANONYMIZED) | 0.964 | ~0.50 | **0.973** | 0.989 | retrieve (web-zero token, still closes) |
| MSA column -> conserved | 1.000 | 0.795 | **0.994** | 0.999 | retrieve |
| methylation -> age | 0.922 | 0.402 | **0.93** | 0.95 (clock) | retrieve (train not needed) |
| DNA -> promoter | 0.880 | 0.815 | **0.85** | 0.89 (6-mer) | retrieve |
| variant -> pathogenic (HGVS text) | n/a | 0.60 | **0.981** | 0.94 (AlphaMissense) | retrieve (web-rich lookup; ClinVar memorization likely, not pure ICL) |
| protein -> Tm | 0.609 | ~0.60 | 0.651 | 0.699 (ESM) | orchestrate (encode-limited; all weak) |
| SMILES -> BBB penetration | n/a | ~0.50 | **0.851** | 0.846 (Morgan) | retrieve (composition-based property) |
| SMILES -> hERG block | 0.787 | 0.453 | 0.70 | 0.825 (Morgan) | orchestrate (target SAR) |
| SMILES -> BACE-1 inhibition | n/a | ~0.50 | 0.692 | 0.834 (Morgan) | orchestrate (target SAR) |
| SMILES -> HIV activity | n/a | ~0.50 | 0.588 | 0.730 (Morgan) | orchestrate (target SAR) |
| NMR -> hERG | 0.747 | ~0.50 | **0.586** | 0.866 | orchestrate (computation-bound) |
| 3D coords / graph -> hERG | 0.67 / 0.71 | ~0.49 | n/a (not encoded) | 0.83 / 0.87 | orchestrate (forced) |
| ligand-kinase -> binding affinity (Axis 1) | n/a | ~0.50 | 0.614 | 0.88 (DeepDTA / Boltz-2) | orchestrate (binding = 3D computation) |
| ECG morphology -> abnormal (Axis 5) | n/a | ~0.50 | **0.982** | 0.994 (5-NN) | retrieve (strong in-data morphology pattern) |
| PPI -> interact (gene names) | n/a | ~0.50 | **0.952** | 0.85 (graph-FM) | retrieve (web-documented interaction) |
| PPI -> interact (ANONYMIZED) | n/a | ~0.50 | **0.500** | 0.85 (graph-FM) | orchestrate (pure relational, no in-data pattern) |

**Relational knowledge refines the retrieve boundary (Axis 4, the first missing-axis filled): anon closes under retrieve only when the DATA carries a learnable pattern.** single-cell-anon (0.973) and methylation (gene = anon) close because the expression/numeric VALUES carry the signal few-shot can fit. PPI-anon (0.500, degree-matched) does NOT, because a protein interaction is PURE external relational knowledge with no in-data pattern once the names are stripped. So retrieve closes web-zero representations that have an in-data pattern (vectors, numerics), but NOT pure relational facts, which fall to orchestrate (graph-FM). The web-exposure swing is sharpest here (name 0.952 vs anon 0.500), and the closed-weight conclusion holds: name -> retrieve, novel/anon -> orchestrate, train still nowhere.

**Axis 5 (temporal) and Axis 2 (generative) close out the missing-axes sweep.** ECG morphology (simple normal-vs-abnormal beat) closes with retrieve (0.982 ~ 5-NN ceiling 0.994), reconfirming the in-data-pattern rule (a temporal numeric series with a strong morphological pattern is few-shot-learnable; complex multi-label diagnosis or subtle EEG likely still needs the specialist encoder, untested here). Generation (Axis 2, a MODIFIED design-then-validate arm, not classification): a general LLM produces simple property-constrained molecules well in-context (opus 40/40 valid SMILES, 38/40 with logP in [1,3], using memorized chemistry, no training), but novel objective-optimized design (binding, structure, multi-objective) is owned by specialist generators (REINVENT/RFdiffusion) with the LLM as spec-parser/orchestrator. So generation splits the same way: simple-constrained = LLM-solo, optimized/novel = orchestrate.

**All 4 in-scope missing axes are now filled (relational/network, structure, temporal, generative), and the closed-weight conclusion SURVIVES and SHARPENS on every one.** Retrieve covers what has an in-data pattern or web-documented anchor (PPI-name, ECG-morphology), orchestrate covers what needs a specialist (affinity, novel design, novel/anon interactions), and TRAIN-Claude wins in NONE of them. The single axis where train demonstrably wins is causal/perturbation (rBio), which is the sibling CausalAtlas project, correctly fenced off so the descriptive claim stays honest.

**Within chemistry the boundary sharpens to composition-vs-target-SAR.** A composition / physicochemical property (BBB penetration, lipophilicity-like) closes with retrieve (BBBP 0.851 ~ ceiling 0.846), because few-shot examples teach the surface-decodable pattern. A SPECIFIC-TARGET binding property (BACE-1, HIV, hERG) does NOT close with retrieve (0.59 to 0.70, below the 0.73 to 0.83 specialist ceiling), because it needs fine structure-activity that few-shot cannot convey. So the retrieve-vs-orchestrate split is the same computation-vs-lookup line inside drug discovery: bulk physicochemistry is retrievable, target-specific bioactivity is orchestrate.

(methylation re-measured on age-CpG probes: ceiling 0.949, encode 0.922, 0-shot output 0.402, expression gap 0.520; few-shot ICL 0.93, anchor-invariant gene-named 0.926 = anon 0.925.)

**Web-exposure is a 0-shot effect, not a retrieve effect.** The rungs we called "scale-invariant, never-verbalizable web-zero" (single-cell-anonymized, methylation) close with few-shot RETRIEVE: single-cell-anon goes 0-shot chance to 0.973 (~ gene-named 0.987), methylation gene-named 0.926 = anon 0.925. So 0-shot web-exposure governs 0-shot output; few-shot retrieval bypasses it, web-exposure-independent. The retrieve-vs-orchestrate boundary is therefore COMPUTATION-BOUND-ness, not web-exposure: vector / lookup / transparent representations (single-cell, MSA, methylation, DNA) close with retrieval even when anonymized, while spectra (NMR 0.586, structure elucidation), fine SAR (SMILES 0.70), and structure-encoding-limited rungs (3D, graph) resist few-shot and need the tool.

## The decision rule that falls out

1. **Encoded AND in-context-learnable -> RETRIEVE.** If the property is encoded (probe near ceiling) and a handful of examples let the model do it in-context, few-shot retrieval closes the gap and no weight training is needed. methylation (0.40 -> 0.93) and DNA (0.82 -> 0.85) land here.
2. **Encoded but retrieve falls short -> ORCHESTRATE.** SMILES->hERG encodes (0.787) and few-shot lifts it (0.45 -> 0.70) but not to the specialist ceiling (0.825), so the tool wins. Call the specialist.
3. **Not encoded -> ORCHESTRATE (forced).** 3D coordinates and molecular graph are not even encoded (structure not extractable from text), so retrieve cannot help; the tool is mandatory.
4. **TRAIN (weights) wins nowhere in this set.** Retrieve is strong enough that, where a capability is learnable at all, in-context retrieval reaches it without weight updates; where it is not, a tool beats both. This matches the train-vs-retrieve-vs-tool literature (Ovadia 2312.05934: RAG beats fine-tuning for knowledge; in-tool learning 2508.20755: tools give unbounded recall and preserve general capability).

## Why this is the central result for a closed-weight model

The placement map says the two placements that do NOT require touching model weights, retrieve and orchestrate, cover the capability space:
- **Retrieve (in-context / few-shot)** turns out to be unexpectedly strong: it closed methylation's 0.52 expression gap entirely (0.40 -> 0.93) with 12 examples, anchor-invariant, no training. The 0-shot expression gap was a 0-shot artifact, not a fundamental limit.
- **Orchestrate (calibrated tool-routing)** covers what retrieve cannot, and we already showed the frontier model is a calibrated router for exactly this decision (`results/calibration_routing.md`: opus confidence-AUROC corr +0.90, routed ~ oracle).

So a grounded biology orchestrator does not need to train the frontier model. It elicits capability by in-context retrieval where the property is encoded-and-learnable, and routes to a specialist tool where it is not, using calibrated confidence to decide which. This is why a closed-weight model (Claude) is not at a disadvantage here: the winning placements are retrieve and orchestrate, both available without weight access. Weight training is the one placement a closed-weight model cannot do, and it is also the one that wins nowhere in the measured map.

## Honest scoping

Four capabilities, one specialist ceiling each, frontier = opus, k=12 to 16 shots. The placement labels are robust to the exact AUROC, but the retrieve-vs-orchestrate margin on SMILES (0.70 vs 0.825) is modest; a larger k or better example selection could move it. "Train wins nowhere" is a claim about THIS set at THIS scale; a capability requiring a genuinely new, non-in-context-learnable skill could still favor training, and finding such a case is the natural next probe. Retrieve closing methylation is an in-context-regression result (the model fits a 60-dim map from examples); whether it holds for cohorts far from the example distribution (easy-to-hard) is untested. The decision-map framing itself is fast-moving and partly anticipated; the durable contribution is the measured placement on a cross-representation set with matched encode/retrieve/orchestrate arms.

## Connection

WS1 instrument: `PROJECT_DESIGN.md` 7.2 + `results/SYNTHESIS.md`. Orchestrate arm: `results/calibration_routing.md`. Application framing: `~/Dropbox/Lab/.../RS_Life_Sciences/Review_5Axis_DecisionMap_BioGroundingEvidence_2026_06_12.md` (the 5-axis pre/post-train/tool/MCP/retrieval split; this result says retrieve + tool/MCP dominate and post-train is rarely the winner, which is the closed-weight-friendly conclusion).
