# Single-cell rung design: the descriptive grounding rung with a built-in web-exposure contrast

*Design doc. 2026-06-12. The 7th modality rung, the flagship modality. Single-cell DESCRIPTIVE grounding (cell-type from an expression vector); the CAUSAL/perturbation question is the sibling CausalAtlas project, OUT OF SCOPE. No em dashes.*

## Why this rung

The deep-research sweep (`docs/MODALITY_LANDSCAPE.md`) ranked single-cell #1: it is a flagship domain AND it has a built-in within-entity web-exposure contrast, exactly the variant text-vs-seq structure that is the project's cleanest test. The same cell has:
- a WEB-ZERO raw form: the numeric expression vector (the model cannot read 20k floats), and
- a WEB-RICH symbolic form: the rank-ordered gene SYMBOLS (Cell2Sentence "cell sentence"), which the model can read because gene names are common in pretraining text.
Cell2Sentence (and C2S-Scale reaching ~95% cell-type accuracy, near the scGPT/Geneformer ceiling) proves the symbolic form is groundable; the raw numeric form should not be. So the rung extends the web-exposure law into omics.

## The task

Binary cell-type classification from a single cell: a recognizable major cell type (e.g. T cell, marked by CD3D/CD3E/CD8A) vs rest, balanced. Binary keeps it comparable to the other rungs (AUROC). Data: a standard annotated PBMC dataset (pbmc68k_reduced with explicit bulk_labels cell types, or pbmc3k with the tutorial mapping); gene symbols as var_names.

## The CEILING caveat (from the sweep, the one 3-vote-verified finding)

Do NOT anchor the ceiling to a zero-shot single-cell FM: scGPT and Geneformer zero-shot are BEATEN by simple baselines (HVG, scVI, Harmony) on cell-type clustering (Genome Biology 2025). The cell-type-from-expression property IS reliably decodable, so the CEILING here is a STRONG SUPERVISED specialist: a logistic-regression / linear classifier on the (HVG-reduced) expression vector, cross-validated. That is the honest "a specialist reads the property" ceiling, expected high (~0.95) for a major cell type.

## Three representations (the web-exposure contrast)

For each cell, take the top-N genes by expression (N ~ 100), and render three ways:
1. **anonymized numeric (web-zero):** the expression VALUES with ANONYMIZED gene labels ("feature_1: 4.2, feature_2: 3.1, ..."). The model has only numbers, no gene identity. Pure web-zero.
2. **named numeric:** gene SYMBOLS + values ("CD3D: 4.2, MS4A1: 0.0, ..."). Names + numbers.
3. **cell sentence (web-rich, C2S):** the top-N gene SYMBOLS only, rank-ordered by expression, no values ("MALAT1 CD3D CD3E IL32 ..."). The proven C2S form.

The load-bearing contrast is anonymized (1) vs cell-sentence (3): the lift is the value of the gene SYMBOLS (web exposure), parallel to variant raw-seq vs gene+HGVS and protein seq vs seq+organism.

## Arms

- **ceiling:** supervised LogReg on the full/HVG expression vector, scaffold-free StratifiedKFold (cells are iid here; optionally hold out by donor if available). Expected ~0.95.
- **LLM-output (8B + frontier panel opus/sonnet/haiku):** each representation -> "probability this cell is a T cell," bare-number protocol. Prediction: anonymized ~chance; cell-sentence rising with scale (C2S evidence).
- **LLM-activation (Qwen3-8B, GPU):** linear probe on the hidden states for each representation. Prediction: anonymized low (the model cannot encode anonymized expression numbers); cell-sentence higher (reads marker names).

## Predicted regime and what it would establish

- anonymized numeric: encoding-limited / expression-limited at the web-zero extreme (like the raw-sequence forms).
- cell sentence: scale-closable expression gap (the model encodes and, at scale or with C2S fine-tuning, verbalizes via gene-name markers).
The within-modality contrast (anonymized vs cell-sentence) is the clean web-exposure test at a fixed ceiling, the project's strongest design, now in the flagship modality and the plan's stated SFM-as-input area.

## Build order (disciplined, gating first)

1. Load the dataset, define the balanced binary cell-type task, gene symbols. [data]
2. GATING: supervised LogReg ceiling on expression -> cell type (confirm ~0.9+, a strong specialist exists). [if it clears, proceed]
3. Build the three representation texts per cell; save a csv (rep, label) per condition.
4. LLM-output panel (8B + frontier) on anonymized vs cell-sentence.
5. LLM-activation arm (Qwen3-8B, GPU) on anonymized vs cell-sentence.
6. Write results + add the rung to PROJECT_DESIGN 7.2.

Implementation: `eval/single_cell_rung.py` (data + gating + representation builder), reuse `frontier_output_panel.py` (add sc rungs) and `activation_arm_dna.py` pattern (text-input 3-arm) for the LLM arms.
