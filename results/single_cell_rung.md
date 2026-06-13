# Single-cell rung: the cleanest web-exposure result in the project

*Results. 2026-06-12. `eval/frontier_output_panel.py` (sc_cellsentence / sc_anon), data `signal/single_cell/pbmc_Tcell.csv` (scanpy pbmc68k_reduced, 384 cells balanced, T cell vs rest). The 7th rung, JK's flagship modality. 8B activation pending (job 3038603). No em dashes.*

## The design: the same cell, two token-identities, one fixed signal

Each PBMC cell is rendered two ways from its top-50 expressed genes:
- cell_sentence (WEB-RICH): the gene SYMBOLS rank-ordered by expression (MALAT1 CD3D CD3E IL32 ...).
- anon (WEB-ZERO): the SAME genes' fixed anonymized IDs in the same order (feat_3650 feat_7123 ...).
Both carry IDENTICAL signal: a bag-of-tokens probe reads T-cell-vs-rest at 0.990 on cell_sentence and the same on anon (the anon IDs are a consistent renaming). The specialist ceiling (supervised LogReg on the full expression vector) is 0.989. So the ONLY thing that differs between the two forms is whether the tokens are web-known gene symbols or web-meaningless IDs, which isolates web-exposure with the signal and ceiling held fixed.

## Result: output (8B + frontier) and activation (8B)

| form | 8B activation | 8B output | haiku output | sonnet output | opus output |
|---|---|---|---|---|---|
| cell_sentence (gene names, web-rich) | 0.983 | 0.497 | 0.965 | 0.977 | 0.993 |
| anon (anonymized IDs, web-zero) | 0.964 | 0.497 | 0.458 | 0.452 | 0.470 |

(specialist ceiling 0.989; bag-of-tokens surface probe 0.992 on BOTH forms.)

Three facts, each load-bearing:
1. **The 8B ENCODES the cell type from BOTH forms** (activation 0.983 and 0.964, both near the 0.992 ceiling). It reads which tokens are present, gene names OR anon IDs alike, the same surface-token-pattern encoding seen in every prior rung (char-n-gram, k-mer, m/z). So encoding does NOT distinguish web-rich from web-zero; the signal is present in the hidden states either way.
2. **No model can verbalize from anon IDs** (8B 0.497, frontier 0.45 to 0.47): feat_3650 means nothing at any scale, even though it carries the identical statistics.
3. **Only the gene-NAME form, and only at frontier scale, verbalizes** (8B 0.497 to haiku 0.965 to opus 0.993).

So the web-exposure law here is entirely in EXPRESSION, factored as token-web-exposure x scale: the model encodes the signal regardless, but it can only SAY the cell type when the tokens are web-known gene symbols AND it is large enough to have learned that CD3D marks T cells.

## Scale: cell-sentence CLOSES with scale (correcting the frontier-only view)

The cell-sentence expression gap is SCALE-DEPENDENT, not flat. The full ladder is 8B 0.497 (at chance, encodes but cannot verbalize) to haiku 0.965 to opus 0.993, a sharp unlock between 8B and the frontier tier, then saturating within it (haiku approx opus). Gene-symbol-to-cell-type knowledge is web-rich enough that the smallest FRONTIER model already has it, but the 8B does not, so the jump is 8B-to-frontier, not within-frontier. The anon form is scale-INVARIANT at chance (8B 0.497, opus 0.470). So this single modality contains BOTH ends of the gradient: the web-rich form (cell-sentence) is a scale-closable expression gap like DNA promoter, the web-zero form (anon) is a scale-invariant one like MS, and the activation is high for both because encoding is surface-token-based.

## Why it matters

This is JK's flagship domain (single-cell), and it extends the web-exposure law into omics, the plan's stated SFM-as-input white-space, with the cleanest possible design: the within-entity notation contrast (cell-sentence vs anon) at a fixed ceiling and a fixed signal, parallel to variant text-vs-seq and protein seq-vs-organism but cleaner because here the signal is provably identical in both forms. Honest scope: a small pilot (n=384, pbmc68k_reduced, one binary cell type); the activation (encoding) side is pending. Cell-type-from-expression is the DESCRIPTIVE grounding question; the CAUSAL/perturbation question is the sibling CausalAtlas project, out of scope.

## Reproduce

Data: `eval` build from scanpy pbmc68k_reduced (top-50 genes, T-cell-vs-rest, gene-name vs anon-ID render); gating supervised ceiling 0.989, bag-of-names probe 0.990. Output: `source ~/.api_keys && PANEL_RUNGS=sc_cellsentence,sc_anon PANEL_N=384 python eval/frontier_output_panel.py`. Raw in `results/frontier_output_panel.json`. Activation: `sbatch run_activation_sc_cayuga.sh` (`eval/activation_arm_sc.py`).
