# Modality landscape for the next grounding rung (deep-research synthesis)

*2026-06-12. Synthesized from a 5-angle deep-research sweep (25 primary sources, 121 extracted claims). NOTE: the workflow's adversarial verifier was rate-limited mid-run, so only 1 claim was 3-vote-confirmed and the rest are SOURCE-EXTRACTED (primary source + quote) but not adversarially re-verified; treat the unstarred claims as well-sourced-but-unverified and re-check before external use. 2026 arXiv IDs flagged. Scoping: Perturb-seq CAUSAL grounding is the sibling CausalAtlas project, EXCLUDED; single-cell DESCRIPTIVE grounding is in scope. No em dashes.*

## TL;DR ranked shortlist (next rungs)

| rank | rung | SFM ceiling | verifiable property | web-exposure of raw form | predicted regime | readiness |
|---|---|---|---|---|---|---|
| 1 | single-cell expression -> cell type/state | scGPT/Geneformer/scFoundation/UCE/CellPLM (caveat below) | cell-type label (curated atlases) | raw vector = web-zero; gene-symbol "cell sentence" = web-rich | within-modality contrast: raw-vector encoding/expression-limited vs cell-sentence SCALE-CLOSABLE | HIGH (CellVerse, Cell2Sentence, scEval ready) |
| 2 | 3D structure (coords) -> binding affinity | Boltz-2 | Kd / binding affinity (FEP+, CASP16) | web-zero raw-numeric, computation-bound | ENCODING-LIMITED candidate (the genuine one we lacked) | MED (need a coord-text protocol) |
| 3 | histopathology H&E -> diagnosis/biomarker | CONCH, Virchow2, UNI | diagnosis/grade/biomarker (TCGA cohorts) | image, perception-mediated | expression-limited / perception-dependent (like the molecular-image rung) | HIGH (19-FM benchmark exists) |
| 4 | DNA methylation array -> age | AltumAge (epigenetic clock) | chronological age (verifiable) | web-zero raw-numeric (CpG beta values) | encoding vs expression OPEN; coarse age may be surface-decodable | MED |

## The crucial caveat that reshapes rung 1

The single-cell SFM "ceiling" is NOT clean. The one 3-vote-CONFIRMED finding of the sweep: in zero-shot, scGPT and Geneformer are OUTPERFORMED by simple baselines (highly variable gene selection, scVI, Harmony) on cell-type clustering across five datasets (AvgBio, ASW). [Genome Biology 2025, https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x, 3-0]. A second large benchmark (scEval, 11 FMs) and a Genome Biology Oct-2025 study (6 FMs: Geneformer/scGPT/UCE/scFoundation/LangCell/scCello) agree the FMs do NOT consistently beat task-specific methods. So for the single-cell rung the CEILING should be the BEST available specialist (supervised classifier or scVI/Harmony clustering), not the zero-shot FM, which is shaky. The cell-type-from-expression property IS reliably decodable (a strong supervised ceiling exists), so the rung is valid; just do not anchor the ceiling to a single FM.

## Rung 1 detail: the single-cell rung has a built-in web-exposure contrast (the killer design)

The single-cell rung maps onto the project's instrument almost exactly, with a clean within-modality notation contrast already in the literature:
- RAW expression vector (web-zero numeric): general LLMs ground it poorly (CellVerse: Qwen/Llama/GPT/DeepSeek show only preliminary grounding = expression/encoding-limited; Biology-Instructions: LLMs poor on multi-omics seq->property without training). [https://arxiv.org/abs/2505.07865 (2025), https://arxiv.org/abs/2412.19191]
- GENE-SYMBOL "cell sentence" (web-rich text): Cell2Sentence rank-orders gene NAMES by expression and an LLM reads it; the scaled C2S-Scale reaches ~95.4% cell-type accuracy, MEETING the SFM ceiling (scGPT 93.1%, Geneformer 94.0%). [https://www.biorxiv.org/content/10.1101/2023.09.11.557287v2.full; the 95.4% is from C2S-Scale scaling work, verify against the primary 2025 paper]
- GPT-4 marker-gene annotation: operates on gene SYMBOLS (text), never the raw count matrix, and matches experts in >75% of cell types, beating SingleR/ScType. [https://www.nature.com/articles/s41592-024-02235-4]

So single-cell is the perfect next rung: the SAME entity (a cell) has a web-zero raw form (expression vector) and a web-rich symbolic form (ranked gene names), exactly the variant text-vs-seq contrast, and it is JK's flagship domain. Prediction: raw-vector activation/output low (expression-limited, the model cannot read 20k floats), cell-sentence output rising with scale/training (C2S evidence). This directly extends the web-exposure law into omics, the plan's stated flagship white-space.

## Rung 2 detail: 3D structure is the genuine encoding-limited candidate we lacked

The modality ladder found NO clean encoding-limited anchor (coarse hERG is surface-decodable from every representation). 3D structure -> binding affinity is the candidate: the property (Kd) requires integrating 3D geometry, a computation a forward pass cannot do, and the raw form (coordinates) is web-zero. Boltz-2 gives a STRONG verifiable ceiling: it predicts structure + binding affinity, approaches physics-based FEP accuracy at >1000x lower cost, Pearson r=0.66 on FEP+, beats all CASP16 affinity participants, EF 18.4 at 0.5% in virtual screening. [https://www.biorxiv.org/content/10.1101/2025.06.14.659707] An LLM reading coordinates-as-text -> affinity is predicted scale-invariant chance (like the MS rung but with a genuinely fine, computation-bound property), which would be the first true encoding-limited point.

## Related-work map (modality x SFM x LLM-grounding)

- **Single-cell FM ceilings + benchmarks:** scFoundation (Nat Methods 2024, https://www.nature.com/articles/s41592-024-02305-7), scEval (11-FM benchmark, https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490), Genome Biology Oct-2025 6-FM benchmark (https://link.springer.com/article/10.1186/s13059-025-03781-6), the zero-shot-FMs-lose study (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x).
- **LLM-on-single-cell:** CellVerse QA benchmark (2505.07865, FLAG 2025), Cell2Sentence (biorxiv 2023) + C2S-Scale, GPT-4 marker annotation (Nat Methods 2024, s41592-024-02235-4), LLM4Cell survey (2510.07793).
- **Multi-omics LLM benchmarks:** Biology-Instructions (2412.19191, LLMs poor without training), COMET (2412.10347), OmicsLM (2605.06728, FLAG 2026).
- **Structure:** Boltz-2 (biorxiv 2025.06). **Genome:** Evo 2 (128k genomes, >90% BRCA1 zero-shot, https://www.researchgate.net/publication/401577941). **Pathology:** 19-FM benchmark CONCH/Virchow2/UNI (Nat Biomed Eng 2025, s41551-025-01516-3). **Methylation:** AltumAge (npj Aging 2022, s41514-022-00085-y).
- **Mechanism analogs the project already cites:** numbers encode>>express (2602.07812, FLAG 2026), CoKE (2510.23127).

## Confirmed white space

The sweep found, per modality: (i) SFM-ceiling papers (scFoundation, Boltz-2, CONCH, AltumAge, Evo2), (ii) LLM-task benchmarks (CellVerse, Biology-Instructions, COMET, OmicsLM), and (iii) single-cell FM surveys (LLM4Cell, scEval). It found NONE that DECOMPOSE encoding-vs-expression against an SFM ceiling ACROSS modalities with a web-exposure law. The existing work asks "which FM is best" or "can the LLM do the task," not "does the LLM ENCODE the property it cannot VERBALIZE, and does that track web-exposure." That decomposition across the modality spectrum remains this project's white space (consistent with the plan's section 7.3 "LEAD WITH THIS"). The closest single-modality analog is the numbers-gap paper (2602.07812).

## Recommendation

Do rung 1 (single-cell, the JK-flagship with the built-in raw-vector-vs-cell-sentence web-exposure contrast) for impact and domain fit, and rung 2 (3D structure -> Boltz-2 affinity) to finally land a genuine ENCODING-limited anchor. Anchor the single-cell ceiling to a strong supervised specialist, not a zero-shot FM.
