# Long-horizon plan: extending the placement instrument beyond descriptive biology

*2026-06-12. From a 5-axis deep-research sweep (`missing-biology-axes-research` workflow; full per-axis findings + sources in the workflow output). Extends the measured 4-arm placement instrument (encode / 0-shot output / retrieve k-shot ICL / orchestrate specialist-ceiling) from the 13 descriptive bio/chem capabilities to the axes biology research uses that our instrument did NOT cover. No em dashes.*

## The 5 missing axes

| axis | cleanest task | specialist ceiling | does ICL/retrieve apply | placement prediction |
|---|---|---|---|---|
| 1 macromolecular structure / docking | sequence/structure -> binding affinity (pKd) | Boltz-2 (FEP+ Pearson 0.66) | applies but FAILS (ICRL underperforms supervised, sometimes worse than text-ICL) | ORCHESTRATE, with a possible TRAIN frontier (multimodal structure head) |
| 2 generative / design | spec -> novel entity -> in-silico validate | RFdiffusion + ProteinMPNN; folding oracles verify | partial / asymmetric (no label to retrieve; novel target) | ORCHESTRATE (LLM drives generator+verifier); needs MODIFIED instrument |
| 3 causal / perturbation | PerturbQA yes/no (knockdown -> DE) | GEARS / rBio (F1 0.786) | NO (base+frontier LLM F1 0.24-0.30) | ORCHESTRATE/TRAIN; EXCLUDED = sibling CausalAtlas |
| 4 relational / network | does A interact-with / regulate B (name vs anon vs seq) | ProLLM/MINT (PPI), TxGNN (KG link) | YES, reuses the project web-exposure knob | RETRIEVE (web-known edge) / ORCHESTRATE (novel) |
| 5 temporal / dynamic | ECG -> arrhythmia class (raw-string vs encoder) | LaBraM (EEG), ECG-FM, MIRA | bounded (generic LLMTime real; bio-signal in-context unproven) | TRAIN-dominant ceiling, ORCHESTRATE-over-features wrapper |

## Fit as-is vs needs a modified instrument

- **Fit the 4-arm descriptive instrument unchanged:** Axis 4 (binary edge property + name-vs-anon knob, reuses `eval/icl_placement.py` + no-LLM neighbor-mean baseline + degree-matched negatives), Axis 1 (sequence/structure -> affinity number/class, the "3D coords -> hERG = orchestrate" row generalized to a real ceiling), Axis 5 (signal -> diagnosis class, with a raw-string-vs-encoder sub-knob).
- **Needs a modified instrument:** Axis 2 generation (a design-then-validate loop, not classification; arms redefined as emit-then-fold-and-score; new mandatory control = a SECOND independent folder to confirm a pass). Axis 3 causal (soft-verifier / RLVR framing; out of scope here).

## Prioritized order (tractability x leverage on the closed-weight conclusion)

1. **Axis 4 relational/network** (DO FIRST). HIGH tractability (STRING/BioGRID/Reactome public+dated, fits as-is, low refusal), extends the flagship web-exposure contrast to a NEW task family (interactions). Predicted retrieve+orchestrate -> STRENGTHENS closed-weight.
2. **Axis 1 structure -> affinity.** HIGH leverage: the cleanest in-set chance to BREAK "retrieve closes everything encodable." Boltz-2 ceiling (weights released). If retrieve stays below Boltz-2 and only a TRAINED multimodal head competes, this is the first real pressure on the closed-weight claim. Most likely: orchestrate wins (LLM calls Boltz-2, does not need to BE it), but report the train cell explicitly. Decontaminate with post-cutoff CAMEO/CASP16.
3. **Axis 5 temporal.** MEDIUM tractability (number-tokenization + Claude seq-refusal gotchas; needs the representation sub-knob), second train-pressure cell; orchestrate-over-features likely rescues closed-weight.
4. **Axis 2 generative.** MEDIUM-LOW (build the design-then-validate loop + dual-folder control; folding oracles compute-heavy). Predicted ORCHESTRATE, least likely to move the conclusion (LLM was never the engine). One untested low-confidence arm worth a clean NO: can a frontier LLM emit a designable sequence purely in-context.
- **Axis 3 causal: EXCLUDED.** Sibling CausalAtlas / Causal_Grounding_Eval. It is the one place TRAIN demonstrably (if weakly) wins (rBio F1 0.786 vs LLM 0.24-0.30), which is exactly why it is fenced off as its own project. Keeping it out is what keeps "train wins nowhere" honest: that claim is about DESCRIPTIVE property prediction, and the causal axis is the known exception next door.

## Expected effect on the closed-weight conclusion

Across the 4 in-scope axes the conclusion most likely SURVIVES and SHARPENS: retrieve+orchestrate keep covering (Axis 4, and the orchestration halves of 1/2/5), and where retrieve fails (affinity, raw temporal signal, generation) the winner is a CALLABLE specialist (Boltz-2, ECG-FM, RFdiffusion), not a weight-trained Claude. The one genuine risk is Axis 1's multimodal structure head; the instrument must report that cell rather than assume orchestrate. The causal axis, correctly fenced as the sibling project, is what keeps the original conclusion from overreaching.

## Data sources

Axis 4: STRING v12, BioGRID, IntAct, Reactome, DrugBank/BindingDB, PrimeKG/Hetionet; DoRothEA/BEELINE (GRN). Axis 1: PDBbind, BindingDB, FEP+ benchmark, CASP16 affinity, MF-PCBA, PoseBusters v2, Tsuboyama-2023 ddG; ceiling Boltz-2/DiffDock-L/AlphaFold3. Axis 5: PTB-XL, MIT-BIH, TUAB/TUEV, MIMIC; ceiling LaBraM/ECG-FM/MIRA. Axis 2: PDB/CATH backbones, ESMFold/AF2/Boltz verifiers, RFdiffusion/ProteinMPNN/REINVENT generators.
