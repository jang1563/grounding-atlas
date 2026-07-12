# Project Design - Bio_Grounding_Eval

*Toward a Grounded Biology Orchestrator: measure the gap, make the signal, map the line. 2026-06-08.*

> **Public evidence scope (2026-07).** The tracked GroundBench registry, Parquet export, and leaderboard
> contain 24 tasks across 9 modalities. Hidden-state encoding probes use open-weight models; frontier
> systems contribute output and routing results. The project does not establish a same-model
> encode-plus-verbalize result for the frontier leaderboard. Web-exposure is treated as one factor and
> a routing prior, not a universal law. With real per-item specialists, routed accuracy is 0.81 versus
> a 0.91 oracle. Placement claims are limited to the measured cells and budgets.

## 1. The gap (measured, not asserted)
Language models ground scientific entities by NAME but poorly from their concrete REPRESENTATIONS (protein/DNA sequence, chemical SMILES/identifier, numeric SFM output). Two measured consequences:
- **Recognition gap** (from a prior identity-recognition study, not measured here): name recognition ~100% vs accession resolution ~2%; rare protein accessions resolve to benign family-level identities (stable but wrong).
- **Over-trust dissociation (exploratory):** shown a specialist model's output, the model reflects source-flagged reliability, but its rejection of a relevance-orthogonal signal is statistically identical for a hazard-flagged and a benign protein. It hedges on the number's magnitude, not on its relevance. Reported across 8-9 of 10 open-weight models, but several scores fall below the inter-rater threshold (kappa ~0.36), so treat as exploratory pending a human-rater pass.
- **Separability exists in the SFM representation:** a classifier on frozen ESM-2 650M embeddings separates a held protein label from a benign class at AUROC 0.9807 (ESM-3 0.942). This is evidence about ESM-2's representation, not the LLM's: there is no LLM-side measurement on the same panel. Whether the LLM has the signal internally and cannot surface it (expression gap) or never encodes it (encoding gap) is **the central open question, not a settled finding**. Call it a probe-LLM gap until measured.

Implication: if it proves an expression gap it is closable by training the read-out, but an LLM-activation probe is needed to decide. (The separate numeric over-trust line does have LLM-internal evidence.)

## 2. Why here, why now
- The adjacent layers are filling in fast (June 2026). Anthropic's **BioMysteryBench** measures task solve-rate through tools (the orchestrate branch). Anthropic's **gget virus / VirBench** (2026-06-08) measures agent *retrieval* accuracy on viral sequence queries against deterministic ground truth. Neither measures whether the model grounds the *content* of what it retrieves or what a specialist model emits (content vs name), nor whether it calibrates trust on that output. The clean complementary slot is a chain: **retrieval (VirBench) -> content-grounding (this project) -> downstream**. It is the content-grounding layer those evals do not run, complementary to them.
- A second space, **publication-bias-in-AI**, is also crowding (e.g. "Dead Science Walking" 2606.04220, 2026-06-02). That is the negative-evidence space NullAtlas already occupies as a substrate; this project leads with the content-grounding instrument (axis B), not that space.
- Eric Kauderer-Abrams names the obstacle: *"there is no single unambiguous source of truth that we could use as the training signal"* (and is acknowledged on the VirBench work). The verifiable-signal substrate (NullAtlas, WS2) speaks directly to it.
- The *finding* that LLMs lean on name/context over sequence content is already published (CoKE 2510.23127, GenomeQA 2604.05774, Rethinking-Protein 2505.20354). Our contribution is the cross-modality matched-control **instrument** plus the encoding-vs-expression **activation probe** (extending Inside-Out 2503.15299), not the finding.
- The work combines eval-driven development, benchmark design, RL environments, tool use, biological
  databases, and single-cell/multi-omics case studies in one reproducible measurement program.

## 3. Workstreams

### WS1 - The grounding eval (the instrument) [MEASURE]
- **Objective:** quantify whether the model grounds a representation from content or from name, with matched controls. Four axes (`docs/FAILURE_MODES.md`): A identity-resolution (measured), **B content-grounding / encoding-vs-expression (the core contribution, currently unmeasured)**, E channel/action-policy (measured), D reliability-relevance (exploratory). Negative-evidence coverage is NOT an axis here; it is NullAtlas's result (WS2), cited not absorbed.
- **Build:** the B-axis instrument = a probe-vs-LLM head-to-head + an LLM-activation probe (to separate encoding from expression) + content-sensitivity (matched/mismatched/scrambled/content-only); deterministic non-LLM-judge scoring; matched controls. A and E reuse the prior study's harness, reframed safety -> capability. See `eval/README.md` for the spec.
- **Source assets:** a prior protein identity-recognition study (aggregate results; supplies A and E) + a prior frozen-ESM-2 separability probe (AUROC 0.9807, the B-axis ceiling anchor). A cross-representation instrument is the design goal; only protein is measured so far.
- **Deliverable:** a clean public eval in `eval/` on a held-out set; a short methods writeup.
- **Ladder status:** T0/T1 done across SMILES, variant, protein. **T2 (apply): SOLVE scored** (`results/t2_apply.md`, `eval/t2_apply.py`): solo grounding does not transfer (output near chance, flat regardless of internal encoding); the solo-to-orchestrate headroom decomposes per task into an expression part (train the read-out, WS3-weights) and an encoding part (orchestrate the specialist), which is the T2 routing rule that seeds the WS3 decision map. T2 propose scored (`results/t2_propose.md`: generation competent, grounded-activity undetermined by the in-distribution probe); evaluate blocked on the axis-D human-rater pass. Caveat on solo "reads": this is named-property recall (property in the prompt), an upper bound on grounding. The scrambled control was run (`results/notation_control.md`, CYP3A4 n=1000 + CYP2D6 replication): the reading IS structure-dependent (scrambled to chance) and notation-invariant, but property-specificity is unestablished (the solubility control is confounded by lipophilicity).

### WS2 - Verifiable signal: the content-property task the B-axis needs [MAKE SIGNAL]
- **Objective:** build the (representation, verifiable-property) task the B-axis head-to-head requires, where the sequence/structure/SMILES is the ground truth (not a contested label). No off-the-shelf asset provides it, so it must be generated; this is why WS2 partly precedes the WS1-B measurement rather than following it.
- **Build:** a generator of matched (representation, content-property) pairs using public or
  appropriately licensed content-property datasets; gate each on a high supervised probe ceiling
  before use. Public source and license details are maintained in `DATA_SOURCES.md`.
- **Deliverable:** a signal spec + a generated sample set in `signal/`, with a verifiability check.
  **Built (v1):** `signal/generate_signal.py` (modality-general generator + gate), `signal/README.md` (spec), and 22,203 records across 7 SMILES ADMET endpoints (`signal/admet/`); all 7 PASS the verifiability gate (cold AUROC 0.75-0.90, leakage drop <0.08, selectivity +0.24-0.38; `signal/verifiability_report.md`). Sequence/variant/metabolite plug in via the same Source interface with an SFM-embedding featurizer, gated on GPU.
  **Built (v2, 2026-06-13):** `signal/gate_multimodal.py` generalizes the gate from SMILES to 19 modality cells under CHEAP local featurizers (Morgan / k-mer / char-ngram / element-fraction / pixel-stat, no SFM), `signal/verifiability_multimodal.md`. **17/19 PASS**, confirming the recipe is modality-general. The 2 fails are **PPI-as-NAME** (named AND anon both at chance, cold 0.50), confirming that PPI's named 0.95 was web-memorization of documented pairs, not content. The clean contrast is single-cell, which PASSES even anonymized (0.917 -> 0.909) because cell identity lives in gene CO-OCCURRENCE, preserved under id-swap.
  **Negative-class expression gap (B, 2026-06-13, `results/negative_expression_gap.md`):** on 4 ADMET endpoints, the open 8B model ENCODES inactive/active near the Morgan ceiling (activation 0.69-0.88 held-out, selectivity-controlled) but VERBALIZES it at chance (0-shot output 0.45-0.51): expression gap 0.19-0.38, encoding gap ~0, replicated cross-family (Qwen3-8B + OLMo-2-7B) on GPU. Largest ROBUST gap on hERG (0.33, n=1250); the first AMES point shrank from 0.376 at n=206 to 0.145 at n=2000, so sizes are read off robust n only. Complements NullAtlas L4 (output-only); does not re-measure it. Honest: open proxy (Claude internals unobservable), anon carries an under-specification confound, AMES n small.

### WS3 - The decision map (the forward build) [MAP THE LINE]
- **Objective:** per capability (read functional class from a sequence; interpret an SFM numeric output; resolve an accession; route to a structure predictor), measure which placement wins: trained into weights / retrieved (MCP/RAG) / orchestrated (call the SFM + trust-gate).
- **Principle:** train the skill (grounding/reading), retrieve the knowledge (facts and calibration that change), orchestrate the heavy specialist compute (you will not out-AlphaFold AlphaFold inside an LLM).
- **Build:** a local open-weight PoC. Train a small model on WS2 signal, measure grounding lift on the WS1 eval, compare against retrieval and orchestrate baselines, and chart the first bio-grounding decision-map points. **Done (`decision_map/DECISION_MAP.md`, `results/ws3_lora.md`):** the LoRA weights PoC lifts the 8B verbalized hERG output from 0.575 to 0.856 (+0.28), recovering MOST of the same-split structural ceiling (Morgan probe 0.899, k-NN 0.901) but staying ~0.04 below it (the earlier "exceeds the 0.787 probe" was a cross-split confound, corrected in review). So P2 (the gap is partly closable by training the read-out) holds at the output level: the model can be trained to surface most of the structural signal it encodes, though on this fingerprint-local property it is a strong SECOND to the cheap specialist, not a winner. The retrieve placement is a label-parrot artifact (beaten by a no-LLM neighbor-mean, follows flipped labels to 0.105); molecular image is the encoding-limited corner (DECIMER 0.97 vs Claude 0.54, only a perception specialist works). For capabilities LACKING a cheap specialist the weights lever would be the placement that delivers, which needs such an endpoint to demonstrate.
- **Source assets:** `BioRLHF/` (pipeline; first data point = a calibration that could not be baked in and was better retrieved), `Evo2/` and ESM-2 (the SFMs to orchestrate), a prior agentic tool-mode harness.
- **Honest prior art to cite, not reinvent:** general weights-vs-retrieval-vs-tool decision (Ovadia 2312.05934; In-Tool Learning 2508.20755; the when-to-retrieve line). This program focuses on bio-specific representation grounding across matched representations and controls.
- **Deliverable:** a decision-map writeup + the PoC in `decision_map/`. **First point done** (`decision_map/DECISION_MAP.md`): for hERG-SMILES the instrument shows the LLM is the WRONG tool. A mandatory no-LLM control (average the k nearest labeled neighbors, never read the query) scores 0.951 dense / 0.913 scaffold-holdout, ABOVE the LLM-in-the-loop retrieve (0.843 / 0.831) and the Morgan probe (0.895); the trained 8B read-out (0.787) and solo (0.45-0.63) are lower still. So every LLM placement is dominated by a cheap fingerprint specialist, and this capability should be orchestrated to a non-LLM tool. The "sparse-pool collapse" was just the baseline degrading (0.554), not a model effect. The second corner (molecular image) is the opposite: solo-image at chance (0.539) and the LLM's OCSR only half-right (Tanimoto 0.54), so only a dedicated perception specialist works. Refined routing axis: not "retrievability" as a model skill but "is there a cheap non-LLM specialist + labeled pool that already solves it." One endpoint so far; CIs ~+/-0.08.

## 4. Program-level outputs
- A grounding + calibration eval that runs on every model release, predicting where the model will silently mis-ground a science-tool output.
- A verifiable-signal recipe for positive-only regions of biology, where training-signal construction is the central bottleneck.
- An empirical, defensible answer to the architecture question the field is deciding by assertion.

## 5. Honest scoping (what this does NOT claim)
- Not a frontier-model training claim. The demonstrable version is a local PoC plus a transferable eval/signal method.
- Not the multimodal build as novelty (crowded: NatureLM, Intern-S1, Prot2Chat, Mol-LLaMA family). This builds on it.
- The vision-as-modality analogy ("ground sequences like pixels") is exposition, not a research claim.
- The durable core is the **B-axis content-grounding instrument** (with A and E as measured supporting findings) and the verifiable-signal substrate (NullAtlas, WS2). The durable parts are those: the decision-map framing is fast-moving, the retrieval layer (VirBench) and the publication-bias space (2606.04220) are filling in, and deterministic-eval alone is no longer a differentiator. The grounding-specific content-vs-name and encoding-vs-expression measurement is what stays distinct.

## 6. One supporting line of safety
The same instrument flags where a grounded model is unsafe, so capability and responsible deployment are measured on one ruler. (One line only; this is a capability project.)

## 7. Modality roadmap (instrument extension, theory-anchored)
Framing: extend the *measurement instrument* to new representations. The novelty is the measurement and a testable hypothesis about grounding (the web-exposure hypothesis in 7.1, since corrected to a two-factor account), not a multimodal model build (NatureLM etc.).

### 7.1 The web-exposure hypothesis (theoretical backbone)
> **Correction (see [`docs/REPORT.md`](docs/REPORT.md)).** This applies to all of section 7: the
> The original single-factor language throughout sections 7.1-7.5 is superseded.
> A later matched experiment (drop the textbook markers, keep the real gene names)
> decomposed the verbalization gap into a **capability-dependent mix of token-familiarity/reasoning and
> mapping-documentation**, not a single law. The effect is real; the mechanism is the correction. Read the
> "law" language below as the working hypothesis it began as, superseded by the two-factor account.

Hypothesis: a modality's internal ENCODING strength (activation-arm AUROC relative to the specialist ceiling) is set by how often that modality's "content -> property" mapping appears in web text. This is a direct generalization of the frequency-encoding result in 2504.12459 (linear representations of a fact form once subject-object co-occurrence crosses ~1-2k in pretraining, and representation quality predicts pretraining frequency). We extend it from text triples to scientific modalities. The expression gap (encoding - output) tracks the same axis, as in the numbers analog 2602.07812 (linear probe >90% vs verbalized 50-70%, "worse where the probe is weaker").

The same entity can be grounded differently across notations: canonical SMILES, gene+HGVS text, raw
sequence, and accessions expose different surface information and training associations. The variant
branch shows a text-vs-sequence difference on the same variants (0.79 vs 0.58), while the SMILES
canonical-vs-randomized comparison is near a floor (0.573 vs 0.553). These matched contrasts support a
notation effect, without establishing one universal mechanism.

Two falsifiable predictions:
- **(P1) Monotonicity:** encoding gap (ceiling - activation) was predicted to decrease with
  content-to-property web co-occurrence. The cross-modality regression is mis-specified, so P1 remains a
  qualitative working hypothesis assessed through within-entity notation contrasts, not a fitted law.
- **(P2) Bottleneck shift:** SMILES / sequence are EXPRESSION-limited (high encoding, chance output, closeable by read-out or finetune, cf. 2602.07812 +3.22%); molecular images are ENCODING-limited (low encoding, perception floor). The 3-arm activation probe separates the two regimes.

### 7.2 Measured anchors (17 representations; open-model probes plus separate frontier output studies)
| modality (property) | ceiling | activation | output | enc gap | exp gap | regime |
|---|---|---|---|---|---|---|
| SMILES (hERG) | 0.825 | 0.787 | 0.453 | 0.038 | 0.334 | expression-dominant |
| SMILES (CYP3A4) | 0.745 | 0.684 | 0.502 | 0.061 | 0.182 | expression-dominant |
| variant (ClinVar, text form) | 0.962 | 0.795 | 0.599 | 0.167 | 0.196 | mixed |
| variant (ClinVar, seq form) | 0.962 | 0.740 | 0.494 | 0.222 | 0.246 | mixed (8B expression-limited) |
| protein (meltome Tm) | 0.699 | 0.609 | 0.486 (8B seq) | 0.090 | 0.123 | encoding-weak + organism-name shortcut, confirmed at frontier: seq-only opus 0.585 (web-poor, weak) vs seq+organism 0.647 (the name grounds it), `results/frontier_output_panel.md` |
| DNA (promoter, non-TATA) | 0.889 | 0.880 | 0.396 (8B) | 0.009 | 0.484 | expression gap CLOSES with scale: output 8B 0.40 anti -> haiku 0.62 -> sonnet 0.80 -> opus 0.82 (web-rich, `results/frontier_output_panel.md`) |
| molecular image (hERG, Qwen2.5-VL) | 0.854 | 0.758 | 0.460 | 0.096 | 0.298 | expression-limited (P2 encoding-limited prediction REFUTED for coarse hERG) |
| spectra (MS, hERG) | 0.825 / 0.667 surf | 0.729 | 0.502 (8B) | 0.096 | 0.227 | expression gap SCALE-INVARIANT: output flat ~0.5 across 8B + haiku/sonnet/opus (web-zero + elucidation-bound; the first scale-invariant verbalization limit) |
| single-cell expr -> T cell, gene-name cell-sentence | 0.989 | 0.983 | 0.497 (8B) | 0.006 | 0.486 | expression gap CLOSES with scale: output 8B 0.497 -> haiku 0.965 -> opus 0.993 (web-rich gene symbols; `results/single_cell_rung.md`) |
| single-cell expr -> T cell, ANONYMIZED ids | 0.989 | 0.964 | 0.497 (8B) | 0.025 | 0.467 | expression gap SCALE-INVARIANT: output ~0.5 across 8B + frontier (web-zero IDs; same signal encoded, never verbalizable) |
| 3D structure (XYZ coords, hERG) | 0.826 | 0.669 | 0.490 (8B) | 0.156 | 0.180 | MOST encoding-limited hERG rep (encoding gap 0.156 largest; 8B gets only atom-composition surface, misses 3D geometry); output scale-invariant chance (opus 0.539) |
| molecular graph (atom+bond list, hERG) | 0.866 | 0.708 | 0.458 (8B) | 0.157 | 0.250 | expression-limited; 8B reads atom-list composition surface, misses bond topology (enc gap 0.157, like 3D) |
| NMR (simulated 13C, hERG) | 0.866 | 0.747 | 0.434 (8B) | 0.119 | 0.313 | expression-limited; 8B reads carbon-shift composition surface (enc gap 0.119) |
| RNA / coding (genomic seq -> coding) | 0.930 (6-mer) | 0.866 | 0.720 (8B) | 0.064 | 0.146 | MILD gap; 8B reads codon structure (output 0.720 >> DNA-promoter 0.396, codon/ORF is web-documented) |
| histopathology H&E (tumor, Qwen2.5-VL) | 0.746 cheap / ~0.9 CONCH | 0.827 | 0.463 (VLM) | ~0.07 vs CONCH | 0.364 | The open-VLM probe exceeds its output; separate frontier output rises to ~0.65. This cross-model comparison motivates, but does not prove, a frontier encoding-vs-verbalization gap; `results/histopath_rung.md` |
| MSA column (AA residues -> conserved) | 0.999 (col stats) | 1.000 | 0.795 (8B) | ~0.00 | 0.205 | Positive control with a strong open-model probe and output. The comparison with methylation is consistent with a mapping-documentation contribution; `results/msa_rung.md` |
| DNA methylation (beta vector -> age) | 0.701 (LR clock) | 0.685 | 0.487 (8B) | 0.017 | 0.198 | The open-model probe approaches the cheap clock while output stays near chance. Compared with MSA, this is consistent with a mapping-documentation contribution but does not isolate it as the only cause; `results/methylation_rung.md` |

**Coverage and deliberate subsumptions.** The 17 rows above span token strings, biological sequence,
images, spectra, 3D coordinates, molecular graphs, expression vectors, epigenetic numeric vectors, and
sequence alignments. Spatial transcriptomics and cell-painting remain separate empirical validations;
representation-type similarity does not by itself establish transfer to those domains.
Regime spectrum: expression-dominant (SMILES, encoding near the ceiling, only output at chance) through mixed (variant, both gaps; the seq form is expression-limited at 8B but opus verbalizes it at 0.80, scale closing the expression gap) to encoding-weak (protein, low ceiling plus an organism-name shortcut).

**Two refinements from the 2026-06-11 deep-review controls (the activation numbers stand; the INTERPRETATION of "encodes" is sharpened).** (1) The SMILES hERG activation signal (0.787) is genuine notation-invariant STRUCTURE, not a canonical-string artifact: it survives randomized SMILES at 0.739 (held-out 0.732). But it does NOT exceed a no-chemistry char-n-gram probe (0.812 on the same randomized SMILES), so "encodes the chemistry" should read as "the property is linearly DECODABLE from the hidden states, as it is from the surface string; the decodable signal is structural but not deeper than a substring probe." The expression GAP (probe >> output, scale/arch/alignment/reasoning-invariant, selectivity- and few-shot-controlled) is the durable claim, not a strong "represents the chemistry" reading. See `results/lipophilicity_control.md`, `results/head_to_head.md`. (2) The re-notation content-sensitivity test now doubles as an axis-A-vs-axis-B DISCRIMINATOR inside the activation probe: on a recognition-heavy endpoint (drug market-withdrawal) the activation reads 0.762 >> output 0.469 with the SAME probe-vs-output shape, but it COLLAPSES to the structure level under randomization (0.762 -> 0.662), revealing that the above-structure part is axis-A entity-recognition + fact-recall (canonical-string identity token), not axis-B content-grounding (which survives, as hERG does). So a probe-vs-output gap counts as content-grounding only if it survives re-notation. See `results/withdrawn_endpoint.md`, `results/layer_profiles.md`, `decision_map/DECISION_MAP.md` third corner. Within-modality P1 is clean: variant text > seq at the SAME ceiling under a gene GroupKFold (activation 0.795 > 0.740, output 0.599 > 0.494), and SMILES canonical vs randomized is a floor. Cross-modality recovery fraction is ceiling-confounded (ceilings span 0.70-0.96), so the encoding-gap absolute (SMILES 0.038 < protein 0.090 < variant 0.167-0.222) and the regime labels carry the comparison, not one monotone number. All arms selectivity-controlled (random-label probe ~0.5). Specialist ceilings are content-grounded and leakage-free: AlphaMissense 0.962 + unsupervised ESM-1v 0.921 on the variant temporal holdout, ESM2 for protein, Morgan FP for SMILES. The 5-model (SMILES) and 6-model (variant) panels confirm architecture/scale/vendor invariance, with one scale effect: the variant web-poor seq floor RISES with model size (sonnet-4-6 0.55 to opus 0.80), unlike the flat-at-chance SMILES output, marking variant-seq as web-poor not web-zero.

### 7.3 Modality ladder (ordered by predicted encoding gap; each gated on a high ceiling)
Each modality's OUTPUT arm is largely pre-run in the literature; the ceiling + activation arms are what this project adds.
| modality | specialist ceiling | output-arm prior art | activation weight | predicted regime |
|---|---|---|---|---|
| Variant / HGVS text | AlphaMissense 0.94, ESM-1v | GPT-4o 0.73 zero-shot (Hu npj 2025), ABOVE chance | light 2-arm; leakage-acute | small enc gap, output nonzero (web-rich text form); MEMORIZATION confound -> temporal holdout + DMS control |
| DNA / RNA (short motif) | NT/DNABERT-2 0.85-0.95, Evo2 | GenomeQA 2604.05774 (GC/motif yes, inference no) | full 3-arm; BPE tokenization risk | medium enc; task-heterogeneous |
| Protein sequence | ESM-2 / ESM-1v | spawned branch | full 3-arm | medium enc, rising with scale |
| SMILES (anchor, done) | Morgan FP 0.825 / 0.745 | near-chance (ours + prior) | full 3-arm | high enc, large exp gap |
| Molecular image | ImageMol 0.82 | MolVision 2507.03283 (image-only 0.15 vs text 0.71) | light 2-arm; open VLM optional | low enc, ENCODING-limited; report OCSR floor as covariate |
| Raw spectra (MS/NMR) | SpecTUS 65% | MolPuzzle (GPT-4o 1.4% exact) | light 2-arm | lowest enc (web-near-zero) |

White space: no one has run a single controlled 3-arm instrument across the modality spectrum and regressed the gap on web-exposure. 2504.12459 did it for text triples; we generalize to chemistry -> biology -> genomics.

**Measured update (4 rungs now run, not predicted):** SMILES expression-dominant and protein encoding-weak-plus-organism-name are confirmed. Variant: the predicted "small encoding gap" was WRONG, it is a MIXED regime (encoding gap 0.167-0.222 against the 0.96 ceiling AND an expression gap; the web-rich text output 0.79 is largely gene-prior clinical recall, not variant-specific grounding, per the DMS and gene-scramble controls). The web-poor seq floor rising with scale (0.55 to 0.80) confirms web-poor-not-web-zero. DNA (promoter, `results/dna_promoter.md`): the predicted "medium encoding gap" was also WRONG, it is the MOST expression-dominant rung (encoding gap 0.009, the smallest; expression gap 0.484, the largest; output 0.396 ANTI-correlated = a TATA/GC heuristic mis-fired on the non-TATA set). The lesson the DNA and SMILES rungs share: the cross-modality encoding-gap magnitude is confounded by how SURFACE-DECODABLE the property is (the DNA ceiling is itself a 6-mer surface probe and the SMILES activation does not beat a char-n-gram), so the model encodes surface string statistics regardless of web-text binding, and the law's clean test stays the within-entity notation contrast (variant text vs seq), not the cross-modality number. Molecular image (hERG, `results/image_rung.md`): the predicted ENCODING-LIMITED regime is REFUTED for this property, two independent ways. An OCSR perception proxy (Morgan on the VLM-transcribed structure) scores 0.759, and a DIRECT open-VLM hidden-state probe (Qwen2.5-VL-7B, `activation_arm_image.py`) scores 0.758, nearly identical, both close to the true-structure ceiling (0.85). So the VLM ENCODES hERG from the image (activation 0.758) and cannot VERBALIZE it (output 0.460), an expression gap, despite imperfect OCSR (0.55 Tanimoto): hERG is a COARSE property whose signal survives a half-right transcription, so image-hERG is expression / orchestration-limited, NOT encoding-limited. P2 (images encoding-limited) is property-granularity-dependent: it holds for FINE-structure tasks (MolVision exact 0.15) where the 0.55-Tanimoto floor is fatal, not for coarse physicochemistry. A genuinely encoding-limited anchor needs a fine-structure property or a non-renderable modality (spectra). Spectra (MS, hERG, `results/spectra_rung.md`): tested as the non-renderable extreme, and P2 FAILS a fourth time. Presented ONLY as a fragment m/z peak list, the LLM still encodes hERG at activation 0.729, ABOVE a binned-m/z surface probe (0.667) and near the structure-elucidation ceiling (0.825), while its output is exactly chance (0.502): reading the m/z numbers as text, it encodes the coarse hERG signal and cannot verbalize it, expression-limited again. CLOSING PRINCIPLE (established across four representations of hERG: SMILES substrings 0.787, DNA k-mers 0.880, image OCSR 0.758, MS m/z 0.729, all expression-limited with encoding gap <= 0.10): a COARSE property is SURFACE-DECODABLE from any representation, so the LLM encodes it from whatever surface that representation exposes and the bottleneck is always EXPRESSION, regardless of how non-renderable the modality is. The ENCODING-LIMITED regime is therefore property-GRANULARITY-dependent, not modality-dependent; it requires a property not surface-decodable (exact structure elucidation, a specific 3D pharmacophore, exact-match identity) where a forward pass cannot compute the answer. This is the corrected, measured form of P2.

**The scale axis** (`results/frontier_output_panel.md`) is an output-only frontier study. DNA promoter
output rises with model scale, whereas the MS-to-hERG output remains near chance. These observations are
consistent with capability and representation familiarity contributing to output, but they are not
frontier hidden-state measurements.

**Generality beyond biology (`results/generality_materials.md`).** In one materials comparison,
element-symbol formulas outperform anonymized-element compositions (0.72-0.84 vs 0.44-0.54). This is
consistent with the notation effect outside biology, but one control domain does not establish a
universal law.

Within-modality control (the cleanest test): variant effect exists as a web-rich text form (HGVS / gene + pathogenic, in ClinVar) AND a web-poor sequence form. Probing both, same biology different surface form, measures the web-exposure effect inside ONE modality. Relevant to GeneLab / space-biology: a novel spaceflight variant is absent from ClinVar, so the output arm should fail while the ceiling/probe still work, a concrete downstream use.

### 7.4 Execution split and a mandatory control
- **Full 3-arm (GPU activation):** SMILES (done), protein (spawned), DNA/RNA. Heavy, separate sessions.
- **Light 2-arm (ceiling + output API):** SMILES endpoints (CYP3A4 done), variant/HGVS, molecular image, spectra. In this project.
- **Control task (Hewitt-Liang 1909.03368), MANDATORY per modality:** a random-label probe to report selectivity (task minus control accuracy). An encoding claim is only defensible with high selectivity; this guards against the probe memorizing rather than reading the model.
- **Representation-invariance fix (the map routes it):** where the same entity is grounded in one notation but not another, normalize the input form before the model reads it (any SMILES to canonical, an accession or HGVS to its sequence) by orchestrating a tool or specialist, or train a representation-invariant read-out that binds an entity's notations together. This is the concrete WS3 placement for notation failures: canonicalize-by-orchestration or train-the-invariance, decided by the same 3-arm instrument, and it turns the recognition gap (axis A) into a capability fix, not only a safety observation.
- Later/heavier candidates kept from prior plan, ceiling-gated first: 3D structure (PDB/mmCIF), metabolite (HMDB), single-cell/spatial (priority domain, later), SFM-embedding-as-LLM-input (scELMo/BioVERSE; widest-open, no behavioral baseline). **SFM-embedding measured (2026-06-13, `results/sfm_embedding_rung.md`): ESM-2 embedding of Meltome proteins -> Tm, ceiling 0.81-0.85, but the LLM reads it at chance both zero-shot (0.47, raw 640-dim as text) and few-shot ICL (0.56, 24 labeled example vectors). So orchestrating an SFM means a TRAINED read-out head on its output, not pasting the embedding into the prompt; the LLM is not an in-context decoder of an abstract embedding space. Output arm only; the activation arm is GPU-deferred.**

Data-format phases unchanged: Phase A text (now) / Phase B raw formats (PDB, VCF, h5ad, mzML) / Phase C images (microscopy, IHC, spatial, cryo-EM; NegResultDB HPA is the text-to-image bridge). The ladder above prioritizes within them. Each representation enters the same machinery: ceiling-gate -> activation/output arms -> content-sensitivity -> selectivity control.

### 7.5 The property-type axis: computable vs empirical (2026-06-13, `results/computable_property_row.md`)
The 7.2 / 7.3 rungs measure empirical properties. Computable properties (atom/ring count, molecular
weight, sequence length, GC content, pI) follow a different axis. In the output study, enough reasoning
tokens close several counting/summing tasks, while pI remains limited by execution reliability and
constant recall. This bounds the web-exposure account and motivates deterministic tools for exact
computation. A 400-token pilot created a truncation artifact; a truncation guard and larger budget
reversed the apparent failure.
