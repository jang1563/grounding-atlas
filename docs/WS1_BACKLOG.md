# WS1 Backlog: maturity and what more to do

*2026-06-09. Companion to `FAILURE_MODES.md` and `../PROJECT_DESIGN.md` section 7. No em dashes.*

## Maturity read: findings solid, eval coverage uneven

The B-axis claim (encoding vs expression + the web-exposure law) is the central claim and SOLID across three modalities (SMILES, protein, variant), with regime spectrum, selectivity control, and leakage-free specialist ceilings. That is enough to carry the main result.

But as an instrument the coverage is a sketch in places, not uniform:
- **Depth is uneven.** SMILES is deep (5-model panel, 2 endpoints, thinking, selectivity, randomized). Variant is rich (6-model panel, activation arm, ESM-1v, DMS, temporal holdout). Protein is thin (1 activation model Qwen3-8B + a Claude axis-A arm), one low-ceiling property (Tm 0.70), no selectivity yet.
- **Axes A and E are protein-only** (FRT reuse), not replicated on chem or DNA.
- **D (reliability-relevance) is exploratory** (kappa 0.36, no human-rater pass).
- **T2 (apply): SOLVE + PROPOSE scored** (`../results/t2_apply.md`, `../results/t2_propose.md`); EVALUATE blocked on the D human-rater pass.
- **Content-sensitivity is partial** on every modality (no rung has all four conditions + re-notation).
- **P1 is shown qualitatively** (monotone-ish), not regressed on a measured web-exposure covariate.
- **Two named confounds are not yet controlled in the measurement** (eval/README): supervision asymmetry (a trained probe vs a zero-shot LLM) and input asymmetry (the probe reads an SFM embedding, the LLM reads raw text). These are the most likely reviewer objections.

## What more to do (by category; weight / relevance)

### A. Modality depth (make each rung as solid as SMILES)
- **protein:** cross-arch panel (Qwen3-32B/Phi-4/gpt-oss), thinking arm, **selectivity control (not yet run)**, a HIGHER-ceiling property (subcellular localization; Tm 0.70 is low so the gaps are small and noisy), composition-preserving + motif-targeted shuffles. [medium, GPU]
- **variant:** activation cross-arch panel + thinking (only 8B activation so far). [medium, GPU]
- **SMILES:** more ADMET endpoints (solubility, permeability, AMES; ceiling-gate each), full content-sensitivity set. [light]

### B. Axis completion (A/E/D, T2)
- **A/E beyond protein:** axis A chem DONE (`../results/axis_a_chem.md`, `../eval/axis_a_chem.py`): within-entity recognition on 35 RDKit-validated famous drugs, name 1.00 > SMILES 0.86 > InChIKey 0.69 (sonnet-4-6), the predicted web-frequency order. Gap is compressed (0.31 vs the protein ~98pt) because ultra-famous InChIKeys are memorized = fame pushes every notation over the web-frequency floor, exactly the web-exposure law; the two ends (web-rare toxin accession ~2%, ultra-famous drug InChIKey 0.69) span the fame axis. Follow-up to widen it: a less-famous molecule set. **Gene/DNA axis A also DONE** (`../results/axis_a_dna.md`, `../eval/axis_a_dna.py`): 40 popular ClinVar genes (trusted data, no curation), symbol 1.00 > UniProt accession 0.60 > rsID 0.025 > sequence 0.025; the rsID/sequence 0.025 REPRODUCES the protein FRT accession ~2% on a broad panel and CONFIRMS the chem prediction (the gap opens to its full 0.975 where the rare notation is genuinely web-rare, vs chem InChIKey 0.69 fame-compressed). Axis A is now measured on three modalities (protein/chem/gene), one instrument, with gap-size tracking notation web-rarity = the web-exposure law over notations. Remaining: axis E (chat-vs-tool action gap, needs the FRT P2 harness). [axis A done x3 modalities; E open]
- **D:** a human-rater pass to lift reliability-relevance off exploratory (current kappa 0.36). **Package built** (`../docs/D_rater_protocol.md`, `../eval/d_agreement.py` with self-test, `../eval/d_rating_sheet_template.csv`): binary rubric + decision rules designed to clear kappa 0.60, plus the recommended D-v2 recast (content-grounded over-reliance on a SCRAMBLED specialist output, the non-scooped version that ties to the content-sensitivity controls). Raw D items stay in `FRT_Pilot_Execution/disclosure/` (disclosure boundary); the package applies there. This same pass unblocks T2-evaluate. Remaining = the human ratings themselves. [package done; needs raters]
- **T2 (apply): SOLVE done, all 7 ADMET endpoints.** Does T1 grounding predict downstream solve? SOLVE is scored (`../results/t2_apply.md`, `../results/output_arm_admet.json`). Two findings: (1) on the 8B 3-arm anchors (hERG/variant/protein) solo does NOT transfer (output near chance regardless of internal encoding = expression gap), and the solo-to-orchestrate headroom decomposes into an expression part (train the read-out, WS3-weights) and an encoding part (orchestrate the specialist) per task = the T2 routing rule and WS3 seed. (2) The frontier ADMET sweep (sonnet-4-6, all 7 endpoints) shows solo NAMED-property recall is ENDPOINT-DEPENDENT, not uniformly chance: oriented AUROC permeability 0.72 / solubility 0.65 / hERG 0.63 / CYP3A4 0.61 / CYP2D6 0.61 read above chance (binned; within-band order not resolved at +/-0.07 CI; 2 of the 5 depend on the label-direction flip), AMES 0.38 (below chance), clearance 0.49 (chance). This is named-property recall (the property is in the prompt), an upper bound on grounding; the scrambled control was since run (`../results/notation_control.md`, CYP3A4 n=1000 + CYP2D6): structure-dependent and notation-invariant confirmed, property-specificity inconclusive (lipophilicity confound). Orchestration still wins for max accuracy (ceiling > solo everywhere). Two methods lessons baked in: a parsing fallback bug (bare prompt -> up to 96% unparsed reads as false "chance"; system-msg fix) and a label-direction resolution (NegBioDB label-1=fail, solubility/permeability inverted, confirmed against assay values). PROPOSE done (`../results/t2_propose.md`): the model generates valid pharmacophore-flavored molecules (93-100% valid SMILES) but the WS2 probe judges them weakly active (mean P 0.14-0.18 vs 0.72 ref); the load-bearing caveat is the narrow probe is OOD on de novo scaffolds, so PROPOSE grounding is undetermined and needs an orchestrated off-distribution verifier (WS3 signal: the bottleneck is the verifier, not the generator). EVALUATE still blocked on the D human-rater pass (package built, see D below). [SOLVE+PROPOSE done; EVALUATE needs D]

### C. Content-sensitivity completion
- All four conditions (matched / mismatched / scrambled / content-only) + re-notation on every modality (currently: SMILES has mismatched + randomized, protein has scramble + mismatch, variant has dual-form + gene-scramble).
- **protein:** composition-preserving shuffle + motif-targeted disruption (the protein README's stated next step; isolates whether the small output lift is just a composition prior). [light-medium]

### D. Law quantification (P1/P2, the headline)
- **P1 regression: DIAGNOSED and reframed (`results/p1_webexposure.md`, v2 `eval/p1_webexposure.py`).** The v1 PMC proxy FAILED because it was a covariate mis-specification, now shown live: the meta-description query `"amino acid sequence" AND pathogenic` returns 143,853 (prose ABOUT sequences) while the form-instance query (a literal sequence window AND pathogenic) returns **0**, so v1 ranked the web-POOREST form as the most exposed and inverted the sign. No cross-modality regression should be fit (web-poor forms not faithfully countable + ceiling-confound 0.70-0.96 + n=5). The valid covariate is the **within-entity notation contrast, ceiling held fixed** (variant text 0.795 > seq 0.740 at one 0.962 ceiling under gene GroupKFold; SMILES canonical ~ randomized floor), which confirms the predicted direction. P1 stands as a within-entity qualitative law plus the regime spectrum. Do NOT claim a fitted P1 law; the reviewer objection is now preempted, not open. [a corpus-scale form-specific covariate remains the open problem]
- **Ladder fill:** DNA/RNA (Evo/NT ceiling, full 3-arm), molecular image (ImageMol ceiling, 2-arm VLM), spectra (SpecTUS, 2-arm), to span the spectrum. [DNA heavy; image/spectra light]
- **P2:** confirm the bottleneck-shift fix (read-out training closes an expression-limited gap, not an encoding-limited one) = WS3.

### E. Methodology hardening
- **selectivity control on every modality: DONE on all rungs (Expanse H100 run, `results/expanse_logs/`).** Protein (the previously-missing rung) measured: structure-probe +0.193, activation **+0.114** (positive, real ESM-grounded signal, lower than the others consistent with encoding-weak). SMILES structure +0.331 / activation +0.301, variant text +0.301 / seq +0.218. See `results/confound_controls.md`.
- **best-layer selection-bias: MEASURED (`results/selection_bias.md`).** Nested-CV held-out-layer protocol in `eval/activation_arm.py:heldout_layer_auroc` was RUN (Expanse H100): measured bias SMILES +0.007, variant-text +0.003, variant-seq +0.012, protein +0.034 (largest, an early-layer spike, exactly as the prior bound predicted). The expression gap is layer-selection-immune and stays large under the held-out estimate; no regime label flips.
- **supervision-asymmetry control: DONE (SMILES/hERG).** 8B few-shot K=10 output = 0.493, unchanged from zero-shot 0.453 and far below activation 0.787 (`eval/fewshot_output.py`, job 3027033), so the probe advantage is not supervision. Extend to other modalities/endpoints if a reviewer asks.
- **input-asymmetry control: DONE for SMILES (`results/confound_controls.md`, `eval/input_asymmetry.py`).** A raw-text char-n-gram probe (no chemistry, no SFM) on the SMILES string scores 0.801 vs LLM output 0.453 (+0.348), nearly matching the Morgan ceiling 0.834. The signal is decodable from the raw string the LLM reads, so the gap is expression, not an SFM input advantage. The complementary orchestrate form (feed the SFM embedding INTO the LLM) needs an adapter and routes to WS3.
- bootstrap CIs everywhere (mostly done).

### F. Model axis
- more open activation models (DeepSeek-V4, Llama-3, Mistral) for cross-arch breadth.
- frontier output cross-vendor (GPT / Gemini) where not done.
- base vs instruct on a second modality (only SMILES has it).

### G. Scale tail
- Qwen3-235B (4-bit, h100) to extend 8B -> 32B -> 235B on the activation arm.

## Priority read

- **Overall** (findings + vision, not exhaustive): the eval is sketch-in-places but the LAW is demonstrated, which is sufficient. The robustness pass is now DONE end-to-end: **D P1** is diagnosed and reframed as a within-entity qualitative law (`results/p1_webexposure.md`, not a fitted regression, which would have been the overclaim); **E** is fully closed with measured numbers (Expanse H100 run, `results/expanse_logs/`, summarized in `results/confound_controls.md` and `results/selection_bias.md`): best-layer selection bias measured (+0.003 to +0.034, protein largest as predicted), protein selectivity measured (+0.114, the last open rung), input-asymmetry controlled for SMILES (raw-text probe 0.801 vs output 0.453). No confound on the encoding claim remains open.
- **For a full research program** (post-join): **A** (modality depth, esp. protein) + **B** (T2 SOLVE now done, `../results/t2_apply.md`; its route decomposition hands WS3 its first decision-map points) + **D ladder fill** are the systematic completion. WS3 (placement) is the natural home for T2-propose/P2, and is now seeded by the T2 routing rule.
