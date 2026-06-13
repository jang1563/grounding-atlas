# Probe vs LLM: encoding vs expression in hERG grounding

*Results section. 2026-06. Instruments: `eval/activation_arm.py` (open-weight 3-arm), `eval/head_to_head.py` (Claude output + content-sensitivity). Data: hERG and CYP3A4 from ChEMBL 36 via NegBioDB. No em dashes.*

## Setup

Three arms on ONE balanced sample under ONE leakage-controlled split:
- **structure-probe** = Morgan fingerprint (radius 2, 2048 bit) + logistic regression = the ceiling (is the signal in the content at all?)
- **LLM-activation** = per-layer linear probe on hidden states (does the LLM ENCODE it internally?)
- **LLM-output** = generate, asked for a single probability (does the LLM VERBALIZE it?)

Encoding gap = ceiling - activation; expression gap = activation - output. Balanced 625/625 (n=1250), Murcko-scaffold GroupKFold(5). Output parser anchored and instrumented (parsed / percent / fallback). The probe is defended by a shuffled-label selectivity control.

## R1. Structure carries the signal; the LLM does not surface it (readout gap, prior art)

The fingerprint reads hERG from SMILES (ceiling 0.825 on the balanced same-set; 0.91 on the full set, see `ceiling_gate.md`). The LLM output stays near chance across vendors:

| output arm | n | AUROC |
|---|---|---|
| Claude haiku-4-5 | 100 | 0.510 |
| Claude sonnet-4-5 | 200 | 0.573 |
| Claude sonnet-4-6 | 100 | 0.581 |
| Claude opus-4-5 | 100 | 0.557 |
| Claude opus-4-8 | 100 | 0.566 |
| open-weight panel (see R2) | 1250 | 0.45-0.52 |

This O1 readout gap is PRIOR ART (GPT-4o SMILES-only ROC-AUC 0.4991, 2501.13824; fingerprint beats LLM-features on scaffold splits, 2402.00024); scope it to zero-shot, no fine-tune (fine-tuned GPT-3 is competitive, Jablonka NMI 2024). Content-sensitivity: swapping a molecule for a valid opposite-label one moves the answer by only mean|delta| 0.085, and re-notating the same molecule (canonical 0.573 vs a randomized-but-valid SMILES 0.553, n=200) barely changes the AUROC, so the LLM answers near-uniformly (around 0.75) regardless of structure or notation. The SMILES notation-robustness here is a FLOOR effect (no surfaced signal to perturb), and is the OPPOSITE of the variant case, where a real output signal is notation-bound (text 0.79 vs raw sequence 0.58): notation-sensitivity is measurable only where grounding actually surfaces, so the variant branch carries the within-entity invariance result and SMILES marks the floor. O2 (fresh hook): the LLM flags invalid / scrambled SMILES (empty or refuse) yet cannot read a valid molecule's property, a validity-known-but-property-unknown dissociation (nearest 2505.16340).

## R2. The LLM encodes it internally: the bottleneck is expression, not encoding

The decisive arm. Same 1250-molecule set, same scaffold split, all three arms:

| model | arch | activation (best layer) | output | encoding gap | expression gap |
|---|---|---|---|---|---|
| Qwen3-8B (instruct) | Qwen3 | 0.787 (L2) | 0.453 | 0.038 | **0.334** |
| Qwen3-32B (instruct) | Qwen3 | 0.806 (L46) | 0.479 | 0.019 | **0.327** |
| Qwen3-8B-Base | Qwen3 | 0.756 (L2) | 0.476 | 0.069 | 0.280 |
| Phi-4 (instruct) | Phi | 0.795 (L17) | 0.521 | 0.030 | **0.274** |
| gpt-oss-20b | gpt-oss | 0.771 (L3) | 0.506 | 0.054 | **0.265** |

*(structure-probe = 0.825 for every row, same balanced set. Parse counts: instruct Qwen3/Phi-4 = 1250 parsed / 0 fallback; base = 777 parsed / 455 fallback; gpt-oss at 12 tokens = 0 parsed (unmeasurable), re-run at 512 tokens = 905 parsed / 345 percent / 0 fallback giving 0.506. Bootstrap CIs in Cayuga logs.)*

- **Expression-dominant.** Every model encodes hERG to near the structural ceiling (activation 0.76-0.81 vs 0.825, encoding gap only 0.02-0.07) yet outputs at chance (0.45-0.52). A general LLM carries near-fingerprint-level structural signal internally and does not surface it.
- **Selectivity control (probe defense 1).** A shuffled-label probe sits at chance (8B hERG: structure 0.505, activation 0.529), giving selectivity 0.320 / 0.258 (Hewitt-Liang 1909.03368). The activation signal is real linear-decodable hERG structure, not the probe overfitting 1250 labels.
- **Supervision-asymmetry control (probe defense 2).** The probe is trained on labels while the output arm is zero-shot, an unfair comparison. Handing the 8B model K=10 in-context labeled examples (few-shot) leaves output at 0.493, essentially unchanged from zero-shot 0.453 and far below the activation probe 0.787 (eval/fewshot_output.py). So the probe advantage is NOT supervision: given labeled examples the model still cannot verbalize hERG. The expression gap is real, not a trained-vs-zero-shot artifact.
- **Layer pattern.** Chance at layer 0, jumps by layer 1-2, then flat 0.73-0.81 across the stack: the signal forms early and persists, it does not wash out toward the output.

## R3. The gap is invariant to five interventions

None of these close the expression gap:

| intervention | result | reading |
|---|---|---|
| scale (8B -> 32B, 4x) | activation +0.019, output +0.026, gap 0.334 -> 0.327 | not a capacity problem (matches Claude 0.51-0.58 flat) |
| architecture (Qwen3 / Phi / gpt-oss) | activation all 0.76-0.81 | not vendor-specific (universality) |
| endpoint (hERG -> CYP3A4) | 0.745 / 0.684 / 0.502, gap structure holds | sizes scale with the endpoint's ceiling |
| alignment (base vs instruct) | encoding 0.756 vs 0.787, both output chance | alignment teaches format (fallback 36% -> 0%), not content |
| reasoning (thinking on, 512 CoT) | 8B 0.479 vs 0.453, 32B 0.467 vs 0.479, gpt-oss 512tok 0.506: no recovery | not a "did not reason enough" problem |

The gap is not a capacity, architecture, endpoint, alignment, or reasoning-effort phenomenon. It is structural.

## R4. Mechanism and the lever

Why the probe beats the output ("LLMs know more than they say", here generalized to a biological property):

- **Two different questions.** The probe asks "is the signal present in the hidden state?" with a supervised readout; the output asks "does the model spontaneously route that signal into its answer?" with no supervision. Presence is not use.
- **Three places the output stalls.** (1) Calibration is never trained: no model card (Qwen3, Phi-4, gpt-oss) optimizes calibrated probability output, so the hidden signal is not mapped to a number. (2) The signal is lost during decoding (2403.09037: hidden knowledge lost in later-token logits; finetuning improves output yet stays below a linear probe). (3) Some models are simply not factual (gpt-oss SimpleQA 0.914 hallucination / 0.067 accuracy, yet activation 0.771).
- **The lever is training the read-out.** 2602.07812 (the numbers analog: probe >90% vs verbalized 50-70%) recovers +3.22% with a probe-loss auxiliary; inference-time levers (scale, alignment, thinking, prompting) do not. CLUE (2510.01591): late layers align with calibrated logits and token-probability methods fail on less-calibrated models.
- **WS3 placement signal.** An expression-limited capability belongs in WEIGHTS (train the read-out), not in prompting or retrieval. The 3-arm decomposition is what tells you which regime you are in.

## R5. Cross-modality prediction: the web-exposure law

Generalizing 2504.12459 (a linear representation of a fact forms once its content-property co-occurrence crosses roughly 1-2k in pretraining, and representation quality predicts pretraining frequency) from text triples to scientific modalities: a modality's internal ENCODING strength tracks how often its "content -> property" form appears in web text. Predicted ladder (`PROJECT_DESIGN.md` section 7), smallest encoding gap first: variant / HGVS text (web-rich, output likely nonzero) < DNA motif < protein sequence < SMILES (this anchor, expression-dominant) < molecular image (encoding-limited, perception floor). Two falsifiable predictions: **P1 monotonicity** (encoding gap decreases as web co-occurrence increases) and **P2 bottleneck shift** (SMILES/sequence are expression-limited and closeable by read-out training; images are encoding-limited). White space: no prior work runs a single controlled 3-arm instrument across the modality spectrum and regresses the gap on web-exposure.

## Caveats

Zero-shot; balanced sample (AUPRC baseline ~0.5); chemistry only so far (hERG + CYP3A4). Best-layer activation is max-over-layers (selection-biased) but the selectivity control shows the signal is non-trivial. The probe is a claim of linear decodability, defended as real by the control task, not a claim of conscious "knowledge". The activation panel is open-weight; the Claude arm is output-only (no hidden states available).

**Interpretation caveats (added 2026-06-11 after a deep adversarial review; these qualify the word "encodes").** The expression-gap ARITHMETIC (probe ~0.79 >> output ~0.45, invariant to scale/architecture/alignment/reasoning, selectivity- and held-out-layer-controlled) is robust. What "encodes" MEANS is softer and is NOT yet settled by these controls:
- **Surface-string vs chemistry (RESOLVED 2026-06-11; `lipophilicity_control.md` + activation re-notation run, log `act_rand_3038486.log`).** A char-n-gram probe with NO chemistry scores 0.845 on the canonical SMILES, near the activation probe (0.787), and SURVIVES re-notation at 0.812 on randomized SMILES (down only 0.033), so the property is robustly decodable from surface substrings on any notation. We then ran the decisive test on the LLM hidden states directly: the ACTIVATION probe on randomized SMILES of the same hERG molecules scores 0.739 max (held-out-layer 0.732), down from canonical 0.787 max / 0.760 held-out. Two findings: (i) the activation signal SURVIVES re-notation (0.739 >> chance), so it is NOT pure canonical-string orthography, there is a notation-invariant structural component; (ii) but the surviving signal (0.739) is BELOW the no-chemistry char-n-gram on randomized (0.812), so the LLM hidden states do NOT exceed a surface substring probe even after randomization. Tellingly, the best layer MOVES from 2 (canonical, very early = surface feature) to 20 (randomized, deeper), i.e. the early-layer canonical signal had an orthographic component that dropped on randomization while a deeper notation-invariant structural component remained. Net: the encoded signal is structural and notation-invariant but does not beat a substring probe, so "encodes chemistry" is not supported over "linearly decodable structural signal." Softened framing confirmed by direct measurement, not just the char-n-gram proxy.
- **hERG-specific vs a single lipophilicity axis (REFUTED, 2026-06-11; `lipophilicity_control.md`).** Ran the residualization test on the structure probe: logP/MW/TPSA alone give only 0.675, and the Morgan probe RESIDUALIZED on those three descriptors still gives 0.768. So most of the structure signal survives lipophilicity removal, the hERG signal is genuine structure not a coarse lipophilicity axis, and the activation probe at 0.787 sits right at this residual-structure level. The "it is just lipophilicity" worry is refuted for the hERG anchor. (The earlier CONFOUNDED verdict was for CYP3A4 on the OUTPUT arm and does not transfer.) STILL OPEN: residualize the ACTIVATION probe features themselves on GPU to confirm the LLM hidden-state signal also survives.
- **Few-shot control (RESOLVED 2026-06-11; log `fewshot_3038487.log`).** The first supervision control (0.493) was run with a degenerate prompt (the balanced query consumed all 625 hERG positives, so the few-shot examples were 5 negatives + 0 positives). `eval/fewshot_output.py` was fixed to reserve the few-shot pool first, and re-run: balanced K=10 in-context examples give FEW-SHOT OUTPUT AUROC = 0.478 (n=1240, all parsed), still near chance and barely above zero-shot 0.453. So given labeled examples the model still cannot verbalize hERG, the probe advantage (0.787) is NOT just supervision, and the expression gap is real, not a trained-vs-zero-shot artifact. (Same conclusion as the buggy run, now valid.)
- **Gate-conditioning.** WS2 admits only endpoints that pass a fingerprint-probe ceiling gate, so the studied properties are pre-selected to be fingerprint-friendly (the corner where a cheap specialist works and the LLM loses). Honest scope: "fingerprint-tractable ADMET," not "biology."
Net (updated 2026-06-11 after the controls completed, local + GPU): the durable, fingerprint-irreducible claim is the expression GAP and its invariance (a readout-vs-verbalization decomposition of the model), not "the model represents the chemistry." All three follow-up controls are now run and CONVERGE: (1) the structure signal is genuine structure, not just lipophilicity (Morgan residualized on logP/MW/TPSA = 0.768); (2) the signal is robustly surface-decodable (a no-chemistry char-n-gram gets 0.812 on randomized SMILES); (3) the LLM ACTIVATION probe on randomized SMILES survives at 0.739 (held-out 0.732) but stays BELOW the char-n-gram, and its best layer shifts from 2 to 20, so the hidden states hold a notation-invariant structural signal that does not exceed a substring probe; (4) the fixed few-shot output is 0.478 (near chance), so the gap is not a supervision artifact. So "encodes chemistry" stays softened to "the property is linearly decodable from the hidden states, as it is from the raw string; the decodable signal is structural and notation-invariant but does not exceed what a surface substring probe achieves." The expression gap itself is fully intact and now triangulated from four directions.

## The randomization control separates content-grounding (axis B) from entity-recognition (axis A) (2026-06-11, `withdrawn_endpoint.md` + `layer_profiles.md`)

The hERG read-out >> verbalization gap above is axis-B CONTENT-GROUNDING: the property is encoded FROM the SMILES structure, notation-invariant (the activation survives randomized SMILES at 0.739), so the probe reads the structural content the model cannot verbalize. A second endpoint, drug market-withdrawal, produces a probe-vs-output gap of the same shape (8B activation 0.762 >> output 0.469, also above the Morgan fingerprint 0.643) but it is NOT the same thing: the randomized-SMILES control drops the activation to the Morgan level (0.662), so its above-structure component is canonical-string-keyed and breaks under re-notation. That is the signature of axis-A ENTITY-RECOGNITION plus fact-recall (the model identifies the known drug from its canonical SMILES, a resolvable identity token, then recalls its documented withdrawal status), not of content read from structure. Per drug, the 8B activation aligns with the frontier model's name-route knowledge (Spearman 0.273, vs 0.035 with the structure route) and the 8B output is uncorrelated (-0.015): the small model encodes the recalled fact and cannot say it.

So the contribution here is methodological, and it sharpens the core rather than adding a second kind of gap: the activation probe picks up BOTH axis-B content-grounding and axis-A recognition-recall, and the re-notation control is what TELLS THEM APART (content survives randomization, recognition does not). A probe-vs-output gap only counts as content-grounding if it survives re-notation, as hERG does; withdrawal shows the same arithmetic arising from recognition, which the control unmasks. (This is the re-notation content-sensitivity test of the plan, `docs/FAILURE_MODES.md` cross-cutting representation-invariance, used as an axis-A-vs-B discriminator.) Separately, withdrawal is the decision map's first cell that routes TO the LLM (LLM-name 0.758 > structure tools, `decision_map/DECISION_MAP.md` third corner), and it routes there via the recognition/recall path, not content-grounding.

## References

Prior art (O1/O2): 2501.13824, 2402.00024, 2505.16340. Mechanism / related work: 2403.09037 (hidden knowledge lost in generation; probe > finetuning), 2510.01591 (CLUE), 2406.15927 (semantic-entropy probes), 2602.07812 (numbers: encode >> express, closest analog), 2504.12459 (frequency -> linear representation, the web-exposure mechanism). Model cards: 2508.10925 (gpt-oss), 2412.08905 (Phi-4), 2505.09388 (Qwen3). Framework: Inside-Out 2503.15299 (encoding vs expression).

## Methods note (superseded run)

An earlier activation run used a random split, mismatched molecule sets across arms, and an untracked parser; it reported probe 0.91 / activation 0.752 / output 0.457 with encoding gap 0.16. It was replaced by the same-set scaffold-split protocol above, under which the encoding gap nearly vanishes (0.02-0.07) and the expression gap is the robust effect. Cayuga job logs retained for audit.
