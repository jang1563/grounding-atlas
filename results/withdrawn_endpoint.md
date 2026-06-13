# WS3 #4: the decision-map circularity-breaker (drug market-withdrawal)

*Results. 2026-06-11. Instrument: `eval/withdrawn_endpoint.py` (structure arms local CPU; LLM arms claude-sonnet-4-6, 500 calls each). The first decision-map cell where the LLM is the right tool. No em dashes.*

## Why this endpoint

The deep review caught that WS2's ceiling-gate admits only fingerprint-friendly endpoints, so every studied corner says "do not use the LLM" by construction, and the map contains no cell where the LLM wins. To break the circularity we need an endpoint where (a) no cheap structure specialist exists and (b) the LLM has another route. Drug market-WITHDRAWAL (why a drug was pulled: cardiotox, hepatotox, abuse, idiosyncratic ADR, commercial) is a clinical/historical fact with little local-substructure basis, and its history is documented for named drugs. Source: the WITHDRAWN / DrugBank / ChEMBL / NCATS agreement set (Mazuz et al., `eyalmazuz/DrugWithdrawn`), 5979 unique molecules, 1685 withdrawn, with drug names.

## Five arms, one balanced sample (n=500, 250 withdrawn / 250 not), AUROC vs withdrawn

| arm | input | AUROC | 95% CI |
|---|---|---|---|
| morgan probe (scaffold CV) | SMILES | 0.662 | 0.615, 0.708 |
| no-LLM k-NN (Tanimoto, scaffold CV) | SMILES | 0.544 | 0.496, 0.591 |
| LLM-output | SMILES | 0.535 | 0.488, 0.582 |
| **LLM-output** | **drug name** | **0.758** | **0.715, 0.795** |
| LLM-output | shuffled (fake) name | 0.503 | 0.458, 0.553 |

(On the full balanced 3370-molecule set the structure arms are Morgan 0.606, k-NN 0.643; the 500-sample numbers above are the matched within-experiment values.)

## Three decisive comparisons, all pass

1. **Circularity broken.** LLM-name 0.758 beats the best structure tool (Morgan 0.662) with NON-overlapping CIs (0.715 > 0.708). For market-withdrawal the LLM (name route) is the best available tool, decisively above the structure specialist and the trivial k-NN. This is the first decision-map cell that routes TO the LLM.
2. **The win is knowledge, not structure.** LLM-name 0.758 vs LLM-SMILES 0.535: given the same molecules but the SMILES instead of the name, the LLM is at chance, exactly like the structure tools. So the LLM cannot read withdrawal from structure either; the advantage comes entirely from recognizing the named drug and recalling its documented history. Consistent with the web-exposure law and the surface-decodability finding (the LLM has no structural edge; its edge is named-entity knowledge).
3. **The knowledge is drug-specific.** LLM-fake-name 0.503 (deranged names, true labels) is exactly chance. So the model is not running a generic "this looks like a withdrawable drug" prior keyed on name plausibility; it is recalling the withdrawal status of the SPECIFIC named drug. (Mirrors the fake-accession control in `axis_a_dna.md`.)

## The decision-map cell it writes

| molecule type, withdrawal endpoint | best tool | AUROC | why |
|---|---|---|---|
| recognized marketed drug | LLM (name route) | 0.758 | documented withdrawal history recalled; no structural basis for structure tools |
| novel / unnamed scaffold | none strong, get assay data | <= 0.66 | Morgan, k-NN, and LLM-from-SMILES all weak; no name to recall |

This is the honest counterpart to the hERG-SMILES corner (where a cheap Morgan/k-NN specialist dominates every LLM placement). The map now has BOTH a "do not use the LLM" cell (fingerprint-tractable property, novel structure) AND a "use the LLM" cell (knowledge-documented property, recognized entity), so its routing verdict is no longer foreordained by the gate.

## Honest caveats

- **Magnitude is modest (0.758, not 0.95).** The name arm includes obscure withdrawn drugs the model does not recognize and salt-form names; the ceiling is the model's withdrawal knowledge, not 1.0. It still clears the structure ceiling with separation.
- **Recognition-bounded.** The win requires the model to recognize the drug. For a genuinely novel molecule there is no name to recall and the cell collapses to the weak-structure corner. The cell is "recognized drug + knowledge endpoint," not "any molecule."
- **Possible dataset exposure.** The model may have seen DrugBank withdrawn tags or the WITHDRAWN database during training. The fake-name control shows it uses the specific real name to recall the status, which is the knowledge route regardless of whether the source was the primary literature or a compiled dataset; for the decision-map purpose (ask the LLM about a known drug) the routing claim holds either way. It does limit the stronger scientific claim "reasons from pharmacological knowledge" vs "recalls a memorized label."
- **Single frontier model, single endpoint.** Demonstrated on sonnet-4-6 and on withdrawal; the cell is illustrative of the class "fingerprint-weak, knowledge-documented, named," not a general law.

## Arm 5 (encoding side): a SURPRISE, not the predicted null (Qwen3-8B, n=2000, job 3038493)

I predicted the activation probe would be weak (~0.6, near Morgan), on the logic that a fingerprint-weak endpoint has little structure to encode. That prediction was WRONG, and the way it is wrong is the most interesting result here:

| arm (8B, same 2000 withdrawal molecules, scaffold split) | AUROC |
|---|---|
| structure-probe (Morgan FP) | 0.643 |
| output (8B verbalized, from SMILES) | 0.469 |
| **activation probe (hidden states, from SMILES)** | **0.762 max / 0.740 held-out-layer** (best layer 27) |

The activation probe (0.762, held-out 0.740) does not just match the structure probe, it EXCEEDS it by ~0.10, and it towers over the model's own output (0.469), a ~0.27 expression gap. So the 8B hidden states, given only the SMILES, encode withdrawal-relevant information BEYOND what Morgan fingerprints capture, and the model cannot verbalize it. Strikingly, this activation level (0.762) lands right at the frontier LLM-NAME level (0.758): the 8B reaches from the SMILES-in-hidden-states what the frontier model reaches from the name-in-context.

The most natural reading is that the model RECOGNIZES the drug from its SMILES and its hidden states then carry what it knows about that drug (including withdrawal status), an internal identity/recall signal it cannot surface as an answer. If so this is axis-A entity-recognition (not the axis-B content-grounding that hERG shows), surfaced through the activation probe; the discriminator below tests exactly that.

### Discriminator (job 3038495, DONE): the above-structure signal is RECOGNITION, not richer structure

The randomized-SMILES activation arm (same 2000 molecules, model sees deterministic re-notations, scaffold split and Morgan on canonical) resolves it:

| 8B arm, withdrawal | canonical SMILES | randomized SMILES |
|---|---|---|
| structure-probe (Morgan, canonical) | 0.643 | 0.643 |
| output | 0.469 | 0.470 |
| activation (max / held-out-layer) | 0.762 / 0.740 | 0.662 / 0.642 |

Under re-notation the activation probe DROPS to the Morgan structure level (0.662 max / 0.642 held-out, vs Morgan 0.643). So the activation signal decomposes cleanly: a notation-invariant STRUCTURE component (~0.64, = Morgan, survives randomization) PLUS a canonical-string-keyed component (~0.10 to 0.12 above Morgan) that DISAPPEARS when the SMILES is re-notated. The above-structure part is exactly the part that made the canonical activation beat Morgan, and it is canonical-string-specific, which is the signature of drug RECOGNITION (the model identifies the known drug from its canonical SMILES string and accesses what it knows about it), not of a richer structural representation. The contrast with hERG is the clincher: the hERG activation signal HELD under randomization (0.787 -> 0.739, it was notation-invariant structure), whereas the withdrawal above-structure signal COLLAPSES (it was recognition). Two endpoints, two mechanisms.

### What withdrawal adds: the control SEPARATES axis-B content-grounding from axis-A recognition

This is a methodological point, not a second kind of expression gap, and it is important to label the axis correctly. hERG is axis-B CONTENT-GROUNDING: probe 0.79 >> output 0.45, and the encoded signal is notation-invariant structure (it survives randomized SMILES at 0.739), so the probe reads the structural content the model will not verbalize. Withdrawal produces the same probe-vs-output arithmetic (activation 0.762 >> output 0.469) but from a DIFFERENT source: the 8B identifies the known drug from its canonical SMILES (a resolvable identity token) and the probe reads the recalled withdrawal fact. The randomized-SMILES control unmasks this, because the above-structure part collapses to the Morgan level under re-notation (0.762 -> 0.662). That is axis-A ENTITY-RECOGNITION plus fact-recall (`docs/FAILURE_MODES.md` A, the name-over-content / identity-token pathway), surfaced through the activation probe, not content read from the structure.

So the lesson sharpens the core axis-B measurement: a probe-vs-output gap counts as content-grounding only if it survives re-notation (hERG), because the same arithmetic can arise from recognition (withdrawal), which the re-notation control exposes. The activation probe captures both axes; the randomization test is the discriminator. (It is the plan's re-notation content-sensitivity condition used as an axis-A-vs-B separator.) Caveat: the randomized control separates content from recognition; within the recognition reading it does not further separate "recalls a memorized fact about drug X" from "reasons from recognized pharmacology," and the dataset-exposure caveat still applies.

### Per-drug agreement: the 8B encodes the same knowledge the frontier verbalizes (`results/peritem_agreement.json`)

Both arms were re-run with per-item dumps and matched by SMILES (intersection n=200 drugs). At the aggregate level the 8B activation (0.79) and the frontier name route (0.796) reach the same AUROC on the shared drugs. Per-drug, a clean 2x2 emerges (Spearman rho):

| pair | rho | reading |
|---|---|---|
| 8B activation vs frontier NAME (knowledge channel) | 0.273 | the 8B hidden states track the frontier's verbalized knowledge |
| 8B activation vs frontier SMILES (structure channel) | 0.035 | and NOT the structure channel |
| 8B activation vs frontier Morgan (structure) | 0.122 | (also low) |
| 8B structure-probe vs frontier Morgan | 0.258 | two structure readers agree at ~0.26 (the baseline) |
| 8B OUTPUT vs frontier NAME | -0.015 | the 8B cannot verbalize the knowledge it encodes |

So the 8B activation aligns with the frontier KNOWLEDGE channel at the same level (0.273) that two structure readers align with each other (0.258), and does NOT align with the structure channel (0.035 to 0.122), while the 8B OUTPUT is uncorrelated with the knowledge (-0.015). This ties the claim down per drug: the small model's hidden states carry the same drug-withdrawal knowledge the frontier model states from the name, and the small model cannot say it. HONEST LIMIT: the absolute correlation is modest (rho ~0.27 at n=200, significant but with a wide CI), so this is directional supporting evidence (knowledge >> structure, output ~0), not a tight per-drug match; two different models with imperfect, partly-overlapping knowledge would not be expected to agree strongly.

## Reproduce

`source ~/.api_keys && WD_N=500 python eval/withdrawn_endpoint.py`. Raw in `results/withdrawn_endpoint.json`. Data `signal/withdrawn/withdrawn.csv`.
