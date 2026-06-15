# WS3 experiment #4 design: a non-fingerprint-local endpoint (break the decision-map circularity)

*Design doc. 2026-06-11. Status: ENDPOINT RESOLVED to drug market-WITHDRAWAL (done); step-3 generalization gating sweep showed fingerprint-weak knowledge endpoints are RARE. Addresses the deep review's sharpest structural objection to the decision map. No em dashes.*

## Gating sweep for a SECOND knowledge endpoint (step 3): fingerprint-weak knowledge endpoints are rare

To generalize the knowledge gap beyond withdrawal, four endpoints were Morgan-gated (balanced, scaffold GroupKFold). Only withdrawal is fingerprint-WEAK; the rest have structural signal and would PASS the WS2 gate, so they cannot show a clean knowledge gap (a structure specialist already works):

| endpoint | Morgan probe | fingerprint-weak? | why |
|---|---|---|---|
| drug market-withdrawal | 0.606 | YES | causes heterogeneous (cardio/hepato/abuse/commercial/manufacturing), no single structural class |
| TDC DILI (hepatotoxicity) | 0.796 | no | curated mechanistic-tox subset is structurally separable |
| teratogenicity / reprotox (lit.) | 0.85 to 0.88 | no | developmental-tox QSAR works |
| controlled substance (DEA schedule I to V, Wikipedia join, n=402) | 0.878 (k-NN 0.833) | no | scheduled by pharmacology = shared scaffolds (opioids, benzodiazepines, amphetamines, barbiturates) |

The pattern is itself a finding and a BOUNDARY on the knowledge gap: an LLM's drug knowledge and a fingerprint both derive from the molecule's pharmacology, so almost any pharmacology-linked property is structure-tractable and the knowledge route has no edge over a cheap specialist. The knowledge gap (LLM-name beats structure, activation encodes recognition beyond structure) appears only where the property is DECOUPLED from pharmacological structure, i.e. driven by heterogeneous or non-pharmacological causes. Withdrawal qualifies (its causes include commercial and manufacturing reasons); DILI/teratogenicity/scheduling do not. The one remaining untested candidate with plausibly heterogeneous causes is the FDA black-box warning (same FDA-safety axis as withdrawal, data in OnSIDES = heavy). So a clean SECOND endpoint is genuinely scarce; the four-endpoint sweep characterizes WHEN the gap appears, which is the honest generalization.

## RESOLUTION (2026-06-11): endpoint = drug market-withdrawal (safety)

After TDC DILI failed the gating check (Morgan 0.796, not weak; numeric IDs, no names), the purely-clinical route was chosen. Drug market-WITHDRAWAL is the clean circularity-breaker: why a drug was pulled from market (cardiotox, hepatotox, abuse, idiosyncratic ADR, commercial) is a clinical/historical fact with little local-substructure basis. GATING CHECK PASSED on the WITHDRAWN / DrugBank / ChEMBL / NCATS agreement set (Mazuz et al., `eyalmazuz/DrugWithdrawn`, columns name + smiles + withdrawn_class, 5979 unique molecules, 1685 withdrawn): balanced scaffold-CV **Morgan probe = 0.606** (0.662 on the 500 sub-sample), **no-LLM Tanimoto k-NN = 0.643** (0.544 on the sub-sample). Both WEAK, both would FAIL the WS2 gate, so no cheap structure specialist exists. The set has real drug NAMES (the famous withdrawals are all present: rofecoxib/Vioxx, cerivastatin/Baycol, terfenadine, cisapride, thalidomide, troglitazone, astemizole, sibutramine), so the LLM-name arm is possible, and a frontier smoke test already returns rofecoxib 0.99 / aspirin 0.05. Implementation: `eval/withdrawn_endpoint.py` (arms 1 to 4 + fake-name control), `eval/run_activation_withdrawn_cayuga.sh` (arm 5, Cayuga job 3038493), data `signal/withdrawn/withdrawn.csv`, results `results/withdrawn_endpoint.{md,json}`.

## The problem this breaks

The decision map (`decision_map/DECISION_MAP.md`) currently says, in every measured corner, "do not use the LLM": for hERG-SMILES a cheap Morgan probe (0.825) and a trivial neighbor k-NN (0.951) dominate every LLM placement (solo 0.45 to 0.63, retrieve 0.84, weights read-out 0.787, LoRA 0.856). The deep review caught the circularity: **WS2's ceiling-gate admits an endpoint only if a fingerprint probe clears a ceiling threshold**, so every studied endpoint is fingerprint-friendly BY CONSTRUCTION, so a cheap structure specialist always exists, so the LLM always loses. The verdict is foreordained by the gate, not discovered. As a forward-build deliverable the map illustrates a routing method but contains no cell where the LLM is actually the right tool, which makes "route to the LLM" vacuous.

To break it we need a cell where (a) NO cheap structure specialist exists (fingerprint probe AND k-NN both weak) and (b) the LLM has a DIFFERENT route to the answer. The LLM's route is not local substructure (the surface-decodability result shows the LLM matches but does not beat a char-n-gram on structure tasks); it is WORLD KNOWLEDGE about named entities. So the cell to look for is a fingerprint-weak, knowledge-heavy property on RECOGNIZED molecules.

## Endpoint: DILI (drug-induced liver injury)

Rationale, point by point against the two requirements:
- **Fingerprint-weak (no cheap specialist).** DILI is idiosyncratic, mediated by dose, reactive-metabolite formation, mitochondrial and immune mechanisms; it is the textbook endpoint where local-substructure QSAR fails. A Morgan-FP scaffold-CV probe was expected near 0.60. GATING CHECK RAN 2026-06-11 on TDC `Tox(name='DILI')` (n=475, balanced 239/236, 345 scaffolds): **Morgan probe scaffold-CV AUROC = 0.796 (AUPRC 0.787)**. This DISQUALIFIES TDC DILI: 0.796 is NOT weak (it would PASS the WS2 gate), so this dataset is fingerprint-tractable and cannot be the circularity-breaker. ALSO its `Drug_ID` is a numeric code, not a drug name, so the LLM-name knowledge arm is impossible here. TWO independent reasons to switch the dataset. (The 0.796 likely reflects that the Xu-2015 curated TDC subset is structurally separable and that 345 scaffolds for 475 molecules makes the scaffold split nearly singleton-per-group, leaking structure.)
- **Knowledge-heavy (the LLM has a route).** Hepatotoxicity of marketed drugs is exhaustively documented (LiverTox, FDA black-box labels, withdrawal history, decades of literature). A frontier LLM has read this. The answer lives in text about the named drug, not in its substructure.
- **Named, recognizable molecules.** TDC `Tox(name='DILI')` ships `Drug_ID` = the drug name (e.g. Acetaminophen), so the LLM-name arm needs no separate name mapping, and these are exactly the marketed drugs the model is likely to recognize (web-exposure law).
- **Public + programmatic.** TDC single_pred Tox DILI (~475 drugs, binary). Fallback if unsuitable: FDA DILIrank (~1000 drugs, 4-class severity, has names) binarized Most+Less vs No.

## Arms (one balanced DILI set, one Murcko-scaffold split)

Structure route (SMILES only, the structure tools' natural input):
1. **Morgan-FP probe**, scaffold GroupKFold. The cheap specialist. Predicted WEAK (~0.6).
2. **No-LLM k-NN** neighbor-mean (the mandatory baseline from the retrieve lesson). Predicted WEAK (a fingerprint-weak endpoint has no informative neighbors).

LLM routes:
3. **LLM-output, SMILES only** (matched input to the structure tools). Predicted WEAK. This is the HARD test (#4a): can the LLM read DILI from structure where fingerprints cannot? The char-n-gram / surface-decodability result predicts NO (LLM approx substring approx Morgan).
4. **LLM-output, NAME given** (the knowledge route, the LLM's natural input for a known drug). Predicted STRONG. This is the circularity-breaker (#4b): the model recalls documented hepatotoxicity.
5. **LLM-activation probe, SMILES** (encoding side, Qwen3-8B). Predicted WEAK, as a consistency check: on a fingerprint-weak endpoint there is little structural signal to encode, so unlike hERG there should be no large expression gap from structure. The DILI signal is recalled from the name, not encoded from the string.

Models: arms 3 and 4 use a FRONTIER model (sonnet-class) because the test is about world knowledge; reuse the `output_arm_admet.py` system-message protocol with two input conditions (name vs SMILES). Arm 5 uses Qwen3-8B (`activation_arm.py` with the DILI csv). Arms 1 and 2 are local CPU.

## The decisive comparisons and the decision-map cell it writes

- **Circularity broken iff arm 4 (LLM-name) > max(arm 1 Morgan, arm 2 k-NN)** by more than the bootstrap CI. If so, the map gains its first genuine "use the LLM" cell.
- **Honest decomposition: arm 4 >> arm 3.** The win is NAME-KNOWLEDGE, not structure-reading, consistent with the web-exposure law and the surface-decodable finding. So the cell is precisely "recognized drug + knowledge-dependent, fingerprint-weak endpoint -> LLM (name route)."
- **Sharp boundary (keeps it honest):** arm 4 needs recognition. For a NOVEL molecule (no known name) arm 4 is unavailable and arms 1 to 3 are all weak, so that corner routes to "generate assay data," not the LLM. The map gets both a real LLM-win cell and its precise limit.

Predicted decision-map row for DILI:

| molecule type | best tool | why |
|---|---|---|
| recognized marketed drug | LLM (name route) | documented hepatotox recalled; structure tools weak |
| novel / unnamed scaffold | none strong, get assay data | fingerprint weak, k-NN weak, LLM-SMILES weak, no name to recall |

## Controls and confounds

1. **Knowledge vs label-memorization.** Arm 4 recalling the documented DILI label IS the knowledge route working; that is the point. Characterize, do not "fix": it is zero-shot (all test drugs held out), and optionally probe the MECHANISM on a sample to confirm it is not blind parroting. Scope the claim as "knowledge recall on documented drugs."
2. **DILI-specific vs generic recognition.** To show arm 4 reads DILI-specific knowledge and not just "this is a drug," add a FAKE-NAME control (swap each SMILES to a random real drug name from the set): if the true-name arm carries DILI signal and the fake-name arm collapses to chance, the signal is drug-specific, not a generic drug prior. (Mirrors the fake-accession control in `axis_a_dna.md`.)
3. **Scaffold leakage** on the structure probe: Murcko-scaffold GroupKFold, as in `activation_arm.py`.
4. **Fingerprint-weakness must be real** (the gating check): Morgan probe AND k-NN both near chance, else DILI is not a circularity-breaker and we switch endpoints.
5. **Class imbalance + small n (~475):** balance the sample or report AUPRC alongside AUROC; report bootstrap CIs; the win must clear the CI.

## Implementation plan

- `eval/fetch_dili.py` (cluster login node, has internet): TDC `Tox(name='DILI')` -> `signal/dili/dili.csv` with columns name, smiles, label; print balance + n.
- `eval/nonfp_endpoint.py` (local CPU for arms 1 to 2, frontier API for arms 3 to 4): Morgan probe + k-NN + LLM-name + LLM-SMILES + fake-name control, bootstrap CIs, JSON out. Reuses `lipophilicity_control.py` (probe/scaffold) and `output_arm_admet.py` (frontier protocol) patterns.
- Arm 5 activation: `activation_arm.py ACT_CSV=dili.csv` on Cayuga a40 (same module-init fix as `run_activation_randomize_cayuga.sh`).
- `results/nonfp_endpoint.md` + a DILI row appended to `decision_map/DECISION_MAP.md`.

## Open decisions

- Endpoint: DILI recommended; confirm the gating Morgan-weakness number first. If Morgan is not weak on TDC DILI (say > 0.7), switch to DILIrank or to a drug-withdrawal / black-box-warning label (even more clearly knowledge-only).
- Frontier model + budget for arms 3 to 4 (475 drugs x 2 conditions x 1 call, small).
