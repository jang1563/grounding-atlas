# T2 (apply) PROPOSE: generate a molecule with a property, probe-judged

*Results section. 2026-06-10. Instrument: `eval/t2_propose.py` (rdkit + sklearn + anthropic, Python 3.13). No em dashes.*

## What this is

T2-solve asks the model to READ a property off a structure (`t2_apply.md` R3, `output_arm_admet.json`); PROPOSE is the generative dual (`eval/README.md` Bridge to T2): ask the model to GENERATE molecules that have the property, then judge each with the WS2 specialist probe (a Morgan-fingerprint LogisticRegression trained on the endpoint's matched data, label-1 = inhibitor). Deterministic judge, no LLM-judge. Endpoints are CYP3A4 / CYP2D6 inhibition, where label-1 = inhibitor needs no direction flip and proposing an enzyme inhibitor carries no acute-toxicity refusal confound (unlike hERG or Ames).

## Result (claude-sonnet-4-6, K=15 asked per endpoint)

| endpoint | returned | valid SMILES | unique | proposal mean P(active) | proposals scored active (P>0.5) | ref P on real actives | ref P on real inactives |
|---|---|---|---|---|---|---|---|
| CYP3A4 | 15 | 14 | 14 | 0.178 | 2/14 | 0.763 | 0.041 |
| CYP2D6 | 15 | 15 | 15 | 0.137 | 1/15 | 0.715 | 0.036 |

K=15 per endpoint is tiny (the "active-rate" is 2 and 1 molecules, so it is shown as raw counts, not a rate); this is a feasibility probe, not a measurement. Two separable readings:

- **Generation competence is high.** 93 to 100 percent of returned strings are valid, unique (canonical-dedup) SMILES, de novo (not memorized named drugs like ketoconazole / ritonavir), and qualitatively carry target-relevant motifs (CYP3A4: azole / amide-piperazine; CYP2D6: basic-amine plus aromatic). The motif claim is eyeballed, not a quantified enrichment vs a random-drug baseline.
- **Grounded activity of the proposals is NOT demonstrated by this judge.** Proposal mean P(active) is 0.14 to 0.18, well below the probe's reference on real actives (0.72 to 0.76) and only a few-fold above its floor on real inactives (0.04). By the specialist probe, the proposals do not strongly carry the inhibition signal.

## The load-bearing caveat: the judge is narrow and the proposals are out of its distribution

The probe is a Morgan-fingerprint classifier trained on one ChEMBL CYP distribution; it judges similarity to the structural features of THOSE actives. The proposals are de novo scaffolds, out of that distribution, so a genuinely active novel molecule would also score low (false negative). The probe references are also resubstitution (train equals test), so 0.72 to 0.76 on real actives is an optimistic ceiling. Therefore 0.14 to 0.18 is a LOWER BOUND on the proposals' true activity, not a calibrated estimate: it shows the proposals do not match the learned actives' fingerprints, which conflates genuine inactivity with novel-scaffold OOD. Settling it needs a judge that generalizes off-distribution (docking, a QSAR model, or assay), which is itself an orchestrate-a-specialist step (WS3).

## Reading for the project

Generation competence (valid, pharmacophore-aware molecules) is real and is a different capability from T2-solve reading. But whether the proposals are genuinely active cannot be decided by the cheap in-distribution probe, so PROPOSE grounding is undetermined here, and the honest conclusion is that T2-propose needs an orchestrated off-distribution verifier rather than a solo answer. This is itself a decision-map (WS3) data point: the propose capability's bottleneck is the VERIFIER, not the generator.

## Caveats

Single frontier model, K=15 per endpoint (small), two endpoints, resubstitution probe references (optimistic), and the OOD limitation above (the central one). No docking or assay confirmation. Toxicity endpoints (hERG, Ames) were excluded to avoid a refusal confound, so this does not test whether the model refuses to propose hazardous actives (an axis-E-adjacent question).

## Reproduce

`python eval/t2_propose.py` (rdkit + sklearn + anthropic; ANTHROPIC_API_KEY from API keys are read from the environment. Raw proposals and scores in `results/t2_propose.json`.
