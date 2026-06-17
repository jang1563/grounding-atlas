# Ceiling-gate results (axis-B candidate screening)

*2026-06-08. Local run (CPU, rdkit + sklearn). Script: `eval/ceiling_gate.py`.*

**Question:** is the property predictable from the representation CONTENT (SMILES)? A high supervised ceiling means the signal is in the representation, so a probe-vs-LLM head-to-head is meaningful there. A low ceiling means there is nothing for the LLM to fail to surface.

**Data:** NegBioDB ADMET (`negbiodb_admet.db`, from ChEMBL 36). Compound-level, any-fail -> fail. Morgan fingerprint (radius 2, 2048 bits). Probes: logistic regression + random forest, both class-weight balanced. 5-fold CV, two splits: random and scaffold (Murcko, the leakage control).

| endpoint | n_cpd | %fail | random AUROC (lr/rf) | scaffold AUROC (lr/rf) | scaffold AUPRC (lr/rf) | baseline AUPRC | verdict |
|---|---|---|---|---|---|---|---|
| **hERG** | 3963 | 15.8 | 0.920 / 0.934 | **0.895 / 0.913** | 0.692 / 0.750 | 0.158 | PASS, top candidate |
| **cyp3a4** | 9929 | 14.8 | 0.874 / 0.910 | **0.830 / 0.883** | 0.545 / 0.671 | 0.148 | PASS |
| **cyp2d6** | 6226 | 10.8 | 0.879 / 0.894 | **0.828 / 0.864** | 0.460 / 0.541 | 0.108 | PASS |
| ames | 517 | 80.1 | 0.925 / 0.922 | 0.847 / 0.853 | 0.958 / 0.956 | 0.801 | HOLD: small N, inverted imbalance, AUPRC barely above baseline |

## Reads
- **Random vs scaffold barely drops** (e.g. hERG 0.934 -> 0.913). The signal is genuine content, not memorized near-duplicates. Contrast the DTI trap (random 0.997 -> cold 0.76 collapse, see `docs/FAILURE_MODES.md`): hERG/cyp3a4/cyp2d6 do not collapse, so they are real.
- **hERG, cyp3a4, cyp2d6 pass the gate** = the property lives in the SMILES content. These are valid axis-B head-to-head candidates. hERG is the strongest (scaffold rf AUROC 0.913, AUPRC 0.750 vs baseline 0.158).
- **ames is held**: only 517 compounds, inverted imbalance (80% fail), and AUPRC sits near the base rate, so the "ceiling" is not informative here.

## Next
Run the probe-vs-LLM head-to-head on hERG (and cyp3a4/cyp2d6): give the LLM the same SMILES, ask pass/fail, score deterministically, and compare against the probe ceiling above. Add the LLM-activation arm (linear probe on the LLM's hidden states) to separate encoding from expression. Heavy/GPU parts move to the GPU workflow (see `eval/README.md` Compute).
