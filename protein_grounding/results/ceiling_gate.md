# Ceiling-gate results, protein branch (axis-B candidate screening)

*2026-06-09. Cayuga GPU (a40), ESM2 650M + sklearn. Script: `eval/ceiling_gate_protein.py`.*

**Question:** is the property predictable from the representation CONTENT (here the amino-acid
sequence, via a protein SFM)? A high supervised ceiling means the signal is in the
representation, so a probe-vs-LLM head-to-head is meaningful. A low ceiling means there is
nothing for the LLM to fail to surface. This is the protein analog of `../../results/ceiling_gate.md`.

**Data:** FLIP meltome (`mixed_split.fasta`, protein melting temperature Tm across many
species). Binarized at the median Tm (50.26 C), length-filtered to 50-512 standard-AA
residues, balanced to 1500 (750/750). Specialist features = ESM2
(`facebook/esm2_t33_650M_UR50D`, 1280-dim) mean-pooled over residues, the protein analog of
the Morgan fingerprint. Probes: logistic regression + random forest, both class-weight
balanced. 5-fold CV, two splits: random (StratifiedKFold) and cluster (MMseqs2 identity
clusters at 30%, GroupKFold), the leakage control.

| property | n | %pos | random AUROC (lr/rf) | cluster AUROC (lr/rf) | cluster AUPRC (lr/rf) | baseline AUPRC | verdict |
|---|---|---|---|---|---|---|---|
| **meltome Tm > 50.26 C** | 1500 | 50.0 | 0.701 / 0.751 | **0.699 / 0.744** | 0.705 / 0.765 | 0.500 | PASS (moderate ceiling) |

## Reads

- **Random ~ cluster: no leakage.** LogReg 0.701 -> 0.699 (drop 0.002); RF 0.751 -> 0.744
  (drop 0.007). The signal survives the sequence-identity split, so it is genuine sequence
  content, not near-duplicate homolog memorization (the protein version of the DTI trap). The
  meltome mixed set is already highly non-redundant (1291 of 1500 are singletons at 30%
  identity), so the cluster split and a random split agree by construction here; the cluster
  GroupKFold still rules out leakage for the 209 sequences that do share a cluster.
- **The ceiling is real but MODERATE.** ESM2 reads thermostability at AUROC ~0.70 (LogReg) /
  ~0.74 (RF), clearly above the 0.50 chance baseline but well below the hERG ceiling (Morgan
  FP LogReg 0.825 on the matched balanced set, 0.895 on the full set). Thermostability
  binarized at the median is a genuinely hard property even for the specialist; this matches
  the FLIP meltome leaderboard (embedding Spearman ~0.6). The gate is about *direction*, not
  saturation: there is real content signal for the LLM to fail to surface, which is what the
  head-to-head needs.
- **Apples-to-apples for the activation arm.** The activation arm's structure-probe uses
  LogReg on the same balanced 1500 set under the same cluster split, so its ceiling is the
  0.699 LogReg number (RF 0.744 is the stronger-specialist reference, reported but not the
  arm's probe).

## Caveat carried forward

Thermostable proteins in this sample are slightly shorter on average (mean 283 vs 311
residues). Mean-pooled ESM2 is length-invariant by construction, so the ESM ceiling is not a
length artifact, but a general LLM reading the raw sequence could in principle exploit length.
A composition/length-matched control (and a composition-preserving shuffle) is the natural
follow-up, the protein analog of the SMILES scramble arm.

## Next

Run the three-arm head-to-head (`eval/activation_arm_protein.py`): ESM2 ceiling, LLM
activation (per-layer linear probe on Qwen3-8B hidden states), and LLM output, all on this
same 1500-protein set under this same cluster split. See `head_to_head.md`.
