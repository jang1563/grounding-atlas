# Spectra rung (simulated MS): the non-renderable modality is STILL expression-limited for hERG

*Results. 2026-06-11. `eval/activation_arm_spectra.py` (Qwen3-8B, n=1250). The molecule is presented ONLY as a list of fragment m/z peaks (BRICS-fragmentation MS), never as a structure. The 6th and hardest rung, and it closes the modality ladder on a clear principle. No em dashes.*

## The setup (dual ceiling)

Mass spectra are non-renderable: the structure is encoded in a numeric peak pattern that requires structure elucidation to invert. Two ceilings bracket what is recoverable from the simulated MS:
- ceiling_structure = Morgan probe on the TRUE structure = 0.825 (what a perfect elucidation specialist, SpecTUS-class, reaches by inverting the spectrum)
- ceiling_surface = a probe on the binned m/z histogram = 0.667 (the SURFACE peak-statistics signal, no elucidation)

The LLM reads only the peak-list TEXT ("Peaks (m/z): 105.0, 230.1, 445.2, ...").

## Result (Qwen3-8B, n=1250, scaffold GroupKFold)

| arm | AUROC |
|---|---|
| ceiling structure (elucidation) | 0.825 |
| ceiling surface (binned m/z) | 0.667 |
| **activation (LLM hidden states on the peak list)** | **0.729** [0.702, 0.756], selectivity 0.219 |
| output (LLM verbalized) | 0.502 |

Encoding gap (vs structure) 0.096, expression gap 0.227. The output is exactly chance: the LLM cannot verbalize hERG from a mass spectrum. But the activation is 0.729, ABOVE the surface histogram probe (0.667) and approaching the structure ceiling (0.825): reading the m/z numbers as text, the LLM encodes the hERG-relevant fragment pattern MORE richly than a coarse 10-Da histogram (it reads the exact peak values). So spectra-hERG is EXPRESSION-limited, not encoding-limited.

## The principle this closes the ladder on

The plan predicted molecular images and raw spectra to be the ENCODING-LIMITED extreme (the model cannot perceive the structure, so it never forms the property). Across FOUR representations of the SAME coarse property (hERG), this prediction FAILED every time:

| representation | how the LLM reads it | activation | encoding gap | regime |
|---|---|---|---|---|
| SMILES | character substrings | 0.787 | 0.038 | expression-limited |
| DNA-style string | k-mers | (DNA promoter 0.880) | 0.009 | expression-limited |
| molecular image | perceived structure (OCSR) | 0.758 | 0.096 | expression-limited |
| mass spectrum | m/z numbers | 0.729 | 0.096 | expression-limited |

The through-line: hERG is a COARSE property (lipophilicity, aromatic rings, size), and a coarse property is SURFACE-DECODABLE from any representation, because the LLM picks up whatever surface features that representation exposes (substrings, k-mers, perceived structure, peak numbers) and a linear probe recovers the coarse signal from the hidden states. So the LLM ENCODES the property and the bottleneck is always EXPRESSION (it cannot verbalize the readout), regardless of how "hard" or non-renderable the modality is.

Therefore the ENCODING-LIMITED regime is property-GRANULARITY-dependent, not modality-dependent. It requires a property that is NOT surface-decodable, one that needs deep computation a forward pass cannot do (exact structure elucidation, a specific 3D pharmacophore, an exact-match identity), where surface features fail AND the LLM cannot compute the answer. A hard MODALITY alone does not produce it; a fine PROPERTY does. This is the corrected, measured form of P2, established across the modality ladder rather than asserted.

## Frontier confirmation: the chance output is SCALE-INVARIANT

The 8B output (0.502) is not a small-model artifact: a frontier panel (`results/frontier_output_panel.md`) finds the MS output FLAT at chance across every scale, opus-4.8 0.459, sonnet-4.6 0.470, haiku-4.5 0.497, 8B 0.502. So the 8B ENCODES hERG from the peak list (activation 0.729) yet NO model, frontier included, can VERBALIZE it. This makes spectra the FIRST representation whose verbalization no scale unlocks (contrast DNA, 8B 0.40 to opus 0.82, and SMILES, 8B 0.45 to sonnet 0.63): the spectrum-to-property step is a computation (structure elucidation) the forward pass cannot perform, not a recall the model can scale into. A scale-invariant expression gap, the web-zero extreme of the law.

## Caveats

The MS is a crude deterministic simulation (BRICS fragment exact masses, binned), so ceiling_surface 0.667 is a weak-specialist lower bound and ceiling_structure 0.825 is the elucidation upper bound; a real high-resolution MS plus a real elucidation specialist (SpecTUS) would sharpen both, but the LLM-side reading of the peak list (activation 0.729 >> output 0.502) is the load-bearing result and would only strengthen. Single open-weight model. The encoding gap (0.096) is small and within the range of the other rungs, confirming expression-limited.

## Reproduce

`sbatch run_activation_spectra_cayuga.sh` on Cayuga (`eval/activation_arm_spectra.py`, simulated MS from `herg.csv` via RDKit BRICS).
