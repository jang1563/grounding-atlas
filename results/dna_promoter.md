# DNA/RNA modality rung: the promoter 3-arm

*Results. 2026-06-11. Instrument: `eval/activation_arm_dna.py` (Qwen3-8B, n=1500 balanced). Data: genomic-benchmarks `human_nontata_promoters`. Adds DNA as the 4th measured modality on the same 3-arm instrument (after SMILES, variant, protein). No em dashes.*

## The rung

The plan's modality ladder (`PROJECT_DESIGN.md` 7.3) lists DNA/RNA as the next full 3-arm modality. The content-property is PROMOTER vs non-promoter (a core, web-discussed genomics concept), and the cheap-specialist CEILING is a 6-mer-frequency logistic regression, the DNA analog of the Morgan fingerprint for SMILES. Gating: the 6-mer probe clears 0.889 (5-fold CV), so this is a verifiable content-property with a high ceiling, as WS2 requires.

## Three arms (Qwen3-8B, one balanced sample n=1500, StratifiedKFold)

| arm | AUROC | note |
|---|---|---|
| ceiling (6-mer LR) | 0.889 | the cheap surface-statistical specialist |
| **activation probe (hidden states)** | **0.880** [0.862, 0.897] | best layer 23, selectivity 0.348 |
| output (verbalized) | 0.396 [0.370, 0.420] | ANTI-correlated, all 1500 parsed |

Gaps: **encoding = ceiling - activation = 0.009** (the smallest in the project), **expression = activation - output = 0.484** (the largest). Regime: strongly EXPRESSION-DOMINANT, more extreme than SMILES.

## Reading

The 8B FULLY encodes promoter-ness from the raw DNA sequence (activation 0.880, essentially at the 0.889 ceiling) yet cannot verbalize it: the output is 0.396, below chance, so the model not only fails to state the property, it states it in the WRONG direction. The anti-correlation is itself a mis-grounding signature, not mere absence: the model has a verbalized promoter notion (likely TATA-box / GC-rich heuristics) and mis-applies it to this NON-TATA promoter set, calling the wrong sequences promoters. The selectivity control (activation 0.348 above its shuffled-label control) confirms the probe reads a real sequence signal, not label memorization.

## The honest caveat (parallels the SMILES surface-decodability finding)

The encoding gap is near zero, but this does NOT mean the model holds a deep regulatory-genomics understanding. The CEILING here is itself a 6-mer surface-statistical probe, so "the activation reads promoter-ness as well as the ceiling" means precisely "the hidden states encode the same surface k-mer statistics a k-mer probe reads." This is the exact DNA analog of the SMILES result, where the activation signal (0.787) did not exceed a no-chemistry char-n-gram probe (0.812): in both modalities the model encodes the property as a SURFACE-STATISTICAL signal of the string (k-mer frequencies for DNA, character substrings for SMILES), linearly decodable from the hidden states, and cannot verbalize it. So the durable claim is the EXPRESSION gap (probe >> output, here 0.880 >> 0.396), not a strong "the model understands promoters" reading.

## What it does to the web-exposure law

The plan (7.3) predicted DNA as "medium encoding gap, task-heterogeneous," expecting the web-poor raw-sequence form to encode WEAKLY. Measured, the encoding gap is the SMALLEST yet (0.009), not medium. This is the same kind of correction the variant rung produced (predicted small encoding gap, measured mixed): the cross-modality encoding-gap magnitude is confounded by HOW SURFACE-DECODABLE the property is (promoter-ness is highly k-mer-decodable, so the model picks it up from surface statistics regardless of web-text binding), which is why the law's clean test is the within-entity notation contrast at a fixed ceiling (variant text vs seq), not the cross-modality encoding-gap number. DNA promoter joins SMILES at the expression-dominant, surface-encoded end of the spectrum; the OUTPUT mis-grounding (anti-correlated, a wrong heuristic mis-fired) is the web-exposure consequence here, not a weak encoding.

## Frontier update (the 8B output was a small-model view)

The 8B output (0.396, anti-correlated) is NOT the whole story: a frontier panel (`results/frontier_output_panel.md`) shows the expression gap CLOSES with scale here, 8B 0.396 to haiku 0.622 to sonnet 0.798 to opus 0.815, approaching the 0.889 ceiling. So frontier models DO verbalize promoter-ness from a raw DNA sequence; promoter recognition is web-rich enough (a textbook genomics task) that capacity unlocks it, like variant-seq rising sonnet 0.55 to opus 0.80. The "expression-dominant" label holds only for the 8B; DNA is a SCALE-DEPENDENT expression gap.

## Reproduce

`sbatch run_activation_dna_cayuga.sh` on Cayuga (data `dna_promoter.csv` from genomic-benchmarks `human_nontata_promoters`, 6-mer gating 0.889). Per-item dump `results/dna_act_peritem.json`. The instrument is `eval/activation_arm_dna.py` (6-mer ceiling + StratifiedKFold + DNA prompt; the DNA counterpart of `eval/activation_arm.py`).
