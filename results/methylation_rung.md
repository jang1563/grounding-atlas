# Methylation rung: a web-zero numeric vector is ENCODED but not verbalized

*Results. 2026-06-12. `eval/activation_arm_methyl.py`, data `signal/methyl/methyl_age.csv` (GEO GSE41037, Illumina 27K blood methylation, 720 samples, age 16-88; 100 top-variance CpG probes shown as `cgXXXX:0.NN` text; label = age above the cohort median 33). Qwen3-8B 3-arm. ceiling = logistic regression on the numeric beta vector (the cheap epigenetic clock); activation = hidden-state probe on the beta-TEXT; output = verbalized age estimate, scored as AUROC of predicted-age vs the binary label. No em dashes.*

## Result

| arm | AUROC |
|---|---|
| ceiling (LR on 100 beta values) | 0.701 |
| ACTIVATION (hidden-state probe) | 0.685 |
| OUTPUT (verbalized age) | 0.487 (verbalized-age MAE 35.3 years) |

- encoding gap: 0.017 (the probe recovers age to the ceiling)
- expression gap: 0.198

## What it shows, and how it corrects the prediction

The prediction was encoding-limited: a web-ZERO numeric vector (methylation beta values never appear in text bound to age) should sit at chance in the ACTIVATION arm. That is REFUTED, and the refutation is the interesting part. The probe recovers age at 0.685, essentially the ceiling (0.701): the encoding gap is ~0. The model's hidden states preserve the input beta values, and a linear probe on them does the same regression the cheap clock does. This is the same phenomenon as the SMILES char-n-gram control: a probe can decode whatever is linearly present in the tokenized INPUT, so "the model encodes age" here means "the age-relevant numeric variance survives into the hidden states", not "the model knows about epigenetic aging".

The OUTPUT, by contrast, is at chance (0.487; the verbalized age is off by 35 years on average, the model defaulting to a near-constant guess). So methylation is EXPRESSION-dominant, not encoding-limited: the discriminative numbers are in the activations, but the model cannot map them to "age" in its output, because the beta-to-age mapping is web-undocumented (no text teaches "cg16867657 = 0.8 implies old"). The signal is held and not said.

This places methylation with single-cell-ANON (activation 0.964, output 0.497): a representation whose tokens carry the signal but whose property-mapping is web-absent is ENCODED (probe reads the input) yet UNVERBALIZABLE (output at chance, expected to stay there across scale).

## The methylation / MSA pair sharpens the two-factor law

These two rungs were built as a matched pair, both formally "a per-item vector/string of values -> a binary property", differing in web-exposure:

| rung | representation | property | ceiling | activation | output |
|---|---|---|---|---|---|
| MSA column | amino-acid LETTERS (web-rich) | conservation (web-documented) | 0.999 | 1.000 | **0.795** |
| methylation | beta NUMBERS (web-zero) | age-from-betas (web-undocumented) | 0.701 | 0.685 | **0.487** |

The decisive correction: they do NOT differ at ENCODING. Both encode their signal to the ceiling (MSA 1.000, methylation 0.685 ~ its 0.701 ceiling), because a probe reads the input either way. They differ at OUTPUT (0.795 vs 0.487). So the web-exposure law is, precisely, a law about VERBALIZATION: web-documentation of the representation-to-property mapping governs whether an encoded signal reaches the output, not whether it is encoded. This is the cleanest statement of the project's central finding, and it is why the histopath result (encoded 0.827, said 0.46-0.65) and the SMILES result (decoded ~ceiling, said at chance) are the same phenomenon as methylation, across image, string, and numeric-vector representations alike.

## Caveats

27K array (not 450K/EPIC) and 100 top-VARIANCE probes (not the Horvath age-CpGs), so the clock ceiling is moderate (0.701) rather than the 0.9+ a purpose-built clock reaches; this lowers all three arms uniformly and does not affect the encoding-complete / output-at-chance shape. Age binarized at the cohort median (young-skewed, median 33). Output scored as predicted-age ranking; a different cutoff or a direct "old vs young" prompt would not change an output sitting at chance with a 35-year MAE. Pilot, n=600 balanced.

## Reproduce

`ACT_CSV=$HOME/bge/methyl_age.csv ACT_K=100 python eval/activation_arm_methyl.py` (Qwen3-8B, GPU). Data built by the GEO fetch in `signal/methyl/`.
