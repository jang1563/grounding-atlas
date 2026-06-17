# Head-to-head, variant branch: specialist ceiling vs LLM, dual surface form (axis B)

*2026-06-09. LLM-output arm (API). Script: `eval/output_arm_variant.py`. Models:
claude-sonnet-4-5-20250929 (primary), meta-llama/Llama-3.3-70B-Instruct-Turbo (refusal-free
cross-check). Ceiling: `ceiling_gate_variant.py` (AlphaMissense).*

**Question:** the specialist reads pathogenicity from the variant content (AlphaMissense AUROC
0.960, holds at 0.965 on the post-2026-01 temporal holdout, `ceiling_gate.md`). Does a general
LLM surface it, and crucially **from which surface form**? The same variant is shown in two forms
with opposite web-exposure, with the identical question (probability pathogenic) and the SAME
anchored parser as the SMILES branch. This is the within-modality test of the web-exposure
hypothesis (`../README.md`, `../../PROJECT_DESIGN.md` section 7).

**Method:** balanced ClinVar sets (1000/1000 main; a separate balanced 1000/1000 strict
post-2026-01 holdout), zero-shot, n=600 sampled per run. Two surface forms plus two leakage
controls, all on the same variants:
- **text**: "Is the human variant `{gene} {HGVS}` pathogenic or benign? ... probability ..." (web-rich)
- **text_nogene**: HGVS only, no gene symbol (intermediate)
- **text_scramble**: gene symbol character-shuffled to a pseudonym, HGVS unchanged (isolates the prior)
- **seq**: a WT protein window + "residue at position k mutated from X to Y", NO gene name (web-poor)

The arm is instrumented (parsed / percent / refusal / fallback) with a parsed-only AUROC.
Decompositions: **text - seq** (web-exposure within one modality), **text - scramble**
(gene-prior reliance), **temporal holdout** (ClinVar-label leakage), **ClinVar - DMS** (clinical
recall vs functional grounding).

## Results: ClinVar main sample (Sonnet 4.5, n=600, 300/300)

| condition | AUROC [95% CI] | answered | parsed-only AUROC | mean_p |
|---|---|---|---|---|
| ceiling: AlphaMissense (matched set) | ~0.96 | - | - | - |
| **text** (gene + HGVS) | **0.789 [0.754, 0.825]** | 599/600 | 0.79 | 0.63 |
| text_nogene (HGVS only) | 0.662 [0.618, 0.701] | 596/600 | 0.66 | 0.64 |
| text_scramble (pseudonym gene) | 0.639 [0.591, 0.680] | 576/600 | 0.64 | 0.68 |
| **seq** (window, no gene) | 0.581 [0.549, 0.612] | 126/600 (**474 refusal**) | _see refusal note_ | 0.50 |

The web-exposure ladder, same variant, surface form only: **gene+HGVS 0.789 > HGVS 0.662 >
scrambled-gene 0.639 > raw seq 0.581**. text - seq = **+0.208**; text - scramble = **+0.150**
(gene-prior reliance); the real gene adds +0.127 over HGVS-only, while a FAKE gene is no better
than no gene (0.639 vs 0.662). The lift is specifically real gene-identity recall.

## Results: strict temporal holdout (post-2026-01 ClinVar, n=600, 300/300)

| condition | Sonnet 4.5 AUROC | Sonnet answered | Llama-3.3-70B AUROC | Llama answered |
|---|---|---|---|---|
| ceiling: AlphaMissense (post-2026-01) | 0.965 | - | 0.965 | - |
| **text** (gene + HGVS) | **0.825 [0.792, 0.857]** | 599/600 | **0.734 [0.692, 0.776]** | 600/600 |
| text_nogene (HGVS only) | n/r (full ladder in main above) | - | **0.530 [0.492, 0.570]** | 600/600 |
| text_scramble (pseudonym gene) | n/r (full ladder in main above) | - | **0.502 [0.456, 0.545]** | 600/600 |
| **seq** (window, no gene) | **0.584 [0.553, 0.616]** | 125/600 (**475 refusal**; parsed-only 0.669) | **0.597 [0.558, 0.642]** | 600/600 (0 refusal) |

text - seq: Sonnet **+0.242**, Llama (refusal-free) **+0.137**. text persists on the holdout
(Sonnet 0.825 = main 0.789; Llama 0.734, matching GPT-4o ~0.73). The ClinVar-date holdout rules
out ClinVar-LABEL memorization specifically; the persistence says the text signal is also a
generalizing gene/region prior, not only label recall. The refusal-free Llama row confirms the
seq drop is a **capability** effect, not just Claude's refusal: with 0 refusals, seq is 0.597,
far below text 0.734. Note the seq floor is **~0.60, not pure chance**: a raw protein sequence
carries some LM-readable biochemical signal (proline/glycine/charge, conservation-like priors),
unlike SMILES which sat at 0.45. The web-poor variant form is web-poorer than its text form but
not as web-poor as SMILES.

Llama gene-prior reliance (text - scramble) = **+0.232**: scrambling the gene name drops Llama to
exactly chance (0.502), and HGVS-only is near chance (0.530). For Llama the ENTIRE text-form
signal is the memorized gene-disease prior; the raw sequence (0.597) actually beats a gene-less
HGVS (0.530), because a context-free "p.Arg1699Gln" carries little while the sequence window has
some biochemical signal. Sonnet leans on the gene too (main: gene adds +0.127 over HGVS-only,
+0.150 over a scrambled gene) but retains some HGVS-string recall that Llama lacks.

## Results: DMS parallel track (ProteinGym BRCA1/PTEN/TP53/MSH2, Sonnet, n=600 pooled)

| condition | AUROC [95% CI] | answered | mean_p |
|---|---|---|---|
| ceiling: AlphaMissense vs DMS damaging | 0.85 - 0.91 | - | - |
| **text** (gene + HGVS) | **0.610 [0.567, 0.649]** | 600/600 | 0.83 |
| text_scramble | 0.582 | 437/600 (160 refusal) | 0.71 |
| **seq** (window, no gene) | 0.515 [0.491, 0.540] | 63/600 (537 refusal) | 0.51 |

On the SAME famous genes but **experimental functional** labels (not memorizable clinical tags):
text drops to 0.610, and gene-prior reliance collapses to text - scramble = **+0.028** (vs +0.150
on ClinVar). The clean gene-matched comparison (BRCA1/PTEN/TP53/MSH2 in both): **ClinVar-label
text 0.723** vs **DMS-functional text 0.610**, a +0.113 clinical-recall gap on identical genes.
So roughly half of the famous-gene text signal that ClinVar rewards is clinical-label recall that
does not transfer to predicting experimental function.

## The refusal finding (instrumentation)

For the **seq** form, Claude returns `stop_reason=refusal` on ~79% of variants (474/600 main,
476/600 holdout, 537/600 DMS), NOT a token truncation (identical at max_tokens 64). The
safety-tuned model freely scores the web-rich gene+HGVS form but refuses to assign pathogenicity
to a raw protein-sequence variant. This is the variant analog of the SMILES O2
validity-vs-property dissociation, here surfacing as a surface-form refusal asymmetry, and it is
why the refusal-free open model (Llama-3.3-70B) is run alongside to read the seq capability floor.

## Cross-modality comparison (same instrument, same anchored parser)

| modality | property | ceiling | output, web-rich form | output, web-poor form | regime |
|---|---|---|---|---|---|
| SMILES | hERG block | 0.825 | n/a (no web-rich symbol form) | 0.453 (at chance) | expression-dominant |
| protein | meltome Tm | 0.699 | n/a | (protein branch) | encoding test |
| **variant** | **ClinVar pathogenicity** | **0.960** | **text 0.79 - 0.82** | **seq 0.58 (0.60 refusal-free)** | **web-exposure axis, within one modality** |

*(SMILES/protein rows: `../../results/head_to_head.md`, `../../protein_grounding/`.)*

## Model panel (strict post-2026-01 holdout, text / seq / scramble; n=600 each)

Mirrors the SMILES branch 5-model panel. The question is whether the two signatures are
model/scale/vendor invariant: (a) the **text > seq** web-exposure gap, and (b) **gene-prior
reliance** (text - scramble). Six models across three vendors and a wide scale range. All seq
AUROCs are refusal-free (answered ~100%) EXCEPT sonnet-4-5 (see refusal note).

| model | vendor / scale | text | seq | scramble | text - seq | text - scramble | seq answered |
|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | Anthropic, small | 0.797 | 0.672 | 0.570 | +0.124 | +0.227 | 599/600 |
| claude-sonnet-4-5 | Anthropic, mid | 0.825 | 0.584* | (main 0.639) | +0.242 | (main +0.150) | 125/600 (79% refuse) |
| claude-sonnet-4-6 | Anthropic, mid (newer) | 0.743 | 0.545 | 0.603 | +0.198 | +0.141 | 584/600 |
| claude-opus-4-8 | Anthropic, large | 0.938** | 0.803 | 0.673 | +0.134 | +0.265 | 600/600 |
| gpt-4o | OpenAI, closed | 0.778 | 0.644 | 0.575 | +0.134 | +0.203 | 600/600 |
| Llama-3.3-70B | Meta/Together, open | 0.734 | 0.597 | 0.502 | +0.137 | +0.232 | 600/600 |

*sonnet-4-5 seq is refusal-confounded (only 125/600 answered, selection-biased upward); the
other five answer the seq form refusal-free. **opus-4-8 text: see leakage caveat below.

Both signatures are invariant. **text > seq for all six** (gap +0.12 to +0.24): the web-rich
symbolic form is universally read better than the web-poor sequence form. **Gene-prior reliance
is universal**: scrambling the gene drops every model (+0.14 to +0.27), to exactly chance for
Llama (0.502). The text-form advantage is substantially the memorized gene-disease prior, in
every model and vendor.

### Two findings the panel adds

- **The seq floor RISES with capability (unlike SMILES).** Refusal-free seq climbs sonnet-4-6
  0.545 < Llama 0.597 < gpt-4o 0.644 < haiku 0.672 < opus-4-8 0.803. SMILES output stayed flat at
  chance across the whole scale range (8B to 32B, haiku to opus); here the web-poor variant form
  is partially readable and a stronger model reads more. This supports the web-exposure account:
  variant sequence is web-poor but not web-zero (protein sequence + biochemistry are in
  pretraining), so it sits ABOVE the SMILES floor and scales with model strength. opus-4-8 reads
  the raw sequence (no gene, no HGVS) at 0.803, approaching the AlphaMissense ceiling (0.965).
- **The seq refusal was sonnet-4-5-specific, not a general safety behavior.** Only sonnet-4-5
  refused the raw-sequence form (79%); the newer sonnet-4-6, opus-4-8, haiku-4-5, gpt-4o, and
  Llama all answer it refusal-free. So the refusal asymmetry is a single-model quirk, and the
  clean capability read comes from the other five. (The DMS-track seq refusal, also sonnet-4-5,
  carries the same caveat.)

### Leakage caveat specific to opus-4-8 (and the reason sonnet-4-5 is the primary)

opus-4-8's training cutoff (~2026-01) coincides with the post-2026-01 holdout boundary, so for
opus those variants are NOT a clean temporal holdout: its text 0.938 may be inflated by
ClinVar-label memorization. Its high SEQ (0.803), which carries no gene/HGVS symbol to recall,
indicates genuine sequence capability rather than pure recall, but the text number is confounded.
This is exactly why the headline uses sonnet-4-5 (cutoff safely before 2026-01): the temporal
holdout is only a valid leakage control for a model whose cutoff predates the boundary. As a
direct check, sonnet-4-6 (newer than 4-5) scores LOWER on holdout text (0.743 vs 0.825), not
higher, so the holdout shows no memorization inflation for the post-4-5 versions that predate the
boundary; opus-4-8 is the one model where the boundary and cutoff collide.

## Activation arm (Qwen3-8B, GPU): does the LLM ENCODE pathogenicity, per surface form?

The third arm (the genuinely novel measurement): a per-layer linear probe on Qwen3-8B hidden
states, run for BOTH surface forms on the same 1500-variant set under a **GroupKFold grouped by
GENE** (so the probe cannot exploit the memorized gene->label prior; the within-modality analog of
the SMILES scaffold split). Open-weight model because hidden states are needed. Script:
`activation_arm_variant.py`.

| form | ceiling (AlphaMissense) | activation (best layer) | output | encoding gap | expression gap | probe selectivity |
|---|---|---|---|---|---|---|
| text (gene + HGVS) | 0.962 | 0.795 (L35) | 0.599 | 0.167 | **+0.196** | +0.277 |
| seq (window, no gene) | 0.962 | 0.740 (L29) | 0.494 | 0.222 | **+0.245** | +0.202 |
| *SMILES anchor (hERG)* | *0.825* | *0.787* | *0.453* | *0.038* | *+0.334* | *-* |

*(SMILES row: same protocol, Qwen3-8B, from `../../results/head_to_head.md`.)*

- **The raw-sequence form is EXPRESSION-limited at 8B, like SMILES.** Qwen3-8B ENCODES variant
  pathogenicity from the bare sequence window at 0.740 while its OUTPUT is 0.494 (chance):
  expression gap +0.245. The signal is inside the hidden states; the 8B model does not verbalize
  it. This reconciles the output panel: the larger opus-4-8 OUTPUTS the seq form at 0.803,
  verbalizing what the 8B model encodes (0.740) but cannot say (0.494). Scale closes the
  expression gap.
- **Web-exposure raises ENCODING too, not just output.** text activation 0.795 > seq activation
  0.740, under the gene GroupKFold (not the gene-memorization shortcut; genes are split). The
  within-modality web-exposure effect appears in the hidden states, the encoding-side mirror of
  the output ladder.
- **The encoding gap is LARGER than SMILES (fractionally), because the specialist is stronger.**
  Qwen3-8B recovers 83% of the AlphaMissense ceiling on text (0.795/0.962) and 77% on seq
  (0.740/0.962), vs 95% of the Morgan-FP ceiling on SMILES (0.787/0.825). AlphaMissense
  (evolutionary + structural, 0.96) is a far harder ceiling to match internally than Morgan-FP
  hERG (0.825). So variants are a MIXED regime: both an encoding gap AND an expression gap, unlike
  SMILES (near-pure expression gap). On the directional prediction (opus's strong seq reading
  implies a small encoding gap): opus's seq OUTPUT 0.803 is a verbalization/expression effect; the
  8B ENCODING gap against the 0.96 ceiling is real and larger than SMILES, not smaller. Encoding
  and expression are separate axes, both nontrivial for variants.
- **The probe reads real signal (selectivity).** Random-label control probe scores 0.52 (text) /
  0.54 (seq); the task probe beats it by +0.277 / +0.202, so the activation AUROC is linear
  decodability of pathogenicity, not probe memorization. Best-layer is max-over-37-layers
  (selection-biased); bootstrap CIs in the log.

## Reads

- **The opposite extreme to SMILES, as predicted.** SMILES has no web-rich symbolic form, so its
  output sits at chance (0.45). Variant effect has one (gene + HGVS), and the LLM reads it: text
  0.79 main / 0.82 holdout, matching GPT-4o ~0.73 (Hu et al, npj Prec Onc 2025). The modality
  axis the project measures is visible in a single instrument.
- **The within-modality ladder is monotone (P1).** Holding the variant fixed and varying only the
  surface form: gene+HGVS (0.79) > HGVS (0.66) > scrambled-gene (0.64) > raw sequence (0.58, and
  mostly refused). The signal tracks how web-exposed the form is, exactly the web-exposure
  prediction, inside one modality and one variant.
- **Web-exposure raises recall AND a prior; the controls partition them.** The temporal holdout
  shows the text signal is not pure ClinVar-label memorization (it persists at 0.82 on
  post-cutoff variants). The DMS track shows it is also not functional grounding (text 0.61 and
  gene-prior 0.03 on experimental labels for the SAME genes). So the text advantage decomposes
  into a generalizing gene/region prior (survives temporal holdout, collapses on DMS) plus
  clinical-label recall (the ClinVar - DMS drop), with little variant-specific grounding.
- **The specialist is the foil.** AlphaMissense (0.965) AND fully-unsupervised ESM-1v (0.921, no
  labels ever seen) both hold on the exact post-cutoff variants where the output arm is read
  (`ceiling_gate.md`); the seq floor and the DMS collapse are grounding gaps, not hard-variant
  artifacts. The instrument distinguishes "grounded by content" (specialist, flat across the
  holdout) from "echoing the symbol" (LLM text, web-exposure-dependent).

## Caveats

- Primary model sonnet-4-5, with a 6-model panel (haiku-4-5, sonnet-4-5/4-6, opus-4-8, gpt-4o,
  Llama-3.3-70B) on the holdout. sonnet-4-5 is the headline because its cutoff predates the
  2026-01 holdout boundary; opus-4-8's cutoff collides with it (text inflation caveat above).
- The ClinVar-date temporal holdout controls ClinVar-LABEL leakage, not all-literature leakage: a
  variant deposited after the cutoff may still appear in a pre-cutoff paper. This is why the
  gene-scramble and DMS controls run alongside.
- text gives gene + HGVS (which encodes the substitution); seq gives the same substitution minus
  the symbol. text - seq is the value of the SYMBOL, biological content held fixed.
- The seq refusal is **sonnet-4-5-specific** (~79%), not a general safety behavior: sonnet-4-6,
  opus-4-8, haiku-4-5, gpt-4o, and Llama all answer the seq form refusal-free. sonnet-4-5's
  full-set seq AUROC is 0.5-diluted and its parsed-only is selection-biased upward; the clean seq
  capability read comes from the other five (0.55 - 0.80, rising with scale).
- That a prompted LLM recalls ClinVar pathogenicity from gene+HGVS is partly PRIOR ART (Hu et
  al). Novel here: the matched surface-form ladder, the temporal/DMS/scramble decomposition, the
  6-model panel showing the seq floor rises with scale (vs flat-at-chance SMILES), and the
  placement on the cross-modality web-exposure axis vs the SMILES anchor.

## Next

- Optional activation arm (`activation_arm_variant.py`, GPU): is the text-vs-seq gap an
  ENCODING gap or an EXPRESSION gap, per form, under the gene GroupKFold? (opus-style strong seq
  reading predicts a smaller encoding gap than SMILES.)
- ESM-1v secondary ceiling (`ceiling_esm1v_variant.py`): an unsupervised specialist holdout
  baseline (rules out "the labels leaked into the specialist").
