# variant_grounding - the variant-effect branch of grounding-atlas

A parallel copy of the axis-B grounding instrument (`../eval/`, `../results/head_to_head.md`),
run on **genetic missense variants** instead of SMILES. Same question, same arms, one balanced
sample, leakage-controlled. This is the **within-modality** test of the project's core
web-exposure hypothesis (`../PROJECT_DESIGN.md` section 7): the same biology in two surface
forms with opposite web-exposure. It is the predicted **opposite extreme** to the SMILES anchor.

## The instrument

Does a general LLM ground a specialist model's signal by **content**? Arms on one variant set:

- **ceiling (specialist analog):** AlphaMissense precomputed pathogenicity (ClinVar AUROC ~0.94)
  + logistic regression, primary; ESM-1v wild-type-marginal LLR (single forward pass, no MSA,
  the methodologically purest specialist, mirrors the hERG fingerprint) secondary. Is the
  signal in the content at all?
- **LLM-output:** the LLM generates a single pathogenicity probability. Does it VERBALIZE it?
  Run in **two surface forms** on the SAME variants (the headline, below).
- **LLM-activation (optional, GPU stretch):** linear probe on the LLM's hidden states. Does it
  ENCODE it internally? Scaffolded for GPU execution (`activation_arm_variant.py`); not required for
  the first result. Here the 2-arm (ceiling + output) result is the deliverable.

This branch leads with the **2-arm** instrument (ceiling + output). The activation arm is the
optional GPU extension, mirroring `../protein_grounding/`.

## The headline: a within-modality web-exposure test

A variant's pathogenicity exists in TWO surface forms over the SAME variants, with OPPOSITE
web-exposure:

| form | what the LLM sees | web-exposure | prediction |
|---|---|---|---|
| **text** | `{gene} {HGVS}` (e.g. "BRCA2 p.Asn372His") | RICH: co-occurs with "pathogenic/benign" constantly in ClinVar, OMIM, abstracts | output ABOVE chance (Hu et al, GPT-4o ~0.73 zero-shot, npj Prec Onc 2025) |
| **sequence** | the mutant protein window, NO gene name | POOR: a raw sequence carrying the substitution has no labeled web presence | output near chance (SMILES-like) |

Probing both forms on the same variants isolates the **web-exposure effect inside one
modality**. The **text-minus-sequence output gap** is the cleanest single test of prediction
**P1 (monotonicity)**: encoding/recall tracks how often the "content -> property" mapping
appears in web text. Contrast with the SMILES result (expression-dominant, output at chance):
SMILES has no web-rich symbolic form, so it has no text arm above chance. That is the modality
axis the project measures.

## Leakage is the central methodological risk (do not skip)

ClinVar is on the web and in training data, so a high text-form number can be **memorized
recall**, not grounding. Five controls partition recall from biology (`eval/README.md`):

1. **Temporal holdout (most important):** restrict to variants first added to ClinVar AFTER the
   model cutoff. Built by diffing dated ClinVar releases (`prepare_data.py` uses the 2025-06,
   2026-01 archived snapshots vs current 2026-06).
2. **Star stratification:** report star-2+ (expert-reviewed) separately from star-1.
   Memorization concentrates in well-known variants.
3. **DMS parallel track:** ProteinGym deep-mutational-scan fitness (experimental, not a
   memorized clinical label). If text-form is high on ClinVar but collapses to chance on DMS
   (same biology, no label to recall), that gap IS the memorization estimate.
4. **Gene-name scramble:** replace the real gene with a pseudonym; the AUROC drop quantifies
   reliance on the memorized gene-disease prior vs the variant itself.
5. **Sequence-vs-text obfuscation:** the (text)/(sequence) split above is itself a control,
   isolating symbol-recall from biology over the identical variant.

Frame: web-exposure raises BOTH encoding and recall; the controls partition them. The
text-minus-sequence gap AND the ClinVar-minus-DMS gap decompose web-exposure from memorization.

## The property and data

ClinVar missense variants, `pathogenic`/`likely pathogenic` (label 1) vs `benign`/`likely
benign` (label 0), review status star-1+, GRCh38, balanced. Each variant is mapped to its
UniProt canonical protein, with a wild-type-residue consistency check, so the same record
carries `{gene, HGVS}` (text form), the mutant sequence window (sequence form), and the
AlphaMissense score (ceiling), all self-consistent. See `eval/README.md`.

## GeneLab / space-biology relevance

A novel spaceflight variant is absent from ClinVar, so the text (output) arm should FAIL on it
while the ceiling (AlphaMissense / ESM-1v) still works. An instrument that flags "grounded by
content vs echoing ClinVar" is directly actionable for novel-variant triage.

## Layout

| Path | What |
|---|---|
| `eval/prepare_data.py` | ClinVar -> UniProt seq -> AlphaMissense -> balanced `variant_clinvar.csv` (text + sequence + score + stars + temporal flag) |
| `eval/prepare_dms.py` | ProteinGym DMS -> `variant_dms.csv` (leakage-free parallel track) |
| `eval/ceiling_gate_variant.py` | AlphaMissense score -> AUROC (primary), stratified by stars and pre/post-cutoff |
| `eval/ceiling_esm1v_variant.py` | ESM-1v WT-marginal LLR ceiling (GPU, secondary) + sbatch |
| `eval/output_arm_variant.py` | dual-form (text / sequence) LLM output arm + gene-scramble control, anchored parser + parsed/percent/fallback |
| `eval/activation_arm_variant.py` | OPTIONAL 3rd arm: linear probe on hidden states (GPU) |
| `data/variant_clinvar.csv` | the balanced ClinVar sample |
| `results/ceiling_gate.md` | is the signal in the AlphaMissense/ESM-1v content? (the gate) |
| `results/head_to_head.md` | the dual-form output result + the web-exposure / memorization decomposition + SMILES contrast |

## Honest scope (carried from the SMILES branch)

- The text-form result is **confounded by memorization** by construction (ClinVar is in
  training data); the leakage controls are not optional decoration, they ARE the measurement.
  A high text number alone proves nothing about grounding.
- That a direct-prompted LLM can recall ClinVar pathogenicity from gene+HGVS is partly PRIOR
  ART (Hu et al, npj Prec Onc 2025, GPT-4o ~0.73). What is novel here is the **matched
  surface-form contrast** (text vs sequence, same variant) and the **temporal/DMS
  decomposition** of recall from grounding, inside the cross-modality instrument.
- The activation arm (if run) measures **linear decodability** of the property from hidden
  states, not "knowledge". Best-layer activation is max-over-layers (selection-biased), with a
  bootstrap CI; a random-label control task bounds selectivity.
- One or a small panel of models to start, zero-shot. No em dashes; verified numbers only;
  capability framing.
