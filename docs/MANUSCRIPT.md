# Encoded, not verbalized: web-exposure governs biological grounding in language models, and tells you when to trust it

*JangKeun Kim. Draft, 2026-06-19. Status: working manuscript. No em dashes. All numbers are pilot-scale; treat individual AUROCs as estimates and read the regimes and orderings, which are stable.*

## Abstract

Language models recognize biological entities by NAME but ground them poorly from their concrete
REPRESENTATIONS (a sequence, a SMILES string, an expression vector). We separate two reasons a model
fails to put a property in its output: it never ENCODED the property, or it encoded it but cannot
VERBALIZE it. Using a single three-arm instrument (a cheap specialist ceiling, a linear probe on an
open model's hidden states, and the model's verbalized output) across 17 representations plus
matched within-entity controls, we find that encoding is near-universal (the encoding gap is under
0.10 for 13 of 17 representations) while verbalization lags far behind (the verbalization gap runs
0.12 to 0.49). The size of the verbalization gap is governed by one thing: how web-documented the
representation-to-property mapping is. The cleanest evidence is a pair of controls with identical
task shape and equal encoding but opposite output (methylation to age, verbalized at chance, versus
MSA conservation, verbalized at 0.795), and a same-data contrast (single-cell classification from
gene NAMES versus ANONYMIZED ids). The law is cross-modality and cross-notation, and it reproduces
outside biology (materials). It interacts with capability multiplicatively: on a controlled
single-cell ladder, the gene-name AUROC rises with model scale (0.76 to 0.98) while the anonymized
form stays pinned at chance at every tier, the most capable model included, so capability reads out
a mapping only where the mapping is web-documented. The consequences are practical. A decision map
across capabilities shows training the weights wins nowhere; the live placements are retrieving
knowledge and orchestrating a specialist, with the frontier acting as a calibrated router at the
task level. And because the model's per-item self-confidence is unreliable in a capability-dependent
way, the a-priori web-exposure of a representation, knowable before any model call, is a more robust
signal for when to trust the model versus defer to a specialist. We package the instrument as a
release-cadence grounding-and-calibration benchmark.

## 1. Introduction

A useful biological AI must ground the CONTENT of a representation, not just its name. Yet a measured
recognition gap is stark: name recognition is near 100% while raw accession resolution is 2 to 28%,
and rare accessions resolve to plausible but wrong family-level identities. The open question this
raises is mechanistic: when a model fails to state a property from a representation, is the
information absent from the model (an encoding failure) or present but unsurfaced (an expression
failure)? The two have opposite engineering implications, and they cannot be told apart from the
output alone.

We answer it with a three-arm instrument and a verifiable-signal substrate, applied uniformly across
the representation-type space. Our contributions:

1. A cross-representation, matched-control **instrument** that separates encoding from verbalization
   (Section 2), and a verifiable-signal substrate where the representation itself is the ground
   truth (so the label is not contestable).
2. A measured **web-exposure law** (Section 3): models encode far more biology than they verbalize,
   and the verbalization gap is set by how web-documented the representation-to-property mapping is,
   not by the modality. Demonstrated with controlled within-entity pairs and a non-biology control.
3. The **capability interaction** (Section 4): model scale closes the verbalization gap only where
   the mapping is web-documented, measured as a clean interaction on a controlled single-cell ladder.
4. A **decision map** (Section 5): training the read-out wins nowhere; retrieving and orchestrating
   cover the space; the frontier is a calibrated router at the task level.
5. A **calibrated-permissioning** lever (Section 6): an a-priori, input-derived web-exposure tag
   predicts when to trust the model better than the model's own self-confidence, whose reliability is
   itself capability-dependent.

We are not the first to report that models lean on name over content (CoKE, GenomeQA,
Rethinking-Protein). The contribution is the instrument, the law, the capability interaction, and the
permissioning lever, with matched controls throughout.

## 2. Instrument and substrate

**The three arms.** For each (representation, binary property) we measure: (i) CEILING, a cheap
non-LLM specialist (Morgan fingerprint, k-mer logistic regression, a foundation-model embedding, an
epigenetic clock), answering whether the property is decodable from the representation at all; (ii)
ENCODING, a linear probe on an open-weight model's last-token hidden states (Qwen3-8B, or Qwen2.5-VL
for images), answering whether the model carries the property forward; (iii) VERBALIZATION, the
model's emitted probability scored by AUROC, answering whether it will say it. Two gaps follow: the
encoding gap (ceiling minus encoding) and the verbalization or expression gap (encoding minus
output). Scoring is deterministic, never an LLM judge.

**Substrate.** Off-the-shelf data does not supply matched (representation, verifiable-property) pairs
where the representation is the ground truth, so we generate them and gate each modality on a high
supervised ceiling before use. A verifiability gate over 19 modality cells passes 17; the two
failures are instructive (protein-protein interaction by name is web-memorization, at chance once
anonymized), which is itself the law in miniature.

**Controls.** Every encoding claim carries a random-label selectivity control (Hewitt and Liang).
Content-sensitivity uses matched, re-notation, and scrambled forms: a probe-vs-output gap counts as
content grounding only if it survives re-notation (otherwise it is entity-recognition or fact
recall, not content). Two honest readings sharpen "encodes": the decodable signal is linear in the
hidden states as it is in the surface string (a no-chemistry character n-gram matches the SMILES
probe), so "encodes" means "linearly decodable," not "represents deep chemistry"; and the durable
claim is the expression GAP, which is invariant to scale, architecture, alignment, reasoning effort,
and is selectivity- and few-shot-controlled.

## 3. The web-exposure law

Across 17 representations spanning token strings, biological sequence, 2D images, spectra, 3D
coordinates, molecular graphs, expression and epigenetic vectors, and sequence alignments (Table 1,
`results/SYNTHESIS.md`):

- **Encoding is near-universal.** For 13 of 17 representations the encoding gap is under 0.10: the
  probe recovers the property nearly to the ceiling, because the hidden states preserve whatever is
  linearly present in the tokenized input. This holds even for a web-zero numeric vector (methylation
  betas, encoding gap 0.017) and anonymized ids (single-cell, 0.025). The exceptions are the
  structure-heavy representations (3D coordinates, molecular graph) where the discriminative signal
  is latent in geometry or bond topology the model gets only as a composition surface.
- **Verbalization lags, governed by web-exposure.** The expression gap runs 0.12 to 0.49. The
  governing variable is whether the representation-to-property mapping is documented in web text.

**The controlled proof.** Two pairs isolate web-documentation from everything else. (1) Methylation
to age and MSA-column to conservation have the same task shape and are both encoded to the ceiling
(0.685 of 0.701; 1.000 of 0.999), yet the MSA output is 0.795 while the methylation output sits at
chance (0.487): the only difference is that amino-acid-to-conservation is web-documented and
beta-vector-to-age is not. (2) Single-cell classification from a cell sentence of GENE NAMES versus
the SAME expression vector with ANONYMIZED ids: encoded equally (probe 0.983 vs 0.964), but the named
form verbalizes (and closes with scale) while the anonymized form stays at chance at every scale. The
law is cross-NOTATION too: the same variant as web-rich HGVS text grounds better than as a web-poor
raw sequence at a fixed specialist ceiling.

**Generality and bound.** The law is not biology-specific: a materials control (metal vs non-metal
from a chemical formula vs an anonymized-element composition) reproduces it (element symbols 0.72 to
0.84, anonymized 0.44 to 0.54). And it is bounded to EMPIRICAL properties: COMPUTABLE properties
(atom count, molecular weight, sequence length), a closed-form function of the representation, are
snap-impossible but reasoning-solvable (AUROC 1.0 given enough reasoning tokens), governed by
reasoning budget, not web-exposure.

**Honest scope.** The activation arm is open-weight-only (frontier internals are unobservable), so the
encoding axis is an open-model property and the frontier enters on the output and routing axes. The
cross-modality encoding-gap magnitude is confounded by how surface-decodable a property is, so the
clean test is the within-entity notation contrast, not a single cross-modality number.

## 4. Capability times web-exposure

Does a bigger or newer model close the verbalization gap? We measured it as a controlled interaction.
Each single cell becomes a cell sentence (top genes by expression), presented two ways: real gene
NAMES (web-documented) or global-consistent ANONYMIZED ids (the same expression vector, only the
human-readable name removed, so a specialist still separates the classes at CV-AUROC ~0.99). We ran a
within-family capacity ladder (Haiku 4.5, Sonnet 4.6, Opus 4.8) plus GPT-4o on two substrates chosen
to need the symbol-to-cell-type prior rather than one marker: CD8-T vs NK and CD14+ vs CD16+
monocyte (Figure 1, `results/benchmark/single_cell/`).

The interaction is clean on both substrates. The gene-name AUROC rises monotonically with capability
(CD8-T/NK 0.826 to 0.871 to 0.978; monocyte 0.763 to 0.929 to 0.983, near the specialist ceiling),
while the anonymized AUROC stays pinned at chance at every tier, the most capable model included
(bootstrap CIs straddle 0.5). The gap WIDENS with capability: web-exposure and capability multiply.
The anonymized arm is also the confound control: if the gain were generic capacity or a larger
corpus, a stronger model would also do better on the anonymized form; that it sits at chance shows
the gain is specifically reading web-documented symbols. The effect is provider-invariant (GPT-4o on
the same curve). A direct three-arm probe on an open model (Qwen2.5-0.5B) confirms the mechanism on
its own activations: it encodes cell type near the ceiling (probe 0.947 on names) while verbalizing
at chance (output 0.461), and that 0.461 is the foot of the name-verbalization curve the frontier
ladder climbs.

The picture is engine times fuel: capability is the engine, web-exposure is the fuel; a bigger engine
goes further only where there is fuel.

## 5. The decision map: route, do not train

If the gap is about what the corpus documented, what should a builder do per capability: train it into
weights, retrieve it, or orchestrate a specialist? Measured across the atlas
(`results/decision_map_placement.md`, `results/ws3_train_placement.md`):

- **Train wins nowhere.** A LoRA read-out lifts the open model's verbalized hERG output from 0.575 to
  0.856 but lands a strong SECOND to the cheap Morgan specialist (0.899), because the trained head
  reads the same surface the specialist does. On the notation-invariance and property-prediction
  cells we tested, training is always at best a strong second.
- **Retrieve and orchestrate cover the space.** Retrieval closes in-data-pattern and web-anchored
  knowledge cells; orchestration wins for heavy specialist compute, canonicalization (any SMILES to
  canonical, an accession to its sequence), and pure computation. On the variant flagship,
  orchestrating AlphaMissense reaches AUROC 0.96, including 0.985 on novel post-cutoff variants,
  above any sequence-reader ceiling.
- **The frontier is a calibrated router.** Across rungs, the frontier model's self-confidence tracks
  its actual grounding at correlation +0.90: it sets low confidence on exactly the web-zero rungs.
  Routing on that continuous confidence reaches the per-rung oracle (0.893 vs 0.894), versus 0.700
  answering everything itself.

We package the instrument as a release-cadence benchmark (grounding-atlas-eval): a model-agnostic
output arm that scores grounding (AUROC and gap to ceiling), calibration (ECE, AURC), and
memorization-transparency, with versioned prompts, released raw outputs, and bootstrap CIs. A
three-model ADMET run reproduces the verbalization gap and is provider-invariant (mean absolute
cross-provider difference ~0.03); capability closes it endpoint-dependently. The run also surfaced a
data-quality lesson worth stating: an apparent "anti-grounding" on the Ames endpoint was a
label-direction error in the source, caught by a structural-alert audit (nitroaromatics, the primary
Ames alert, were enriched in the class labeled non-mutagenic); corrected, all models ground Ames at
~0.68. A benchmark's largest risk is label provenance, not the model.

## 6. Calibrated permissioning: when to trust

Per-item, the model cannot tell you where it is right. On the property-prediction arm the frontier
model's explicit self-confidence is essentially uncorrelated with its per-item correctness, and
implicit confidence recovers only 1 to 2 of the roughly 16 to 20 points of oracle headroom. Worse,
self-confidence's reliability is itself capability-dependent: on the single-cell permissioning test
(Figure 2), the well-calibrated models lower their confidence on the anonymized form (Opus 0.25 to
0.03, Sonnet 0.22 to 0.07), so confidence routing works for them, but Haiku is equally confident on
the anonymized form (0.35 to 0.35) and GPT-4o is more confident (0.16 to 0.22), confidently wrong, so
confidence routing collapses to near chance (accuracy at 50% coverage 0.54 and 0.56).

The a-priori web-exposure tag, whether a representation's mapping is web-documented, is knowable
before any model call and is uniformly safe (accuracy at 50% coverage 0.75 to 0.82; it wins or ties
the deferral curve for every model). The permissioning rule that follows: decide what to let a model
answer versus route to a specialist from the input's web-exposure, not from the model's self-report.
This is the safety face of the same ruler: capability and responsible deployment measured together.

## 7. Related work

The finding that models lean on name and context over sequence content is established (CoKE
2510.23127, GenomeQA 2604.05774, Rethinking-Protein 2505.20354). The frequency-encoding result that
linear representations of a fact form once co-occurrence crosses a threshold (2504.12459), and that
scale helps the head but not the long tail (Kandpal et al. 2023, Mallen et al. 2023), are the
natural-language precedents we generalize to scientific modalities under an encoding-held-equal
control. Encode-greater-than-verbalize has latent-knowledge precedent (Burns et al. 2023; the
acquisition-utilization gap, Kazemnejad et al. 2023). On calibration, verbalized confidence is known
to separate per-item correctness poorly (Kadavath et al. 2022, Xiong et al. 2024), with prompt and
reasoning-mode caveats (Tian et al. 2023, Yoon et al. 2025); we add the granularity split (calibrated
at the task level, not the item level) and the capability-dependence of self-confidence's
reliability. On deferral, input-derived signals exist (Mallen adaptive retrieval; Hybrid-LLM); the
mapping-level web-exposure tag, benchmarked head-to-head against self-confidence and framed as
biosafety permissioning, is what we add. The instrument is complementary to adjacent evals that
measure retrieval (VirBench) or tool solve-rate (BioMysteryBench): retrieval, then content-grounding
(this work), then downstream.

## 8. Limitations

Pilot scale throughout (n typically a few hundred to ~2000 per rung; the capability ladder at
n=200/model). The activation arm is open-weight-only by necessity. The cross-modality encoding gap is
ceiling-confounded, so the within-entity controls carry the comparison. An early prediction that
images are encoding-limited was refuted for coarse properties (a coarse property is surface-decodable
from any representation); the encoding-limited regime is property-granularity-dependent, not
modality-dependent. The capacity-versus-corpus confound in the capability ladder is mitigated by the
within-family design and the anonymized control, not eliminated. And, as the Ames episode shows,
label provenance must be audited.

## 9. Reproducibility

Code is Apache-2.0, data CC-BY-SA-4.0; the matched-pair tables are released as a Hugging Face dataset.
The benchmark uses versioned prompts, fixed decode, released raw model outputs, full provenance
manifests, and bootstrap CIs on every metric. Adding a model is one flag. See `README.md`,
`results/SYNTHESIS.md` (Table 1, the 17-representation master table), `results/benchmark/`,
`docs/capability_web_exposure.md`.

## Figures and tables

- **Table 1** (`results/SYNTHESIS.md`): the 17-representation three-arm master table (ceiling,
  activation, output, encoding gap, expression gap, regime).
- **Figure 1** (`results/benchmark/single_cell/interaction.png`): capability times web-exposure on two
  substrates; gene-name AUROC rises with scale, anonymized stays at chance, with 95% bootstrap CIs.
- **Figure 2** (`results/benchmark/single_cell/deferral.png`): permissioning; the a-priori web-exposure
  tag versus the model's self-confidence as deferral signals.
