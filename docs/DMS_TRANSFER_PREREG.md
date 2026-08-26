# DMS zero-target-label transfer contract and preregistration template

*Design contract and preregistration template. Drafted 2026-07-25. The exact
claim-bearing statistical choices have not yet been externally registered. This
document reports no transfer, decoder, or causal result.*

## Question and claim boundary

Can a decoder whose layer, preprocessing, parameters, calibration, and decision
rule are selected using only source-DMS labels predict whether the signed
assay-normalized mutant effect relative to its matched physical WT/control meets
the preregistered retained/neutral-function criterion in independent,
protein-family-disjoint target DMS data when no target label is available before
immutable target predictions are written?

A passing experiment may support **zero-target-label, assay-outcome-aligned
transferable representation evidence** for the named model, source assays, target
assays, representations, and family partition. It does not by itself establish
human-like biological knowledge, spontaneous verbal access, natural use by the
unperturbed model, a causal activation gap, clinical pathogenicity, organismal
fitness, or a physical law.

The primary truth level is T5: an assay-derived interventional effect. Every claim
is conditional on the recorded construct, assay system, score transformation,
replicate policy, and threshold.

## Release architecture: aggregate schema 1 and raw-assay schema 2

The two schemas are mechanically independent. Validators reject missing or
additional fields rather than accepting free-form extensions.

Schema 1 retains the four historical pilot containers:

1. `dms_preregistration.v1.json`;
2. `dms_label_free_inputs.v1.json`;
3. `dms_replicates.v1.json`; and
4. `dms_outcomes.v1.json`.

It serializes each manifest as one canonical JSON object, not JSONL. Its bundled
writer is deliberately limited to the aggregate ProteinGym pilot and rejects
confirmatory mode.

Schema 2 starts from raw or upstream-resolvable assay observations and uses seven
exact objects:

1. an authoritative source lock with dataset, license, assay, reference, raw-record,
   and metadata identities;
2. one global sequence-to-protein-family map;
3. a label-free source/target input manifest;
4. a raw replicate manifest split into mutant observations, physical baseline pools,
   physical WT/control observations, and exact item-to-baseline links;
5. a frozen assay-transformation specification;
6. a deterministically recomputed exclusion ledger; and
7. a deterministically recomputed outcome manifest.

Baseline observations are physical source observations rather than copies attached
to each mutant. A shared control distribution is represented once and linked to
permitted items under a frozen reuse policy, maximum multiplicity, assay, batch,
condition, and matched-set contract. Literal WT observations use the same structure.

Every downstream schema-2 object binds its prerequisites by canonical SHA-256, and
outcomes are replayed from the exact passing mutant and baseline observations.
Reference and mutant sequence checksums are firewalled across source and target
independently of accession aliases. The local readiness API requires usable
two-class source outcomes, two source families, two-class target outcomes, and at
least eight evaluable target families.

These checks establish local deterministic consistency only. Self-contained source,
family-authority, and registration hashes cannot authenticate themselves, so schema
2 always returns `NOT_READY_FOR_CONFIRMATORY_EXECUTION`,
`confirmatory_eligible=false`, and a missing external trust index. A separate future
verifier must authenticate those anchors out of band.

### Authentication and exact-registration boundary

This document freezes the scientific schema, firewall order, required controls,
non-lowerable family minimum, and adjudication logic. It is a **contract and
preregistration template**, not the final immutable statistical preregistration.
Before any claim-bearing execution, an externally timestamped and authenticated
registration must bind at least:

- the exact primary metric and its direction;
- the exact confidence-interval construction and superiority rule for every control;
- the minimum total and per-class target item counts, in addition to the eight-family
  floor;
- the source-frozen decision threshold and any calibration estimand;
- the family-bootstrap design and number of draws;
- the multiplicity family and correction across layers, prompts, datasets, models,
  endpoints, and secondary analyses; and
- the positive-claim threshold and explicit `NOT_SUPPORTED` and `NOT_ADJUDICATED`
  decision rules.

The external registration must also authenticate the source lock, global family map,
sealed target-outcome identity, and the registration timestamp before any actor or
process involved in selection, fitting, calibration, thresholding, or intervention
design can access target outcomes. A canonical hash stored only inside the release
cannot authenticate its own chronology or prove who had label access. Until the
external trust index verifies these anchors and exact statistical choices, every
locally valid execution remains nonconfirmatory and `NOT_ADJUDICATED`.

Target prediction is a separate execution product, not a replacement for either
dataset schema. It must be an exact label-free artifact with a durable freeze receipt
and is later joined to outcomes by typed `item_id`.

## Assay-normalized mutant-versus-matched-WT/control target

For variant \(i\) in assay \(A\), define

\[
e_i =
s_A\frac{
\operatorname{Agg}_A[T_A(y^{mut}_{ir})]
-
\operatorname{Agg}_A[T_A(y^{base}_{jr})]
}{c_A},
\qquad
Y_i = \mathbf{1}[e_i \ge \tau_A],
\]

where \(T_A\) is the frozen assay-specific transformation, `Agg` is the frozen
replicate aggregation rule, \(s_A\in\{-1,+1\}\) is the frozen orientation,
\(c_A>0\) is the frozen scale, and the orientation is chosen so larger \(e_i\)
always means more retained/neutral function under the assay contract. Sign reversal is part of the
registered derivation, not a post-hoc edit.

Here \(y^{base}_{jr}\) denotes the preregistered matched physical baseline
observations: literal WT when valid, or another explicitly justified physical
control when the assay contract requires it.

The continuous \(e_i\) is retained for calibration and threshold-robustness
analysis. A “fraction of WT” endpoint is permitted only when the source
measurement is on a valid ratio scale with a valid WT denominator. Log-enrichment,
relative-fitness, or WT-centered scores instead use a preregistered difference or
log-ratio. Raw scores from different assays are never pooled as if they shared a
unit.

Each assay contract requires:

- the exact score column, raw unit, transformation, orientation, and code hash;
- the biological meaning of a higher source score and a higher derived effect;
- the physical WT/control source and batch/condition matching keys;
- the scale definition and the data permitted to estimate it;
- the neutral/retained-function threshold, source, unit, and inclusive operator;
- exact label-0 and label-1 meanings;
- replicate aggregation order and uncertainty estimator;
- missing, censored, conflicting, duplicate, and indeterminate-value rules; and
- permitted construct backgrounds, sequence contexts, and variant classes.

Thresholds and normalizations may differ by assay. Cross-assay transfer requires a
common semantic orientation—label 1 means the preregistered retained/neutral
function contrast—not numerical equality of raw scores.

## Variant and row identity

Every label-free row requires:

- `item_id`, `entity_id`, `task_id`, `biological_question_id`, and
  `task_family_id`;
- upstream dataset, assay-record, and variant identifiers;
- `assay_id`, construct/background ID, and experimental condition;
- canonical protein accession and version;
- checksum of the exact assay-WT construct sequence;
- a canonical ordered edit list containing reference residue, 1-based position,
  and alternate residue;
- variant type and checksum of the derived mutant sequence;
- a construct-to-canonical-reference map checksum when the construct differs;
- `split_group_id` and `split_group_scope="protein_family"`;
- non-null `intervention_pair_id`; and
- representation kind, condition, and representation checksum.

Reference residue, coordinate, and mutant-sequence reconstruction must validate.
Multi-mutants are excluded from the first primary unless explicitly registered.
Codon-resolved assays retain nucleotide/codon identities; synonymous codons
encoding one protein substitution may be combined only under the frozen
duplicate/replicate rule.

`intervention_pair_id` deterministically identifies the mutant relative to its exact
assay-WT construct, including assay, construct/background, variant, and WT reference
set. It is a variant/construct identity, not necessarily the functional baseline used
to derive \(e_i\), and it is not a replicate ID. In schema 2, the exact outcome
contrast is additionally identity-bound by the physical `baseline_pool_id` and
`baseline_link_id` records; a non-WT control cannot be substituted merely by retaining
the mutant-WT pair ID. Alternate prompt or representation renderings of one construct
pair share the pair ID; unrelated mutants do not. Every pair must be nested within
exactly one protein-family group.

## Replicate and QC policy

The preregistration distinguishes biological and technical replicates and freezes:

- transformation-before-aggregation versus aggregation-before-transformation;
- mutant-to-matched-physical-baseline batch and condition matching;
- minimum passing replicate counts and read/count-depth thresholds;
- outlier, replicate-discordance, high-uncertainty, and zero-denominator rules;
- batch correction and weighting, if any;
- duplicate and conflicting-score handling;
- missingness, censoring, and excluded-variant policy; and
- the complete reason-code vocabulary.

Every outcome row records qualified mutant replicate IDs, qualified physical-baseline
observation IDs, exact baseline pool/link identities, aggregate values,
assay-normalized effect, uncertainty or an explicit unavailable reason, `qc_status`,
reason codes, and the strict integer `target_label` in `{0, 1}` for included rows.
The release includes all exclusions and their reasons. Filtering chosen after
viewing model predictions, or filtering on favorable outcome direction, is
prohibited.

## Protein-family partition

Protein families are constructed once across the union of all candidate source
and target proteins using a frozen database release, clustering algorithm,
identity and coverage thresholds, code version, and seed. The preregistration
stores the resulting item-to-family map checksum.

- Every item resolves to exactly one `protein_family` group.
- Source and target item, protein, WT-sequence, and family sets are disjoint.
- Exact or homologous constructs are assigned before source/target partitioning.
- All renderings, replicates, mutant-to-assay-WT construct pairs, and item-bound
  baseline links remain within one family.
- Unresolved or proxy grouping is non-confirmatory.

At least **eight held-out target protein-family groups** must contribute to a
confirmatory transfer estimate. Source groups do not count toward this minimum.
A one-protein or four-protein target may support only a pilot, regardless of row
count.

## Pinned source plan

### Existing local ProteinGym assets: pilot only

The only local DMS assets currently in scope are:

- `variant_grounding/data/variant_dms_BRCA1_HUMAN_Findlay_2018.csv`
- `variant_grounding/data/variant_dms_PTEN_HUMAN_Mighell_2018.csv`
- `variant_grounding/data/variant_dms_P53_HUMAN_Kotler_2018.csv`
- `variant_grounding/data/variant_dms_MSH2_HUMAN_Jia_2020.csv`

They are balanced derived snapshots made by
`variant_grounding/eval/prepare_dms.py`. Their legacy binary orientation is
`1 = damaging`, obtained by reversing ProteinGym `DMS_score_bin`; the saved CSVs
do not retain the continuous `_dms` value, raw replicate lineage, an explicit
matched WT aggregate, or the release-grade QC contract. They represent only four
proteins/families. Therefore they are useful for parser, identity, orientation,
and end-to-end pilot tests, but they cannot populate a confirmatory source or
target artifact without re-ingestion from pinned upstream records.

The strict pilot builder re-ingests the unbalanced raw ProteinGym tables rather
than those legacy derivatives. The 2026-07-25 local source lock is:

| Source | Valid single substitutions | SHA-256 |
|---|---:|---|
| ProteinGym reference catalog | 217 assay records | `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308` |
| `BRCA1_HUMAN_Findlay_2018.csv` | 1,837 | `0c4cbf6cd3195828a68d16faaf597b4f54b8a9d761524fd473cab438e6cd6fc5` |
| `MSH2_HUMAN_Jia_2020.csv` | 16,749 | `85a206825e197e57bb7bf09e94700b8ce80f22a34878e86b6751ce64350968f5` |
| `P53_HUMAN_Kotler_2018.csv` | 1,048 | `ee231b0832493701089c9d916e97e4b9acc734ddfdc12f52a4c7021ec2a36bd8` |
| `PTEN_HUMAN_Mighell_2018.csv` | 7,260 | `16cad49f65130f316544f425d9082d666678fad4541b69b6672e72c4cedaf53b` |

`python eval/dms_contract.py --out <directory>` currently produces 26,894
identity-bound inputs and outcomes: 21,766 retained/neutral (`1`) and 5,128
damaging/loss-of-function (`0`). The canonical artifact digests are
`2fda857d710ddf5efe1d35936bdc0490f20196285cfe057ea66cf80045523e24`
(preregistration),
`eae2fd9ddebde235fae82fee6d5032875f87d55868e54841cc353869f6630e35`
(label-free inputs),
`22340daa888e33ac52e45c2144464f3839dad9adf776a74fce439539f0cd1721`
(explicitly unavailable aggregate-only replicate manifest), and
`8f3c845250e9f16bfe93a99f9510461fc4d3dabf24a00eb358d894f1ffe7eff8`
(outcomes). The producing `eval/dms_contract.py` digest is
`9f6a01722434f8d4d3378b688b4081c70802d9416747d5f11b6e6e2cc388b4fa`.
These are data-contract integrity results, not model evidence.

### MaveDB candidate registry and source-readiness screen

The initial exact four-record lock remains in
`signal/dms/mavedb_candidate_lock.v1.json`. Its source-readiness screen is:

| Gene | Score-set URN | Candidate-stage status | Principal blocker |
|---|---|---|---|
| BRCA1 | `urn:mavedb:00000097-0-2` | `COUNT_LINEAGE_PARTIAL` | Count and replicate lineage exists, but the multi-child meta-analysis and normalization are not yet replayed from a frozen assay specification. |
| PTEN | `urn:mavedb:00000054-a-1` | `PROCESSED_REPLICATES` | Processed replicate scores exist, but deposited count lineage is absent and the calibration text contains a scale inconsistency that must fail closed. |
| MSH2 | `urn:mavedb:00000050-a-1` | `AGGREGATE_ONLY` | The deposited score is a three-replicate aggregate without deposited replicate columns or substantive counts. |
| TP53 | `urn:mavedb:00000059-a-1` | `IDENTITY_BLOCKED` | Fragment offsets, duplicate protein substitutions, and incomplete nucleotide-to-protein identity prevent an unambiguous release mapping. |

The expanded machine-readable screen is
`signal/dms/mavedb_candidate_registry.v2.json`. It contains 20 exact URNs,
14 core candidates and 6 conditional candidates after strict target-sequence
identity validation, sorted by URN. The core genes are CBS, CHEK2,
CYP2C9, G6PD, GCK, HMBS, KCNH2, KCNQ4, LDLR, MSH2, NUDT15, PTEN, RHO, and
TPMT. The conditional genes are BRCA1, BRCA2, MTHFR, PALB2, TP53, and TPK1.
TPK1 is `IDENTITY_BLOCKED`: its 585-nt target translates to 194 amino acids,
while deposited protein coordinates extend through residue 243; 796 syntactic
substitutions are out of bounds and 1,207 in-range substitutions disagree with
the target reference, leaving 2,132 strict sequence-valid missense substitutions.
The strict MTHFR count is 12,464 after excluding 20 stop-loss `p.Ter657X`
records that are not missense substitutions. Each record is
`candidate_not_ingested` and the registry-level `confirmatory_eligible` flag is
false.

The registry freezes exact decoded metadata and score-body byte counts and SHA-256
digests for all 20 records. It freezes an exact substantive count body only for CBS
and BRCA1; the other count hashes and byte counts are null with explicit blockers.
MaveDB count and score columns are assay-specific and non-prescriptive, so count
availability does not imply that a score can be reproduced. Distinct genes and the
registry's inferred family labels likewise do not establish protein-family
disjointness.

`eval/mavedb_source_lock.py` is the candidate-stage online verifier. It validates
transport completeness, decoded content hashes, exact metadata and table schemas,
row/accession identity, active public license and processing state, substantive
versus identifier-only counts, mapping history, target sequence identity, and the
MaveDB API contract. Its output is forced to `not_ingested` and `not_derived`; it
rejects `CONFIRMATORY_READY` even when a caller supplies syntactically valid
registration hashes.

As of 2026-07-26, complete candidate-stage source locks are materialized for the CBS
low- and high-B6 conditions. The mapped endpoints remained intermittent, but each
successful lock passed the unchanged transport, OpenAPI, metadata, tabular, mapping,
target, and readiness validators before an atomic write. Each mapped body digest was
then reproduced by a separate network replay. No partial or decoded response body was
retained. These are source snapshots, not schema-2 outcome artifacts, and they remain
`COUNT_LINEAGE_PARTIAL`, `not_ingested`, and `not_derived`.

### CBS low-B6 native-adapter boundary

`eval/cbs_mavedb_adapter.py` and
`signal/dms/cbs_adapter_status.v1.json` implement the first assay-specific
candidate adapter, for `urn:mavedb:00000005-a-5`. This is a native-lineage and
admission-status layer, not a schema-2 outcome adapter. It validates the exact
candidate-registry record and can structurally replay a complete offline MaveDB
source lock plus its decoded metadata, score, count, and mapped-variant bodies.
Metadata, score, and count bodies must also match the registry's independent body
digests. The original low-B6 adapter registry did not independently pin mapped and
OpenAPI bodies; the paired CBS registry now closes that gap by binding the exact
complete low/high source-lock artifact bytes, canonical and bundle digests, OpenAPI
identity, mapped-body identity, mapping-contract digest, and observed mapping-error
count. A persisted adapter status still cannot claim decoded-body structural replay
unless the reader supplies those exact source inputs.

The 32 native channels are nonnegative, depth-normalized relative allele
frequencies per one million reads, with literal `NA` missingness; they are not
integer raw-read counts. `controlNS*` and `controlS*` are non-mutagenized wild-type
amplicon channels used to estimate PCR/sequencing false positives. They are not a
functional wild-type fitness baseline. The deposited `score` is an aggregate, not
a raw or processed replicate. The eight channels per role must not be called eight
replicates until an exact sample map binds dose, biological culture, sequencing run,
and tiled region.

[Sun et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6993387/) describes
filtering selected and non-selected allele frequencies against the corresponding
error controls, collapsing equivalent codons, subtracting the error-control
frequencies, calculating a selection/non-selection enrichment ratio, regularizing
uncertainty, and scaling by nonsense and synonymous medians. The final paper defines
low B6 from the 0 and 1 ng/ml conditions and reports a functional-deleterious rule
of upper 95% CI below 0.60 at 5% FDR; it reports 0.22 for the separate high-minus-low
B6-remediability rule. These are assay-specific functional rules, not clinical
pathogenicity thresholds. The complement of the 0.60 deleterious rule is not a
published neutral/retained-function equivalence rule because it includes uncertain
measurements. The earlier 0.45/0.29 manuscript values are excluded. The paper's
well-measured screen additionally requires pre-selection allele frequency above
0.005% and standard error below 0.2, but these statements do not supply the missing
row-level replay lineage.

The paper's Additional File 3,
`13073_2020_711_MOESM3_ESM.xlsx`, was inspected read-only from the official
supplement archive. The fetched workbook was 10,005,946 bytes with SHA-256
`878365975f62da42c4113214958d4bd60ba7f4d3fe269eee1a722465ca6470aa`.
It contains 22,536 low-B6 pre-score rows and 11,478 experimental low-B6 score
rows. These observations support the status audit but the workbook is not retained
or body-validated by the adapter, so it cannot authenticate replay parameters.

CBS remains `COUNT_LINEAGE_PARTIAL`, `candidate_not_ingested`,
`outcome_status=not_derived`, and `confirmatory_eligible=false`. Admission requires
the exact sample-role/depth map, TileSeq parameter sheet and software revision,
codon-to-protein collapse map, QC and uncertainty operations, synonymous/nonsense
anchor membership, explicit admissible functional baseline, and a native
multichannel replay extension. The current scalar schema 2 cannot silently encode
these operations.

### CBS paired B6-response target: candidate contract implemented, outcome not registered

The preferred next CBS endpoint is the signed within-variant high-minus-low B6
fitness difference. Its primary population contains only protein variants with
measured, QC-admissible values in both conditions; it must preserve each condition's
uncertainty and the exact nucleotide-to-protein collapse lineage. Missing-condition
scores are excluded from the primary analysis rather than imputed.

The continuous difference is primary. Among variants first classified as deleterious
at low B6, a separate binary remediability rendering may use the final paper's
lower-95%-confidence-bound-above-0.22 rule at 5% FDR. It must not substitute the
low-B6 deleterious rule, whose upper 95% confidence bound is below 0.60. The target
is assay-level rescue in the yeast complementation system, not clinical response to
pyridoxine. A positive high-minus-low score alone means only higher measured
high-B6 fitness; it is not sufficient for the published remediability classification.

`signal/dms/cbs_b6_pair_registry.v1.json` now independently anchors the exact
low/high complete source locks, metadata, score, count, OpenAPI, and mapped-variant
body digests; the shared 1,656-nt CBS target; and the signed, measured-only contrast.
`eval/cbs_b6_pair_contract.py` implements:

- a high-B6 candidate adapter for `urn:mavedb:00000005-a-6`, whose native table
  has 10,802 rows, `score/sd/se`, and four channels each for `nonselect`,
  `select`, `controlNS`, and `controlS`;
- all-or-none exact metadata/score/count replay with target, license/state,
  schema, accession order, numeric, and missingness checks;
- an accession-independent one-to-one join on MaveDB's primary `hgvs_nt` key,
  with exact `hgvs_splice` and `hgvs_pro` agreement on shared rows;
- codon-copy collapse only when the complete `score/sd/se` value and missingness
  patterns agree exactly within each condition; and
- unconditional rejection of imputation, delta values, confidence intervals,
  labels, outcome manifests, and confirmatory promotion.

The pair registry freezes an expected structural intersection of 10,098 nucleotide
rows, with 1,380 low-only rows and 704 high-only rows. Those shared nucleotide rows
contain 7,365 distinct nonmissing protein-HGVS values. Independently intersecting
the two condition-level protein-HGVS sets gives 7,395; it is not the
nucleotide-join-derived count, and neither quantity is a strict-missense count. A
validator may claim structural replay only while the exact low and high source bytes
are supplied again. Its `complete` structural count means only that `score/sd/se`
are nonmissing in both deposited rows; it does not establish the paper's
allele-frequency, SE, control-filtering, or other QC requirements. A bounded
implementation-time replay matched all frozen structural counts and validated the
available high metadata/score/count bodies, but emitted no delta or outcome and
retained no decoded source bodies. The persisted
`signal/dms/cbs_high_b6_adapter_status.v1.json` and
`signal/dms/cbs_b6_pair_status.v1.json` therefore remain no-body candidate
statuses with `outcome_status=not_derived` and `confirmatory_eligible=false`.

The complete low/high source locks are now materialized and independently replay
confirmed. Their mapping contracts retain 152 low-B6 and 150 high-B6 current mapping
errors rather than implying error-free mapping. This resolves source snapshot
availability only. Outcome execution still requires authenticated post-count TileSeq
score reconstruction, condition-specific QC, a frozen codon provenance map, a biological
sample/resampling graph, the locally selected joint bootstrap, CI construction, and
external registration.

`eval/cbs_b6_uncertainty_contract.py` and
`signal/dms/cbs_b6_uncertainty_status.v1.json` implement the next prerequisite
boundary. The production method is now locally hash-locked to joint post-count
TileSeq
bootstrap with exactly 10,000 draws and a deterministic seed derived from the
pair-registry digest and serialized as fixed-width lowercase hexadecimal. Direct
covariance remains a reference schema, but selecting it
now requires a versioned amendment; zero covariance and independent-condition
fallback are prohibited. The status records no delta, CI, label, or outcome and
remains `confirmatory_eligible=false`. A schema-valid declaration, local method lock,
or self-reported registration receipt cannot authenticate evidence or promote
readiness.

`eval/cbs_b6_public_native_evidence.py` and
`signal/dms/cbs_b6_public_native_evidence_lock.v1.json` freeze the official
Additional File 3 workbook at 10,005,946 bytes and SHA-256
`878365975f62da42c4113214958d4bd60ba7f4d3fe269eee1a722465ca6470aa`.
Exact-body OOXML replay recomputes the six visible sheets, dimensions, header-row
hashes, formulas, defined names, external links, and custom XML. It finds only ordinal
replicate labels: eight columns per
measurement role for low B6 and four for high B6. The paper reports two biological
replicate cultures per condition and two sequencing runs, but the public workbook
does not bind its ordinal columns to dose, culture, library/run, tile, or shared
ancestry. The paper-linked software repositories supply publication-window commit
candidates and a generic two-replicate, 15-tile CBS test fixture, not an authenticated
B6-specific paper parameter sheet or exact executed revision. Consequently the
joint-bootstrap resampling graph remains unauthenticated and execution fails closed.
Even if the missing graph confirms those two cultures, claim-bearing percentile CIs,
threshold labels, and outcomes remain `NOT_ADJUDICATED`: the preregistered floor is
eight effective independent biological blocks per condition branch, and bootstrap
draws do not count as experimental support. Execution additionally requires a
hash-bound Python/NumPy/PCG64DXSM runtime manifest and top-level resample-degeneracy
and quantile Monte Carlo stability gates. Published remediability further requires
the exact synonymous and nonsense empirical-null memberships, missense testing
universe, p-value construction, FDR implementation, threshold derivation, and
paper-compatible CI method; generic Benjamini-Hochberg must not be assumed.

This variant-aligned CBS target can strengthen T5
assay-outcome alignment and a later local model-intervention test, but one protein
cannot establish protein-family-disjoint transfer or independent-domain replication.
A CBS-only causal run is pilot-only and `NOT_ADJUDICATED` under the current
eight-biological-group gate. Variants, codons, tiles, and ordinal channels may not be
relabelled as independent protein-family groups.

Every admitted assay still needs a dedicated adapter that translates its native
counts or processed replicates, physical WT/control evidence, transformations,
orientation, threshold, QC, and duplicate/codon rules into schema 2. The expanded
pool makes an eight-family design possible in principle; only a frozen
sequence-derived family map can establish that it is actually family-disjoint.

## Zero-target-label firewall

The controlled runner must execute in this order:

1. Freeze the preregistration, source and target input manifests, sealed target
   outcome digest, replicate digest, and combined family map.
2. Make the target outcome and replicate-value files inaccessible to the model,
   layer-selection, fitting, calibration, thresholding, and intervention
   processes.
3. Select the layer and fit all preprocessing, decoder parameters, calibration,
   and decision rules using source labels only.
4. Apply the frozen source decoder and every source-trained control to the exact
   target input rows.
5. Write and hash the label-free target activations and predictions with model,
   prompt, code, decoder, item-map, group-map, and input-manifest identities.
6. Commit the prediction freeze receipt.
7. Only then expose target outcomes to a separate scorer and attach them by an
   exact, duplicate-free, bijective typed-identity join.

The firewall audit requires zero source-target overlap in item IDs, protein IDs,
WT sequence checksums, variants, and protein-family IDs. Target-distribution
adaptation, target-fitted scaling, target-label calibration, target threshold
selection, and target-informed layer or prompt selection are forbidden in the
primary. A separately registered unlabeled transductive analysis must use weaker
language.

## Required source-trained controls and analysis

All controls use the same source training rows and exact target evaluation rows:

- identically regularized raw-input decoder;
- layer-0 decoder;
- preregistered simple surface-feature decoder; and
- multiple source-label permutations, frozen before target evaluation.

The primary metric, confidence interval, control contrasts, family-cluster
bootstrap, class support, minimum item count, and multiplicity rule are frozen
before target labels are opened. Thresholded target metrics use only a
source-frozen decision threshold. Calibration is secondary unless transported
without target fitting.

This template deliberately does not assign the exact values for those statistical
choices. They must appear in the externally authenticated immutable registration
described above; selecting them only after a locally sealed run, even before the
scorer opens labels, is not an admissible substitute.

Success requires all of the following:

- family-disjoint positive target transfer under the locked primary metric;
- superiority to every required source-trained control under the locked
  confidence-interval rule;
- agreement between the predicted and measured signed mutant-minus-baseline
  direction;
- a clean target-label firewall and exact artifact joins;
- at least eight held-out target families; and
- replication on another independent DMS dataset or model family.

One successful transfer without replication is preliminary evidence, not the
registered confirmatory claim.

## Artifact validity and adjudication

The release is **malformed** and rejected before statistical analysis if it has an
inexact schema; an invalid or ambiguous variant; duplicate, missing, extra, or
non-bijective identities; target fields in the label-free manifest; a missing
pair; a pair crossing family groups; an invalid score-to-label derivation; missing
replicate/source lineage; source-target identity or biological-group overlap; or
a checksum mismatch.

A syntactically valid release is **`NOT_ADJUDICATED`**, rather than a negative
transfer result, if environmental separation or label-access order cannot be
verified, biological family grouping is unavailable, fewer than eight target
families remain, required controls are incomplete, source/target semantics cannot
be aligned, the target prediction was not durably frozen before label access, or
minimum class/item support is not met. In schema 1, a direct identity-overlap
firewall violation is malformed and rejected rather than converted into this
status.

Only a valid and fully adjudicable artifact whose locked statistical gates fail is
a negative (`NOT_SUPPORTED`) result. Lower-rung engineering or source-decoder
evidence remains reportable but cannot be promoted to transfer, natural-use, or
causal language.

## Implementation map

Existing code:

- `variant_grounding/eval/prepare_dms.py`: legacy ProteinGym pilot ingestion;
- `eval/benchmark_tasks.py`: task registration, biological group fields, and
  row-level `intervention_pair_field`;
- `eval/probe_common.py`: train-only frozen decoder, label-free activation and
  prediction artifact, and later label-attached evaluation artifact;
- `eval/model_hooks.py`: residual-stream capture for later causal work;
- `eval/causal_intervention.py`: exact label-free execution receipts and
  intervention artifacts;
- `eval/analyze_causal.py`: separated causal-availability, natural-use, and
  routing adjudication;
- firewall and artifact-contract tests covering the three modules above. These
  are not in the public release: they exercise decoder-artifact helpers that this
  repository does not yet ship.

Implemented schema-1 pilot infrastructure:

- `eval/dms_contract.py`: four-artifact identity, family, pair, QC, and firewall
  validation for the strict aggregate ProteinGym schema-1 pilot. Schema 1
  mechanically rejects `confirmatory` mode;
- `eval/latent_transfer.py`: source-only fit, target label-free prediction freeze,
  replay-verified grouped layer selection and decoders, exact release-hash and
  multi-identity bindings, exclusive-create local freeze receipt plus detached
  commitment, exact outcome join, controls, effect-monotonicity report, and
  family-bootstrap analysis. The authoritative upstream outcome-manifest digest
  remains distinct from the canonical transfer-row digest recomputed at scoring.
  Its schema-1 adjudication is always `NOT_ADJUDICATED`;
- adversarial schema, leakage, orientation, pair nesting, source-target overlap,
  and freeze-order tests for the two modules above. These are not in the public
  release: they read the raw ProteinGym reference tables, which stay out of the
  repository.

Implemented candidate-source and schema-2 infrastructure:

- `signal/dms/mavedb_candidate_registry.v2.json`: 20 exact, fail-closed MaveDB
  candidates with metadata/score locks, count-lock availability, orientation
  evidence, provisional non-release family labels, and explicit blockers;
- `eval/mavedb_source_lock.py`: live candidate snapshot and validation of exact
  MaveDB bytes, table/accession integrity, license and record state, mapping
  history, target sequences, and caller-declared source-readiness evidence. It
  cannot emit an ingestion or outcome claim and rejects `CONFIRMATORY_READY`.
  Persisted validation replays the exact nested transport, metadata, tabular,
  mapping, and readiness invariants, while the source-bundle digest binds every
  top-level field except the digest itself;
- `eval/cbs_mavedb_adapter.py` and
  `signal/dms/cbs_adapter_status.v1.json`: offline, candidate-only CBS low-B6
  native-lineage validation; exact registry/body cross-binding; decimal
  relative-frequency semantics; explicit error-control, aggregate-score, mapped
  body, OpenAPI, and scalar-schema firewalls; and unconditional rejection of
  assay-bundle or outcome emission;
- `eval/dms_raw_contract.py`: mechanically independent schema 2 with exact
  source, family, input, raw observation, transformation, exclusion, and outcome
  artifacts; physical shared-baseline linkage; fixed cross-assay class semantics;
  arithmetic-mean-only within-role and across-match aggregation; deterministic
  mutant/baseline aggregates, effect, label, and matched-set standard-error
  replay; immediate rejection of non-finite transformation, aggregate, contrast,
  orientation, affine, and uncertainty results; global reference/mutant checksum
  firewalls; and local source/target support gates. Its confirmatory eligibility
  is always false without external authentication; and
- `tests/test_mavedb_source_lock.py`, `tests/test_mavedb_candidate_registry.py`,
  `tests/test_cbs_mavedb_adapter.py`, and `tests/test_dms_raw_contract.py`:
  transport truncation, empty counts,
  readiness overclaim, candidate-registry drift, coherent sequence-alias,
  control-cloning/reuse, QC exclusion, orientation, and replay attacks.

Still required for a confirmatory execution:

- assay-specific adapters that translate selected authoritative MaveDB count or
  processed-replicate fields, physical WT/control evidence, and native
  transformations into schema 2. The generic candidate source lock does not
  perform that translation;
- an externally authenticated source/trust index and a frozen global
  sequence-derived protein-family map with at least eight evaluable held-out
  target families;
- true paired mutant-minus-matched-baseline prediction deltas, multiple
  independently frozen source-label permutations, and an independent replication.
  When the registered functional baseline is not literal WT, its prediction input,
  representation identity, and physical baseline link must be frozen explicitly;
- execution in an environment that keeps outcome files inaccessible until the
  prediction receipt is committed. The current receipt detects subsequent local
  file mutation but is not remote attestation or proof of who accessed labels.

The causal-use track remains orthogonal. Transfer evidence can pass the
assay-outcome-alignment axis, but natural use requires targeted erasure plus
independent rescue, and the current `CAUSAL_ACTIVATION_GAP_SUPPORTED` code status
requires the compatible steering, erasure-rescue, and bidirectional routing suite
specified in `docs/B1_CAUSAL_UPGRADE_DESIGN.md`. That status is local to its locked
task, dataset, model, and held-out items. In the future reporting taxonomy, a
replicated causal activation gap additionally requires an independent biological
dataset and model family; reserve **biological causal activation gap** for the
conjunction of that replicated causal evidence with a passed zero-target-label
alignment/transfer gate under compatible target semantics.
