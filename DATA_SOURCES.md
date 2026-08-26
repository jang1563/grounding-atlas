# Data sources, truth levels, and attribution

This repository distributes small derived representation-label tables. The table below is a source
inventory and licensing summary. It is **not yet a complete row-level provenance manifest**.

The representation is generally an input, not the ground truth. Targets may be exact functions,
software outputs, empirical measurements, annotations, database assertions, or intervention effects.

## Truth-level taxonomy

| Level | Target source | Examples | Required interpretation |
|---|---|---|---|
| T0 | exact algorithmic/formal | length, count, visible-column entropy | exact under a named parser and convention |
| T1 | simulator/software defined | RDKit descriptor, simulated NMR/MS | reproducible under version, parameters, and seed |
| T2 | empirical assay | hERG, solubility, permeability, Tm | conditional on assay protocol, units, and uncertainty |
| T3 | observational/curated annotation | cell type, tumor, coding status | conditional on cohort and annotation policy |
| T4 | database assertion | ClinVar classification | assertion at a release, submitter, and review status |
| T5 | causal/interventional target | Perturb-seq, MPRA, DMS | assay effect under a defined intervention and experiment; not evidence that a model causally uses the target |

Predictability from a representation does not prove label correctness, causal dependence, or absence
of batch effects and assay noise.

Target/truth alignment and model causal use are orthogonal. For empirical T5 targets, the former is
operationalized only as alignment with the named assay outcome. A valid T5 target can support that
bounded alignment test, but only a separate controlled model-intervention suite can test causal use.
Conversely, a local activation-to-output effect does not authenticate the target or show
cross-dataset transfer. Future **biological causal activation-gap** wording requires both a passed
alignment/transfer gate and replicated causal evidence in an independent biological dataset and model
family.

## Source inventory

`Redistributable` is a good-faith project assessment, not legal advice. Upstream terms control.

| Config or data | Upstream lineage | Truth level | Terms | Redistributable status and scientific limit |
|---|---|---:|---|---|
| default ADMET endpoints | public ChEMBL / TDC / MoleculeNet lineage sources | T2 | source dependent | derived release intended with attribution; exact assay/version lineage is required per row |
| default computable | RDKit / Biopython functions on released inputs | T0-T1 | software/source dependent | yes for derived values; record software version and parameters |
| `admet_tdc` | Therapeutics Data Commons / MoleculeNet | T2 | MIT / source specific | intended with attribution; preserve dataset version and endpoint semantics |
| `affinity` | Davis et al. 2011 kinase benchmark lineage | T2 | public benchmark | intended with citation; record unit and transformation |
| `clinvar` | NCBI ClinVar | T4 | US public domain | yes; classification is a versioned submitter assertion, not adjudicated biological truth |
| `dna_promoter` | public promoter-sequence sets | T3 | source specific | source accession and construction policy required |
| DMS pilot and transfer candidates | legacy ProteinGym v0.1 derivatives; initial four-record lock in `signal/dms/mavedb_candidate_lock.v1.json`; 20-record screen in `signal/dms/mavedb_candidate_registry.v2.json`; CBS low-B6 status in `signal/dms/cbs_adapter_status.v1.json`; CBS low/high registry, complete source locks, public native-evidence lock, pair statuses, and uncertainty status in `signal/dms/cbs_b6_pair_registry.v1.json`, `cbs_low_b6_source_lock.v1.json`, `cbs_high_b6_source_lock.v1.json`, `cbs_b6_public_native_evidence_lock.v1.json`, `cbs_high_b6_adapter_status.v1.json`, `cbs_b6_pair_status.v1.json`, and `cbs_b6_uncertainty_status.v1.json` | T5 | legacy source specific; the MaveDB records reported CC0 in the 2026-07-26 validated source snapshots | every record remains `candidate_not_ingested`; both CBS condition snapshots are complete and hash-pinned but remain `COUNT_LINEAGE_PARTIAL`, with no derived outcome; the paired contract freezes a variant-aligned, no-imputation high-minus-low target and locally locks joint post-count TileSeq bootstrap, but cannot emit deltas or labels without an authenticated CBS-specific sample/dependency graph, processed-input boundary, runtime, QC, independent-block and stability gates, empirical-null/FDR/CI lineage, and registration; schema-2 admission additionally requires continuous effects, physical WT/control observations or resolvable lineage, a frozen sequence-derived family map, mutant-to-assay-WT construct pair IDs, and exact item-to-baseline links |
| `ecg` | ECG5000, UCR Time Series Archive | T3 | public archive | intended with citation; nonbiology control/medical time series |
| `generality` | periodic-table, mineral, material, and metabolite references | T0-T4 | source specific | source-specific attribution required |
| `graph` / `nmr` / `structure3d` | representations derived from hERG-task SMILES using RDKit | representation T1; label T2 | derived/source dependent | representation is software-defined; hERG target retains its empirical source lineage |
| `histo` | PatchCamelyon derived from Camelyon16 | T3 | CC0 / MIT lineage | intended with citation; retain patient/slide/site identifiers where license permits |
| `materials` | materials-formula data of Materials Project lineage | T3-T4 | CC-BY lineage | intended with attribution; record exact release and target construction |
| `methyl` | GEO GSE41037, Illumina 27K blood methylation | T2-T3 | public GEO record | derived summary intended with citation; record sample, cohort, preprocessing, and exact age target |
| `msa` | Pfam family alignments | T0 derived label | CC0 lineage | yes; conservation target is an algorithmic statistic of the released column under fixed rules |
| `ppi` | STRING-lineage protein interactions | T4 | CC-BY 4.0 lineage | intended with attribution; database association, not direct causal truth |
| `protein_meltome` | Meltome Atlas, Jarzab et al. 2020 | T2 | publication/data terms | intended with citation; record assay context, protein ID, and replicate policy |
| `rna` | Ensembl-lineage coding/noncoding sequences | T3-T4 | public database terms | intended with attribution; record release, transcript ID, and construction policy |
| `single_cell` | public PBMC scRNA-seq of 10x lineage | T3 | public dataset terms | intended with citation; record exact study, donor, cell barcode, preprocessing, and annotation |
| `withdrawn` | Mazuz et al., DrugWithdrawn (`eyalmazuz/DrugWithdrawn`) | T4 | public repository terms | intended with citation; historical database status |

Terms such as "public" do not replace a versioned source citation. Before a release claim, each row or
resolvable group must be tied to the exact source.

## Required provenance manifest

Each task release must provide a machine-readable manifest with:

- exact upstream title, URL or accession, release/version, and access date;
- raw source record ID for each entity;
- source checksum and derived-table checksum;
- truth level and target definition;
- assay or annotation type, organism or experimental system;
- units, threshold, label direction, and transformations;
- software version, parameters, tolerance, and random seed for T0-T1 derivations;
- replicate aggregation, uncertainty, conflict, duplicate, and missing-value policy;
- stable entity and condition IDs;
- biological dependency and split-group ID;
- representation-generation procedure;
- license, redistribution status, and required citation; and
- code commit that created the released row.

For database assertions such as ClinVar, also record submitter, review status, conflict state, and
release date. For empirical assays, retain assay identifiers and individual measurements where
redistribution permits rather than only a thresholded binary label.

## Task-specific release blockers

- **DMS:** the existing balanced four-protein derivative is pilot-only. It discards the continuous
  ProteinGym score and lacks a release-grade WT reference, replicate/QC lineage, globally
  namespaced protein-family map, mutant-to-assay-WT construct-pair contract, and exact functional
  baseline linkage. The 20-record MaveDB registry is a candidate screen, not a release; distinct genes
  and provisional family labels do not prove family disjointness. Only CBS and BRCA1 currently have
  exact substantive count-body locks in that registry, and count availability alone does not prove
  score recomputability. The CBS native adapter confirms that its 32 count channels are
  depth-normalized relative allele frequencies, not integer raw reads; its `controlNS`/`controlS`
  channels are sequencing-error controls, not a functional WT baseline; and its deposited score is
  an aggregate, not a replicate. On 2026-07-26, complete low- and high-B6 MaveDB source locks were
  materialized, reloaded, and independently cross-bound to exact OpenAPI, metadata, score, count,
  mapped-variant, mapping-summary, and target identities. Independent mapped-endpoint replays
  reproduced both decoded body hashes; no decoded response body was retained. The pair registry now
  freezes those source-lock artifacts, the 16 high-condition channels, exact core-body identities,
  target hash, primary `hgvs_nt` join, annotation agreement, codon-copy rule, missingness, and signed
  contrast. It explicitly rejects a paired-replicate interpretation, imputation, delta uncertainty
  without the locally selected joint bootstrap, and all outcome materialization. Complete source
  snapshots do not authenticate native TileSeq computation or convert normalized frequencies into
  raw reads. The official Additional File 3 workbook is now independently hash-locked and its six
  visible sheets are inventoried in `cbs_b6_public_native_evidence_lock.v1.json`. It exposes only
  ordinal replicate columns—eight per low-B6 measurement role and four per high-B6 role—with no
  formula, defined name, hidden sheet, external link, or custom XML that binds those columns to
  dose, culture, sequencing library/run, or tile. The paper-linked repositories provide
  publication-window code candidates and a generic CBS test fixture, but neither identifies the
  exact paper-executed revisions nor authenticates the B6-specific parameter sheet. The uncertainty
  prerequisite therefore locally hash-locks a 10,000-draw joint post-count TileSeq bootstrap with a
  registry-derived seed serialized as fixed-width lowercase hexadecimal while emitting only a
  no-body `not_derived` status. Its input scope is normalized relative-frequency channels, not
  FASTQ, alignment, integer-count generation, or depth normalization. The paper-reported two
  biological cultures per condition fail the contract's conservative eight-independent-block
  percentile-CI gate; 10,000 draws cannot increase biological support. Column order cannot
  establish pairing, and no fallback to zero covariance, direct covariance, or independent
  condition resampling is allowed without a versioned amendment. A separate bootstrap-runtime
  manifest and the published synonymous/nonsense empirical-null, FDR, and CI implementation must
  also be authenticated before any remediability label. Every admission still needs an assay-specific baseline,
  transformation, QC, duplicate/codon, and equivalence audit plus the frozen global family map. The
  current DMS transfer document is a contract and
  preregistration template: confirmatory use additionally requires an externally timestamped,
  authenticated registration binding the exact primary metric, confidence-interval superiority rule,
  item/class minima, bootstrap design, and multiplicity correction before target-label access.
- **ADMET/hERG:** generic endpoint names are insufficient. The six `admet/*` tasks and three
  alternate `herg/*` renderings are exploratory until exact assay source, protocol context, units,
  binary threshold, and replicate policy are documented.
- **Methylation:** the prompt target must exactly match the derived age label and preprocessing must be
  versioned.
- **Single-cell:** cell-level rows require donor/study provenance and donor/study-aware splits.
- **Histopathology:** retain patient/slide/site groups and state the central-region label definition.
- **Variant sequence:** the historical `variant/seq` rows lack stable IDs and explicit mutation
  identity. They are quarantined until regenerated with gene, position, REF, ALT, context, and
  release-specific assertion provenance.
- **Protein embedding:** `protein/esm2_emb` is exploratory because its current reference score 0.633
  does not pass the 0.65 predictability threshold.
- **NMR/3D:** a reference computed from original SMILES must be labeled external-information context,
  not a representation-matched ceiling.

Where an upstream row identifier is absent, the export uses `source_id=snapshot_row:<index>` and
declares the applicable `entity_id_scope`. Such IDs are deterministic only within that source-table
snapshot and do not satisfy the required upstream-record provenance field.

## Explicit exclusions

- **AlphaGenome-derived scores** under `signal/regulatory/` are not redistributed because their terms
  restrict use. Users regenerate them under their own authorization.
- Large re-fetchable databases under domain-specific `data/raw/` directories, including AlphaMissense,
  ClinVar releases, UniProt, and ProteinGym, remain outside the tracked tree.
- Secrets, provider keys, and restricted raw source artifacts are never part of the dataset release.

## Licensing

Repository code is Apache-2.0. Released derived datasets use CC-BY-SA 4.0 because some ADMET labels
derive from share-alike ChEMBL-lineage sources. This project license does not override upstream
terms. Users must cite and comply with every source represented in their selected configuration.
