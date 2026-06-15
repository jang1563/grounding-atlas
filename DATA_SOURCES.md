# Data sources and attribution

This repository and its companion dataset distribute small, derived
(representation, verifiable-property) tables built from public datasets. The
curation, the matched-pair construction, and the computable labels are released
under Apache-2.0; the underlying molecules, sequences, labels, and images retain
their original sources' terms. Cite the upstream source when you use a config.

"Redistributable" below is a good-faith assessment for non-commercial research
reuse with attribution, the norm for ML benchmark sharing. It is not legal advice;
the items marked **verify** should be confirmed before a public release.

| config / data | upstream source | terms | redistributable (derived, with attribution) |
|---|---|---|---|
| default (ADMET endpoints) | ADMET assays compiled in the author's Negative_result_DB from public sources (ChEMBL / TDC lineage) | source-dependent | yes (derived labels) — **verify** the compilation's upstream terms |
| default (computable) | RDKit / Biopython computed on the same molecules | deterministic functions | yes |
| `admet_tdc` (bace/bbbp/hiv) | Therapeutics Data Commons / MoleculeNet | MIT / CC | yes |
| `affinity` | Davis et al. 2011 kinase set | public benchmark | yes |
| `clinvar` | NCBI ClinVar | US public domain | yes |
| `dna_promoter` | public promoter sequences | public genomic | yes |
| `ecg` | ECG5000 (UCR Time Series Archive) | public | yes |
| `generality` | periodic-table / minerals / materials / metabolite reference sets | public reference | yes |
| `graph` / `nmr` / `structure3d` | derived from the hERG SMILES via RDKit (graph, simulated spectrum, 3D coords) | derived | yes |
| `histo` | PatchCamelyon (from Camelyon16) | CC0 / MIT | yes |
| `materials` | materials formulas (Materials Project lineage) | CC-BY | yes (attribute) |
| `methyl` | GEO GSE41037 (Illumina 27K blood methylation, 720 samples) | public GEO | yes (summary betas) — **verify** methylation-reuse norms |
| `msa` | Pfam protein-family alignments | CC0 | yes |
| `ppi` | protein-protein interactions (STRING lineage) | CC-BY 4.0 | yes (attribute) |
| `protein_meltome` | Meltome Atlas (Jarzab et al. 2020) | public | yes |
| `rna` | coding / noncoding RNA sequences (Ensembl lineage) | public | yes |
| `single_cell` | PBMC scRNA-seq | public (10x / atlas) | yes — **verify** the exact source's license |
| `withdrawn` | Mazuz et al., DrugWithdrawn (`eyalmazuz/DrugWithdrawn`) | public | yes |

## Explicitly excluded (kept out of the tracked tree)

- **AlphaGenome scores** (`signal/regulatory/*.csv`): AlphaGenome is free for
  non-commercial use only, so its derived scores are gitignored, not redistributed.
  Regenerate with `signal/regulatory/` and your own API key.
- Large re-fetchable reference DBs (`**/data/raw/`): AlphaMissense, ClinVar releases,
  UniProt, ProteinGym, re-fetched via the per-branch setup scripts.

## Before a public release

- **methylation** (GSE41037): assessed clear. Public GEO summary beta values from a
  published aging study, redistributable as a derived table with citation.
- **single-cell** (PBMC): assessed clear. Public PBMC scRNA-seq (10x lineage), freely
  reusable; cite the specific dataset.
- **ADMET compilation** (the `default` config, the largest): confirmed by the author to come
  only from public sources (ChEMBL / TDC / MoleculeNet lineage), no proprietary or
  no-redistribution source. Redistributable as a derived table with attribution.

License: because some ADMET labels derive from ChEMBL (CC-BY-SA 3.0), the derived data
inherits **share-alike**, so the datasets (the `signal/` tables here and the companion Hugging
Face dataset) are released under **CC-BY-SA 4.0**. The code stays Apache-2.0. AlphaGenome
scores are not redistributed (non-commercial).
