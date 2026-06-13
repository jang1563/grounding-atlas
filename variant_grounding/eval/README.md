# variant_grounding/eval - the variant-effect branch of the axis-B instrument

The same grounding instrument as the SMILES branch (`../../eval/`), run on genetic missense
variants. The headline is the **dual-form** within-modality test: the same variant in a web-rich
text form and a web-poor sequence form (`../README.md`).

| Arm | SMILES branch | variant branch (here) | Question |
|---|---|---|---|
| ceiling (specialist) | Morgan fingerprint + LogReg | **AlphaMissense score** (primary), **ESM-1v LLR** (secondary) | is pathogenicity in the variant content? |
| LLM-output | LLM generates a probability | same, in **two surface forms** (text / sequence) + controls | does it VERBALIZE it, and from which form? |
| LLM-activation (optional) | linear probe on hidden states | same, per form, gene-grouped split (Cayuga GPU) | does it ENCODE it internally? |

This branch leads with the **2-arm** result (ceiling + output). The activation arm is the
optional GPU extension.

## Data and the self-consistent triple

`prepare_data.py` builds one row per ClinVar variant carrying all three forms self-consistently:

- **ClinVar** (`variant_summary`, pinned 2026-06 release): GRCh38 single-AA missense, germline
  classification `Pathogenic`/`Likely pathogenic` (1) vs `Benign`/`Likely benign` (0), review
  status star >= 1, single gene. The p.HGVS three-letter change is parsed from `Name`.
- **UniProt** (human reviewed SwissProt FASTA): gene -> canonical accession + sequence. The
  **WT-residue consistency QC** drops any variant where the UniProt residue at the position does
  not equal ClinVar's wild-type AA. This makes every kept variant self-consistent with the
  sequence window AND with AlphaMissense (which is UniProt-canonical based). 184,534 of 198,197
  candidates map (93%); 13,310 dropped on residue mismatch (isoform/coordinate differences).
- **AlphaMissense** (`aa_substitutions`, streamed once): the `am_pathogenicity` score per
  (accession, one-letter sub). Coverage 98.5%.

Outputs: `variant_clinvar_full.csv` (all 184k mapped, with strata flags), `variant_clinvar.csv`
(balanced 1000/1000 main sample), `variant_clinvar_post2026_01.csv` (balanced strict temporal
holdout). Columns: `id,gene,hgvs_p,uniprot,pos,wt,mut,sub1,label,stars,rsid,first_seen,post_cutoff,am,seq_len,win_pos,wt_window`.

`prepare_dms.py` builds the leakage-free parallel track from ProteinGym DMS (BRCA1, PTEN, TP53,
MSH2: clinically famous genes, so memorized text recall is strongest there). Same schema, label =
1 if damaging (low experimental fitness), oriented and sanity-checked against AlphaMissense.

## Leakage control (the five partitions, see `../README.md`)

1. **Temporal holdout** = `first_seen`/`post_cutoff`, built by diffing the 2025-06, 2026-01, and
   2026-06 ClinVar releases. `post_2026_01` (variants absent from the 2026-01 snapshot) is the
   strict holdout. 10,072 such variants.
2. **Star stratification** = `stars`; every arm reports star-1 vs star-2+ separately.
3. **DMS parallel track** = `prepare_dms.py` (experimental fitness, not a memorizable label).
4. **Gene-name scramble** = the `text_scramble` condition in `output_arm_variant.py`.
5. **Sequence-vs-text obfuscation** = the `text` vs `seq` conditions (the headline).

For the optional activation arm, the leakage control is a **GroupKFold grouped by gene**, so no
variant of a training gene appears in test (the within-modality analog of the scaffold split).

## Scripts

| File | What | Where |
|---|---|---|
| `prepare_data.py` | ClinVar -> UniProt -> AlphaMissense -> balanced CSVs + temporal flag | login node (CPU) |
| `prepare_dms.py` | ProteinGym DMS -> the leakage-free parallel track | login node (CPU) |
| `setup_data_cayuga.sh` | download all raw data + run both prep scripts | login node (CPU) |
| `ceiling_gate_variant.py` | AlphaMissense AUROC, stratified by stars + temporal bin (the gate) | CPU |
| `ceiling_esm1v_variant.py` | ESM-1v WT-marginal LLR ceiling (secondary, unsupervised) | GPU |
| `run_ceiling_esm1v_cayuga.sh` | sbatch for the ESM-1v ceiling | a40 |
| `output_arm_variant.py` | the dual-form (text/seq) output arm + gene-scramble, anchored parser | CPU/API |
| `activation_arm_variant.py` | OPTIONAL: per-form linear probe on hidden states, gene split | GPU |
| `run_activation_cayuga.sh` | sbatch for the activation arm (override `ACT_MODEL` for the panel) | a40 (h100 alt) |

## Run order

```bash
# 0. data (CPU). Locally: the two prepare_*.py after downloading data/raw; on Cayuga:
bash setup_data_cayuga.sh

# 1. ceiling gate (is the signal in the content?) - CPU, instant
python ceiling_gate_variant.py

# 2. the dual-form output arm (the headline) - API. Load keys first, never print them:
set -a; source ~/.api_keys; set +a
VG_CSV=../data/variant_clinvar.csv VG_COND=text,seq,text_scramble,text_nogene \
  VG_MODEL=claude-sonnet-4-5-20250929 python output_arm_variant.py        # main + strata
VG_CSV=../data/variant_clinvar_post2026_01.csv VG_COND=text,seq python output_arm_variant.py  # strict holdout
VG_CSV=../data/variant_dms.csv VG_COND=text,seq \
  VG_TASK_Q="damaging to protein function or tolerated" VG_TASK_POS="DAMAGING to protein function" \
  python output_arm_variant.py                                            # DMS parallel track

# 3. (optional GPU) secondary ceiling + activation arm on Cayuga
sbatch run_ceiling_esm1v_cayuga.sh
sbatch run_activation_cayuga.sh
```

The output arm runs against any provider via `VG_PROVIDER` (anthropic|openai) and `VG_MODEL`.
Reuses the SMILES branch venv (`~/bge/venv`, no `kernels` package) and HF cache for the GPU arms.

## What is reused verbatim from `../../eval/`

`parse_prob` (anchored last-number parser, percent handling, fallback count) and the
parsed/percent/fallback instrumentation from `head_to_head.py`; `bootstrap_ci`, `load_model`,
`chat_input`, and the per-layer best-layer-as-max-over-layers probe shape from
`activation_arm.py`. Swapped: Morgan FP -> AlphaMissense / ESM-1v; Murcko scaffold -> gene group;
single SMILES prompt -> the dual-form (text / sequence) prompt pair.
