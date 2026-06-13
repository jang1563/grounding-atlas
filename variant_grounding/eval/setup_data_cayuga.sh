#!/bin/bash
# One-time data setup for the variant branch. Runs on a login node (CPU only):
#   bash setup_data_cayuga.sh
# Downloads UniProt human reviewed FASTA, three dated ClinVar releases (current + two cutoff
# boundaries for the temporal holdout), AlphaMissense precomputed aa-substitution scores, the
# ProteinGym DMS reference + a few clinically famous gene DMS files; then runs prepare_data.py
# and prepare_dms.py. Pure Python + curl, no GPU, no MMseqs.
set -euo pipefail
cd "$(dirname "$0")"
RAW="../data/raw"
mkdir -p "$RAW/dms" "../results/logs"

dl() { [ -s "$RAW/$2" ] || curl -sL --retry 3 --max-time "${3:-1800}" -o "$RAW/$2" "$1"; echo "  have $2"; }

echo "downloading raw data -> $RAW"
dl "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true+AND+organism_id:9606&format=fasta&compressed=true" uniprot_human_sprot.fasta.gz 600
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-06.txt.gz" clinvar_2026-06.txt.gz
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz" clinvar_2026-01.txt.gz
dl "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2025-06.txt.gz" clinvar_2025-06.txt.gz
dl "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz" AlphaMissense_aa_substitutions.tsv.gz 2400
dl "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv" DMS_substitutions_reference.csv 300

PGBASE="https://huggingface.co/datasets/ICML2022/ProteinGym/resolve/main/ProteinGym_substitutions"
for id in BRCA1_HUMAN_Findlay_2018 PTEN_HUMAN_Mighell_2018 P53_HUMAN_Kotler_2018 MSH2_HUMAN_Jia_2020; do
  [ -s "$RAW/dms/${id}.csv" ] || curl -sL --max-time 300 -o "$RAW/dms/${id}.csv" "$PGBASE/${id}.csv"
  echo "  have dms/${id}.csv"
done

# venv reused from the SMILES branch (torch / transformers / sklearn; no `kernels` package).
source "$HOME/bge/venv/bin/activate" 2>/dev/null || echo "  (activate a venv with numpy/sklearn first)"
python -u prepare_data.py
python -u prepare_dms.py
echo "done -> ../data/variant_clinvar.csv, variant_clinvar_post2026_01.csv, variant_dms*.csv"
