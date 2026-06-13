#!/bin/bash
# One-time data setup for the protein branch on Cayuga. Run on a login node (CPU only):
#   bash setup_data_cayuga.sh
# Installs the MMseqs2 static binary (no conda/root), downloads FLIP meltome, binarizes Tm
# at the median, balances, clusters at PG_MINID identity, and writes data/protein_meltome.csv.
set -euo pipefail
cd "$(dirname "$0")"

PGE="${PGE:-$HOME/pge}"
MMDIR="$PGE/mmseqs"

if [ ! -x "$MMDIR/bin/mmseqs" ]; then
  echo "installing MMseqs2 static binary -> $MMDIR"
  mkdir -p "$PGE"
  url="https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz"
  curl -sL --max-time 300 -o "$PGE/mmseqs.tar.gz" "$url"
  tar -xzf "$PGE/mmseqs.tar.gz" -C "$PGE"
  rm -f "$PGE/mmseqs.tar.gz"
fi
export MMSEQS_BIN="$MMDIR/bin/mmseqs"
"$MMSEQS_BIN" version || { echo "mmseqs failed (try mmseqs-linux-sse41 if AVX2 is unsupported)"; exit 1; }

module load anaconda3/2023.09-3 2>/dev/null || true
source "$HOME/bge/venv/bin/activate"

export PG_N="${PG_N:-1500}"
export PG_MINID="${PG_MINID:-0.3}"
export PG_RAW="${PG_RAW:-meltome_mixed.fasta}"
export PG_OUT="${PG_OUT:-protein_meltome.csv}"
python -u prepare_data.py
echo "done -> $PG_OUT"
