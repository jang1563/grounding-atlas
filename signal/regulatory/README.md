# signal/regulatory/ (AlphaGenome ceiling)

The specialist-ceiling arm for the regulatory rung (`docs/ALPHAGENOME_CEILING_DESIGN.md`).
`alphagenome_score.py` scores variants with [AlphaGenome](https://deepmind.google.com/science/alphagenome)
and writes a per-variant scalar ceiling, used as the regulatory-effect label.

## Setup
```bash
pip install -U alphagenome
# free non-commercial key:
export ALPHA_GENOME_API_KEY=...   # https://deepmind.google.com/science/alphagenome
```

## Run
```bash
python signal/regulatory/alphagenome_score.py             # variants_demo.csv (smoke test)
AG_VARIANTS=known_plus_novel.csv python signal/regulatory/alphagenome_score.py
```

Input CSV columns: `chrom,pos,ref,alt,rsid,set` where `set` is `known_eqtl`
(web-rich, eQTL/GWAS-documented) or `novel` (web-poor, unannotated). Output:
`variant_scores.csv` with `max_abs_raw` / `max_abs_quantile` / `max_abs_raw_rnaseq`.

## Pre-check (the cheap first step in the design doc)
Confirm the known_eqtl vs novel split is separable on the AlphaGenome score
before building the LLM output / activation arms. The script prints the
per-`set` mean. Then the rung asks: does the LLM verbalize the known set (and
collapse under a scrambled rsID, `eval/notation_control.py`) while staying at
chance on the novel set that AlphaGenome reads (the orchestrate-won prediction)?

## Notes
- AlphaGenome scores are predictions, not measured ground truth: treat the label
  as predicted-structural with a proxy-gold gap, and stratify by cell-state data
  density (`docs/DATA_DENSITY_RUNG_DESIGN.md`), since AlphaGenome is itself
  trained on cell-line-biased public data.
- Free for non-commercial use only (DeepMind terms). Not committed: API keys, scores.
