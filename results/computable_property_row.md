# Computable-property row: the execution axis of property knowledge

*2026-06-13. WS1 add-on. No em dashes. Reuses the axis-B 3-arm instrument (`eval/`), the
verifiable-signal schema (`signal/generate_signal.py`), and the NegResultDB / Meltome / promoter
sources. Companion code: `signal/generate_computable.py`, `eval/output_arm_computable.py`,
`eval/bridge_test_pi.py`, and the `ACT_PROMPT` / `ACT_PARSE` hooks in `eval/activation_arm.py`.*

## Claim

The 17 measured rungs all use EMPIRICAL properties (hERG, Tm, age, pathogenicity): the label
needs an experiment, the structure-probe ceiling is < 1.0, and the verbalization gap is set by
the web-exposure law (`PROJECT_DESIGN.md` section 7). This row adds the orthogonal column,
COMPUTABLE properties (atom count, ring count, molecular weight, length, GC content, pI), where
the label is a closed-form function of the representation (RDKit / Biopython computes it exactly),
so the structure-probe ceiling is 1.0 by construction. That makes the row a dissociation control:
it isolates a failure mode the web-exposure law does not govern.

Finding, in one line: **computable properties are snap-impossible but reasoning-solvable.** The
frontier model cannot emit a computable value as a gestalt (it must execute), and given enough
reasoning tokens it computes counting and summing properties at rank-perfect accuracy. The only
exceptions are properties that need empirical constants (pI, logP), which sit on the seam between
computable and empirical.

## Method

Same representation handed to the model, scored deterministically (no LLM judge). Two regimes,
switched by the token budget (`CMP_MAXTOK`):

- **snap** (max_tokens <= 32): force a bare number. Metric = FALLBACK RATE (fraction of items
  where no number is produced). Measures whether the property is gestalt-answerable.
- **reason** (max_tokens large): allow a reasoning trace, parse the FINAL number. Metric =
  COMPLETION RATE (finished / total) and ACCURACY-AMONG-COMPLETED (exact-match, MAE, AUROC).

Each computable property is also binarized at its median so AUROC lands on the same 0.5-to-ceiling
axis as every other rung. Sources: NegResultDB SMILES (1500 cpds), Meltome proteins (1500),
promoter DNA (1500, 251 bp). Model: claude-sonnet-4-6.

### Methodology trap (the reason this row almost reported the opposite result)

A first reason-regime pass capped max_tokens at 400 (chosen for speed). Every hard property hit
`stop=max_tokens` mid-enumeration, and the last-number parser grabbed a PARTIAL count, so
n_carbon looked like exact 0.30 / MAE 16 / AUROC 0.41 (below chance) and mol_wt 0.05 / MAE 418.
Inspecting raw outputs showed the model was counting correctly and simply running out of budget.
Two fixes: a truncation guard (`stop_reason == max_tokens` -> NaN "truncated", never parse a
partial), and a larger budget (2500 to 4000). At max_tokens 2000 the same n_carbon items finish
and are correct (true 22 -> 22, 24 -> 23).

General lesson: **measuring a computable property without enough reasoning budget manufactures a
spurious "model failure."** A capability eval must give the model room to execute, inspect raw
outputs before trusting aggregates, and treat a below-chance AUROC as a red flag, not a result.
This is the honest-hype-check theme (`docs/field_message.md`) in miniature.

## Results (corrected: truncation guard ON, 2500-4000 tokens, 100% completion)

| modality | property | computation type | snap could-not-answer | reason exact | MAE | reason AUROC |
|---|---|---|---|---|---|---|
| smiles | mol_wt | sum atomic masses | 86% | 0.86 | 3.3 g/mol | **1.00** |
| smiles | n_carbon | count atoms | 73% | 0.64 | 0.36 | **1.00** |
| smiles | n_hbd | count N-H/O-H | 53% | 0.71 | 0.43 | 0.94 |
| protein | length | count residues (<=512) | n/a | 0.58 | 1.08 | **1.00** |
| dna | gc_content | count ratio over 251 bp | n/a | 0.67 | 2.5 pp | **1.00** |
| dna | n_a | count A over 251 bp | n/a | 0.25 | 3.75 | **1.00** |
| protein | isoelectric_point | Henderson-Hasselbalch + pKa | n/a | 0.25 | 1.28 pH | **0.79** |

(snap fallback gradient, N=120: logp 99%, n_hba 97%, mol_wt 86%, n_carbon 73%, n_hbd 53%,
n_aromatic_rings 38%. Even the most pattern-like computable property cannot be reliably
gestalt-answered.)

## Three-way structure within "computable"

1. **Snap-impossible.** Unlike an empirical gestalt (a hERG probability the model emits as one
   number), every computable property needs execution; under a 16-token cap the model opens a
   reasoning preamble and never reaches a number (53 to 99% fallback).
2. **Counting and summing: reasoning closes them.** n_carbon, mol_wt, length, gc_content, n_a all
   reach AUROC 1.0 (rank-perfect). Exact-match is high for short tallies (mol_wt 0.86) and degrades
   with tally length (n_a over 251 bases: exact 0.25, MAE 3.75, but AUROC still 1.0, so it ranks
   perfectly while miscounting by a few). The only ceiling here is the reasoning-token budget,
   which scales with instance size; giant molecules and long sequences would truncate even at 4000.
3. **Formula needing empirical constants: only partial.** isoelectric_point is the single property
   with AUROC < 1.0 (0.79), at 100% completion (not truncation). A raw-trace spot-check shows the
   model does not run rigorous Henderson-Hasselbalch; it counts charged residues and ESTIMATES pI
   from charge balance (true 5.40 -> 6.49, 9.14 -> 9.5). The bottleneck is the empirical constants
   (pKa values), which the model recalls only approximately. logP (Crippen fragment table) is
   expected to behave the same way.

### Bridge test (does supplying the constants close pI?)

**Result (+pKa table supplied, rigorous Henderson-Hasselbalch instructed):** at N=40, ranking AUROC
rose only modestly, 0.79 -> 0.834 (parsed 36/40), with exact-match still ~0.19 and MAE ~1.34 pH.
An initial N=12 gave a noisy 0.97 that the larger run CORRECTED downward, exactly the small-sample
over-claim the N=12 caveat warned about (and the reason the larger-N run was worth doing). So
supplying the empirical constants gives a SMALL ranking lift (+0.04), NOT closure. A raw-trace
spot-check shows why: the model still commits gross charge-balance errors with the constants in
hand. One protein (true pI 5.40): it counted 34 positive vs 44 negative groups (net acidic, so
pI ~5), yet concluded "basic, pI approx 9.7", an error of ~4 pH in the wrong direction. So the
seam is mainly EXECUTION-reliability limited (the Henderson-Hasselbalch root-solve), with the
constant-recall component real but small. Net: pI sits on the computable-empirical seam, and
neither lever closes it (the scale curve below reaches only ~0.84 at opus).

### Scale / recovery curve (haiku to opus, `results/computable_scale_sweep.json`)

Running the reason regime across the scale ladder separates two computable patterns, the analog
of section 7.3's scale-closable vs scale-invariant split for empirical rungs:

| property | haiku | sonnet | opus | pattern |
|---|---|---|---|---|
| n_carbon (counting) AUROC | 0.98 | 1.00 | 1.00 | rank closes early; PRECISION climbs (exact 0.30 -> 0.80 -> 0.90, MAE 1.2 -> 0.1) |
| isoelectric_point (seam) AUROC | 0.50 | 0.80 | 0.84 | rises then SATURATES below 1.0; exact stays ~0.2 (MAE ~1 pH) |

Counting is scale-closable like a web-rich empirical rung: the ranking is solved from haiku up,
and scale buys exact-count precision. The pI seam is a third pattern: scale partially closes it
(chance at haiku to 0.84 at opus) but saturates well short of counting, and the exact value stays
approximate. Combined with the bridge test (supplied constants give only a +0.04 lift to 0.83 at
N=40, the N=12 0.97 being small-sample noise), neither lever closes the seam: scale reaches 0.84,
constants reach 0.83, both short of counting's 1.0. It is bottlenecked mainly by reasoning
reliability (the root-solve), with constant-recall a small secondary component. Within computable, then:
counting / summing close with reasoning and scale; formula-plus-constants only partially closes,
needs a tool for the constants, and still carries residual execution error.

## Framework: two orthogonal axes, meeting at a seam

| | snap-answerable? | accuracy source | closes via |
|---|---|---|---|
| EMPIRICAL (hERG, Tm, the 17 rungs) | yes (gestalt) | web-documented knowledge | retrieve / scale (web-exposure law) |
| COMPUTABLE (count, sum, length) | no (must execute) | reasoning execution | reasoning tokens or a tool |
| SEAM (pI, logP) | no | algorithm + memorized constants | barely: supplied constants give only +0.04 (AUROC 0.79->0.83, N=40), scale only ~0.84; execution-reliability-bound, neither lever closes it |

This row anchors the execution corner that the 17 empirical rungs never touched, and it BOUNDS the
web-exposure law: web-exposure governs empirical verbalization, not computable. The two axes are
not fully orthogonal, because pI/logP need a small empirical (memorized, hence web-sensitive)
constant component. The result aligns with `decision_map_placement.md` (computable -> orchestrate /
compute lane) and `docs/field_message.md` (the frontier model executes and routes, it does not
need to "know").

## Caveats

- Single model (claude-sonnet-4-6), small N (12 to 40 per property), moderate-size instances. The
  token-budget ceiling binds with instance size and was not characterized here (giant molecules /
  long sequences truncate even at 4000).
- AUROC is on the median-binarized label; exact-match and MAE are on the raw value among completed
  items. snap fallback rates are from a larger separate sample (N=120), descriptive context only.
- A reasoning model with extended thinking (opus / o-series) is the natural follow-up for the
  recovery curve and for whether the pI seam closes with more deliberation alone.

## Reproduce

```
python signal/generate_computable.py --modality smiles --props all
python signal/generate_computable.py --modality protein --source protein_grounding/data/protein_meltome.csv
python signal/generate_computable.py --modality dna --source signal/dna_promoter.csv --seq-col smiles
# reason regime, truncation guard (CMP_MAXTOK > 32 selects it):
CMP_N=15 CMP_MAXTOK=2500 CMP_FORCE=1 python eval/output_arm_computable.py
CMP_MAXTOK=4000 CMP_MODALITY=protein python eval/output_arm_computable.py
python eval/bridge_test_pi.py        # the pI constants-supplied bridge test
```

Files: `signal/computable/<modality>/<prop>/` (pairs.jsonl + verifiability.json + per-prop CSV),
`results/output_arm_computable_*_reason.json` (max_tokens field = provenance),
`results/output_arm_computable_smiles_snap.json`, `results/bridge_test_pi.json`.
