# WS3: is there a cell where TRAIN (weights) wins? The "no cheap specialist" hypothesis

*2026-06-13. Tests the PROJECT_DESIGN section 3 open hypothesis: "for capabilities LACKING a cheap
specialist the weights lever would be the placement that delivers." Resolves it. No em dashes.*

## The test

TRAIN wins a cell only if a trained read-out's OUTPUT beats every alternative on the same cold
split: `max(cheap-featurizer, retrieve/ICL, orchestrate-specialist)`. The LoRA hERG-SMILES PoC
(`results/ws3_lora.md`) recovered output to 0.856, near but BELOW the cheap Morgan/k-NN ceiling
(0.90), so train lost there. The hypothesis is that a cell with NO cheap specialist would flip it.

Candidates = the rungs whose cheap featurizer is weak: NMR, MS, 3D coordinates (all hERG, the
benign anchor). For each we ask: is the cheap specialist really weak, and is there a HEAVY
specialist that orchestrate uses instead.

## Findings (local cheap ceilings,  scaffold GroupKFold)

| candidate | cheap featurizer (local) | 8B encode (best-layer) | retrieve/ICL | orchestrate specialist | TRAIN verdict |
|---|---|---|---|---|---|
| NMR -> hERG | binned 13C-shift histogram = **0.706** (sel +0.19) | 0.747 | 0.586 | elucidation -> Morgan 0.866 | NO: cheap ~ encode, orchestrate dominates |
| 3D coords -> hERG | shape descriptors ~0.61 | 0.669 | n/a | Morgan from the coords = 0.825 | NO: the coords ARE the structure, orchestrate is trivial |
| MS -> hERG | binned m/z histogram = 0.667 | 0.729 | 0.586 | SpecTUS elucidation -> Morgan ~0.72-0.825 | MARGINAL: weakest cheap, but elucidation specialist exists |

Two structural reasons train cannot win these:
1. **Where the cheap featurizer is WEAK (3D, MS), a HEAVY structure-recovery specialist exists.**
   For 3D the coordinates are the full molecule, so orchestrate just computes the Morgan fingerprint
   (0.825). For MS a structure-elucidation specialist (SpecTUS-class) recovers the structure, then
   Morgan. So "weak cheap specialist" does not mean "no specialist"; the heavy tool dominates.
2. **Where there is no structure to recover (NMR read as a surface peak list), the cheap featurizer
   reads the SAME surface the LLM does.** The local binned-shift histogram scores 0.706, essentially
   tied with the 8B encode (0.747 best-layer, and the unbiased held-out number is lower, cf. the
   SFM-embedding rung where best-layer 0.71 fell to 0.60). A trained read-out reads surface features
   the cheap featurizer already captures; it does not exceed them.

The hERG anchor makes this especially clean: hERG is a COARSE, surface-decodable property
(`results/spectra_rung.md`), so every representation of the molecule is either (a) invertible to a
structure that a fingerprint specialist reads, or (b) a surface a cheap featurizer reads. There is
no gap that only a trained LLM read-out fills.

## Conclusion: the weights-lever niche is EMPTY for property prediction

Train wins NOWHERE among the property-prediction cells, and the hypothesized "no cheap specialist"
escape does not exist: the weak-cheap cells retain a heavy structure-recovery specialist
(orchestrate), and the no-structure cells are read just as well by a cheap surface featurizer. A
trained LLM read-out is structurally a STRONG SECOND, never the exclusive winner, because it reads
the same surface a cheap featurizer reads and cannot out-compute a heavy specialist. This matches
the LoRA-SMILES result (recovered to, not past, the cheap ceiling) and is consistent across the
17-rung sweep.

## Tested: notation-invariance is ALSO orchestrate-won (the redirect was wrong)

A first revision proposed notation-invariance as train's home, with the variant text-vs-seq gap
(0.79 vs 0.58) as the ready target. TESTED 2026-06-13: it does NOT give train a win either, for two
reasons, both orchestrate:
1. **The property has a dominant specialist that GENERALIZES.** AlphaMissense (the `am` column of
   variant_clinvar.csv) scores AUROC 0.960 overall, 0.959 on known variants, and 0.985 on
   post-cutoff (novel) variants (n=98). So orchestrate not only beats the LLM, it beats it MOST on
   exactly the novel variants where a trained reader was supposed to have an edge (the GeneLab /
   spaceflight use case). A trained seq-reader, per the LoRA-SMILES precedent (recovers to, not
   past, the achievable ceiling), would land near the seq ceiling (ESM-1v ~0.92), still below
   AlphaMissense.
2. **The notation gap closes by CANONICALIZE-orchestration.** Reformatting the raw sequence +
   position to gene+HGVS (a deterministic conversion) yields the web-rich text form the model
   already grounds at 0.79. So even the notation conversion is an orchestrate placement.

The "train the invariance skill" framing wins only if you FORBID tool-use and demand the LLM ground
content intrinsically, a constraint the orchestrator architecture rejects.

## Final verdict: train wins nowhere; route, don't train

Across property prediction AND notation-invariance, the weights lever is empty: every
representation-grounding capability is covered at least as well by a non-train placement, a cheap
featurizer (Morgan / k-mer / histogram), a heavy specialist (AlphaMissense 0.96 / AlphaFold /
SpecTUS / structure-recovery to fingerprint), deterministic canonicalization (seq to HGVS), or
retrieval (ICL / k-NN). A trained LLM read-out reads the same surface a cheap featurizer reads and
cannot out-compute a heavy specialist, so it is structurally a strong-second everywhere. For a
tool-using science agent the conclusion is clean and closed-weight-friendly (cf.
`methylation-posttrain-poc`): the two live placements are RETRIEVE (knowledge / in-data-pattern,
even anonymized) and ORCHESTRATE (computation / specialist / canonicalize); TRAIN has no home.
(Caveat: this is "no home for the OUTPUT-property or notation capability." A genuine train-only
niche would need a capability with no tool, no lookup, and no canonical form at all; none has
surfaced in the 17-rung sweep.)

## Empirical confirmation (MEASURED 2026-06-13, Cayuga job 3040970)

The pre-registered LoRA-output confirmation ran on Cayuga (Qwen3-8B, `eval/cayuga_ws3_lora_cells.sbatch`,
data `eval/prep_lora_cells.py`; results in `results/ws3_lora_cells.json`). Measured:

| cell | base | finetuned (LoRA-8B) | bar TRAIN must STRICTLY beat | verdict |
|---|---|---|---|---|
| variant_seq | 0.515 | 0.652 (lift +0.137) | AlphaMissense 0.96 (novel 0.985); seq ceiling ~0.92 | LOSES decisively (distant 2nd) |
| spectra_ms | 0.536 | 0.706 (lift +0.17) | cheap 0.667, retrieve 0.586, elucidation ceiling 0.825 | NO WIN (firmed): 0.706 inside cheap's same-split CI; dominated by elucidation ceiling |

Under the strict criterion (beat cheap AND retrieve AND specialist), TRAIN wins NEITHER cell, so "train
wins nowhere" holds empirically. variant_seq is decisive: LoRA lifts the raw-sequence read-out
0.515 -> 0.652 (it learns, loss 0.80 -> 0.34) but lands far below AlphaMissense (0.96) and even the seq
specialist ceiling (~0.92), a distant second. spectra_ms is the honest NEAR-MISS and the most
interesting cell: LoRA reaches 0.706, edging the cheap binned-m/z featurizer (0.667) and retrieve
(0.586), so where structure elucidation is impractical (SpecTUS ~65%, GPT-4o exact 1.4%) the trained
read-out is the best PRACTICALLY-available placement. But the +0.04 over cheap is within noise at
n=481, and the elucidation specialist ceiling (0.825) strictly beats it, so it is not a clean win.

**Firming the spectra_ms borderline (`eval/firm_spectra_ms.py`, the EXACT same split, bootstrap CIs).**
Reproducing the LoRA's deterministic train/test split (769/481, confirmed identical) and scoring the
same n=481 held-out test set: the cheap binned-m/z probe is 0.668 [95% 0.619, 0.717] and the LoRA
0.706 sits INSIDE that CI, so the +0.04 edge is noise, not a real win; the perfect-elucidation
orchestrate ceiling (structure Morgan on the true SMILES) is 0.824 [0.787, 0.861], dominating both
(real MS elucidation, SpecTUS ~65%, lies between). So the lone borderline DISSOLVES: the trained
read-out is statistically indistinguishable from the cheap featurizer and dominated by the elucidation
ceiling.

Net: TRAIN WINS NOWHERE, now fully confirmed with no surviving borderline. variant_seq loses decisively
to AlphaMissense; spectra_ms, once firmed, is indistinguishable from the cheap m/z probe and below the
elucidation ceiling. The weights-lever niche is empty; the live placements are RETRIEVE and ORCHESTRATE.
(Honest-process note: the strong-second prediction was registered before the run; spectra_ms first
looked like a slight beat, but the same-split CI firming showed it was noise, the fifth measurement-vs-
prediction check of the session and the one that closed the last gap.)

## Status and confirmation

Local evidence: the NMR cheap ceiling (0.706, this run) and the 3D structure-recovery argument
settle two of three candidates. The marginal MS cell can be confirmed empirically by a LoRA-MS
output run on Cayuga (adapt `eval/ws3_lora.py` to the BRICS-MS representation), but the LoRA-SMILES
precedent (recovers to the cheap ceiling, not above) predicts a strong second, not a win.

## Caveats
8B encode numbers are best-layer (selection-biased upper bounds; the held-out-layer number is lower,
so the "train ~ cheap" gap is if anything wider). MS is a crude BRICS simulation. Single cold split.
The redirect (skills) is a hypothesis, not yet measured.
