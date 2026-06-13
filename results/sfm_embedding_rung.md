# SFM-embedding rung: can the LLM read a property out of a specialist's embedding?

*2026-06-13. The orchestrate-condition input (PROJECT_DESIGN 7.3 confound 2, 7.4 "widest-open, no
behavioral baseline"). Capability-neutral: ESM-2 embeddings of Meltome proteins -> thermostability.
Code: `eval/sfm_embed_meltome.py` (stage 1, embeddings + ceiling), `eval/sfm_embedding_output.py`
(stage 2, LLM arms). No em dashes.*

## Question

The decision map says "orchestrate the heavy specialist." But orchestrate HOW: paste the
specialist's output (an embedding) into the prompt, or put a trained head on it? This rung feeds
an ESM-2 embedding to the LLM and measures whether the LLM can read a property from it, two ways:
zero-shot (raw vector as text) and few-shot ICL (labeled example vectors in context). No behavioral
baseline for this existed.

## Setup

ESM-2 (esm2_t30_150M, 640-dim, mean-pooled) of 320 Meltome proteins, generated locally on MPS
(no HPC). Property = thermostability, tercile-extreme label (top vs bottom third of Tm, the median
split is too diluted: ceiling 0.633, terciles 0.754, quartiles 0.835). Ceiling = a probe on the
embedding under a Meltome-cluster GroupKFold (leakage control). Capability-neutral; the dual-use
ESM panel was deliberately not used.

## Results (sonnet-4-6, query n=50, K=24 ICL examples, PCA-16)

| arm | AUROC |
|---|---|
| ceiling: probe on full 640-dim embedding | 0.811 |
| ceiling: probe on PCA-16 embedding | 0.846 |
| LLM activation: 0.5B hidden states on embedding-text, best layer | 0.710 (selective +0.22) |
| LLM zero-shot, raw 640-dim embedding as text | 0.466 (chance) |
| LLM few-shot ICL, PCA-16 vectors + 24 labeled examples | 0.562 (at / near chance, n=50) |
| raw-sequence output arm (existing protein rung) | 0.486 (8B) / 0.585 (opus) |

## Read

The signal is strongly present in the embedding (ceiling 0.81 to 0.85), and the LLM cannot read it
either way. Zero-shot is chance (0.47): a raw SFM embedding handed as text is opaque, the same
web-zero-numeric-vector pattern as methylation and anonymized single-cell ids. Few-shot ICL barely
moves it (0.56, within noise of chance at n=50) and stays far below the PCA-16 ceiling (0.85): with
24 labeled example vectors the LLM still cannot learn the decision boundary in an abstract embedding
space in context.

But the LLM DOES encode it internally. A linear probe on a 0.5B model's hidden states reading the
same embedding-text recovers Tm at 0.710 (selective +0.22, best layer), well above its own output
(0.47 to 0.56) and within ~0.12 of the ceiling. So this is an EXPRESSION gap like the numeric-vector
rungs (methylation, single-cell-anon): the embedding-text signal IS carried into the activations, it
is just not verbalized or usable through the prompt. The orchestrate failure is a verbalization /
read-out gap, not an encoding gap, and the fix is a trained head (on the embedding, or on the LLM's
activations), not prompting (zero-shot or ICL).

**Consequence for the decision map.** Orchestrating an SFM means calling it and putting a TRAINED
read-out head on its output (the ceiling probe), NOT pasting the embedding into the prompt. The LLM
is not an in-context decoder of an abstract embedding space. This sharpens "orchestrate the
specialist" (`decision_map_placement.md`): the specialist's value reaches the answer through a
trained head, not through the LLM's context window.

**Contrast to flag (not yet controlled).** Methylation, an INTERPRETABLE numeric vector (per-CpG
beta values), is reported ICL-closable (`methylation-posttrain-poc`, 0.40 -> 0.93 via retrieve/ICL),
while this ABSTRACT ESM embedding is not (0.56). The hypothesis: in-context decoding works on
numeric vectors whose dimensions carry real-world meaning or whose neighbors can be retrieved, but
not on a learned embedding whose axes are opaque. A direct same-harness comparison would confirm it.

## Caveats

All three arms now measured, but the activation arm is a 0.5B LOCAL PROXY (Qwen2.5-0.5B on MPS; the
26GB box cannot host the 8B), and its best-layer AUROC is a selection-biased upper bound (no nested
held-out-layer correction here, unlike `activation_arm.py`). Qwen3-8B on Cayuga is the comparable
run (same script, ACT_MODEL=Qwen/Qwen3-8B) and would likely push activation closer to the ceiling.
n=50 query makes the ICL AUROC noisy (treat 0.56 as at-chance, not a real lift). One property (Tm
extremes), one ESM size (150M), one model (sonnet-4-6). ICL sees 24 examples vs the probe's 164, so
it is not a sample-matched comparison; the qualitative result (chance vs a 0.85 ceiling) is the
claim, not the exact ICL number. A larger n / K and a 650M embedding would tighten it.

## Files
`signal/sfm_embedding/meltome_esm2.npz` (+ `_ceiling.json`), `results/sfm_embedding_output.json`,
`results/run_sfm_embed.log`, `results/run_sfm_output.log`.
