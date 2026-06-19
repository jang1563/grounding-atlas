# Capability x web-exposure, and the a-priori permissioning lever

*2026-06-19. A short synthesis binding three measured results: the web-exposure law, how model
capability interacts with it, and the deferral rule that follows. Figures in
[`results/benchmark/single_cell/`](../results/benchmark/single_cell/).*

## The law (recap)

Across 17+ representations, LLMs encode far more biology than they verbalize: a linear probe on an
open model's hidden states recovers a property near a specialist ceiling, but the verbalized answer
lags. The size of that verbalization gap is governed by how web-documented the representation ->
property mapping is, not by the modality. The controlled proofs are within-pair: methylation vs MSA
(identical task shape, both encoded to ceiling, opposite output), and single-cell with gene names vs
anonymized ids. See [`results/SYNTHESIS.md`](../results/SYNTHESIS.md).

## Capability x web-exposure: the interaction

The web-exposure law leaves a question: does a bigger or newer model close the gap? We measured it
cleanly on single-cell classification. Each cell becomes a cell-sentence (its top genes), presented
two ways: real **gene names** (web-documented) or **global-consistent anonymized ids** (the same
expression vector, only the human-readable name removed, so a specialist still separates the classes
at CV-AUROC ~0.99). We ran a within-family capacity ladder (Haiku 4.5 -> Sonnet 4.6 -> Opus 4.8) plus
GPT-4o, on two substrates chosen to need the symbol -> cell-type prior rather than one marker:
CD8-T vs NK and CD14+ vs CD16+ monocyte.

The result (figure `interaction.png`, n=200/model, 95% bootstrap CI) is a clean interaction on both
substrates:

- The **gene-name** AUROC rises monotonically with capability: CD8-T/NK 0.826 -> 0.871 -> 0.978;
  monocyte 0.763 -> 0.929 -> 0.983, Opus near the specialist ceiling.
- The **anon** AUROC stays pinned at chance at every tier, even Opus (CIs straddle 0.5).
- The gap widens with capability. Web-exposure and capability **multiply**: capability is the engine,
  web-exposure is the fuel. A bigger engine goes further only where there is fuel; with the names
  removed, no capability tier moves off chance.

The anon arm is also the confound control. If Opus's name gain were just "a bigger or more recent
training corpus," a more capable model would also do better on anon (it is a better pattern-finder).
But Opus is at chance on anon. So the gain is specifically reading web-documented symbols, not generic
capacity, and the specialist's ~0.99 shows the discriminative information is in the vector the whole
time. Provider-invariant: GPT-4o sits on the same curve.

## The permissioning lever

If per-input competence cannot be read from the model, what decides when to trust it? We compared two
deferral signals on the pooled name+anon items (figure `deferral.png`): the model's own
**self-confidence** (|P-0.5|) versus an **a-priori web-exposure tag** (answer name, defer anon),
which is knowable before any model call.

Self-confidence is **capability-dependent, and that is the problem**. The well-calibrated models lower
their confidence on anon (Opus 0.25 -> 0.03, Sonnet 0.22 -> 0.07): they know they are guessing, so
confidence routing works for them. But Haiku is equally confident on anon (0.35 -> 0.35) and GPT-4o is
more confident (0.16 -> 0.22): confidently wrong, so confidence routing collapses to ~chance
(accuracy at 50% coverage 0.54 / 0.56). The **a-priori tag is uniformly safe** (accuracy at 50%
coverage 0.75 to 0.82; AURC wins or ties for every model), because deferring a web-zero input needs no
model call and never depends on the model knowing its own limits.

## Why it matters

The conclusion is a calibrated-permissioning rule for biological AI: you can decide, before any model
call, whether to trust a frontier model on a given representation, from how web-documented that
representation -> property mapping is. It is computable a priori and model-invariant, whereas the
model's self-report is unreliable in a way that is itself unpredictable across models. The one-line
prescription: **do not ask the model, look at web-exposure.**

## Scope and relation to prior work

Pilot scale (n=200, two PBMC pairs, output arm). The within-Claude-4 ladder is the cleanest available
capacity axis; the open-weight 8B foot of the curve is in `SYNTHESIS.md`. The interaction is the
biology-domain, encoded-equal-by-construction version of a known natural-language phenomenon: accuracy
tracks pretraining frequency and scale helps the head but not the tail (Kandpal et al. 2023; Mallen et
al. 2023). The gene-name-vs-anonymized control has an encoder-side precedent (Mahbub et al. 2025); the
a-priori-tag-vs-self-confidence comparison, framed as biosafety permissioning, is the piece this adds,
and it refines the finding that confidence can beat input-difficulty for abstention (true here only for
the well-calibrated models, while the a-priori signal is robust by construction).
