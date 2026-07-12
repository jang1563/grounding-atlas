# Capability x web-exposure, and the a-priori permissioning lever

*2026-06-19. A short synthesis binding three measured results: the web-exposure effect, how model
capability interacts with it, and the deferral rule that follows. Figures in
[`results/benchmark/single_cell/`](../results/benchmark/single_cell/).*

> **Evidence scope (see [`REPORT.md`](REPORT.md)).** Hidden-state probes here use open-weight models;
> the frontier ladder is output-only. The corrected mechanism is a capability-dependent mix of token
> familiarity/reasoning and mapping documentation, not a single factor.

## Observed grounding gaps

Across the 17-representation pilot sweep, open-model probes often recover linearly decodable signal
that open-model output does not surface. Frontier output is measured separately, so this is not a
same-model frontier encoding claim. The output differences are associated with a capability-dependent
mix of token familiarity/reasoning and mapping documentation.
The strongest matched comparisons are methylation vs MSA
(similar task shape, strong probe results, opposite output), and single-cell with gene names vs
anonymized ids. These comparisons support the multi-factor account but do not isolate one cause. See
[`results/SYNTHESIS.md`](../results/SYNTHESIS.md).

## Capability x web-exposure: the interaction

The observed web-exposure effect leaves a question: does a bigger or newer model close the gap? We measured it
cleanly on single-cell classification. Each cell becomes a cell-sentence (its top genes), presented
two ways: real **gene names** (web-documented) or **global-consistent anonymized ids** (the same
expression vector, only the human-readable name removed, so a specialist still separates the classes
at CV-AUROC ~0.99). We ran a within-family capacity ladder (Haiku 4.5 -> Sonnet 4.6 -> Opus 4.8) plus
GPT-4o, on two substrates chosen to need the symbol -> cell-type prior rather than one marker:
CD8-T vs NK and CD14+ vs CD16+ monocyte.

The result (figure `interaction.png`, n=200/model, 95% bootstrap CI) is an interaction on both
substrates:

- The **gene-name** AUROC rises monotonically with capability: CD8-T/NK 0.826 -> 0.871 -> 0.978;
  monocyte 0.763 -> 0.929 -> 0.983, Opus near the specialist ceiling.
- The **anon** AUROC stays pinned at chance at every tier, even Opus (CIs straddle 0.5).
- The gap widens with capability in this panel; with the names removed, no tested tier moves off
  chance.

The anon arm shows that readable gene symbols matter, but it does not separate documentation from
token familiarity and reasoning. A later marker-depleted control found that the relative contribution
changes with capability: stronger models retained more performance from familiar but less canonical
gene sets. A direct three-arm open-model probe (Qwen2.5-0.5B) supplies same-model pilot evidence on its
own activations and output; the frontier ladder remains output-only.

## The permissioning lever

If per-input competence cannot be read from the model, what decides when to trust it? We compared two
deferral signals on the pooled name+anon items (figure `deferral.png`): the model's own
**self-confidence** (|P-0.5|) versus an **a-priori web-exposure tag** (answer name, defer anon),
which is knowable before any model call.

Self-confidence is **capability-dependent, and that is the problem**. The well-calibrated models lower
their confidence on anon (Opus 0.25 -> 0.03, Sonnet 0.22 -> 0.07): they know they are guessing, so
confidence routing works for them. But Haiku is equally confident on anon (0.35 -> 0.35) and GPT-4o is
more confident (0.16 -> 0.22): confidently wrong, so confidence routing collapses to ~chance
(accuracy at 50% coverage 0.54 / 0.56). In this panel, the **a-priori tag is competitive** (accuracy
at 50% coverage 0.75 to 0.82; AURC wins or ties for the tested models). This is a pilot result, not a
model-invariant guarantee.

## Why it matters

The practical conclusion is to use input-derived tags as one prior in a calibrated routing policy.
They do not replace model confidence, real specialist predictions, or deployment-specific validation.
The per-item study remains the binding result: routed accuracy is 0.81 versus a 0.91 oracle, and the
best thresholds call specialists on most items.

## Scope and relation to prior work

Pilot scale (n=200, two PBMC pairs, output arm). The within-Claude-4 ladder is the cleanest available
capacity axis; the open-weight 8B foot of the curve is in `SYNTHESIS.md`. The interaction is the
biology-domain, encoded-equal-by-construction version of a known natural-language phenomenon: accuracy
tracks pretraining frequency and scale helps the head but not the tail (Kandpal et al. 2023; Mallen et
al. 2023). The gene-name-vs-anonymized control has an encoder-side precedent (Mahbub et al. 2025); the
a-priori-tag-vs-self-confidence comparison is the piece this adds, and it refines the finding that
confidence can beat input difficulty for abstention in some models.
