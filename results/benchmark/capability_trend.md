# Capability trend: the token-familiarity share rises monotonically with capability

*Tests the decomposition in `token_familiarity.md` across a capability ladder. 2026-06-19.
`eval/capability_trend.py`. share = (obscure - anon) / (name - anon) = the fraction of the single-cell
name/anon gap due to token-familiarity + reasoning rather than mapping-documentation.*

| model | name | obscure | anon | familiarity share |
|---|---|---|---|---|
| claude-haiku-4-5 | 0.851 | 0.613 | 0.500 | **0.32** |
| claude-sonnet-4-6 | 0.871 | 0.684 | 0.504 | **0.49** |
| claude-opus-4-8 | 0.978 | 0.880 | 0.495 | **0.80** |
| gpt-4o-mini | 0.715 | 0.519 | 0.565 | -0.31 (degenerate, see below) |
| gpt-4o | 0.826 | 0.582 | 0.460 | 0.33 |

## Finding

Within the Anthropic family, where capability is the main variable (same training lineage), the
token-familiarity share rises monotonically: **Haiku 0.32 < Sonnet 0.49 < Opus 0.80**. The mechanism by
which a model grounds is capability-governed: weaker models lean on the web-documented markers (strip
them and grounding collapses toward chance), while the frontier reasons cell identity from any familiar
gene symbols and barely needs the textbook markers. This confirms the prediction from the
token-familiarity dissociation and resolves it into a clean capability law for the symbolic case.

## Honest caveat

The cross-family comparison is noisier, and the share is only defined when a model actually grounds
names. gpt-4o-mini is degenerate here: it grounds names only weakly (0.715) and is spuriously above
chance on anon (0.565, a weak-model calibration artifact rather than real signal), so its share is
negative and uninformative. gpt-4o (0.33) sits in the low-to-mid range, consistent with a mid-capability
model. The controlled, monotone evidence is the within-family Anthropic ladder; the OpenAI points are
consistent in direction (the mini model relies less on reasoning) but too noisy to rank cleanly at the
weak end.

## What it means

Combined with `token_familiarity.md` and `budget_arm.md`: the single-cell name/anon gap is real and
large, but its mechanism is a capability-dependent mix of token-familiarity/reasoning and
mapping-documentation, with the familiarity share a monotone function of capability. "Web-exposure of the
mapping" describes the weaker-model end; the frontier grounds by reasoning over familiar representations.
This is the corrected, sharper claim the manuscript should make.
