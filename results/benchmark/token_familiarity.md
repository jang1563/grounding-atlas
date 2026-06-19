# Token-familiarity vs mapping-documentation: decomposing the "web-exposure" gap

*The discriminating experiment for the bias audit. 2026-06-19. `signal/single_cell/build_cd8t_nk.py
--drop-famous`, task `single_cell/cd8t_nk:obscure`.*

## The question

The single-cell name-vs-anon pair was our cleanest "web-exposure" evidence: the same CD8-T/NK cells
ground from gene NAMES (0.83-0.98) but are at chance from anonymized ids (0.50). We attributed the gap
to web-documentation of the gene->cell-type mapping. But that conflates two things: the model may ground
because the TOKENS are familiar (real gene symbols it can reason over) or because the specific MAPPING is
web-documented. To separate them we add a third condition: the same cells as real, familiar gene symbols
but with the web-FAMOUS markers (GZMB, NKG7, CD8A, ...) dropped. A head still reads CD8-T vs NK from the
remaining genes at 0.979, so the signal is intact; only the textbook, web-documented markers are gone.

## Result (n=200)

| model | name (markers) | obscure (no markers) | anon (hashes) | documentation = name-obscure | token-familiarity = obscure-anon | familiarity share of the gap |
|---|---|---|---|---|---|---|
| claude-opus-4-8 | 0.978 | 0.880 | 0.495 | +0.10 | +0.39 | ~80% |
| claude-sonnet-4-6 | 0.871 | 0.684 | 0.504 | +0.19 | +0.18 | ~50% |
| gpt-4o | 0.826 | 0.582 | 0.460 | +0.24 | +0.12 | ~33% |

## Finding: the mechanism is capability-dependent, and "web-exposure" over-attributes

Strip the documented markers and the frontier barely drops (opus 0.978 -> 0.880) while a weaker model
collapses (gpt-4o 0.826 -> 0.582). The name/anon gap decomposes into two factors whose mix tracks
capability:

- **Token-familiarity + biological reasoning** (does better with real gene symbols, even obscure ones,
  than with alien hashes): the DOMINANT factor for the frontier (opus +0.39, ~80% of the gap). Opus
  reasons cell identity from any familiar gene context; it barely needs the textbook markers.
- **Mapping-documentation** (recalling the specific marker->type fact): larger for weaker models
  (gpt-4o +0.24, ~67%); the documented markers are a recall crutch that opus mostly does without.

The token-familiarity share is capability-ordered: opus ~80% > sonnet ~50% > gpt-4o ~33%. So the anon
failure is largely that the tokens are ALIEN (nothing to reason over), not that the mapping is
undocumented, especially for the frontier. This is consistent with the budget arm: methylation (numeric,
no familiar tokens to reason over, an empirical clock) stays at chance even for opus even with a
reasoning budget, because there the knowledge is genuinely absent.

## Honest caveat and the revised claim

The obscure set still contains a few immune-documented genes (CCL3/4, IL32, LTB), so documentation is not
fully removed: the documentation contribution is a lower bound and the familiarity contribution an upper
bound. Even so, the capability ordering is unambiguous.

Revised claim: the verbalization gap is governed by BOTH token-familiarity / reasoning-amenability AND
mapping-documentation, in a capability-dependent mix; "web-exposure of the mapping" alone over-attributes,
especially for the frontier. The honest headline is the two-factor decomposition, not a single web-exposure
axis. The name/anon contrast remains a real, large effect; what this corrects is its mechanism.
