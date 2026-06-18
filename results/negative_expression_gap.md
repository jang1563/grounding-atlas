# Negative-class expression gap (B, the non-overlapping cell)

*2026-06-13. The one piece of "B" (verifiable-negative signal) that does NOT overlap NullAtlas: apply the 3-arm grounding instrument to the NEGATIVE class. NullAtlas owns the data (32.8M confirmed negatives), the ML/JEPA/RL signal, the L4 LLM-discrimination benchmark, and even the negative web-exposure gradient. The only thing it does not do is decompose the LLM's negative-handling into encode-vs-verbalize. That is this cell. No em dashes.*

## Question

NullAtlas's headline is "models without negative data produce excessive false positives", measured at the ML / LLM-output level. We ask the layer below: does the LLM **encode** confirmed-inactive compounds internally (a probe separates active/inactive at a high ceiling) yet fail to **verbalize** it (0-shot output at chance)? If yes, the false-positive behavior is an expression gap, not an encoding gap, and it should be largest exactly where the negative is web-poor (the "no effect" that is never published).

## Instrument

`eval/negative_expression_gap.py` (frontier output arm, local API) + `eval/activation_arm.py` on GPU (`run_neg.sh`, Qwen3-8B / OLMo-2-7B, singularity). One balanced confirmed-active(fail) vs confirmed-inactive(pass) set per endpoint from NegBioDB `admet.db` (`eval/data/neg_*.csv`), one Murcko-scaffold GroupKFold split, three arms (AMES additionally re-run on the Hansen 2009 Ames benchmark, 6512 compounds 3503 mutagen / 3009 non-mutagen, for a robust n=2000; `eval/data/neg_ames.csv`):
- **ceiling**: Morgan r2/2048 logistic probe (signal is in the structure)
- **activation**: open-weight hidden-state linear probe, held-out-layer (nested GroupKFold, selection-bias controlled) + shuffled-label selectivity
- **output**: 0-shot verbalized active-probability (same open model; plus a frontier Claude panel)

Endpoints span web-exposure of the NEGATIVE: hERG-safe and CYP-non-inhibitor are better documented; AMES non-mutagen ("this is safe") is the least published.

## Arm 1+2+3 (Qwen3-8B, the same-model expression gap)

| endpoint | probe | **activation** (held-out) | output (0-shot) | encoding gap | **expression gap** | act selectivity |
|---|---|---|---|---|---|---|
| hERG | 0.837 | 0.779 | 0.454 | 0.058 | **0.325** | 0.265 |
| CYP3A4 | 0.784 | 0.691 | 0.506 | 0.093 | **0.185** | 0.169 |
| CYP2D6 | 0.763 | 0.725 | 0.477 | 0.038 | **0.248** | 0.243 |
| AMES (admet n=206) | 0.863 | 0.868 | 0.492 | −0.005 | 0.376 | 0.414 |
| **AMES (Hansen n=2000)** | 0.724 | 0.733 | 0.588 | −0.009 | **0.145** | 0.238 |

**All 4 endpoints: expression gap > encoding gap.** Activation (LLM hidden state) reaches 0.69–0.87, near the Morgan ceiling, while 0-shot output sits at 0.45–0.59 (chance to slightly above). So the active/inactive boundary IS in the model's internal representation; the model surfaces little of it zero-shot. The shuffled-label control collapses activation to about 0.5, so selectivity (0.17–0.41) confirms the probe reads real signal, not a fit (this is the control the earlier "encodes chemistry = a substring probe" critique demanded).

**AMES needed a robustness check, and it changed the story.** The first AMES run used the admet gold-tier set with only 103 inactive/class, giving a dramatic gap (0.376) on a wide CI (output 0.492 [0.42, 0.56]). Re-running on the standard Hansen 2009 Ames benchmark (3009 confirmed non-mutagens, 1000/class, a broader chemical space) tightens the CI threefold and SHRINKS the gap to 0.145 (Qwen) / 0.118 (OLMo): the 0.376 was a small-sample over-estimate, and output rose from chance (0.49) to 0.59. The expression gap survives robustly (activation 0.73 > output 0.59, encoding gap ≈ 0, selectivity 0.24, cross-family), but AMES is NOT the sharpest endpoint; it is mid-pack, smaller than hERG (0.33). So web-poor negatives still show encode-without-verbalize, but the SIZE is endpoint-specific and must be read off robust n, not a 200-molecule point. *(Label-direction note, 2026-06-18: the admet-set AMES used NegBioDB `pass/fail`, later found inverted via a structural-alert audit — `eval/analyze_ames.py`, label-0 is the nitroaromatic-rich mutagenic class. The admet AMES output sat at chance (0.49), so its direction is moot here, and the load-bearing result uses the Hansen benchmark's own correct mutagen/non-mutagen labels, so this endpoint's conclusion is unaffected.)*

## Arm 3 detail: frontier output panel (Claude), named vs anon

The output arm across a scale curve and an anonymization (target named vs generic phrasing). asym = FP(inactive called active) − FN(active called inactive):

```
                NAMED asym                 ANON asym            output AUROC (named)
endpoint   haiku  sonnet  opus      haiku  sonnet  opus      haiku  sonnet  opus
hERG       +0.95  -0.90  -0.11      +0.88  -0.58  +0.53      0.48   0.49   0.63
CYP3A4     +0.31  -0.47  +0.32      +0.96  -0.98  +0.66      0.46   0.57   0.64
CYP2D6     +0.64  -0.58  -0.23      +0.93  -0.98  +0.61      0.57   0.55   0.73
AMES       +0.59  -0.47  +0.62      +0.83  -0.51  +0.30      0.41   0.35   0.45
```

- **Bias direction is model prior, not scale (non-monotonic).** haiku over-calls active (asym +), sonnet over-calls inactive (asym −), opus is mixed. So NullAtlas's "universal false-positive" is model-specific in direction; what is universal is the failure to ground the negative (output AUROC ≪ ceiling for all).
- **Anonymization breaks it: negative grounding is web-lookup.** opus named-hERG is balanced (−0.11) but anon flips to over-call (+0.53), and opus output AUROC drops named→anon on all 4. The frontier model's apparent negative-calibration was recall of web-documented safe drugs by target name, not structural grounding. This is the negative-class version of the PPI 0.95→0.50 collapse.

## Cross-family check: OLMo-2-7B reproduces the gap

| endpoint | probe | **Qwen3-8B** act / out / gap | **OLMo-2-7B** act / out / gap |
|---|---|---|---|
| hERG | 0.837 | 0.779 / 0.454 / **0.325** | 0.793 / 0.510 / **0.283** |
| CYP3A4 | 0.784 | 0.691 / 0.506 / **0.185** | 0.688 / 0.488 / **0.200** |
| CYP2D6 | 0.763 | 0.725 / 0.477 / **0.248** | 0.753 / 0.496 / **0.257** |
| AMES (admet n=206) | 0.863 | 0.868 / 0.492 / 0.376 | 0.883 / 0.647 / 0.236 |
| **AMES (Hansen n=2000)** | 0.724 | 0.733 / 0.588 / **0.145** | 0.700 / 0.582 / **0.118** |

Two independent families (Qwen3, Alibaba, BPE/tiktoken; OLMo-2, AllenAI, Dolma corpus, GPT-NeoX BPE) give the SAME picture: activation(held-out) near the Morgan ceiling, output below it, encoding gap small, selectivity 0.23–0.41 (both pass the shuffled-label control on every endpoint). **The expression gap is model-family-invariant, not a Qwen artifact**, and it holds at the corrected AMES size (Qwen 0.145, OLMo 0.118 on the robust Hansen n=2000). The one family difference: OLMo verbalizes slightly better (output higher than Qwen on every endpoint). Ranking by robust gap, the order is roughly hERG > CYP2D6 > CYP3A4 ~ AMES, so the larger-n endpoints (hERG n=1250, CYP2D6 n=1348) are the load-bearing ones, not AMES.

(Llama-3.1-8B was gated, Mistral-7B-v0.3 needs sentencepiece the container lacks; OLMo-2-7B is the BPE-tokenizer cross-family check that ran.)

## Reads

1. The negative false-positive behavior is an **expression gap**: the open 8B model encodes confirmed-inactive at near-ceiling internally and verbalizes it at or near chance (output below ceiling on every endpoint), with selectivity confirming real encoding.
2. The gap is present across endpoints with encoding intact (activation ≈ ceiling), so web-exposure governs verbalization not encoding. Its SIZE is endpoint-specific and must be read off robust n: AMES looked largest at n=206 (0.376) but shrank to 0.145 at n=2000, so no single endpoint should be over-ranked from a small sample.
3. The frontier panel shows the verbalization side is **model-prior-directional** (haiku active / sonnet inactive / opus mixed) and **web-lookup-dependent** (anonymization reintroduces the over-call). High frontier numbers on named negatives are memorization, not grounding.
4. The gap is **model-family-invariant**: Qwen3-8B and OLMo-2-7B (different team, corpus, tokenizer) reproduce it on all 4 endpoints, so it is a property of how general-purpose LMs handle the negative class, not one model's quirk.

## Honest limits

- **Open proxy, not Claude.** Claude has no exposed hidden states, so the encode side is measured on an open 8B proxy. The valid claim is "a general-purpose 8B LM encodes what it cannot verbalize, and Claude's output behavior mirrors the verbalization side", not a statement about Claude's internals.
- **anon confound.** The generic phrasing ("a cardiac ion channel") is more ambiguous than the named target, so part of the named→anon drop is under-specification, not only web-lookup removal. Direction is robust (opus AUROC named ≥ anon on all 4); the mechanism is partly confounded.
- **Small n inflates the point estimate (a caught error).** The admet AMES set had 103 inactive/class and gave gap 0.376 on a wide CI; re-running on the Hansen 2009 benchmark (1000/class, broader chemistry) corrected it to 0.145. The robustness check changed the ranking (AMES is mid-pack, not sharpest), so trust the larger-n endpoints (hERG n=1250, CYP2D6 n=1348) and read sizes off robust n only.
- **Not a standalone headline.** This is one cell complementing NullAtlas L4 (the encode-vs-verbalize decomposition it does not run), not a new benchmark.
