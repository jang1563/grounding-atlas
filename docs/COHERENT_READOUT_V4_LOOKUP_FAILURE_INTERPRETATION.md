# Coherent Readout V4: lookup failure and positional composition residual

Date: 2026-08-03  
Official status: `V4_LOOKUP_COMPONENT_FAIL`  
Qualifier: disclosed post-freeze implementation repair

## 1. What V4 established

The locked Qwen2.5-1.5B-Instruct run completed all 448 synthetic,
non-biological calls. All confirmatory decisions were reconstructed from the
raw float32 full-vocabulary matrix, not generated text or two-label-only
scores.

| Family | Confirmatory accuracy | Native-label accuracy | Gate |
|---|---:|---:|---:|
| Property retrieval | 64/64 = 1.0000 | P 1.0000; Q 1.0000 | pass |
| Codebook lookup | 104/128 = 0.8125 | X 0.6250; Y 1.0000 | fail |
| Composition | 196/256 = 0.765625 | X 0.578125; Y 0.953125 | fail |

Output engineering passed in all families: every row had a unique global
argmax inside the permitted label pair, with mean family-label probability
mass above 0.99998. The failure is therefore not malformed output or label
escape.

No registered composition alternative dominated. The intended rule matched
196/256 predictions (0.765625), the frozen V3 heuristic 192/256 (0.75), and
constant Y 176/256 (0.6875). V3's exact fitted heuristic did not replicate as
a dominant V4 rule.

The terminal interpretation is narrower than a composition or activation gap:
the model retrieved the explicitly labeled property perfectly but did not
reliably bind a supplied arbitrary codebook to its output.

## 2. Repair disclosure

The original frozen analyzer stopped before aggregation because it compared
the execution-record schema tag with the intentionally distinct fixture-cell
schema tag. Repair1 verified that this was the only disagreement across all
448 pairs, removed only the cell tag from a deep-copied in-memory merge view,
and delegated raw-logit replay and every decision rule to the untouched frozen
analyzer.

The repair changed no raw artifact, prompt, score, threshold, bootstrap,
policy, or status rule and executed zero model forwards. A provisional outcome
was exposed during diagnosis before the repair lock; the result must therefore
retain the `disclosed_post_freeze_implementation_repair` qualifier.

Official bindings:

- repair-plan canonical SHA-256:
  `bba4bf08c63d77b25d4a4725a0a2103d0868d65ffd090b26f5013a701d75e0e5`;
- analysis file SHA-256:
  `9f6a02863dd84299f859480b11e53b37b0b51b02b758b77035ce2e857812e5d9`;
- analysis canonical SHA-256:
  `49ab2b0aa18f9da7e8e795ef703058028ca5fc451344721e129e9af4efa442a2`;
- raw-logit file SHA-256:
  `4f281510fb70f5f678eee3680999193b559970cf823e67d7c03f6b50bde80928`.

## 3. Post-hoc lookup decomposition

This section is exploratory and does not change the terminal status.

The nominal 128 lookup rows reduce to only 16 distinct prompt/token/logit rows,
each repeated identically across eight world IDs. The lookup family contains
no world-specific string. Its degenerate world-bootstrap interval therefore
does not provide independent prompt- or world-level replication.

Each cell below represents eight identical executions. Bold Y entries are
errors because the intended code is X.

| Target / mapping | P-rule first, X-first | P-rule first, Y-first | Q-rule first, X-first | Q-rule first, Y-first |
|---|---:|---:|---:|---:|
| P / identity; answer X | X | **Y** | **Y** | X |
| P / swapped; answer Y | Y | Y | Y | Y |
| Q / identity; answer Y | Y | Y | Y | Y |
| Q / swapped; answer X | **Y** | X | X | X |

Thus three of 16 unique prompts fail, repeated to produce 24/128 errors. All
intended-Y prompts succeed; intended-X accuracy is only 40/64. The errors are
not explained by a first-option, last-option, first-rule, constant, mapping-only,
or any three-factor deterministic policy. An exact post-hoc truth-table rule
requires all four manipulated factors: target property, mapping, rule order,
and option order. This is descriptive evidence for asymmetric Y-default
behavior with higher-order order sensitivity, not a mechanism.

## 4. Post-hoc matched composition decomposition

Every composition row was matched exactly to its standalone lookup row using
world, target property, mapping, rule order, and option order. Retrieval was
correct for every corresponding semantic condition.

- Composition agreed with its lookup prediction in 194/256 rows (75.78125%).
- When lookup was correct, composition was correct in 171/208 rows (82.2115%).
- When lookup was wrong, composition was nevertheless correct in 25/48 rows
  (52.0833%).
- Therefore composition neither merely inherited nor monotonically amplified
  standalone lookup behavior.

Fact position produced a strong matched residual:

| Matched outcome | Pair count |
|---|---:|
| Both target positions correct | 84 |
| Target-first only correct | 28 |
| Target-second only correct | 0 |
| Neither correct | 16 |

Accuracy was 112/128 (87.5%) with TARGET FACT first and 84/128 (65.625%)
with TARGET FACT second, a 21.875-point difference. The exact paired McNemar
diagnostic is `p = 7.45058e-9`. Even among the 104 standalone-lookup-correct
keys, target-first-only correct occurred 21 times and the reverse never
occurred (`p = 9.53674e-7`). These p-values summarize this fixed deterministic
panel; they are not population inference across models or prompt families.

The defensible discovery is:

> Standalone retrieval success, and even matched standalone lookup success, do
> not guarantee joint execution. The joint prompt has a strong TARGET/OTHER
> positional sensitivity concentrated when TARGET FACT is displayed second.

Because standalone lookup failed its confirmatory gate, this cannot be called
a pure composition gap. Candidate explanations include joint-prompt
interference, positional weighting, TARGET/OTHER binding, and interactions with
codebook or option presentation. V4 does not distinguish their internal causes.

## 5. Next authorized design: V5 independent-codebook binding

The next meaningful tier remains behavioral and synthetic. It must repair the
coverage defect rather than proceed to activation.

1. Give every world a distinct, tokenizer-preflighted single-token property
   pair and output-code pair so no lookup prompt is duplicated.
2. Balance every output token as correct/incorrect and first/second equally;
   rotate token pairs across worlds to measure label prior directly.
3. Separate one-rule direct readout from two-rule selection, then cross query
   position, rule order, option order, and at least two preregistered prompt
   paraphrases.
4. Match lookup and composition with the exact same world-specific codebook.
5. Fully cross TARGET/OTHER order and include a target-only condition to
   distinguish distractor binding from absolute position.
6. Cluster uncertainty by genuinely distinct world/codebook prompts, not
   duplicated rows.
7. Require retrieval and lookup to pass before estimating a confirmatory
   composition residual. No V5 behavioral status may authorize activation.

Only after this tier succeeds on new worlds should a separately preregistered
causal study ask whether an item-specific representation can selectively rescue
joint execution without changing matched controls.

## 6. Biology, physical-law, and verification boundaries

A better eventual biology question is:

> Given an independently verified biological entity-state relation, can the
> model retrieve the state and apply a newly supplied reporting rule robustly
> across fresh code vocabularies, paraphrases, and fact/rule/option order; and,
> only after standalone retrieval and lookup are matched, does joint execution
> show an item-specific deficit that a preregistered causal intervention can
> selectively rescue?

V4 is not evidence of a physical law. At most, multi-model and multi-scale
replication could establish an empirical regularity about order-sensitive
binding. Artifact-level verification is strong: prompts, token IDs, raw logits,
row hashes, gates, and outputs are replayable without a model call. Verifying
that the stored logits originated from the claimed runtime still requires a
locked model rerun.

## 한국어 요약

V4의 공식 상태는 `V4_LOOKUP_COMPONENT_FAIL`이다. Property retrieval은
64/64로 완벽했지만 codebook lookup은 104/128, composition은 196/256이었다.
출력 channel은 모든 행에서 정상이다. 따라서 핵심 실패는 정보를 찾지 못한
것이 아니라, 찾은 속성을 새로 제시된 임의 출력 규칙에 안정적으로 결합하지
못한 것이다.

사후 분해에서는 lookup 128행이 실제로 16개 고유 prompt의 8회 반복임이
드러났다. 세 고유 prompt만 실패했고 모두 X 정답을 Y로 출력했다. 이는 단순
first/last-option 규칙이 아니라 Y 기본 편향과 4-factor order interaction이다.

Composition에서는 TARGET FACT를 첫 번째에 둘 때 87.5%, 두 번째에 둘 때
65.625%였다. 동일한 128쌍 중 28쌍이 정답에서 오답으로 바뀌었고 반대는
0쌍이었다. Standalone lookup이 맞은 조건에서도 두 번째 TARGET 위치에서
추가 실패가 남았다. 그러나 lookup 자체가 gate를 실패했으므로 이를 순수한
composition gap이나 activation gap이라고 부를 수 없다.

다음 V5는 activation이 아니라, world마다 서로 다른 single-token property와
codebook을 사용하고 one-rule/two-rule, paraphrase, query/rule/option/fact order를
완전 균형화한 독립 codebook-binding 실험이어야 한다. 생물학으로 확장하려면
retrieval과 lookup을 먼저 독립적으로 통과시킨 뒤 joint execution의 잔차와
item-specific causal rescue를 물어야 한다. 이는 물리 법칙의 증명이 아니다.
