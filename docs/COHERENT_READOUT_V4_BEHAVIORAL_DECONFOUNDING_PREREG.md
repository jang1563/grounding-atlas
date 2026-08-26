# Coherent Readout V4 behavioral-deconfounding preregistration

Freeze date: 2026-08-02  
Mode: prospective development, synthetic and non-biological  
Model: `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

## 1. Question and scope

V3 stopped before intervention because its fit bank failed behavioral admission.
Post-hoc, all 512 V3 predictions followed one exact alternative rule: X iff the
queried property was P, the mapping was identity, and the target fact appeared
first; otherwise Y. V3 also displayed answer options only as “X or Y,” leaving
a Y/default or last-option effect unbalanced.

V4 prospectively asks:

1. Can the locked model retrieve a property from an explicitly labeled target
   fact, invariant to target-fact and answer-option order?
2. Can it look up an arbitrary codebook mapping, invariant to codebook-rule and
   answer-option order?
3. When both components are available, can it compose retrieval with lookup,
   invariant to all three order factors?
4. If composition fails, does one of five registered alternative policies
   predict behavior better than the intended ledger?

V4 is terminal and behavior-only. It fits no direction, inspects no activation,
selects no layer, and authorizes no downstream model call. Its maximum claim is
component localization among output engineering, property retrieval, codebook
lookup, and their synthetic composition in this one locked model.

## 2. Frozen static bank

The exact static artifacts are:

| artifact | path | SHA-256 |
|---|---|---|
| builder | `signal/syntax/build_coherent_readout_v4_behavioral_deconfounding_bank.py` | `2af597f4031db819c94a5b4ec6d6845a513e0484eaa9a6da8ab61ce9c6e178da` |
| bank | `signal/syntax/coherent_readout_v4_behavioral_deconfounding_bank.json` | `8d988b36d99677e798628b932fea86efd68317cd4a8653bbcd2f12a3294021c2` |
| bank canonical JSON | same logical bank | `dd46347c62f1f3823a9813b5d1a302e0e26d86d90c1a4d5d0e28f523dfa73f39` |
| bank manifest | `signal/syntax/coherent_readout_v4_behavioral_deconfounding_bank.manifest.json` | `ad957af864564c9e79cb9b2330a2e66076385ceeac5eead833909a4c59b252cb` |

The bank contains eight new worlds, `behavior_world_001` through
`behavior_world_008`, with entity strings not used by V3. No V3 row, activation,
axis, world, or fitted quantity enters V4 estimation or gating.

The neutral system message is exactly:

> Follow the user's labeled task. Reply with exactly the requested
> single-character label and nothing else.

It does not enumerate X/Y or P/Q. Each user prompt explicitly labels TARGET and
OTHER roles where facts are present and states that displayed option order
carries no meaning.

## 3. Families and factorial coverage

Exactly 448 next-token calls are frozen.

### Composition: 256 rows

For every world, fully cross:

- target property: P or Q;
- mapping: identity (`P->X`, `Q->Y`) or swapped (`P->Y`, `Q->X`);
- target-fact order: target first or target second;
- rule order: P rule first or Q rule first; and
- displayed code order: X then Y or Y then X.

This is `8 worlds x 2^5 = 256` rows and 32 full factorial strata. The
non-target fact always contains the opposite property and is explicitly labeled
OTHER FACT. The intended answer is `mapping[target_property]`.

### Property retrieval: 64 rows

For every world, cross target property, target-fact order, and displayed
property order (P then Q or Q then P). This is `8 x 2^3 = 64` rows and eight
strata. No codebook is present. The intended answer is the TARGET FACT property.

### Codebook lookup: 128 rows

For every world, cross given property, mapping, rule order, and displayed code
order. This is `8 x 2^4 = 128` rows and 16 strata. No fact retrieval is required.
The intended answer is `mapping[given_property]`.

Every semantic retrieval item appears in four order permutations, every lookup
item in four, and every composition item in eight. The fixture stores the
semantic-bundle ID and fixed permutation index; runner and analyzer reconstruct
them independently.

## 4. Registered policies

The intended ledger and five alternatives are fixed before V4 model calls:

1. `intended_compositional_rule`: `mapping[target_property]`;
2. `frozen_v3_heuristic`: X iff property=P, mapping=identity, and target fact is
   first; otherwise Y;
3. `last_displayed_option`: the second displayed code;
4. `first_displayed_codebook_rule_output`: the code on the first displayed rule;
5. `constant_y`; and
6. `constant_x`.

Policy-match rates are computed on composition rows from the unique global
argmax token. They are descriptive when composition passes. When composition
fails, an alternative qualifies as a dominant registered failure policy only
if its match rate is at least `0.90` and at least `0.10` above intended-rule
accuracy. If several alternatives share the exact highest qualifying match,
the classification is `MULTIPLE_REGISTERED_HEURISTICS`; otherwise the unique
highest qualifying policy is named. If none qualifies, the classification is
`NO_REGISTERED_HEURISTIC_DOMINANT`.

The five unique-policy labels are respectively
`V3_HEURISTIC_DOMINANT`, `LAST_DISPLAYED_OPTION_HEURISTIC_DOMINANT`,
`FIRST_DISPLAYED_RULE_OUTPUT_HEURISTIC_DOMINANT`,
`CONSTANT_Y_HEURISTIC_DOMINANT`, and `CONSTANT_X_HEURISTIC_DOMINANT`.

## 5. Immutable plan and execution

Before loading model weights, the runner must write a complete tokenizer-only
plan, design, tokenization receipt, and dependency lock. They must bind all 448
rendered prompts, token IDs, attention masks, correct/incorrect contextual label
token IDs, row order, source files, package versions, model files, analyzer, and
tests. They report zero model calls and forbid generation and logit processors.

Before execution, the runner rebuilds the fixture and the complete tokenized
plan and requires exact equality. It then writes an immutable attempt receipt.
Any attempt, partial output, completed output, or analysis artifact forbids
re-entry. Execution consists of exactly one teacher-forced prompt forward per
row with `use_cache=False` and no generation.

The raw float32 next-token vocabulary matrix is stored with shape
`(448, 151936)`, alongside exact row IDs and row hashes. This allows the analyzer
to reconstruct global argmaxes, ties, label logits, probability mass, and every
confirmatory accuracy without trusting runner summaries. Execution records and
the manifest must reconstruct exactly from the call plan and raw sidecar.

## 6. Accuracy and channel definitions

Confirmatory behavioral accuracy is one only when the unique full-vocabulary
argmax is exactly the intended label token. A tie or non-family maximum is
incorrect. Binary two-label preference accuracy is reported descriptively but
is never substituted for confirmatory accuracy.

For each family, output engineering requires:

- no global-argmax tie in any row;
- unique global argmax inside that family's two allowed labels in at least
  `0.95` of rows; and
- mean probability mass of the two family labels at least `0.95`.

Failure of any artifact or channel requirement has engineering precedence over
behavioral component statuses.

## 7. Frozen behavioral gates

For property retrieval and codebook lookup separately, all of the following are
required:

- overall confirmatory accuracy at least `0.95`;
- accuracy at least `0.95` in every full factorial stratum;
- fraction of semantic bundles for which every order permutation is correct at
  least `0.95`;
- accuracy at least `0.90` in every world; and
- the lower endpoint of the world-cluster bootstrap 95% interval at least
  `0.90`.

Composition requires:

- overall confirmatory accuracy at least `0.95`;
- native-X accuracy at least `0.95` and native-Y accuracy at least `0.95`;
- accuracy at least `0.90` in every full five-factor stratum;
- fraction of world-by-property-by-mapping bundles with all eight order
  permutations correct at least `0.95`;
- accuracy at least `0.90` in every world; and
- the lower endpoint of the world-cluster bootstrap 95% interval at least
  `0.90`.

The bootstrap uses NumPy `default_rng(260805)`, 10,000 draws, eight worlds
sampled with replacement per draw, world-mean accuracy, and the 0.025/0.975
quantiles. All family analyses reuse the same frozen world-index matrix.

## 8. Terminal status hierarchy

Exactly one terminal status is emitted, in this precedence order:

1. `V4_ENGINEERING_INVALID`;
2. `V4_RETRIEVAL_AND_LOOKUP_COMPONENTS_FAIL`;
3. `V4_RETRIEVAL_COMPONENT_FAIL`;
4. `V4_LOOKUP_COMPONENT_FAIL`;
5. `V4_COMPOSITION_FAIL_COMPONENTS_PASS`; or
6. `V4_BEHAVIORAL_COMPOSITION_QUALIFIED`.

If composition passes while either direct component family fails, qualification
is still forbidden and the corresponding component-failure status is emitted.
No status authorizes an activation experiment. A qualified result is only a
prerequisite for designing a new, separately preregistered causal tier with new
worlds and freshly fitted quantities.

## 9. Interpretation and nonclaims

- Retrieval pass + lookup pass + composition fail supports a synthetic
  behavioral composition gap in this prompt family; it does not establish an
  internal activation gap.
- Retrieval failure localizes the observed problem no later than labeled target
  selection/property reporting under this test.
- Lookup failure localizes it no later than arbitrary rule binding/reporting.
- A registered heuristic match is predictive evidence, not a unique or causal
  mechanism.
- Qualification shows robust synthetic task performance only.

V4 cannot establish biological knowledge, latent-knowledge possession, a
causal activation gap, a representation's necessity or sufficiency, a physical
law, or model-family generality. Those require separate datasets, controls,
models, and causal interventions.

## 10. Verifiability boundary

The complete plan, prompts, raw full-vocabulary float32 rows, label mappings,
global decisions, bootstrap indices, thresholds, and analysis are locally
reconstructible from frozen artifacts. Model and tokenizer identity are bound by
revision and cached-file hashes. Independent verification can replay all
analysis without a model call; verification that the recorded logits truly came
from the claimed runtime still requires rerunning the locked model.

## 11. 한국어 요약

V4는 activation 연구가 아니라 **행동 confound 분리 실험**이다. 새 8개
world에서 property retrieval, codebook lookup, 두 연산의 composition을 각각
분리하고, target-fact 순서, rule 순서, option 순서를 완전 균형화한다. 정답은
항상 full-vocabulary의 유일한 global argmax로 판정한다.

V3에서 사후적으로 관찰된 3-factor heuristic, last-option, first-rule,
always-Y, always-X를 모두 V4 model call 전에 등록한다. Retrieval과 lookup은
통과하지만 composition만 실패할 때에만 synthetic behavioral composition
gap이라고 부를 수 있다. 이것은 latent knowledge나 causal activation gap의
증명이 아니다.

448개 raw full-vocabulary logit row를 float32로 저장하므로 analyzer는 runner
summary를 신뢰하지 않고 모든 argmax, tie, probability mass, accuracy, bootstrap
결과를 다시 계산할 수 있다. 어떤 결과도 후속 activation call을 자동 승인하지
않는다.
