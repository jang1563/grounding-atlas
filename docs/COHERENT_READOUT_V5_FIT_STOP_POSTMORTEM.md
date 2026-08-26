# Coherent Readout V5: fit-stop result and serial-recency discovery

Date: 2026-08-03  
Official status: `V5_FIT_COMPONENT_ADMISSION_FAIL`  
Scope: frozen confirmatory fit decision plus clearly labeled post-hoc decomposition  
New model calls used by the post-hoc analysis: `0`

## English

### 1. Frozen decision

V5 was a legitimate preregistered and falsifiable experiment. Its first live
gate failed, so the experiment stopped after exactly 448 fit-baseline forwards.
No localization, activation patch, layer selection, or holdout forward was
executed.

| Registered quantity | Observed | Required | Gate |
|---|---:|---:|---|
| Property retrieval | 38/64 = 0.59375 | >= 0.95 | fail |
| Codebook lookup | 65/128 = 0.5078125 | >= 0.95 | fail |
| Fixed direct prerequisites | 73/128 | 128/128 | fail |
| TARGET-first composition | 69/128 = 0.5390625 | >= 0.95 | fail |
| TARGET-second error | 26/128 = 0.203125 | >= 0.25 | fail |
| First-minus-second accuracy | -0.2578125 | >= 0.10 | fail |
| Registered panel margin gap `G` | -1.850037 | lower bound > 0 | fail |

The registered finite-panel stability interval for `G` was
`[-2.273672, -1.413423]`, and all eight fit worlds had negative
first-minus-second gaps. The full 16-pair-per-world diagnostic was almost
identical: `G_full = -1.843154` with interval
`[-2.226692, -1.413390]`.

This was not an output-channel failure. All 448 rows had a unique global argmax
inside the registered label pair. Mean probability mass on the two valid labels
was 0.999925 for retrieval, 0.999356 for lookup, and 0.999161 for composition.
The frozen analyzer independently reconstructed the full-vocabulary logits,
answer ledger, margins, hashes, and stopping decision.

### 2. Meaningful post-hoc finding

This section is exploratory. It does not change the frozen status or authorize
any downstream V5 phase.

The preregistered TARGET-first deficit hypothesis did not merely fail; it
reversed almost deterministically.

| Exact composition pairs | Both correct | First only | Second only | Neither | Second has larger margin |
|---|---:|---:|---:|---:|---:|
| Full factorial | 69 | 0 | 33 | 26 | 127/128 |
| Fixed intervention panel | 37 | 0 | 16 | 11 | 64/64 |

The full-panel paired fixed-set diagnostic is `p = 2.3283e-10`; the fixed
intervention-panel value is `p = 3.0518e-5`. These summarize the registered
deterministic panels and are not population inference over prompts or models.

The model showed a coherent later-item preference across all three component
families:

- retrieval selected the last displayed output option in 58/64 rows (90.6%);
- lookup selected the right-hand code of the last codebook rule in 111/128
  rows (86.7%); and
- composition selected the last rule's right-hand code in 195/256 rows
  (76.2%).

The balanced factorial identifies three dominant composition contrasts:

| Task-relative cue | Accuracy, early | Accuracy, late | Late-minus-early margin contrast | Same sign across worlds |
|---|---:|---:|---:|---:|
| TARGET rule position | 0.40625 | 0.9296875 | +4.377494 | 8/8 |
| Correct option position | 0.5390625 | 0.796875 | +2.294503 | 8/8 |
| TARGET fact position | 0.5390625 | 0.796875 | +1.843154 | 8/8 |

When all three relevant cues were early, accuracy was 0/32 and mean margin was
`-2.9733`. When all three were late, accuracy was 32/32 and mean margin was
`+5.8963`. A descriptive additive model using only target-second,
target-rule-last, and correct-option-last explained `R^2 = 0.6927` of the
composition-margin variance and predicted the observed margin sign in
217/256 rows (84.8%). This model is fitted and evaluated on the same fit panel;
it is a compact description, not cross-validated prediction.

Matching each composition prompt to its standalone lookup condition shows why
the higher TARGET-second accuracy is not compositional competence:

- TARGET-first composition agreed with lookup in 122/128 rows. Of 63 lookup
  errors, only five became correct in composition.
- TARGET-second composition agreed with lookup in 91/128 rows. Of 63 lookup
  errors, 37 became correct in composition, while none of the 65 correct lookup
  cases became wrong.

Thus the final fact position supplies a strong response-adjacent cue that can
override failed standalone lookup. It does not establish stable rule binding.

### 3. Integrity and error audit

The frozen and post-hoc audits found:

- 448 unique records resolving to 448 registered fit cells;
- zero fixture/record answer-ledger mismatches;
- zero raw-row SHA, reconstructed-logit, margin, argmax, or valid-label
  mismatches;
- zero semantic-pair invariant violations; and
- exact reproduction of every per-world registered gap.

The sign reversal is model behavior under the frozen prompt scaffold, not an
analysis-sign, answer-key, token-ID, or artifact-binding error.

### 4. What changed from V4

V4 placed `TASK` and valid-output text after the fact block. V5 placed the
codebook, valid outputs, and `TASK` before the facts, then ended immediately on
the two fact lines. TARGET-second is therefore the most response-adjacent
task-relevant fact in V5.

| Study | TARGET-first accuracy | TARGET-second accuracy | First-minus-second accuracy | First-minus-second margin |
|---|---:|---:|---:|---:|
| V4 | 0.875 | 0.65625 | +0.21875 | +0.1944 |
| V5 | 0.5390625 | 0.796875 | -0.2578125 | -1.8432 |

Within V5, TARGET order is an exact same-answer matched manipulation, so the
reversal is causal for this fixed scaffold. Across V4 and V5, however, query
placement was not the only change: symbols, wording, entity strings, and world
construction also changed. The defensible cross-study conclusion is therefore
**scaffold-sensitive sign instability**. The hypothesis that query placement
and semantic recency caused the cross-version reversal requires a fresh
factorial ablation.

### 5. Scientific boundary

V5 supports no causal activation mediation claim because the behavioral
construct was never admitted. Captured fit activations do not license fitting a
direction after the failed gate, reversing the registered gap sign, selecting
correct pairs, or opening localization/holdout.

It also does not test latent knowledge: every relevant symbolic fact was
provided explicitly in the prompt. It supports no biological claim, no model
family generalization, and no physical law. What it does establish is narrower
and useful:

> For Qwen2.5-1.5B-Instruct on this fixed factorial scaffold, rule, option, and
> fact recency combine into a strong later-item policy; the TARGET-order effect
> changes sign across scaffolds and is therefore not yet a portable activation
> gap.

### 6. Prospective V6: scaffold x reactivation

V5 localization and holdout worlds must remain unopened. V5 fit results may be
used only as disclosed hypothesis-generating data. V6 should use fresh worlds
and identify the proposed cause rather than flip the V5 estimand post hoc.

Disposable calibration may select prompt clarity or model size using component
accuracy only, never observed gap magnitude or sign. Before freezing the V6
confirmatory bank, both scaffold variants should reach retrieval and two-rule
lookup accuracy at least 0.98 overall and 0.90 at every registered factor level
across two paraphrases. If 1.5B cannot qualify, the model should be prospectively
fixed to a larger candidate before confirmatory worlds are generated.

The confirmatory factors should be:

1. task/valid-output block before versus after facts;
2. TARGET fact first versus second;
3. semantic competing OTHER fact versus token/length-matched inert filler;
4. target-identity reactivation cue versus matched neutral nonce cue; and
5. balanced mapping, rule order, and output order.

The target cue may repeat the target entity but must contain no property or
answer code. Every prompt must end at the same token-position-matched `ANSWER:`
cue. The primary intervention site is the `resid_post` state at `ANSWER:`, which
has seen the entire prompt and avoids interpreting the TARGET property's
absolute token position as the response state. The TARGET-property site remains
a secondary encoding analysis.

The primary behavioral quantities are:

\[
g_s=M_{F,s}-M_{S,s}, \qquad H=g_{post}-g_{pre},
\]

\[
G=M_{S,pre}-M_{F,pre},
\]

and, with target-identity versus neutral post-fact cues,

\[
B_F=M_{F,target}-M_{F,neutral},\quad
B_S=M_{S,target}-M_{S,neutral},\quad J=B_F-B_S.
\]

`H` tests scaffold-dependent order reversal, `G` prospectively defines the
early-target deficit suggested by V5, and `J` tests whether a non-answer-bearing
target cue selectively reactivates that deficit.

The next causal stage is authorized only if fresh fit worlds satisfy all of:

- retrieval and lookup each at least 0.95 overall and 0.90 per world, scaffold,
  paraphrase, and registered factor level;
- every fixed-panel direct prerequisite correct;
- `G` accuracy gap at least 0.10, margin lower bound above zero, and at least
  6/8 positive worlds;
- `H` lower bound above zero and at least 6/8 positive worlds; and
- `J/G >= 0.30`, `J` lower bound above zero, and at least 6/8 positive worlds.

Only then should a fit-world response-site reactivation axis be residualized
from answer, mapping, and order axes. Localization must require cross-fit
activation separation, bidirectional rescue and damage of at least `0.30G`,
specificity of at least `0.20G` beyond dose-matched answer, null, and
wrong-entity-cue shams, and all identity checks. A single shallowest passing
layer is then frozen for untouched 16-world holdout, with at least 12/16
positive worlds and no reselection.

### 7. Biology and law expansion

A defensible later biology question is:

> When a model can directly retrieve an independently verified biological
> entity-state relation, does a novel reporting code plus a competing relation
> create a use deficit that a non-answer-bearing entity cue and response-site
> intervention selectively restore?

The panel must be time-stamped and independently reviewed, use mutually
exclusive labels, aliases and paraphrases, counterfactual-override and
fictional-entity controls, and prohibit post-baseline item filtering. A positive
result would be operational evidence for accessible-but-unused biological
knowledge, not proof of memorization or of how knowledge entered the weights.

This is not a physical law. Only preregistered multi-model, multi-scale, and
multi-template replication of a quantitative target-distance x distractor-load
relation, including out-of-distribution extrapolation, could justify calling it
an empirical computational scaling law.

## 한국어

### 1. 동결 판정

V5는 사전등록되고 반증 가능한 정당한 실험이었다. 첫 실모델 gate가
실패했으므로 정확히 448회의 fit-baseline forward 뒤에 중단했다.
Localization, activation patch, layer 선택, holdout forward는 실행하지 않았다.

| 등록 지표 | 관측값 | 기준 | 판정 |
|---|---:|---:|---|
| Property retrieval | 38/64 = 0.59375 | >= 0.95 | 실패 |
| Codebook lookup | 65/128 = 0.5078125 | >= 0.95 | 실패 |
| 고정 직접 선행조건 | 73/128 | 128/128 | 실패 |
| TARGET-first composition | 69/128 = 0.5390625 | >= 0.95 | 실패 |
| TARGET-second error | 26/128 = 0.203125 | >= 0.25 | 실패 |
| First-minus-second accuracy | -0.2578125 | >= 0.10 | 실패 |
| 등록 패널 margin gap `G` | -1.850037 | lower bound > 0 | 실패 |

`G`의 등록된 finite-panel stability interval은
`[-2.273672, -1.413423]`이며 8개 fit world가 모두 음수였다. 전체
factorial 진단도 `G_full=-1.843154`로 거의 동일했다.

출력 channel은 정상이다. 448개 전 행에서 global argmax가 tie 없이 등록된
두 라벨 중 하나였고, 두 valid label의 평균 probability mass도
retrieval 0.999925, lookup 0.999356, composition 0.999161이었다. 따라서
형식 오류가 아니라 행동 구성의 실패다.

### 2. 의미 있는 사후 발견

이 절은 탐색적 분석이며 동결 status를 바꾸거나 다음 V5 단계를 승인하지
않는다.

TARGET-first deficit 가설은 단순히 약하게 실패한 것이 아니라 거의
결정론적으로 반전되었다.

| 정확한 composition 쌍 | 둘 다 정답 | First만 정답 | Second만 정답 | 둘 다 오답 | Second margin 우세 |
|---|---:|---:|---:|---:|---:|
| 전체 factorial | 69 | 0 | 33 | 26 | 127/128 |
| 고정 intervention panel | 37 | 0 | 16 | 11 | 64/64 |

모델은 세 component에서 일관되게 뒤에 나온 항목을 선호했다.

- Retrieval: 마지막 output option 선택 58/64 (90.6%).
- Lookup: 마지막 codebook rule의 RHS 선택 111/128 (86.7%).
- Composition: 마지막 rule의 RHS 선택 195/256 (76.2%).

균형 factorial에서 가장 큰 세 composition contrast는 target rule last
`+4.377494`, correct option last `+2.294503`, TARGET fact second
`+1.843154`였고 모두 8/8 world에서 같은 방향이었다. 세 cue가 모두 앞쪽이면
0/32 정답, 모두 뒤쪽이면 32/32 정답이었다. 이 세 cue만 사용한 탐색적
가산 모형은 margin 분산의 `R^2=0.6927`을 설명하고 217/256의 정오 부호를
맞췄다. 이는 같은 fit panel에서 적합·평가한 기술 모형이지 외부 예측 검증이
아니다.

TARGET-first composition은 standalone lookup과 122/128에서 같은 답을 냈고,
63개 lookup 오답 중 5개만 바로잡았다. TARGET-second에서는 일치가 91/128로
낮아졌지만 lookup 오답 63개 중 37개가 정답으로 바뀌었고, lookup 정답 65개는
하나도 손상되지 않았다. 즉 마지막 fact가 응답 인접 cue로 작동해 실패한
lookup을 덮어쓴 것이며, 안정적인 composition 능력의 증거가 아니다.

### 3. 무결성과 V4 비교

동결 분석과 별도 사후 분석에서 fixture binding, answer ledger, raw-row SHA,
logit, margin, argmax, valid label, semantic pair invariant 불일치는 모두
0건이었다. 역전은 분석 부호나 정답표 오류가 아니라 동결 scaffold에서의
모델 행동이다.

V4는 facts 뒤에 `TASK`와 valid outputs를 두었지만, V5는 이 블록을 facts
앞에 두고 두 fact 줄로 prompt를 끝냈다. 그 결과 V5에서는 TARGET-second가
응답에 가장 가까운 의미 항목이다.

| 실험 | TARGET-first | TARGET-second | 정확도 차이 F-S | Margin 차이 F-S |
|---|---:|---:|---:|---:|
| V4 | 0.875 | 0.65625 | +0.21875 | +0.1944 |
| V5 | 0.5390625 | 0.796875 | -0.2578125 | -1.8432 |

V5 내부의 TARGET 순서 변화는 같은 정답을 유지한 정확한 matched
manipulation이므로 이 scaffold에 대한 순서 효과는 인과적이다. 그러나
V4와 V5 사이에는 query 위치 외에도 symbol, 문구, entity, world 구성이 함께
변했다. 따라서 확정 가능한 cross-study 결론은 **scaffold에 따른 효과 부호의
불안정성**이다. Query placement와 recency가 반전의 원인이라는 설명은 새
factorial 실험으로 검증해야 하는 가설이다.

### 4. 과학적 경계

Behavioral construct가 admission을 통과하지 못했으므로 V5는 causal
activation mediation을 지지하지 않는다. 이미 저장된 fit activation을 이용해
사후적으로 방향을 뒤집거나, 정답 쌍만 고르거나, localization/holdout을 여는
것도 허용되지 않는다.

V5는 필요한 symbolic fact를 prompt에 직접 제공했으므로 latent knowledge
실험도 아니다. Biology, model family 일반화, physical law도 지지하지 않는다.
지지되는 결론은 다음처럼 좁다.

> Qwen2.5-1.5B-Instruct의 이 고정 factorial scaffold에서는 rule, option,
> fact recency가 강한 later-item policy를 만들며, TARGET-order 효과는
> scaffold에 따라 부호가 바뀌므로 아직 운반 가능한 activation gap이 아니다.

### 5. 다음 V6: scaffold x reactivation

V5 localization/holdout world는 열거나 재사용하지 않는다. V5 fit은 공개된
가설 생성 자료로만 사용한다. Disposable calibration에서는 component
accuracy로만 prompt 명료도나 model size를 정하고 gap 방향·크기를 선택에
사용하지 않는다. 두 scaffold가 retrieval 및 two-rule lookup 전체 0.98 이상,
각 factor level 0.90 이상을 두 paraphrase에서 달성해야 confirmatory bank를
동결한다. 1.5B가 통과하지 못하면 더 큰 모델을 먼저 고정한다.

V6 factor는 다음과 같다.

1. TASK/valid-output block의 facts 전/후 배치;
2. TARGET fact first/second;
3. 의미 있는 경쟁 OTHER fact/token-length matched inert filler;
4. target-identity reactivation cue/matched neutral nonce cue;
5. 균형화된 mapping, rule order, output order.

Target cue에는 target entity만 넣고 property나 answer code는 넣지 않는다.
모든 prompt는 동일 token 위치의 `ANSWER:` cue로 끝낸다. 주 intervention
site는 prompt 전체를 본 `ANSWER:`의 `resid_post`이며 TARGET-property site는
보조 encoding 분석으로만 둔다.

새 fit world에서 component gate를 모두 통과하고, V5형 early-target gap `G`,
scaffold interaction `H`, 선택적 reactivation `J`가 각각 사전등록된 양의
방향으로 재현될 때만 causal localization을 실행한다. 구체적으로
`G` accuracy gap 0.10 이상, `G`와 `H`의 lower bound 양수, 각각 6/8 world
양수, `J/G >= 0.30`, `J` lower bound 양수를 요구한다.

그 뒤에만 response-site reactivation axis를 fit world에서 적합한다.
Localization은 cross-fit separation, rescue와 reverse damage 각각 `0.30G`
이상, answer/null/wrong-entity sham 대비 specificity `0.20G` 이상 및 identity
검사를 요구한다. 가장 얕은 단일 통과 layer를 고정하고, untouched 16-world
holdout에서 동일 기준과 12/16 양수 world를 다시 요구한다.

### 6. 생물학과 법칙 확장

더 나은 생물학 질문은 다음과 같다.

> 모델이 독립적으로 검증된 biological entity-state 관계를 직접 회수할 수
> 있을 때, 새 reporting code와 경쟁 관계가 use deficit을 만드는가? 그리고
> 정답 정보를 포함하지 않은 entity cue 및 response-site intervention이 이를
> 선택적으로 복구하는가?

Time-stamped reviewed panel, 상호배타적 label, alias/paraphrase,
counterfactual-override 및 fictional-entity control, baseline 이후 item filtering
금지가 필요하다. 양성 결과는 accessible-but-unused biological knowledge의
조작적 증거가 될 수 있지만 memorization이나 학습 경로의 증명은 아니다.

이 현상은 물리 법칙이 아니다. 여러 모델, scale, template에서
target-distance x distractor-load의 사전등록 정량 관계가 OOD extrapolation까지
재현되어야 비로소 경험적 computational scaling law라고 부를 수 있다.

## Artifact bindings

- V5 preregistration:
  `0a2b5fef4329bf5fad3a004a8a03da2b4f00f0ebc344359e5f2857073e439f1e`
- plan manifest:
  `56c26948d45192e00df95643ba472d70b2792d87c88b46f70a2c3ae54d3b8f7f`
- fit execution manifest:
  `c7bef521fac2431759e4466c549d7f26180913a2e127437c576428605af07411`
- fit records:
  `e0883f62110d040da29d841c43b0f6c24f7dac35c79a5b464384cf9409f8c96d`
- fit activations:
  `67790740ba89d0b9dcdaeb67a8adecf0397fe81acddb09d1ffb4d7361c91af5c`
- frozen fit analysis:
  `a14b1f5a75d4ea0768cf843432e1371ec3717b2f4bf543e4eb7c913e434b886e`
- post-hoc analyzer:
  `2942e0b41257b8b65cdd8b45bf9a8abb63ad11cbc4eff3e597ee91f37bef5336`
- post-hoc analysis:
  `668edaaa5582331e0a0425c9afcb1ae459bff7bf08142a62f0fec457ac2a4d9d`
- post-hoc analyzer test:
  `cb978f3d1773985ce843a936b4861ed3d33a375d24e64f57994c97e43f0904c9`
