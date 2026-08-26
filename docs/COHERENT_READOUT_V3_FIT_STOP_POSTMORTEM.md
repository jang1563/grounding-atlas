# Coherent Readout V3: frozen fit-stop postmortem

Date: 2026-08-02  
Scope: descriptive analysis of already completed direction-fit artifacts  
Status: exploratory postmortem; not part of the frozen V3 preregistration

## English

### Frozen outcome

V3 stopped at the first behavioral-admission gate. No localization or holdout
forward was executed. The emitted status is `FIT_STOP_BASIS_LOCK_INVALID`, but
the artifact fields show that this is a composite authorization failure:

- `basis_engineering.pass = true`;
- `engineering_pass = true`;
- `behavioral_admission_pass = false`; and
- `localization_baseline_authorized = false`.

The frozen label must not be renamed after execution. Scientifically it should
be read as “fit-stage composite authorization failed,” not as evidence that the
numerical basis construction failed.

### Confirmatory gate results

- Native-answer accuracy: `320/512 = 0.625` (required: `>= 0.95`).
- Native-X recall: `64/256 = 0.25`; native-Y recall: `256/256 = 1.00`.
- Accuracy by queried property and codebook:
  - `P / identity = 0.50`;
  - `P / swapped = 1.00`;
  - `Q / identity = 1.00`;
  - `Q / swapped = 0.00`.
- Counterfactual accuracy by transition and codebook:
  - `P -> Q / identity = 1.00`;
  - `P -> Q / swapped = 0.00`;
  - `Q -> P / identity = 0.50`;
  - `Q -> P / swapped = 1.00`.
- Every global argmax was a unique X or Y; mean X/Y probability mass was
  `0.9999456`. The failure is therefore not an output-channel or tie failure.
- The text-flip gap was positive in all 16 worlds: mean `G = 4.1524`, frozen
  world-bootstrap 95% interval `[4.0941, 4.2198]`. This shows logit sensitivity
  to the text counterfactual, but failed endpoint accuracy prevents interpreting
  it as compositional competence.

### Post-hoc failure structure

Across all 512 rows, with no exceptions, the observed prediction rule was:

> predict X iff queried property is P, the codebook is identity, and the target
> fact is first; otherwise predict Y.

This three-factor rule reconstructed `512/512` predictions. No subset of one or
two frozen factors reconstructed more than `448/512`. Distractor property,
target entity slot, and world changed confidence but never the predicted class.
The model emitted Y on `448/512` rows.

A representative semantic-equivalence failure is the world-001 P/identity
prompt with a Q distractor. When the target fact was first, the expected and
predicted answer was X (`X=34.4981`, `Y=28.7888`). Swapping only the two labeled
fact lines retained the expected X but produced a strong Y (`X=29.0891`,
`Y=37.0335`). Under the swapped codebook, a queried Q that clearly maps to X
also produced Y (`X=29.4357`, `Y=34.3477`).

The answer ledger and text-counterfactual construction were independently
rechecked and are correct. The most economical explanation is a strong Y/default
plus positional heuristic. V3 always displayed answer choices as “X or Y,” so
answer-option recency was not counterbalanced and is confounded with genuine
rule use. This explanation is a prospective V4 hypothesis, not a confirmatory
V3 claim.

### Fit-only geometry

The frozen numerical eligibility rule passed at every layer, but eligibility is
not scientific separability. The content and distractor unit directions had
cosines `-0.999962`, `-0.999742`, `-0.927639`, `-0.399067`, and `-0.356125` at
layers 8, 12, 16, 20, and 24. Thus early-layer content and distractor projectors
were almost identical up to sign. No intervention was authorized, so these are
descriptive fit-set facts only.

### Consequence

V3 supports no causal recomposition, latent-knowledge, activation-gap, biology,
physical-law, or model-family claim. It does support the narrower conclusion
that the preregistered admission gate successfully prevented a strongly biased,
position-sensitive policy from being misread as codebook composition.

The next prospective tier must use new worlds, counterbalance fact order,
codebook-rule order, and answer-option order, and separately test property
retrieval, codebook lookup, and their composition before fitting any activation
axis. A future causal tier must additionally require prospective
content-versus-distractor projector separation.

## 한국어

### 동결된 결과

V3는 첫 behavioral-admission gate에서 중단되었다. Localization 및 holdout
forward는 실행하지 않았다. 출력 status는 `FIT_STOP_BASIS_LOCK_INVALID`이지만,
artifact 내부에는 `basis_engineering.pass=true`, `engineering_pass=true`,
`behavioral_admission_pass=false`, `localization_baseline_authorized=false`로
기록되어 있다. 따라서 이는 수치 basis 실패가 아니라 **복합 진행 권한
lock의 실패**로 해석해야 한다. 실행 후 status 이름은 소급 변경하지 않는다.

### Confirmatory gate 결과

- Native-answer accuracy: `320/512 = 0.625` (기준 `>= 0.95`).
- Native-X recall: `64/256 = 0.25`; native-Y recall: `256/256 = 1.00`.
- `P/identity=0.50`, `P/swapped=1.00`, `Q/identity=1.00`,
  `Q/swapped=0.00`.
- 모든 global argmax는 tie 없는 X 또는 Y였고, 평균 X/Y probability mass는
  `0.9999456`이었다. 즉 output channel 자체의 실패는 아니다.
- Text-flip gap은 16개 world 모두 양수였고 `G=4.1524`, 95% world-bootstrap
  interval은 `[4.0941, 4.2198]`이었다. 이는 text counterfactual에 대한 logit
  민감도이지만, endpoint accuracy 실패 때문에 compositional competence로
  해석할 수 없다.

### 사후적으로 관찰된 실패 구조

512개 전 행에서 예외 없이 다음 규칙이 예측을 재현했다.

> queried property가 P이고, identity codebook이며, target fact가 첫 줄일
> 때만 X; 그 외에는 Y.

모델은 `448/512`회 Y를 출력했다. 정답 ledger와 text-counterfactual 생성은
정확했다. 가장 간단한 설명은 강한 Y/default bias와 위치 heuristic의 결합이다.
V3는 항상 answer option을 “X or Y” 순서로 제시했기 때문에 option recency가
진짜 rule-following과 분리되지 않았다. 이는 V4에서 prospectively 검증할
가설이며 V3의 confirmatory claim은 아니다.

### Fit-only geometry와 결론

5개 layer의 수치 eligibility는 모두 통과했지만, content와 distractor unit
direction cosine은 layer 8/12/16에서 각각 `-0.999962`, `-0.999742`,
`-0.927639`였다. 초기 layer의 두 projector가 부호를 제외하면 거의 같다는
뜻이다. Intervention은 승인되지 않았으므로 이는 fit-set descriptive fact일
뿐이다.

V3는 causal recomposition, latent knowledge, activation gap, biology, physical
law, 또는 model-family 일반성을 지지하지 않는다. 다만 preregistered gate가
강한 label/position bias를 compositional mechanism으로 오해하는 것을 실제로
차단했다는 결론은 가능하다.

다음 prospective tier는 새 world에서 fact order, codebook-rule order,
answer-option order를 완전 counterbalance하고, property retrieval, codebook
lookup, composition을 activation fitting 이전에 분리 검증해야 한다. 이후의
causal tier에는 content-versus-distractor projector separation gate도 추가한다.

## Frozen artifact bindings

- `fit_baseline_records.jsonl`:
  `9faff1d3c95bdc21de730a0d5eb1e8e8c49e8fa26c0eae2d61698a964d070c70`
- `fit_basis_analysis.json`:
  `e85b7a1bd7d5cab9947c85db9ded379ead1dddd7d41a5caf6de94634925572ef`
- `fit_basis_lock.json`:
  `e628bf75c04fea5c20cd6e16b1b86e469eeb9b911a3949c32e91ca641ee61790`
- V3 call plan:
  `fb95a8e5314b806a3aa3e35b33e13ff164e550726721c2305098fa07f1cbb0dd`

