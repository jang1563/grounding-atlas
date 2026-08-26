# Coherent readout v2 causal-binding result

Final status: **`NONSPECIFIC_DECISION_STATE_TRANSFER_REPLICATED`**

## English

This development-only, non-biological study tested whether the X-first versus
Y-first/line-2 behavioral gap reflects an item-specific relational state or a
generic late decision state in Qwen2.5-1.5B.

The prospective 48-pair holdout replicated the behavioral gap (`G=5.064`). At
the preregistered layer-24 final-token site, paired full-state replacement was
strong and bidirectional:

- forward: `F/G=0.590`, 48/48 positive pair effects;
- reverse damage: `N/G=0.851`, 48/48 positive pair effects.

However, same-pair and unrelated correct-X sources transferred almost exactly
as much as the matched source. The four preregistered exact-length specificity
ratios were `0.00081`, `0.00132`, `0.00012`, and `-0.00103 G`; every specificity
gate failed.

The meaningful conclusion is therefore not item-specific binding. Layer 24
contains a causally transferable but nonspecific line-order/X-Y decision state.
This result argues against interpreting the original behavior as proof of latent
item knowledge awaiting activation at this site.

It does not establish necessity, a unique mediator, biology, latent knowledge,
a natural activation gap, model-family generality, a general variable-binding
mechanism, or a physical law.

Recommended next experiment: a new preregistered content-by-routing factorial
with train-only directional erasure/rescue and label-renaming controls. Its goal
should be to separate token/position routing from content, not to repeat
full-state patch verification. Biology should remain closed until a selective,
content-specific causal effect replicates on a new holdout.

## 한국어

이 비생물학적 development 연구는 X-first와 Y-first/line-2 행동 격차가
item-specific relational state인지, 아니면 late generic decision state인지
Qwen2.5-1.5B에서 검정했다.

새 48-pair holdout에서 행동 격차가 재현됐고(`G=5.064`), 사전 등록된 layer-24
final-token site의 paired full-state replacement는 강한 양방향 효과를 보였다.

- forward: `F/G=0.590`, 48/48 pair에서 양수;
- reverse damage: `N/G=0.851`, 48/48 pair에서 양수.

하지만 같은 pair의 반대 선언 source와 unrelated correct-X source도 matched
source와 거의 똑같이 작동했다. 사전 등록된 네 exact-length specificity 비율은
`0.00081`, `0.00132`, `0.00012`, `-0.00103 G`였고 모든 specificity gate가
실패했다.

따라서 의미 있는 결론은 item-specific binding이 아니다. Layer 24에는
인과적으로 전달 가능한 nonspecific line-order/X-Y decision state가 있다. 이
결과는 원래 행동을 “활성화만 기다리는 item-level 잠재지식”의 증거로 해석하는
가설에 반대한다.

Necessity, 고유 mediator, 생물학, 잠재지식, 자연적 activation gap,
model-family 일반성, 보편적 variable-binding mechanism 또는 물리 법칙은
확립되지 않았다.

다음 실험은 full-state patch를 반복 검증하는 것이 아니라, 새 preregistration과
holdout을 사용한 content-by-routing factorial 및 train-only directional
erasure/rescue여야 한다. Label-renaming control로 token/position routing과 content를
분리해야 하며, selective content-specific causal effect가 재현되기 전까지 biology는
계속 닫아 두는 것이 타당하다.

## Bound artifacts

- `qwen2.5-1.5b-instruct/analysis.json`
- `qwen2.5-1.5b-instruct/analysis.md`
- `qwen2.5-1.5b-instruct/analysis_manifest.json`
- `qwen2.5-1.5b-instruct/design.json`
- `qwen2.5-1.5b-instruct/plan_manifest.json`

Primary X/Y estimands, activation sidecars, effects, and bootstraps are
artifact-auditable. Full-vocabulary argmax/tie/mass diagnostics are
runner-committed and replay-verifiable because the complete vocabulary rows were
not persisted.
