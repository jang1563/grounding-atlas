# GSE96583 multi-donor expression-context discovery

Run date: 2026-07-30  
Model: `claude-haiku-4-5-20251001`  
Data: eight unstimulated SLE donors from GSE96583  
Execution: 1,120/1,120 unique calls parsed exactly

## Decision

| claim gate | result | reason |
|---|---|---|
| Gate A: TCR/CD8 and cytotoxic opposition in `T_plus_cytotoxic` | **PASS** | both donor-level components passed; exact IUT `p=0.01953125` |
| Gate B: NK-receptor and cytotoxic concordance in `NK_receptor_plus_cytotoxic` | **FAIL** | receptor component had the expected strict sign in only 6/8 donors |
| full hierarchical extension | **FAIL** | Gate B failed after Gate A passed |
| prompt invariance | **FAIL** | interaction intervals were not contained in `±0.03` |
| neutral-sham equivalence | **FAIL** | sham intervals crossed zero but were too wide to fit inside `±0.03` |

This is a meaningful asymmetric result, not a generic validation success.
The earlier TCR/CD8-versus-cytotoxic evidence competition replicated at the
donor level, while the stronger claim that NK-receptor evidence is
donor-uniform was prospectively falsified.

## English interpretation

### What replicated

Within the expression-selected `T_plus_cytotoxic` context:

- TCR/CD8 matched-deletion effect on P(CD8):
  `+0.1142`, 95% donor CI `[+0.0929,+0.1356]`, exact sign-flip
  `p=0.00390625`, expected sign in 8/8 donors.
- Cytotoxic matched-deletion effect on P(CD8):
  `-0.0648`, CI `[-0.1274,-0.0023]`, exact `p=0.01953125`,
  expected sign in 7/8 donors.
- TCR-minus-cytotoxic separation:
  `+0.1791`, CI `[+0.1167,+0.2415]`, two-sided `p=0.000257`;
  the separation was positive in all eight donors.

The corresponding absolute unmasked-minus-module effects had the same
directions: TCR/CD8 `+0.0986 [+0.0658,+0.1314]` and cytotoxic
`-0.0556 [-0.0973,-0.0140]`. Thus the registered relative contrasts are not
explained solely by neutral deletion moving in the opposite direction.
However, neutral-sham equivalence was not established, so the strongest valid
wording remains “directional leverage relative to a matched deletion, with
concordant absolute change.”

### What did not replicate

Within `NK_receptor_plus_cytotoxic`:

- NK-receptor/identity aggregate effect:
  `-0.0528`, CI `[-0.0863,-0.0193]`, exact `p=0.015625`.
- Cytotoxic effect:
  `-0.1164`, CI `[-0.1925,-0.0402]`, exact `p=0.00390625`,
  expected sign in 8/8 donors.

The receptor mean and interval were negative, but the prospective consistency
rule required at least seven strictly negative donor effects. Only six passed:
donor 101 was slightly positive (`+0.0083`) and donor 107 was exactly zero.
Therefore the receptor component, Gate B, and the full hierarchical claim
failed. The strict 54-cell matching sensitivity gave the same decision.

The receptor-minus-cytotoxic difference was `+0.0635` with CI
`[-0.0290,+0.1561]` (`p=0.149`). The data therefore do not establish that
cytotoxic evidence is stronger than receptor evidence, even though that is a
useful next hypothesis.

### The biologically useful localization

The broad “cytotoxic” category is not behaving as a homogeneous biological
program:

- In `T_plus_cytotoxic`, 19/32 selected targets were `CCL5`; their
  donor-equal descriptive effect was approximately zero (`-0.0055`).
  Seven `GNLY`-selected cells had a much larger descriptive effect
  (`-0.2317`; four donors), and four `GZMB`-selected cells were also negative
  (`-0.1358`; three donors).
- In `NK_receptor_plus_cytotoxic`, `GNLY` occurred in 16/24 cells across seven
  donors and had a descriptive effect of `-0.1570`; `CCL5` occurred in four
  cells across two donors and did not show the same direction.
- Receptor targets were heterogeneous: `FCGR3A` was more negative
  descriptively than `KLRC1`, while many receptor-targeted cell effects were
  exactly zero.

These are post-hoc, expression-selected strata. They support a prospective
target-specific experiment, not individual-gene biological causality.

The result argues against a single smooth coefficient for an entire marker
module. A better mathematical candidate is a sparse or zero-inflated
cue-gating model with target-specific slopes and donor/context random effects.
The model emitted only seven distinct raw probabilities, with 1,044/1,120
outputs equal to `0.15`, `0.25`, `0.75`, or `0.85`; this is more consistent
with discrete category switching than calibrated additive evidence
integration. That model is a new hypothesis, not a discovered law.

### Better biology question

> When lineage-identity and effector-state signals coexist, which axis does
> the model actually use, and is the failure to use a receptor cue caused by
> missing knowledge or by context-dependent access to known knowledge?

This separates three constructs that the present binary annotation question
mixed together:

1. knowledge of what a marker means;
2. use of that marker in a cell-state decision;
3. donor- or context-dependent routing of that evidence.

The next discriminating study should prospectively balance `GNLY`, `CCL5`,
`FCGR3A`, `KLRD1`, and `KLRC1` target strata across donors; freeze an
independent marker-knowledge probe; and cross exact-token mask, exact-token
rescue, full-name/functional-description rescue, and matched-neutral deletion.
An orthogonal CITE-seq or sorted-cell protein reference would remove the
same-expression annotation circularity. A latent causal activation-gap claim
would additionally require an open-weight model and a hidden-state
erasure/rescue or activation-patching intervention.

### Scope

This experiment is verifiable as an input-output causal audit: raw outputs,
the randomized plan, donor/context assignments, hashes, and donor-level
statistics are preserved and independently recomputable. It is not a physical
law, a biological gene perturbation, proof of annotation truth, proof that
knowledge is stored but inactive, or a hidden-state causal activation gap.

Prompt interactions were not equivalent, so all effects are four-form
averages and are prompt-dependent. Neutral sham effects were small and
compatible with zero, but their intervals did not establish `±0.03`
equivalence.

## 한국어 해석

### 재현된 것

`T_plus_cytotoxic` 발현 문맥에서:

- TCR/CD8 효과는 `+0.1142`
  (95% donor CI `[+0.0929,+0.1356]`, exact `p=0.00390625`, 8/8 donor).
- cytotoxic 효과는 `-0.0648`
  (CI `[-0.1274,-0.0023]`, exact `p=0.01953125`, 7/8 donor).
- 두 효과의 분리는 `+0.1791`
  (CI `[+0.1167,+0.2415]`, `p=0.000257`)이며 8/8 donor에서 양수였다.

따라서 TCR/CD8 증거와 cytotoxic 증거가 CD8/NK 판단에서 서로 반대
방향으로 작용한다는 기존 발견은 donor-level로 재현되었다. 절대
unmasked-minus-module 변화도 같은 방향이었지만, neutral sham의
`±0.03` 등가성은 입증되지 않았다. 따라서 “절대적이고 deletion
특이적인 작동”보다 “matched deletion 대비 방향성 leverage”라고
표현하는 것이 정확하다.

### 재현되지 않은 것

`NK_receptor_plus_cytotoxic` 문맥에서 receptor 평균은 음수였고
통계적으로도 0에서 떨어져 있었지만, 사전등록한 donor 일관성 조건을
통과하지 못했다. 8명 중 6명만 엄격히 음수였고, donor 101은 약한
양수, donor 107은 정확히 0이었다. 반면 같은 문맥의 cytotoxic 효과는
8/8 donor에서 음수였다.

따라서:

- Gate A는 **통과**;
- Gate B는 **실패**;
- 전체 receptor 확장 주장은 **실패**이다.

이는 “receptor cue도 모든 donor에서 안정적으로 NK 방향으로
활성화된다”는 강한 가설을 반증한 의미 있는 결과다.

### 의미 있는 생물학적 발견

`cytotoxic_effector`를 하나의 균질한 프로그램으로 보면 안 된다.
`CCL5` 선택 세포에서는 효과가 거의 없었지만, `GNLY` 선택 세포에서는
큰 NK 방향 효과가 나타났다. receptor 쪽도 `FCGR3A`, `KLRD1`,
`KLRC1` 사이에서 효과가 달랐고 zero effect가 많았다.

따라서 더 적절한 모델은 “모듈 전체의 연속적 가중치”가 아니라,
특정 cue가 문맥과 donor에 따라 켜지거나 꺼지는 sparse/zero-inflated
gating 모델이다. 하지만 이것은 다음 실험에서 검증할 수학적
가설이지 물리 법칙이 아니다.

더 좋은 생물학 질문은 다음과 같다.

> lineage identity와 effector state 신호가 함께 있을 때 모델은 어느
> 축을 실제로 사용하는가? receptor cue를 사용하지 않는 이유는
> 지식 부재인가, 아니면 이미 아는 지식의 문맥 의존적 접근 실패인가?

다음 실험은 donor-balanced `GNLY/CCL5/FCGR3A/KLRD1/KLRC1` strata,
독립적인 marker-knowledge probe, token mask/rescue, full-name 또는
functional-description rescue를 결합해야 한다. 잠재지식의 causal
activation gap까지 주장하려면 open-weight 모델에서 hidden-state
erasure/rescue 또는 activation patching이 추가로 필요하다.

### 검증 가능성과 경계

현재 결과는 raw response, plan, donor/context 배정, hash, donor-level
통계를 통해 재계산 가능한 input-output causal audit이다. 그러나
물리 법칙, 생물학적 gene perturbation, annotation truth, 잠재지식의
증명, 또는 hidden-state causal activation gap은 아니다.

## Provenance note

The raw checkpoint binds the runner, prompt, input, preregistration, and plan
hashes. The imported helper file was not bound per record. Its post-run hash
and runtime versions are preserved in
`execution_dependency_manifest.json`; that attestation documents the
uninterrupted run but does not retroactively create per-record helper binding.
Future execution code must include that dependency hash in every checkpoint
record.

The target-stratum and raw-quantization values in this document are
recomputed from the checkpoint by `eval/analyze_gse96583_context_discovery.py`;
machine-readable and rendered outputs are `discovery_localization.json` and
`discovery_localization.md`.
