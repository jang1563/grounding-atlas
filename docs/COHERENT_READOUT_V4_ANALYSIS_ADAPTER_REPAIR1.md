# Coherent Readout V4 analysis-adapter repair 1

Repair date: 2026-08-03  
Mode: post-freeze technical amendment; no new model call  
Repair ID: `coherent-readout-v4-analysis-adapter-repair1`

## 1. Trigger and preserved failure

The frozen V4 behavior run completed all 448 preregistered calls and wrote its
immutable raw artifacts. The first invocation of the frozen analyzer stopped
before aggregation or output with:

```text
BehavioralDeconfoundingAnalysisError:
record differs from frozen cell field: schema_version
```

The frozen runner writes execution records with
`coherent-readout-v4-behavioral-deconfounding-record-v1`. The independently
frozen fixture registry correctly tags cells with
`coherent-readout-v4-behavioral-deconfounding-cell-v1`. The analyzer's adapter
blindly compared the two type tags as if they occupied one namespace. They are
intentionally different.

Across the 448 one-to-one record/cell pairs, `schema_version` is the sole
overlapping-field disagreement. The other 17 overlapping fields agree in every
pair. This is a metadata-adapter defect, not a prompt, logit, scoring,
threshold, bootstrap, policy, or status defect.

## 2. Frozen inputs that must not change

| artifact | SHA-256 |
|---|---|
| V4 preregistration | `af63bb4fabcff96486adf9715c5ade276a44fe5f07dc07add02cb452836e0bdb` |
| frozen runner | `4773f48ec1fa6d7e9b6a685456e3729183d1ff66682f14c41954243f4d60d28a` |
| frozen analyzer | `7b8b460714d146371d6a24a2459bb6a19c7fa979bb4a4ecd86ecec6bfc2b1175` |
| dependency lock | `14e3cc08b61df3d5e05036897640e925c8f0ee54e7e767591dde486aa49c5bbb` |
| plan manifest | `fc0446884ff51c9fb2b9fc4d1f99490adbf99faaf3637b6463ebdc82b73ffb1a` |
| design | `1b6d841db45025452d6bc3f845ea3872718cd15e4d8aa339aac6b4056827de23` |
| tokenization receipt | `a7c600a7af28366f2dc369b60adcf2277b86bf0121e862306bfb527af8f84543` |
| behavior attempt | `56e06041ae81b88b7dfda1cb928fdeef15f117b0124a68b147961a1bf4f59b53` |
| behavior records | `a96554156f91f7b6721cd5abdb21f3694f30ab6efd3f55f0aec4de840e14d041` |
| raw full-vocabulary logits | `4f281510fb70f5f678eee3680999193b559970cf823e67d7c03f6b50bde80928` |
| behavior execution manifest | `3ebf2262111376d0870c0189016d62eed83e46ef9b231cf12f3edc28433c4132` |

The raw matrix remains little-endian float32 with shape `(448, 151936)` and
logical-f32 SHA-256
`3beb4b6e09ef91d23c94e961008a99044b49798d5b9747d9750192761bba9a23`.

## 3. Sole permitted repair

The separately named repair adapter must:

1. verify all original hashes and call the untouched frozen runner's full
   artifact validator;
2. require exactly 448 record-schema tags and 448 cell-schema tags of their
   respective frozen types;
3. require one-to-one, order-preserving cell IDs and the exact set of 18 shared
   keys;
4. require equality of every shared field other than `schema_version`;
5. remove only the cell-level `schema_version` from an in-memory copy of the
   merge view, preserving the record-level type tag and every disk byte;
6. pass that adapted view to the untouched frozen analyzer's raw-sidecar replay
   and preregistered `analyze_records` decision tree; and
7. write only separately named `repair1` outputs.

Any second mismatch, schema deviation, raw-row mismatch, hash change, or
decision-rule change must stop the repair. The adapter, tests, this memo, and
all input hashes are frozen in a separate zero-forward repair plan before the
official repaired aggregation.

## 4. Outcome-exposure disclosure

During independent diagnosis, a delegated read-only audit executed the proposed
in-memory suppression and exposed a provisional gate outcome before the
repair1 plan was frozen. No artifact was written or changed, and the unique
schema collision and minimal repair had already been identified. Nevertheless,
repair1 is **not claimed to be outcome-blind**. It is a disclosed post-freeze
implementation repair applied to preregistered estimands and thresholds.

Accordingly, the mechanically recovered metrics may be described as the V4
preregistered analysis **with a post-freeze implementation-repair qualifier**.
It is forbidden to claim that the original frozen analyzer completed unchanged.
Any new metric, threshold, subset, policy, or interpretation remains
exploratory.

## 5. Claim boundary

Repair1 executes zero model forwards and zero biological calls. It cannot
authorize activation work or establish a causal mechanism, latent knowledge,
an activation gap, biology, a physical law, or model-family generality. All V4
terminal statuses retain the original behavior-only scope.

## 한국어 요약

원 analyzer는 record schema와 fixture-cell schema라는 서로 다른 type tag를
같아야 한다고 잘못 비교해 aggregation 전에 중단되었다. Repair1은 원본
runner/analyzer/data를 전혀 수정하지 않고, 448개 쌍에서 `schema_version`만
유일한 충돌임을 확인한 뒤 cell 쪽 tag만 메모리상의 merge view에서 제거한다.
그 다음 원 frozen analyzer가 raw logits와 모든 preregistered gate를 그대로
재실행한다.

진단 과정에서 repair lock 전에 잠정 결과가 한 번 노출되었으므로 완전한
outcome-blind prereg 실행이라고 부르지 않는다. 결과에는 반드시
`post-freeze implementation repair` 한정어를 붙인다. 추가 model/biology
호출은 없고, activation·잠재지식·물리 법칙 주장을 승인하지 않는다.
