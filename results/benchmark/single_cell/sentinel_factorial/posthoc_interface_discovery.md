# GSE96583 sentinel post-hoc interface discovery

## English

The registered donor-recurrent three-token output-surface hypothesis did not pass. A large repeated structure in the raw replies is a coarse, prompt-contingent readout failure; this does not distinguish an absent token effect from an effect obscured by that readout. Canonical `ab_pa` uses donor 107 for the entire full-triple mean and donor 1039 for the only negative matched GNLY value.

- Anchor gate: **FAIL** (four-component dual-scale exact IUT p=1); endpoint: `anchor_gate_failed`.
- The held-out matched GNLY mean is `-0.012500` under canonical `ab_pa`, but `-0.140625` after the four-form average. The parent receptor-context GNLY means are `-0.255714` and `-0.157024`, respectively. The similar four-form means are descriptive only: the cell selection and donor support differ.
- The numeric replies are not a coherent binary probability surface. For the same sentinel input, mean `P(A)+P(B)-1` is `+0.565000` under AB order and `+0.364750` under BA order; the mean answer-order shift after pairing queried targets is `+0.151792`.
- `407/480` replies (`84.8%`) assign at least `0.75` to whichever class is queried. Only `16/120` AB and `2/120` BA complementary pairs are within `±0.03` of summing to one, and all `120/120` cell-condition inputs vary across forms.
- Therefore this is evidence of a structured **elicitation/output-readout failure** for this model revision and template family, not proof of latent stored knowledge, hidden-state activation failure, biology, or a physical law.

Canonical donor order: `101, 107, 1015, 1016, 1039, 1244, 1256, 1488`.

| subset | a vector | h vector | r=a-h vector | r mean |
|---|---|---|---|---:|
| `GNLY` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]` | -0.0125 |
| `NKG7` | `[0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | +0.0875 |
| `CCL5` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | +0.0000 |
| `GNLY+NKG7` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | -0.0125 |
| `GNLY+CCL5` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | -0.0125 |
| `NKG7+CCL5` | `[0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | +0.0875 |
| `GNLY+NKG7+CCL5` | `[0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | `[0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | +0.0750 |

### Canonical donor drivers

- Donor 107: unmasked aligned CD8 score=`0.75`; NKG7 and full-triple target masks both move it to `0.15`, while their matched masks are `0.85` and `0.75`. Its `r(T)=+0.60` supplies 100% of the `+0.075` full-triple donor mean and opposes the registered NK-directed hypothesis.
- Donor 1039: GNLY target masking leaves the aligned CD8 score at `0.85`, while the matched GAPDH mask moves it to `0.75`. Thus its `r(G)=-0.10` is comparator-driven (`a=0`, `h=+0.10`), and it supplies the entire `-0.0125` matched GNLY mean.

### Readout coherence

| raw set | calls / levels | four-level mass | AB complement residual mean [range] | BA residual mean [range] | paired answer-order mean [range] |
|---|---:|---:|---:|---:|---:|
| held-out sentinel | 480 / 5 | 99.8% | +0.565000 [-0.70, +0.70] | +0.364750 [-0.60, +0.60] | +0.151792 [-0.300, +0.400] |
| parent | 1120 / 7 | 93.2% | +0.340071 [-0.10, +0.77] | +0.229036 [-0.10, +0.70] | +0.107982 [-0.350, +0.435] |

The held-out raw outputs occupy only five levels; `0.15/0.25/0.75/0.85` account for `99.8%` of calls. Full per-form ranges and exact residual distributions are in the JSON artifact.

A falsifiable next mathematical model is `y = Q(alpha[answer order, queried target] + b[donor] + beta[mask] + interactions)`, where `Q` is a coarse quantizer. The route-scale diagnostics above are larger than the canonical GNLY (`|mean|=0.0125`) and triple (`|mean|=0.075`) effects. This equation has not been fitted; it is a next model, not a variance decomposition, invariant, or physical law.

## 한국어

사전등록한 donor-recurrent 3-token 출력-surface 가설은 통과하지 못했다. Raw 응답에서 두드러진 반복 구조 중 하나는 거칠고 prompt 의존적인 readout 실패지만, 이것만으로 실제 token 효과가 없는지 readout에 가려졌는지는 구분할 수 없다. 정규 `ab_pa`에서 full-triple 평균은 전적으로 donor 107이 만들고, 음의 matched GNLY 값은 donor 1039 하나에서만 나온다.

- Anchor gate: **실패** (4-component dual-scale exact IUT p=1); endpoint: `anchor_gate_failed`.
- Held-out matched GNLY 평균은 정규 `ab_pa`에서 `-0.012500`, 4-form 평균에서 `-0.140625`이다. Parent receptor-context의 대응 값은 각각 `-0.255714`, `-0.157024`이다. 4-form 평균의 유사성은 기술적 발견일 뿐이며, 선택된 cell과 donor support가 달라 독립 재현이 아니다.
- 동일 입력에서 수치 응답은 일관된 이항 확률을 이루지 않는다. 평균 `P(A)+P(B)-1`은 AB 순서에서 `+0.565000`, BA 순서에서 `+0.364750`이고, queried target을 짝지은 평균 answer-order 이동은 `+0.151792`이다.
- `407/480` 응답(`84.8%`)은 질문한 class에 최소 `0.75`를 부여한다. 합이 1에서 `±0.03` 이내인 상보 query pair는 AB `16/120`, BA `2/120`뿐이며, `120/120` cell-condition 입력이 form에 따라 달라진다.
- 따라서 이는 이 model revision과 template family에서의 구조적 **elicitation/output-readout 실패**의 증거다. 잠재 지식의 증명, hidden-state activation 실패, 생물학적 인과, 또는 물리 법칙의 증거는 아니다.

검증 가능한 다음 수학 모델은 `y = Q(alpha[answer order, queried target] + b[donor] + beta[mask] + interactions)`이다. 여기서 `Q`는 거친 양자화 함수다. 위 route-scale 진단값은 정규 GNLY (`|mean|=0.0125`)와 triple (`|mean|=0.075`) 효과보다 크다. 아직 적합하거나 분산분해한 모델이 아니며, 불변식이나 물리 법칙이 아니라 다음 검증 가설이다.

### Donor 107 / 1039

- Donor 107: unmasked aligned CD8 score=`0.75`에서 NKG7 및 full-triple target mask가 모두 `0.15`로 이동한다. `r(T)=+0.60` 하나가 full-triple 평균 `+0.075` 전체를 만들며 사전등록된 NK 방향과 반대다.
- Donor 1039: GNLY target mask는 `0.85`로 변화가 없고, matched GAPDH mask만 `0.75`로 이동한다. 따라서 `r(G)=-0.10`은 target 효과가 아니라 comparator 효과(`a=0`, `h=+0.10`)이며 matched GNLY 평균 전체를 만든다.

## Post-hoc boundary / 사후분석 경계

These diagnostics were chosen after the confirmatory responses were observed. They use one model revision and template family, the same eight-donor SLE control cohort, expression-selected cells, text-token masking, and uncalibrated coarse outputs. They do not validate deposited cell labels or identify gene/pathway causality, latent knowledge, hidden-state activation, a mathematical invariant, or a physical law.

Analysis code SHA-256: `4d41552419142ff84cd7797a2db8a20179c57efbcba76a31ee85d2355273b46c`.

이 진단은 확증 응답을 본 뒤 선택한 사후분석이다. 하나의 model revision과 template family, 동일한 8-donor SLE control cohort, 발현으로 선택된 cell, text-token masking, 보정되지 않은 거친 출력만 사용한다. 따라서 기탁 cell label의 진실성, gene/pathway 인과, 잠재 지식, hidden-state activation, 수학적 불변식 또는 물리 법칙을 검증하지 않는다.
