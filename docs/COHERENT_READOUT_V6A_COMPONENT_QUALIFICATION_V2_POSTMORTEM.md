# Coherent Readout V6A qualification V2: contextual-token postmortem

Date: 2026-08-03  
Status: descriptive replay of sealed logits; V2 terminal status unchanged  
Model calls added: `0`

## English

### Frozen result and scope

The preregistered V2 analysis remains
`V6A_QUALIFICATION_COMPONENT_FAIL`. It was engineering-valid and evaluated all
384 planned direct-component rows, but it did not authorize V6A topology or an
activation intervention. This postmortem neither rescored nor reopened that
result.

The postmortem independently revalidated the frozen plan, records, six
full-vocabulary float32 logit shards, model/tokenizer dependencies, and terminal
analysis. It then performed a zero-forward descriptive replay using the natural
assistant continuation surface already represented in the tokenizer:

```text
ANSWER: + U+0020 + registered glyph
```

Thus a semantic answer such as `gamma` was represented by the single decoded
token `" gamma"`, not by the different bare token `"gamma"`. Here `gamma`
stands for the corresponding one-character registered glyph.

### Measurement mismatch

V2's tokenizer receipt had already recorded two distinct IDs for every one of
the 32 glyphs:

- the token used by the glyph in prompt text, which decodes as one ASCII space
  plus the glyph; and
- the registered continuation token, which decodes as the bare glyph.

The frozen preflight required each ID to be internally stable but did not
require the prompt-occurrence ID to equal the answer-continuation ID. The model's
full-vocabulary maxima consistently used the former, natural token family.
Consequently, the frozen registered-pair gate reported `0/384` global maxima in
the bare-token pair even though the natural-token maximum set was inside the
semantic answer pair on `384/384` rows.

| Metric | Frozen bare-token contract | Descriptive natural-token replay |
|---|---:|---:|
| Property retrieval, pairwise correct | 256/256 | 256/256 |
| Lookup, pairwise correct | 125/128 | 127/128 |
| Unique correct full-vocabulary maximum | 0/384 in registered pair | 383/384 |
| Full-vocabulary maximum set contained in answer pair | 0/384 in registered pair | 384/384 |
| Exact natural-answer ties | not applicable to corrected pair | 1/384 |

Natural answer-pair probability mass was also concentrated on the intended
channel: its minimum was `0.9738629` for retrieval and `0.9526037` for lookup;
the corresponding means were `0.9993118` and `0.9977632`.

### Three diagnostic rows

- Call 62 (`lambda` versus `mu`) changed from a bare-token margin of `-0.625`
  to an exact natural-token tie at `0.0`. Both natural answer tokens were joint
  full-vocabulary maxima. This remains an ordinary failure.
- Call 207 changed from a bare-token margin of `-1.375` to a natural-token
  margin of `+2.0`; the expected natural token was the unique global maximum.
- Call 250 changed from a bare-token tie to a natural-token margin of `+3.875`;
  the expected natural token was the unique global maximum.

The descriptive natural-token result therefore has one real unresolved row,
not three, but it still cannot become a V2 PASS.

### Consequence for the next study

The legitimate repair is prospective:

1. preserve the V2 FAIL and its original gates;
2. register the answer surface as exactly one ASCII space plus one glyph before
   any new forward;
3. require exact one-token prefix extension, exact decode, and equality between
   every prompt-occurrence token ID and its answer-continuation token ID;
4. use behaviorally unseen, tokenizer-compatible symbols and fresh prompt
   identities; and
5. treat ties as ordinary errors under fixed-panel budgets rather than as a
   posthoc tie-break or a universal one-row veto.

This finding identifies an output-channel measurement error. It is not evidence
of an activation gap, latent knowledge, biology, a physical law, or model-family
generality. The fresh V6A-R2 topology design must establish a replicated
behavioral topology effect before a separate V6B activation intervention can
be designed.

### Reproducibility

- Read-only analyzer:
  `eval/analyze_coherent_readout_v6a_component_qualification_v2_posthoc.py`
- Descriptive artifact:
  `results/benchmark/single_cell/coherent_readout_v6a_component_qualification_v2_posthoc/qwen2.5-7b-instruct/contextual_token_posthoc_analysis.json`
- Descriptive artifact SHA-256:
  `8b3ed9a43241286fb72e10edea0d27f2a2ead113c482422d5340ff51f6438ed8`
- Source V2 terminal-analysis SHA-256:
  `595cff448a3f72011e119f37556c95e11ea0fe4c4daef7d79541f507f4987cb8`
- Postmortem model calls, generation calls, and composition calls: `0`, `0`,
  and `0`.

## 한국어

### 동결 결과와 핵심 발견

V2의 사전등록된 최종 상태는 그대로
`V6A_QUALIFICATION_COMPONENT_FAIL`이다. 384개 direct-component row는
engineering-valid하게 실행됐지만, V6A topology 또는 activation intervention을
승인하지 않았다. 이번 사후분석은 이 결과를 재채점하거나 PASS로 바꾸지 않는다.

문제는 응답 의미가 아니라 응답 token의 표면형 등록이었다. V2는 `ANSWER:` 뒤의
공백 없는 glyph token을 정답으로 등록했다. 그러나 tokenizer와 모델이 실제로
사용한 자연스러운 다음 token은 `ASCII 공백 1개 + glyph 1개`가 합쳐진 단일
token이었다. V2 receipt에는 두 ID가 모두 기록됐지만, prompt 속 glyph token
ID와 answer continuation ID의 일치를 요구하지 않았다.

Natural-token으로 이미 저장된 full-vocabulary logits를 다시 읽으면:

- property retrieval: `256/256`;
- lookup: `127/128`, exact tie 1개;
- expected natural token이 unique global maximum: `383/384`;
- global maximum set이 natural answer pair 내부: `384/384`;
- 새 model forward: `0`.

Call 207과 250은 자연 token에서 각각 margin `+2.0`, `+3.875`로 교정됐다.
Call 62는 `lambda`와 `mu`가 모두 global maximum인 정확한 tie로 남았다. 따라서
실제 미해결 행은 1개이지만, 이것으로 V2 FAIL을 소급 변경할 수는 없다.

### 다음 단계의 의미

정당한 수정은 새 연구에 대해 사전적으로 수행해야 한다. R2는 정답 표면형을
`공백 1개 + glyph 1개`로 고정하고, exact decode·한-token prefix extension·
prompt occurrence와 continuation의 동일 token ID를 forward 전에 증명해야 한다.
또한 행동적으로 미노출이고 tokenizer-compatible한 새 심볼 및 prompt ID를
사용해야 한다.

이 결과는 **output-channel measurement mismatch**를 확인한 것이다. Activation
gap, 잠재지식, biology, 물리 법칙을 증명한 결과는 아니다. 먼저 fresh V6A-R2
discovery와 confirmation에서 행동적 topology 효과가 재현되어야 하며, 그 뒤에만
별도 V6B activation intervention 설계를 승인할 수 있다.
