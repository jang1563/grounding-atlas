# Coherent readout v2: frozen syntax-selection result

Status: **`SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT`**  
Execution date: 2026-08-02  
Scope: permanent development-only, non-biological syntax selection

## Outcome

**English.** The 0.5B smoke artifact was valid and exercised no selection
authority. The authoritative Qwen2.5-1.5B run produced valid raw artifacts, but
none of the four preregistered answer contexts passed the conjunctive gate.
Therefore no syntax was selected and the conditional biological stage made zero
model calls. No threshold, candidate, item, form, or fallback was changed.

**한국어.** 0.5B smoke artifact는 유효했지만 선택 권한은 없었다. 선택 권한을
가진 Qwen2.5-1.5B raw artifact도 완전히 유효했으나, 사전등록한 네 answer
context 모두 결합 gate를 통과하지 못했다. 따라서 syntax를 선택하지 않았고
조건부 biological stage의 모델 호출은 0회다. 임계값, 후보, 문항, form 또는
fallback을 변경하지 않았다.

## Frozen execution chain

Both 256-row plans were written before either forward pass. Each plan manifest
reported `PLAN_AND_DESIGN_FROZEN_NO_FORWARD`, `model_calls=0`, and no output
sidecar.

| role | design file SHA-256 | zero-call plan SHA-256 | raw calls | final status |
|---|---|---|---:|---|
| Qwen2.5-0.5B smoke | `1a0e2144...f2d72d` | `d787e554...711cff` | 256 | `SYNTAX_SMOKE_ANALYSIS_COMPLETE_NO_SELECTION_AUTHORITY` |
| Qwen2.5-1.5B selection | `ed0cf164...ff0001` | `94608ea6...5738e6` | 256 | `SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT` |

The runner SHA-256 was
`6387eec115f1a8ee64149334d9a7934bcc3229ea1c81952b2a2fa0db86ae6c34`;
the independent analyzer SHA-256 was
`02ca4234a05c20bd2fe7ce00b2b0d9e50d2b6803eaf2394b2d1d87552f5717ae`.
Both execution manifests bind 256 JSONL rows to a finite little-endian float32
sidecar of shape `(256, 151936)`. File hashes and logical matrix hashes were
independently reconstructed.

## Authoritative 1.5B metrics

`X-first` pools the two forms in which the X mapping line occurs first. The
Y-first records are separated by whether the declaration denotes the concept on
mapping line 1 (correct label Y) or line 2 (correct label X). `range` is the
number of 16 items passing the registered four-form range limit; every candidate
needed 16/16.

| candidate | native | native-correct | X-first | Y-first, line 1 | Y-first, line 2 | range | eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| assistant prefill | 64/64 | 50/64 | 32/32 | 16/16 | 2/16 | 0/16 | no |
| system + user exact | 64/64 | 49/64 | 32/32 | 16/16 | 1/16 | 0/16 | no |
| user-only exact | 64/64 | 54/64 | 32/32 | 16/16 | 6/16 | 3/16 | no |
| v1 `Label:` control | 13/64 | 13/64 | — | — | — | 2/16 | no |

All six registered pair estimands (`M_O,D_O,M_R,D_R,M_I,D_I`) failed strict
equivalence for every candidate. The three exact-output contexts also missed the
61/64 native-correct floor and the form/truth-cell floors. The v1 control failed
the native-channel floor: its greedy tokens were `Label` for 32 records, `label`
for 19, X for 12, and Y for one. Although its within-X/Y contrast favored the
correct label in 55/64 records, a non-label token was the global maximum in
51/64, so that forced two-label contrast is not a native model choice.

## Meaningful failure pattern

The exact-output contexts solved the narrow output-channel problem: each emitted a
native opaque token on 64/64 records, with mean two-token vocabulary mass from
0.9912 to 0.99996. They did **not** solve relational binding.

Across all three contexts, X-first forms were perfect (32/32). Y-first forms were
also perfect when the declaration referred to mapping line 1 (16/16), but
collapsed to 2/16, 1/16, and 6/16 when it referred to line 2, whose correct label
was X. Across all Y-first records, the three contexts chose Y in 30/32, 31/32,
and 26/32 cases. Every X choice was correct; all errors were excess Y choices.
The median correct-minus-wrong logit margin was strongly positive for X-first
forms (4.014, 4.704, 7.209) and negative for the Y-first/line-2 cell (-1.432,
-2.011, -0.232). The failure affected all 8/8 lexical pairs for prefill and
system/user and 7/8 for user-only. Thus output-format adherence and semantic
mapping correctness dissociated under a controlled order-by-remapping
intervention.

세 exact context 모두 X가 첫 mapping label이면 32/32, Y가 먼저이면서 선언
대상이 1행이면 16/16이었다. 그러나 Y가 먼저이고 선언 대상이 2행(정답 X)이면
2/16, 1/16, 6/16으로 붕괴했다. 즉 단순 Y-first 효과보다 **첫 mapping label과
선언 대상 행 위치의 상호작용**이 더 정확한 기술이다.

**Interpretation.** On this fixed bank and model, the dominant failure is a
prompt-level `Y-first × declared-line-position` relational-binding interaction,
not absence of a native answer channel. This is a bounded factorial interface
result. It is not evidence that a biological answer was latent, not an activation
gap, and not a physical law.

**해석.** 이 고정 bank와 모델에서는 native answer channel 자체가 없는 것이
아니라, mapping line의 순서와 opaque label Y가 결합할 때 관계 binding이
무너지는 것이 핵심 실패다. 즉 출력 형식 준수와 의미적 mapping 정확성이
분리된다. 이는 제한된 factorial interface 결과이며 잠재 생물학 지식,
activation gap 또는 물리 법칙의 증거가 아니다.

## Why biology stopped

The outcome-blind biological projection registry and a fully disjoint eight-donor
fixture were frozen before syntax outcomes. Their entry contract nevertheless
requires one eligible 1.5B syntax. That condition is false, so the biological
fixture was never rendered into a model plan and no biological forward occurred.
Using the numerically best failed context would be an unregistered fallback and
would carry a known mapping-order confound into biology.

## Legitimate next experiment

The next meaningful expansion is a new preregistered mechanistic experiment on
the discovered binding failure, not another post-hoc prompt search. Pair an
X-first correct prompt with its matched Y-first failure and localize causal rescue
using layer/span activation patching, reverse patches, unrelated-pair null patches,
native-output rescue, and held-out lexical pairs/model-family replication. Such a
study could test a causal **instruction-binding bottleneck**. Biology should resume
only after a readout passes the same permutation-invariance gate and an orthogonal
gold/intervention design is available.

## Key artifacts

- Smoke: `qwen2.5-0.5b-instruct/analysis.json`
- Authoritative selection: `qwen2.5-1.5b-instruct/analysis.json`
- Frozen syntax preregistration: `docs/COHERENT_READOUT_V2_SYNTAX_SELECTION_PREREG.md`
- Outcome-blind biological projection registry:
  `signal/syntax/coherent_readout_v2_bio_projection_registry.json`
- Conditional biology preregistration:
  `docs/COHERENT_READOUT_V2_BIOLOGICAL_DEV_PREREG.md`
