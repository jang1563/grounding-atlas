# Coherent readout v2 causal decision-state transfer preregistration

Freeze date: 2026-08-02  
Mode: development-only, non-biological mechanistic follow-up  
Primary model: `Qwen/Qwen2.5-1.5B-Instruct`  
Primary prompt: `system_user_exact`  
Intervention family: paired post-block residual-state replacement at the final
context token

## 1. Status and hard boundary

This is a new study. It is not a continuation of the stopped biological
coherent-readout pipeline. The prior authoritative result remains
`SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT`; no result here can select a syntax,
authorize a biological forward pass, or revise that status.

The study asks one narrower question:

> At a frozen decoder-block output and the final context token, can a residual
> state from an X-first/line-1 rendering causally transfer the correct-X decision
> to its matched Y-first/line-2 rendering, and is that transfer item-specific
> rather than a generic correct-X decision state?

Even a complete pass can support only a model-, prompt-, site-, and bank-specific
causal state-transfer claim. It cannot establish biology, latent knowledge, a
natural activation gap, underactivation, a unique circuit, a general
variable-binding mechanism, model-family generality, or a physical law.

## 2. Outcome exposure and frozen inputs

The contrast and primary prompt are explicitly discovery-derived. The existing
eight-pair syntax bank and its Qwen2.5-1.5B outputs were inspected before this
document was frozen. They may be used only for layer localization and engineering
checks; they contribute zero held-out inferential units.

Frozen outcome-exposed inputs:

- syntax bank file SHA-256:
  `d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8`;
- authoritative syntax analysis SHA-256:
  `dda99af4f3dfe4dda808e39b37f2e491cfd75af7766f1dc666c5fb577135415f`;
- authoritative syntax raw-record file SHA-256:
  `f085506de6bc1501cabfc43e25520f5397f4b97e486d2c5a31cbefd7e26ac453`;
- discovery bank: 8 lexical-pair clusters, 16 items;
- primary prompt choice: `system_user_exact`, chosen because the prior fixed bank
  showed 16/16 matched X-first successes and 1/16 Y-first/line-2 successes.

Prospective held-out inputs:

- fixture:
  `signal/syntax/coherent_readout_v2_causal_binding_bank.json`;
- fixture file SHA-256:
  `2c40ba0c796202059056aec4535fd7656eab2b446d8895816bbae2034ebcbcdb`;
- fixture canonical SHA-256:
  `b2fad50672456a725274bc91df38e82f265ff969f0e7b0f15cfd81657c6b12f9`;
- manifest SHA-256:
  `fadc7eda8edc92e2b355a23db33898660265c2d2949cf1c7724172398b0dd8ac`;
- builder SHA-256:
  `b2526ce87ddcba51ea6af951fe8bbec724c41924fb6288373ce4c96cb4100359`;
- 48 new lexical-pair clusters, two declared classes per pair, 96 items;
- no held-out native output or intervention output was observed by the builder.

The 24 reciprocal unrelated-source dyads and the same-pair counterfactual-source
identities are part of the frozen fixture. Pair order groups identical
primary-prompt token-length signatures before forming dyads. Primary transfer
uses all 48 lexical pairs. Same-pair specificity is gated only on the 31 lexical
pairs whose two fixed controls are exactly length matched. Unrelated specificity
is gated only on the 22 reciprocal dyads whose four items are exactly length
matched in both directions (all eight item-direction cells). These eligibility
sets are computed from frozen input IDs
before any forward and stored in the plan; they cannot depend on outputs. All
nonmatched cases remain in descriptive ITT tables. Pair is the transfer
dependency unit; reciprocal dyad is the unrelated-control dependency unit.

## 3. Exact prompt and paired renderings

The primary messages are exactly:

```text
system: Follow the user's label mapping. Your entire response must be exactly one character: X or Y. Do not write any other text.
user: Classify the record from its explicit declaration.
{line_1}
{line_2}
{declaration}
Return exactly X or Y.
```

Chat-template flags are `add_generation_prompt=true`,
`continue_final_message=false`, and `enable_thinking=false`. Contextual answer
tokens must be the single tokens X=`55` and Y=`56`; otherwise execution stops.

For every item, choose the mapping that assigns its declared class to X. Define:

- **D (source rendering):** the declared class is mapping line 1, whose label is
  X;
- **R (recipient rendering):** the same two mapping lines are swapped, placing
  label Y on line 1 and the declared class/label X on line 2.

D and R have the same item, declaration, class words, mapping, correct token X,
token multiset, and sequence length. Only the mapping-line order differs. Both
are valid prompts; `source` and `recipient` are used instead of implying that one
input is intrinsically clean or corrupted.

All 16 discovery and all 96 holdout items remain in their assigned analysis.
Item inclusion may not depend on baseline correctness or patch response.

## 4. Model and execution lock

- model ID: `Qwen/Qwen2.5-1.5B-Instruct`;
- revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- cached `model.safetensors` SHA-256:
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`;
- cached `config.json` SHA-256:
  `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670`;
- cached `tokenizer_config.json` SHA-256:
  `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`;
- cached `tokenizer.json` SHA-256:
  `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`;
- effective chat-template text SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`;
- tokenizer ID and revision: identical to the model lock;
- architecture: 28 Qwen2 decoder blocks, residual width 1536;
- device: MPS;
- every parameter and persistent buffer: float32 on MPS after loading;
- batch size: 1, model in evaluation mode, `torch.inference_mode()`,
  `use_cache=false`, explicit `sdpa` attention implementation, no generation,
  no logits processor, no gradient;
- hook site: output of `model.model.layers[layer]` (`resid_post`), not HF
  `hidden_states[-1]` and not the final RMS normalization;
- token site: final attended context token (`-1`) only;
- patch: complete residual-vector replacement, strength `1.0`;
- the hook must fire exactly once and must be removed after every forward.

The runner, analyzer, `eval/model_hooks.py`, preregistration, fixtures, model and
tokenizer config files, tokenizer chat template, installed dependency versions,
runtime platform class, input IDs, attention masks, and every plan identity are
SHA-bound. Both the complete
discovery plan and the holdout baseline/patch template are written before the
first new model forward.
Before every execution phase, the runner rebuilds the full plan from the frozen
fixtures and locked tokenizer and requires exact equality with the disk plan;
self-consistent edits to the plan/design cannot authorize execution.

## 5. Intervention conditions

For each item and layer, use the following separately; controls may not be
averaged into one null.

1. `forward_paired`: D of the same item -> R.
2. `forward_same_pair_x`: D from the opposite declaration in the same lexical
   pair -> R. Its native correct answer is also X.
3. `forward_unrelated_x`: D from the frozen unrelated pair with the same truth
   polarity -> R. Its native correct answer is also X.
4. `reverse_paired`: R of the same item -> D.
5. `reverse_same_pair`: R from the opposite declaration in the same pair -> D.
6. `reverse_unrelated`: R from the frozen unrelated pair with the same truth
   polarity -> D.
7. `identity_r`: R -> itself.
8. `identity_d`: D -> itself.

The correct-X controls make the key distinction explicit: if paired and
unrelated D states rescue R equally, the result is a generic X-decision transfer,
not item-specific relational binding.

## 6. Staged execution and stop rules

### Stage A: plan freeze and engineering gates

Write `design.json` and `plan_manifest.json` with zero outputs and
`model_calls=0`. Refuse overwrite of non-identical artifacts. Before scientific
analysis, require:

- cached weight, tokenizer, model configuration, code, fixture, and manifest
  hashes match;
- all planned D/R pairs have equal token counts and identical token multisets;
- all answer tokens and source identities resolve exactly;
- every new discovery baseline reproduces the prior frozen prompt/input identity
  and complete float32 full-vocabulary-row SHA-256;
- an unhooked duplicate changes X/Y logits by at most `1e-4`;
- both self-patches change X/Y logits by at most `1e-4`;
- hook count is exactly one and no hook remains installed;
- final-block paired replacement reproduces the corresponding source X/Y logits
  within `1e-4`. This is an engineering positive control, not a scientific
  finding.

Failure yields `ENGINEERING_STOP_INVALID_PATCH_EXECUTION`. No threshold may be
relaxed.

### Stage B: outcome-exposed layer localization

On the original 16 discovery items only, run all eight intervention conditions at
the frozen block grid:

```text
[0, 4, 8, 12, 16, 20, 24, 27]
```

This is a coarse localization grid; no exact-layer claim is authorized. Layer 27
is an engineering-only source-copy control and is never selectable. Let `ell*` be
the earliest layer in `[0,4,8,12,16,20,24]` satisfying the transfer rule below.
Specificity does not participate in layer selection. If no selectable layer
passes transfer, stop with
`LOCALIZATION_STOP_NO_TRANSFER_LAYER` and make no held-out patch call.

If only layer 27 passes, stop with `LOCALIZATION_STOP_ONLY_FINAL_COPY_CONTROL`.
`ell*` denotes only the earliest tested passing layer at this grid resolution,
not an exact boundary or interval. The final block is not called a discovered
mechanistic site merely because it copies the source decision state.

### Stage C: prospective holdout behavioral admission

At the frozen holdout, first execute only unpatched D and R baselines. Continue to
patching only if all conditions hold across all 96 items:

- D native-X global-argmax and correct rate >= 0.95;
- R native-Y global-argmax and incorrect rate >= 0.75;
- no global-argmax ties;
- mean `{X,Y}` probability mass >= 0.95 separately for D and R;
- the 48-pair cluster bootstrap 95% lower bound for mean `G` is above zero;
- mean `G >= 1.0` logit, where `G_i = E_i(D)-E_i(R)`.

The primary analysis is intention-to-treat over all 96 frozen items. A secondary
baseline D-correct/R-incorrect cohort may be reported only because its identity
is frozen in the baseline-admission artifact before any holdout patch is run.

Failure yields `HOLDOUT_STOP_BEHAVIOR_NOT_REPLICATED`, makes zero held-out patch
calls, and cannot be interpreted as a negative internal-mechanism result.

### Stage D: holdout patching

Run the eight conditions at `ell*` only. Neither the holdout baseline nor any
patch result may change the layer, site, strength, items, controls, or thresholds.

## 7. Scores and estimands

For every output, store the raw X and Y logits, full-vocabulary global maximum,
ties, log-sum-exp, `{X,Y}` probability mass, and SHA-256 of the transient
little-endian float32 full-vocabulary row. Store every unpatched final-token
block activation in a finite little-endian float32 sidecar with row and logical
matrix hashes.

The persisted X/Y estimands and activation sidecars are independently
artifact-auditable. Full-vocabulary rows are not persisted: their maxima, ties,
log-sum-exp, label mass, and row digest are frozen-runner commitments and are
replay-verifiable from the locked model and inputs, but cannot be independently
recomputed from the result files alone. No stronger artifact-verifiability claim
is permitted.

Define the X-directed margin:

```text
E_i(c) = z_X(c) - z_Y(c)
G_i    = E_i(D) - E_i(R)
F_i    = E_i(R <- D_i) - E_i(R)
N_i    = E_i(D) - E_i(D <- R_i)
```

`F` is forward operational sufficiency and `N` is reverse damage. Reverse damage
is not called necessity. For each direction, define two separate specificity
contrasts:

```text
S_F,same = F_paired - F_same_pair_x
S_F,unrel = F_paired - F_unrelated_x
S_N,same = N_paired - N_same_pair
S_N,unrel = N_paired - N_unrelated
```

Aggregate the two items within pair first. Report means, medians, every cluster
value, leave-one-cluster-out means, positive-cluster counts, and percentile
cluster-bootstrap intervals. F, N, and same-pair controls use lexical-pair
clusters. Unrelated-control contrasts first aggregate reciprocal source/target
pairs into the frozen 24 dyads. Exact-length unrelated specificity uses its frozen
22-dyad subset. The holdout uses the same 10,000 resample index draws with seed
260802 for every metric. Discovery layer selection uses no p-value and contributes
no confirmatory interval.

Recovery fractions divide aggregate effects by aggregate `G`; item-wise ratios
are secondary because small item denominators can be unstable.
If an aggregate `G` denominator, or a leave-one-cluster-out `G` denominator
required by a rule, is not strictly positive, that recovery fraction is recorded
as undefined and the corresponding gate fails. The analyzer must still emit the
frozen stop or nonspecific status; an undefined diagnostic subset may not abort
analysis or alter transfer-only layer selection.

## 8. Frozen decision rules

### Discovery transfer layer

A grid layer passes transfer-only when all hold:

- mean `F/G >= 0.30` and mean `N/G >= 0.30`;
- at least 7/8 discovery pairs have positive pair-mean F;
- at least 7/8 discovery pairs have positive pair-mean N;
- every leave-one-pair-out F/G and N/G is >= 0.20;
- all Stage A engineering gates pass globally; they are a prerequisite rather
  than a per-layer scientific threshold.

### Held-out transfer

`DECISION_STATE_TRANSFER_REPLICATED` requires:

- F/G and N/G point estimates each >= 0.30;
- bootstrap 95% lower bounds for F and N are above zero;
- at least 36/48 pair means are positive in each direction;
- unique global argmax lies in `{X,Y}` for >= 0.95 of each paired-patch arm;
- mean `{X,Y}` mass decreases by no more than 0.02 from its matched baseline;
- identity and final-block engineering gates pass.

On the frozen secondary failure cohort, report forward correction and reverse
damage flip rates. A descriptive rescue criterion is point estimate >=0.25 with
bootstrap lower bound >0.10; it cannot replace a failed continuous-margin gate.
A forward flip means patched `X-Y > 0`; a reverse flip means patched `X-Y < 0`.
These are explicitly within-channel margin-sign flips, not global-argmax claims.
Both use 10,000 pair-cluster bootstrap draws with seed 260802.

### Held-out item specificity

`ITEM_SPECIFIC_DECISION_STATE_TRANSFER_REPLICATED` additionally requires all four
target-minus-control contrasts to be at least `0.20 G` and have bootstrap 95%
lower bounds above zero. Both same-pair contrasts must be positive in at least
24/31 frozen exact-length lexical pairs; both unrelated contrasts must be positive
in at least 17/22 frozen exact-length reciprocal dyads. Every component is an
intersection-union requirement; no averaging across controls, directions, or
dependency units is allowed. A transfer failure cannot coexist with a passing
scientific specificity status.

If transfer passes but specificity fails, report
`NONSPECIFIC_DECISION_STATE_TRANSFER_REPLICATED`. If transfer fails, report
`NO_REPLICATED_CAUSAL_TRANSFER_AT_FROZEN_SITE`. These statuses do not imply that
no relevant state exists elsewhere.

## 9. Prohibited adaptations

After the first new forward, do not:

- select a different prompt, candidate, item subset, lexical pair, model, layer
  grid, patch strength, token position, control, threshold, seed, or status rule;
- choose only native-correct source/recipient pairs for the primary analysis;
- use 0.5B behavior to alter 1.5B inference;
- interpret final-layer copying as localization by itself;
- treat a correct-X null rescue as item-specific binding;
- average failed controls, search additional spans, or run post-hoc layer rescue
  under this preregistration;
- use this experiment to reopen the biological fixture.

Any later span-aligned patching, directional occlusion, content-by-routing
factorial, other prompt, new answer tokens, or model-family replication requires a
new preregistration and new holdout.

## 10. Claim language

The maximum passing claim is:

> At a frozen post-block final-token site in Qwen2.5-1.5B under the exact
> system/user prompt, paired residual-state replacement bidirectionally transfers
> part of the line-order-dependent X/Y decision on a disjoint lexical holdout;
> within the preregistered exact-length subsets, paired sources are more effective
> than both frozen correct-X controls.

Without specificity, replace the final clause with “the transfer is consistent
with a nonspecific label/line-order decision state rather than item-specific
binding.” Forward
rescue alone is intervention sensitivity. Bidirectional full-state replacement
still does not prove necessity, a unique mediator, or a bottleneck; those require
separate selective occlusion and content/routing intervention designs.

## 한국어 요약

이 실험은 중단된 biological readout의 재개 단계가 아니라 별도의
비생물학적 인과-development 연구다. 기존 8 lexical pair는 layer localization에만
사용하고, 새 48 pair·96 item은 native behavior와 patch 결과를 모두 보지 않은
holdout으로 동결했다.

각 item에서 정답 X인 두 prompt를 만든다. D는 X가 첫 mapping line이고 선언 대상이
1행이며, R은 같은 두 line의 순서만 바꿔 Y가 먼저 나오고 선언 대상 X가 2행에
있다. D의 final-token residual을 R에 넣는 forward patch와 R을 D에 넣는 reverse
patch를 함께 검사한다. 같은 pair의 반대 선언 D와 고정 unrelated D도 모두 정답
X인 대조군이므로, 이들이 paired D와 똑같이 rescue하면 item-specific binding이
아니라 nonspecific label/line-order decision state로 판정한다.
Item specificity 판정은 사전에 동결한 exact-length subset(같은-pair 31개,
reciprocal unrelated dyad 22개)에만 적용하고, primary transfer는 48 pair 전체를
사용한다.

기존 16 item에서 `[0,4,8,12,16,20,24]` coarse layer만 selection에 사용하고,
27번 layer는 source-copy engineering control로만 둔다. 고정 규칙으로 선택한 가장
이른 nonfinal layer 하나만 96-item holdout에서 검정한다.
Holdout의 D/R 행동 격차가 먼저 재현되지 않으면 patch를 실행하지 않는다. 전체
gate가 통과해도 가능한 주장은 이 모델·prompt·site에서의 causal state transfer로
제한된다. 생물학, 잠재지식, activation gap, underactivation, 일반 variable-binding
법칙 또는 물리 법칙은 주장할 수 없으며 biological 실행도 계속 금지된다.
저장된 X/Y logit과 activation sidecar는 artifact-level 검증이 가능하지만, 저장하지
않는 full-vocabulary row의 global-max·mass 통계는 잠긴 모델과 입력으로 재실행해야
검증할 수 있다.
