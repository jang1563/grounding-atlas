# Coherent Readout V6A-R2: natural-token topology design

Registration state: `FROZEN_BEFORE_ANY_R2_MODEL_FORWARD`  
Date: 2026-08-03  
Mode: fresh 16-world, behavior-only main study with components-first admission  
Pinned model: `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`

This document is a new design. It does not edit, reopen, or overwrite the
frozen V6A design or either V6A qualification record. The fixture, builder,
runner, analyzer, tests, tokenizer contract, model-asset registry, execution
plan contract, and dependency-lock contract were completed and reviewed before
this state transition. Zero V6A-R2 model forwards occurred before the exact
registration state `FROZEN_BEFORE_ANY_R2_MODEL_FORWARD` was set. This is the
only execution-enabled state; every other state must fail closed. The one-shot
registered plan must bind the final source and artifact hashes before the first
model load.

Pre-execution Repair 1 is recorded in
`COHERENT_READOUT_V6A_R2_PRE_EXECUTION_REPAIR1.md`. The first zero-forward plan
emission was retired after mandatory analyzer replay exposed a false
prompt-versus-cell `prompt_id` comparison. It contained no attempt, execution,
record, raw-logit, or analysis artifact and issued zero model calls. Repair 1
changes only that analyzer invariant and its regression tests; it must preserve
the scientific registry, rendered prompts, stage hashes, and ordered seven-field
record identities exactly before any behavioral execution is authorized.

## 1. Relation to V2 and the reason for R2

The V2 component qualification is terminally sealed as
`V6A_QUALIFICATION_COMPONENT_FAIL`. Its registered bare-character continuation
analysis cannot be rescored into a PASS.

A posthoc engineering inspection of the already saved V2 logits found that,
when the answer candidates were represented by their natural leading-space
tokens, property retrieval was correct in `256/256` rows and lookup had 127
unique correct maxima plus one exact correct/distractor top tie; every maximum
was in the contextual answer pair. This is exploratory evidence about an
output-token mismatch, not confirmatory evidence and not a qualification
result. In particular, R2 does not adopt `127/128` as a target, count that tie
as correct, or tune a margin threshold to the posthoc result.

The descriptive source is
`contextual_token_posthoc_analysis.json`, SHA-256
`8b3ed9a43241286fb72e10edea0d27f2a2ead113c482422d5340ff51f6438ed8`.
The R2 dependency lock must bind this file, its zero-forward analyzer, and the
unaltered V2 terminal analysis separately; the posthoc artifact itself grants
no execution authorization.

V2 exposed the direct-retrieval and direct-lookup templates, including the
retrieval topology crossing. It contained zero two-fact composition calls and
zero single-TARGET composition calls. R2 therefore discloses direct-component
template exposure while keeping its composition endpoints, single-TARGET
controls, symbols, entities, keys, and prompt hashes fresh. R2 is a staged main
study, not another standalone model-qualification study.

## 2. Exact natural-continuation contract

The system message is exactly:

```text
Follow the user's instructions. The assistant message is prefilled with ANSWER:. Continue it with exactly one ASCII space followed by exactly one registered character from the valid output choices, and nothing else.
```

The assistant prefill is exactly `ANSWER:` with no trailing space. Rendering
uses:

```text
add_generation_prompt=false
continue_final_message=true
enable_thinking=false
```

The last attended response-site token is the colon in `ANSWER:`. Under the
pinned tokenizer it must be token ID `25`; the implementation preflight must
re-establish and hash this fact rather than assume it.

For every registered glyph `s`, define its sole answer surface and token by

\[
S_s=\texttt{U+0020}\,\Vert\,s,\qquad t_s=\operatorname{encode}(S_s).
\]

`S_s` is one ASCII space followed by exactly one Unicode glyph. Before any
model forward, tokenizer-only preflight must prove all of the following for
every glyph and every registered prompt context:

1. `S_s` is exactly one token and `decode([t_s]) == S_s` byte for byte;
2. appending `S_s` to the complete rendered assistant prefix extends its input
   IDs by exactly `[t_s]`, without retokenizing the prefix;
3. every occurrence of `s` in every user-prompt slot is tokenized as the same
   token ID `t_s`;
4. all 64 glyphs and their token IDs are distinct, every ID is absent from
   both the tokenizer's `all_special_ids` and its bound added/control-token
   registry (`added_tokens_decoder` or the pinned equivalent), and none is a
   special or control token;
5. topology mates have identical total token count, response-site index,
   response-site token ID, and attention-mask shape; and
6. no prompt is truncated.

Bare-glyph token variants are not registered answers. They may not replace,
be pooled with, or contribute probability mass to `t_s` in any accuracy,
margin, selection, or inferential calculation. Full-vocabulary logits remain
full vocabulary, so a bare variant can still make a registered strict-global
prediction incorrect.

## 3. Fresh worlds and exact symbol registry

R2 has eight discovery and eight confirmation worlds. Each world has two
globally unique property glyphs and two globally unique code glyphs. The exact
ordered registry is:

```text
àáâä åæçè éêíî óôöø úüýþ ŻЧАč đŚЯł КŁœś
şšżž ŞÜУО ÅНЕШ ЮіЦÎ ÈĐГт уфхц чшэя µÀÁÂ
```

Each consecutive group of four is assigned as
`property_0, property_1, code_0, code_1`. Groups 1--8 are discovery and groups
9--16 are confirmation. Within each split, odd world indices have `g=-1` and
even world indices have `g=+1`.

The registry contains 64 distinct glyphs. It retains 42 behaviorally unseen
glyphs from the unexecuted V6A main registry and replaces 22 glyphs that fail
the R2 one-natural-token structural contract. Replacement was based only on
the pinned tokenizer, not on model logits. None of the 32 V2 qualification
glyphs is reused.

The replacement provenance is exact and tokenizer-only. In original slot
order, the structurally invalid glyphs were:

```text
ă ą ć ę ı ń ő ơ ư ǎ ǐ ǒ ǔ ǝ ǥ ǧ ǫ ǯ ǵ ǹ ǻ ǽ
```

Their replacements in the same slot order are:

```text
Ż Ч А Ś Я К Ł Ş Ü У О Å Н Е Ш Ю і Ц Î È Đ Г
```

The eligible candidate universe was fixed to single NFC code points with a
Unicode letter category in `U+00C0..U+02AF` or `U+0370..U+052F`, using Unicode
database `15.1.0`. It excluded ASCII `A`--`Z`, every V2--V5 answer glyph, all
32 executed V6A-qualification glyphs, and all 64 original V6A main glyphs. A
candidate was eligible only if `" " + glyph` encoded as one non-special token
and decoded exactly. The 45 eligible candidates were ordered by the bytewise
SHA-256 of

```text
V6A-R2-main-replacement-v1|{UTF-8 glyph}
```

with glyph code point as the final tie-break; the first 22 were assigned to
the invalid slots above. This rule, its resolved list, and all token IDs must
be reproduced independently before plan freeze. It performs no model forward
and reads no V2 logit or accuracy value.

World identities are fixed as:

```text
world_id      = v6a_r2_{discovery|confirmation}_world_{role_index:03d}
target_entity = v6a_r2_{role}_entity_t_{role_index:03d}
other_entity  = v6a_r2_{role}_entity_o_{role_index:03d}
instance_key  = V6A-R2-{D|C}-W{role_index:03d}-N{(global_index*6151)%10000:04d}
```

Here `global_index` is 1--8 for discovery and 9--16 for confirmation. The
zero-forward implementation preflight must reverse-check every R2 glyph,
token ID, entity, key, user-prompt hash, and rendered-chat hash against V2--V5,
both terminal V6A qualification attempts, and every other R2 world. The
builder owns static-field checks; the runner owns tokenizer- and rendered-chat
checks. Discovery and confirmation must be token-, entity-, key-, and
prompt-disjoint.

## 4. Frozen prompt topology and literal blocks

R2 preserves the original V6A topology question:

> Holding wording, semantics, answer, content factors, and the colon response
> site fixed, do TASK placement and valid-answer-set placement separately or
> jointly change the sign of TARGET fact-order sensitivity?

For composition, define four unchanged blocks:

- `H`: instance header and the two codebook rules;
- `A`: valid-output line;
- `Q`: task line; and
- `F`: labeled-facts header and fact line or lines.

The fully crossed placement topology is:

| `q` | `a` | Block order | Historical analogue |
|---:|---:|---|---|
| -1 | -1 | `H -> A -> Q -> F` | V5-like |
| -1 | +1 | `H -> Q -> F -> A` | mixed |
| +1 | -1 | `H -> A -> F -> Q` | mixed |
| +1 | +1 | `H -> F -> Q -> A` | V4-like |

Within the two-fact `F` block, `o=-1/+1` places TARGET first/second and swaps
only the two fact lines. The semantic answer is invariant within each topology
octet.

The composition header is exactly:

```text
INSTANCE KEY: {instance_key}.
CODEBOOK RULES
RULE: {property} maps to {code}.
RULE: {property} maps to {code}.
```

The answer block is exactly:

```text
VALID OUTPUTS (display order carries no meaning): {code}, {code}.
```

The composition task block is exactly:

```text
TASK: Read the one-character property in TARGET FACT, find its exact CODEBOOK RULE, and return only that rule's right-side character.
```

The two-fact block is exactly:

```text
LABELED FACTS
TARGET FACT: {target_entity} has property {target_property}.
OTHER FACT: {other_entity} has property {other_property}.
```

The single-TARGET control uses the same `H`, `A`, and `Q`, with:

```text
LABELED FACTS
TARGET FACT: {target_entity} has property {target_property}.
```

Property retrieval uses the instance header and the same `(q,a)` placements:

```text
INSTANCE KEY: {instance_key}.
```

```text
VALID OUTPUTS (display order carries no meaning): {property}, {property}.
```

```text
TASK: Copy the one-character property stated in TARGET FACT; return only that character.
```

```text
LABELED FACTS
TARGET FACT: {target_entity} has property {target_property}.
OTHER FACT: {other_entity} has property {other_property}.
```

Codebook lookup is not topology-moved and is exactly:

```text
INSTANCE KEY: {instance_key}.
CODEBOOK RULES
RULE: {property} maps to {code}.
RULE: {property} maps to {code}.
VALID OUTPUTS (display order carries no meaning): {code}, {code}.
GIVEN PROPERTY: {target_property}.
TASK: Find the rule whose left-side property exactly matches GIVEN PROPERTY; return only that rule's right-side character.
```

No literal text, punctuation, capitalization, block placement, or chat flag may
change after the implementation registration is frozen.

## 5. Content design and calls

Raw content/display factors are:

- `p=-1/+1`: TARGET property 0/1;
- `m=-1/+1`: identity/swapped property-to-code mapping;
- `r=-1/+1`: property-0/property-1 rule first;
- `v=-1/+1`: code-0/code-1 valid option first; and
- `o=-1/+1`: TARGET fact first/second.

Task-relative coordinates remain

\[
u=-pr,
\]

where `u=-1/+1` means the TARGET rule is first/last, and

\[
w=pmv,
\]

where `w=-1/+1` means the correct option is first/last. The native correct-code
sign remains `-pm`.

Two-fact and single-TARGET composition use the resolution-IV half-fraction

\[
pmuw=g.
\]

For each world, free signs `(p,m,u)` enumerate all eight combinations and

\[
w=gpmu,\qquad r=-pu,\qquad v=pmw=gu.
\]

The topology factors are fully crossed. Lookup is expanded from the earlier
fraction-matched eight rows to the complete `p*m*r*v` factorial of 16 rows.

| Family | Construction | Calls/world | Discovery | Confirmation | Total |
|---|---|---:|---:|---:|---:|
| two-fact composition | 8 content profiles x `(o,q,a)` | 64 | 512 | 512 | 1,024 |
| property retrieval | `p*v*o*q*a` | 32 | 256 | 256 | 512 |
| codebook lookup | complete `p*m*r*v` | 16 | 128 | 128 | 256 |
| single-TARGET composition | 8 content profiles x `(q,a)` | 32 | 256 | 256 | 512 |
| **total** | | **144** | **1,152** | **1,152** | **2,304** |

No row may be selected or removed by correctness, margin, glyph identity,
topology effect, or agreement with V2--V5.

## 6. Components-first execution firewall

The complete 2,304-call plan, record identities, raw-logit shard allocation,
stage order, and all implementation dependencies must be hash-frozen before
the first R2 model forward.

The exact execution revision is
`coherent-readout-v6a-r2-natural-token-topology-exec-v1`. For every planned
call, define a pre-forward record identity from exactly this canonical JSON
object:

```text
{
  "block_call_index": ...,
  "cell_id": ...,
  "execution_block": ...,
  "execution_stage": ...,
  "global_call_index": ...,
  "prompt_id": ...,
  "stage_call_index": ...
}
```

`record_identity_id` is the lowercase SHA-256 of that canonical object. It is
stored in the frozen plan and echoed unchanged in the execution record. It
does not include `call_plan_sha256`, avoiding self-reference. The separate
outcome-dependent `record_id` is computed only after the raw row and permitted
record fields exist.

Execution then has exactly two behavioral stages:

1. `discovery-components`: execute only discovery retrieval, full lookup, and
   single-TARGET rows: `8 * (32 + 16 + 32) = 640` calls. The admission analyzer
   may read only these 640 records and their raw logits. It may not read,
   estimate, or report a two-fact composition or topology outcome.
2. `remaining-main`: only after discovery components pass, execute all
   remaining 1,664 calls in one sealed phase and in the frozen order:
   512 discovery two-fact topology calls, 640 confirmation component calls,
   then 512 confirmation two-fact topology calls. There is no interim analyzer,
   effect-based stop, glyph replacement, model replacement, retry, or prompt
   repair between these blocks.

The remaining-stage outputs must nevertheless be physically block-separated:

- discovery-topology: raw-logit shards `0--7` and its own record stream;
- confirmation-components: raw-logit shards `8--17` and its own record stream;
- confirmation-topology: raw-logit shards `18--25` and its own record stream.

Before the confirmation-component gate, the final analyzer may enumerate the
three block paths and verify opaque file metadata such as name, byte size, and
whole-file SHA-256. It must deserialize and semantically validate only the
confirmation-component record stream and shards `8--17`. It must not open or
parse either topology record stream, load topology shards `0--7` or `18--25`,
materialize topology row hashes, or compute or read any topology diagnostic.
Topology execution records contain frozen identities and raw-logit bindings
only; the executor must not store derived topology diagnostics. The analyzer
must reject any pre-gate topology diagnostic field rather than consuming it.

After all no-forward environment, dependency, MPS-kernel, memory, and disk
preflights pass, each behavioral stage must write its immutable attempt
artifact before model loading. The exact attempt statuses are
`V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_ATTEMPT_STARTED_IMMUTABLE` and
`V6A_R2_REMAINING_MAIN_EXECUTION_ATTEMPT_STARTED_IMMUTABLE`. Any interruption
after that write permanently invalidates the stage: retry, overwrite, partial
resume, and deletion-based recovery are not part of the protocol.

Model loading delegates to the hash-bound sealed V2 MPS loader. Every call is
one `torch.inference_mode()` forward with the registered input IDs and mask,
`use_cache=false`, `logits_to_keep=1`, and `return_dict=true`. The retained row
is exactly `output.logits[0,-1,:]` converted to finite little-endian float32
with shape `(152064,)`. Component records may store diagnostics that the
analyzer independently reconstructs. Topology records may not store margins,
predictions, correctness, maxima, probabilities, or any other derived outcome.

The remaining-stage attempt must hash-bind both the completed discovery
execution manifest and the exact passing discovery authorization. Before model
loading, the runner must replay the discovery artifacts and analyzer read-only
and require exact equality with that authorization. No analyzer is invoked
between the three remaining execution blocks.

If discovery components fail, R2 stops permanently after 640 calls. If they
pass, confirmation runs regardless of the sign or magnitude of the discovery
topology outcome. An engineering interruption makes the affected stage
invalid; partial behavioral results do not authorize resume or inference.

The final analyzer must validate and gate confirmation components before it
deserializes, reads, or interprets topology outcomes from either split. On a
confirmation-component failure it must return without invoking either topology
loader. The failure suppresses topology inference from both splits even though
the sealed remaining stage has already completed.

## 7. Component metrics and prospective fixed-panel budgets

For a row with correct natural token `t_c` and distractor natural token `t_d`,
define:

- **pairwise natural-token correctness:**
  `logit(t_c) > logit(t_d)`; and
- **strict unique-global correctness:** `t_c` is the sole argmax over the full
  vocabulary.

An exact pairwise or global tie is incorrect. There is no tolerance, secondary
tie-break, or repeat-forward adjudication.

The following gates are applied independently and unchanged to discovery and
confirmation components. Both pairwise and strict unique-global accuracy must
meet every overall and per-world line below.

| Family | Overall | Per world |
|---|---:|---:|
| property retrieval | at least 251/256 | at least 29/32 |
| single-TARGET composition | at least 251/256 | at least 29/32 |
| full-factorial lookup | at least 126/128 | at least 15/16 |

Additional pairwise natural-token gates are:

- every retrieval property label: at least `15/16`;
- every single-TARGET code label: at least `15/16`;
- every lookup code label: at least `7/8`;
- every registered retrieval `p`, `v`, `o`, `q`, and `a` level:
  at least `116/128`;
- every registered single-TARGET content/display and topology level
  (`p`, `m`, `r`, `v`, `u`, `w`, `q`, and `a`): at least `116/128`;
- every registered lookup `p`, `m`, `r`, and `v` level: at least `58/64`;
- every retrieval `q*a` group: at least `58/64`;
- every retrieval `o*q*a` group: at least `29/32`; and
- every single-TARGET `q*a` group: at least `58/64`.

The overall `0.98` target is retained prospectively from V2 and translated by
ceiling to `251/256` and `126/128`. The `0.90` marginal targets are likewise
translated exactly to `116/128`, `58/64`, and `29/32`. The registered `7/8`
lookup-label budget is explicit: it allows one failure in an eight-row label
panel rather than silently converting a nominal `0.90` threshold into `8/8`.
It is not chosen to accept the V2 tied row: that row is outside the fresh R2
panel and an equivalent R2 tie would count as an ordinary incorrect row.

These are deterministic error budgets on the complete registered fixed panel,
not binomial estimates, iid samples, confidence intervals, or population
claims. There is deliberately no universal single-row veto for a tie, a
global maximum outside the answer pair, or a non-correct global maximum. Such
rows are ordinary failures in the relevant pairwise and/or strict metric, and
the frozen aggregate, world, label, factor, and scaffold budgets decide
admission.

## 8. Engineering and firewall gates

In addition to the behavioral budgets, admission requires every engineering
gate to pass exactly:

1. fixture and call plan independently rebuild byte-for-byte;
2. model ID, pinned revision, tokenizer, chat template, model assets, dtype,
   device, attention implementation, and loader policy match their frozen
   registries;
3. every natural-token, prompt-occurrence, prefix-extension, colon-site,
   topology-shape, no-truncation, and answer-ledger check passes;
4. all 2,304 cell IDs, prompt IDs, user-prompt hashes, rendered-chat hashes,
   and record IDs are unique and match the plan;
5. discovery/confirmation and historical symbol, token, entity, key, and
   prompt firewalls pass;
6. stage call counts and order are exactly `640` then `1,664`, with zero model
   calls before the registered first call;
7. every retained full-vocabulary logit row, shard, record, manifest, and
   dependency file replays and hash-validates; and
8. generation is unused: each call is one deterministic teacher-forced prompt
   forward scored only at the registered colon response site.

Attempt artifacts, execution manifests, block bindings, and records use closed
schemas: unknown keys are engineering-invalid. Attempts bind the execution
revision, plan/stage hashes, global and stage ranges, block order/counts,
preflight, model/loader policy, and the hashes of the plan manifest, design,
dependency lock, token receipt, fixture, runner, and analyzer. Execution
manifests additionally bind the attempt, ordered record stream(s), ordered
raw-logit shards, phase and cumulative call counts, and the same frozen source
artifacts. Production planning accepts no fixture, dependency, tokenizer,
path, or integration-test override and can write only once to the exact
registered result root.

Behavioral underperformance is a component failure, not an engineering
invalidity. An engineering invalidity cannot be converted into a component
result.

## 9. Registered topology estimands

For each two-fact row, let

\[
M=\operatorname{logit}(t_{\mathrm{correct}})
 -\operatorname{logit}(t_{\mathrm{distractor}}).
\]

For world `z`, content profile `c`, and topology `(q,a)`, define the matched
TARGET-second advantage

\[
D_z(q,a)=\frac18\sum_c
 [M_{z,c,o=+1,q,a}-M_{z,c,o=-1,q,a}].
\]

The primary endpoint is

\[
R_z=D_z(-1,-1)-D_z(+1,+1).
\]

The registered sign-reversal hypothesis requires both

\[
D(-1,-1)>0
\]

and

\[
D(+1,+1)<0.
\]

TASK- and answer-set-placement contrasts remain

\[
Q_z=\frac12[D_z(-1,-1)+D_z(-1,+1)]
 -\frac12[D_z(+1,-1)+D_z(+1,+1)],
\]

\[
A_z=\frac12[D_z(-1,-1)+D_z(+1,-1)]
 -\frac12[D_z(-1,+1)+D_z(+1,+1)].
\]

The residual joint placement interaction is reported from the complete
`o*q*a` factorial as the `q*a` factorial coefficient of the matched `o`
effect:

\[
J_z=\frac14[D_z(-1,-1)-D_z(-1,+1)-D_z(+1,-1)+D_z(+1,+1)].
\]

Secondary estimands compare two-fact and single-TARGET composition at matched
content and `(q,a)` levels.

## 10. Discovery, confirmation, and inference

World is the inferential and resampling unit. Use exactly 10,000 percentile
bootstrap resamples with seed `260806`. These are fixed-panel stability bounds,
not population confidence intervals over glyphs, prompts, models, or biological
systems.

Only after both splits pass their component gates may topology be interpreted.
The maximum topology result requires, separately in discovery and confirmation:

- `D(-1,-1)` bootstrap lower bound above zero and positive in at least 6/8
  worlds;
- `D(+1,+1)` bootstrap upper bound below zero and negative in at least 6/8
  worlds;
- `R` bootstrap lower bound above zero and positive in at least 6/8 worlds;
- V5-like second-minus-first accuracy at least `0.10`; and
- V4-like first-minus-second accuracy at least `0.10`.

These two topology accuracy endpoints use pairwise natural-token correctness
(`logit(t_c) > logit(t_d)`). Strict unique-global correctness is an output-
channel component-admission guard, not a replacement topology estimand.

`Q`, `A`, and the residual joint interaction are always reported after
component admission with their fixed-panel intervals. No dominant placement
block is selected unless its registered contrast has the same sign and a
nonzero interval in both discovery and confirmation.

## 11. Status hierarchy

Use the first applicable terminal status:

1. `V6A_R2_ENGINEERING_INVALID`;
2. `V6A_R2_DISCOVERY_COMPONENT_FAIL`;
3. `V6A_R2_CONFIRMATION_COMPONENT_FAIL`;
4. `V6A_R2_NO_REPLICATED_TOPOLOGY_EFFECT`;
5. `V6A_R2_ORDER_EFFECT_WITHOUT_SIGN_REVERSAL`;
6. `V6A_R2_SCAFFOLD_SENSITIVE_ORDER_REVERSAL_SUPPORTED`.

Implementation registration is complete in the exact state
`FROZEN_BEFORE_ANY_R2_MODEL_FORWARD`, set before any R2 model forward. A
discovery-component PASS is an intermediate execution authorization for the
sealed 1,664-call remaining stage, not a scientific result.

## 12. Claim boundary and activation-stage authorization

The component stages can support only this claim:

> At the registered natural one-token continuation channel, the pinned
> Qwen2.5-7B-Instruct revision met the prospective direct-retrieval,
> full-factorial-lookup, and single-TARGET fixed-panel error budgets for the
> relevant split.

They cannot establish or rescue the V2 bare-token qualification result.

Maximum V6A-R2 success supports only:

> In the pinned Qwen2.5-7B-Instruct revision and this registered synthetic
> prompt family, moving fixed TASK and valid-answer blocks around a fixed fact
> block causally changes the sign of TARGET-order sensitivity, with replicated
> factorial contrasts identifying the contribution of each block placement.

It does not establish an activation gap, latent knowledge, biology, a physical
law, a universal recency mechanism, or model-family generality.

Only replicated V6A-R2 topology support authorizes the design—not execution—of
a separate V6B activation experiment. V6B must use another token-disjoint
fit/localization/holdout bank, freeze the colon-site causal intervention before
any V6B forward, include answer, null, and generic-cue shams, and make no use
of V5 localization/holdout artifacts.

## 한국어 요약

V6A-R2는 V2 qualification을 다시 채점해 살리는 절차가 아니다. V2는
`V6A_QUALIFICATION_COMPONENT_FAIL`로 종료되며, leading-space token을 사용한
사후 결과는 출력 token 불일치를 시사하는 탐색적 engineering 근거일 뿐이다.

R2의 assistant prefill은 공백 없는 `ANSWER:`이고, 정답 표면은 정확히
`ASCII 공백 1개 + 등록 glyph 1개`이다. 이 전체가 하나의 natural token이어야
하며, prompt 안의 같은 glyph도 반드시 같은 token ID를 가져야 한다. Bare
glyph token을 대신 사용하거나 두 variant의 확률을 합치는 것은 금지한다.

Discovery 8개와 confirmation 8개의 fresh world를 사용한다. World당 two-fact
composition 64개, retrieval 32개, full-factorial lookup 16개, single-TARGET
32개로 총 144 calls이며, 전체는 2,304 calls이다. 먼저 discovery component
640 calls만 실행한다. 통과하면 discovery topology 512개, confirmation
component 640개, confirmation topology 512개를 합친 나머지 1,664 calls를
중간 효과 판독 없이 모두 실행한다. Confirmation component가 통과하기 전에는
세 block의 출력은 물리적으로 분리되며, 어느 split의 topology record나 logit도
역직렬화하거나 읽거나 해석하지 않는다. 이 gate가 실패하면 topology loader를
호출하지 않고 즉시 종료한다.

각 call의 pre-forward `record_identity_id`와 exact execution revision을 plan에
고정한다. 모든 preflight가 통과한 뒤 model load 전에 immutable attempt를 먼저
기록하며, 중단 시 retry/resume은 금지한다. Component record만 재구성 가능한
diagnostic을 저장하고 topology record는 raw-logit binding 외 결과 파생값을
저장하지 않는다. Remaining stage 시작 전 discovery 실행과 PASS authorization을
read-only로 완전히 재생하며, 세 remaining block 사이에는 analyzer를 호출하지
않는다.

Component gate는 pairwise natural-token 정확도와 strict unique-global-correct
정확도를 분리해 고정된 error budget으로 평가한다. Tie와 answer-pair 밖
global maximum은 해당 metric의 오답이지만 단일 row만으로 전체 실험을
중단시키는 별도 veto는 없다. `7/8` lookup-label 기준도 사전에 명시된
fixed-panel budget이며 iid 표본의 신뢰구간이 아니다.

현재 문서와 구현은 R2 model forward가 한 번도 수행되기 전에 정확히
`FROZEN_BEFORE_ANY_R2_MODEL_FORWARD` 상태로 동결되었다. 최대 성공도
synthetic prompt family 안의 행동적 topology 효과만 지지한다.
Activation gap, 잠재지식, biology, 물리 법칙은 증명하지 않으며, activation
단계 V6B의 설계는 discovery와 confirmation에서 topology 효과가 재현된 뒤에만
허용된다.
