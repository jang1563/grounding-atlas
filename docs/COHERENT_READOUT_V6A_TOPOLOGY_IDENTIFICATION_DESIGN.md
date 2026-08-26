# Coherent Readout V6A: prompt-topology identification design

Design state: `FROZEN_BEFORE_ANY_QUALIFICATION_OR_V6A_MODEL_FORWARD`  
Date: 2026-08-03  
Mode: synthetic, behavior-only, prospective bridge to a separate V6B causal study  
Provisional primary model: `Qwen/Qwen2.5-7B-Instruct`

## 1. Why V6A is separate from V6B

V4 and V5 produced opposite TARGET-order effects, but they changed several
prompt properties together. Most importantly, V4 placed `TASK` and the valid
answer set after the facts, whereas V5 placed both before the facts. V5 then
ended immediately on the fact block and showed strong last-rule, last-option,
and last-fact preferences.

V5 stopped before activation use with
`V5_FIT_COMPONENT_ADMISSION_FAIL`. Reversing its registered direction or using
its unopened localization/holdout worlds would be invalid. V6A therefore asks
a fresh behavioral identification question:

> Holding wording, semantics, answer, content factors, and the response site
> fixed, do TASK placement and valid-answer-set placement separately or jointly
> change the sign of TARGET fact-order sensitivity?

V6A contains no activation patch and cannot support an activation-gap claim.
Its purpose is to identify a stable, prospectively confirmed behavioral
contrast before V6B freezes a response-site causal estimand in another fresh
bank.

## 2. Component-only model qualification firewall

Model qualification is disposable and precedes the confirmatory V6A bank.

- Eight qualification worlds use entities, instance keys, property symbols,
  code symbols, and prompt hashes that never appear in V6A.
- Qualification contains direct property retrieval and direct two-rule
  codebook lookup only.
- It contains zero two-fact or single-TARGET composition calls. Consequently,
  qualification cannot observe the composition order gap, composition topology
  interaction, or any composition outcome. It can observe direct-retrieval
  scaffold accuracy, which is handled by the prospective firewall below.
- The candidate model is fixed prospectively to
  `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`. The move from 1.5B is justified
  only by V5's failed direct-component gate, not by its gap magnitude or sign.
- The selected prompt wording is frozen before qualification and may not be
  repaired after observing a model output.

Qualification uses 32 topology-crossed retrieval prompts and 16 complete
factorial lookup prompts per world: `8 * 48 = 384` calls. It passes only if:

1. retrieval and lookup accuracy are each at least `0.98` overall;
2. each is at least `0.90` in every world, answer label, and registered marginal
   factor level;
3. retrieval is at least `0.90` in every registered `q*a` and `o*q*a` group;
4. every row has a unique full-vocabulary argmax inside its registered label
   pair;
5. every answer is exactly one contextual continuation token;
6. all topology mates render without truncation and with the registered fixed
   assistant response site; and
7. no prompt, entity, property symbol, or code symbol overlaps V2--V5 or V6A.

Engineering-valid underperformance is
`V6A_QUALIFICATION_COMPONENT_FAIL`; invalid execution/replay is
`V6A_QUALIFICATION_ENGINEERING_INVALID`. Neither authorizes a V6A model
forward.

Because qualification retrieval crosses `(o,q,a)`, its raw outputs could reveal
direct-copy scaffold sensitivity, although they cannot reveal the composition
endpoint below. This entire V6A design—including the literal block text in
Section 4, symbol allocation, fraction, estimands, gates, and status hierarchy—is
therefore frozen and included in the qualification dependency hash before any
qualification forward. The qualification analyzer may report only registered
accuracy gates, not retrieval-margin topology contrasts.

## 3. Fresh confirmatory worlds

V6A freezes 16 prompt-unique, token-disjoint worlds:

| Role | Worlds | Foldover `g` | Calls/world | Calls |
|---|---:|---:|---:|---:|
| discovery | 8 | 4 negative / 4 positive | 136 | 1,088 |
| confirmation | 8 | 4 negative / 4 positive | 136 | 1,088 |
| total | 16 | 8 / 8 | | **2,176** |

Every world has two globally unique property characters and two globally unique
code characters. The resulting 64 characters are absent from V2--V5,
qualification, every other V6A world, and the opposite property/code role.
Discovery and confirmation are therefore token-disjoint, not merely
combination-disjoint.

The ordered symbol registry is fixed to:

```text
àáâä åæçè éêíî óôöø úüýþ ăąćč đęıł ńőœś
şšżž ơưǎǐ ǒǔǝǥ ǧǫǯǵ ǹǻǽт уфхц чшэя µÀÁÂ
```

Each consecutive group of four is assigned as
`property_0, property_1, code_0, code_1`. Groups 1--8 are discovery and groups
9--16 are confirmation. Within each role, odd world indices have `g=-1` and
even indices have `g=+1`. World IDs, entities, and keys are deterministic:

```text
world_id      = v6a_{role}_world_{role_index:03d}
target_entity = v6a_{role}_entity_t_{role_index:03d}
other_entity  = v6a_{role}_entity_o_{role_index:03d}
instance_key  = V6A-{D|C}-W{role_index:03d}-N{(global_index*6151)%10000:04d}
```

Here `global_index` is 1--8 for discovery and 9--16 for confirmation. The future
builder must reverse-check these symbols, entities, keys, and prompt hashes
against the complete qualification registry before it can write a V6A fixture.

Tokenizer-only preflight must prove that every registered character is one
token in every prompt occurrence and as an assistant continuation. Entity and
instance strings are also unique. All rendered user prompts and complete chat
inputs must have unique hashes.

Confirmation artifacts are generated and hash-locked before discovery model
execution, but the analyzer may not read confirmation outcomes until discovery
component admission issues a one-shot entry authorization. Confirmation runs
regardless of the observed discovery topology effect once components are
admitted; effect-based optional stopping is prohibited.

## 4. Response topology

Every example uses a final assistant prefill `ANSWER:`. Rendering must use
`add_generation_prompt=false`, `continue_final_message=true`, and
`enable_thinking=false`. The final attended token is the colon in the frozen
assistant prefill, and the next token is the scored one-character answer.

Tokenizer preflight must prove that every topology mate has the same total token
count, final response-site index, response-site token ID, and attention-mask
shape. This controls response-site absolute position while moving only frozen
text blocks.

For composition, define four blocks:

- `H`: instance header and two codebook rules;
- `A`: the unchanged valid-answer-set line;
- `Q`: the unchanged task line; and
- `F`: the labeled-facts header and fact lines.

Two placement factors are crossed:

- `q=-1/+1`: `Q` before/after `F`;
- `a=-1/+1`: `A` before/after `F`.

The exact topology is:

| `q` | `a` | Block order | Historical analogue |
|---:|---:|---|---|
| -1 | -1 | `H -> A -> Q -> F` | V5-like |
| -1 | +1 | `H -> Q -> F -> A` | mixed |
| +1 | -1 | `H -> A -> F -> Q` | mixed |
| +1 | +1 | `H -> F -> Q -> A` | V4-like |

The wording and punctuation of every block are identical across placements.
Within `F`, `o=-1/+1` places TARGET first/second and swaps only the two fact
lines. The semantic answer never changes within a topology octet.

### 4.1 Literal frozen prompt contract

The system message is exactly:

```text
Follow the user's instructions. Your entire response must be exactly one registered character from the valid output choices. Do not write any other text.
```

The assistant prefill is exactly `ANSWER:`. User prompts contain only the
following literal line templates, joined by a single newline with no blank
lines. Braces denote deterministic fixture substitutions, not optional text.

Shared composition header `H`:

```text
INSTANCE KEY: {instance_key}.
CODEBOOK RULES
RULE: {property} maps to {code}.
RULE: {property} maps to {code}.
```

The two RULE lines are ordered by `r`. Identity mapping maps property 0/1 to
code 0/1; swapped mapping maps property 0/1 to code 1/0.

Composition answer block `A`:

```text
VALID OUTPUTS (display order carries no meaning): {code}, {code}.
```

Composition task block `Q`:

```text
TASK: Read the one-character property in TARGET FACT, find its exact CODEBOOK RULE, and return only that rule's right-side character.
```

Two-fact block `F`:

```text
LABELED FACTS
TARGET FACT: {target_entity} has property {target_property}.
OTHER FACT: {other_entity} has property {other_property}.
```

The last two lines are swapped only by `o`. The single-TARGET control uses the
same `H`, `A`, and `Q`, while `F` is exactly:

```text
LABELED FACTS
TARGET FACT: {target_entity} has property {target_property}.
```

Property retrieval uses header `H_r`:

```text
INSTANCE KEY: {instance_key}.
```

Its answer, task, and two-fact blocks are exactly:

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

`H_r`, the retrieval answer/task blocks, and the retrieval fact block follow the
same four `(q,a)` block orders. The two fact lines are swapped only by `o`.

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

No punctuation, capitalization, header, or task wording may change after
qualification begins. The builder must instantiate this literal contract and
the runner must reconstruct it independently before V6A execution.

## 5. Content and task-relative coordinates

Raw content/display signs are:

- `p=-1/+1`: TARGET property 0/1;
- `m=-1/+1`: identity/swapped property-to-code mapping;
- `r=-1/+1`: property-0/property-1 rule first;
- `v=-1/+1`: code-0/code-1 valid option first; and
- `o=-1/+1`: TARGET fact first/second.

V6A additionally records the task-relative coordinates discovered in V5:

\[
u=-pr,
\]

where `u=-1/+1` means the TARGET rule is first/last, and

\[
w=pmv,
\]

where `w=-1/+1` means the correct option is first/last. The native correct-code
sign is `-pm`.

## 6. Resolution-IV composition fraction

The topology factors `(o,q,a)` are fully crossed. The content coordinates
`(p,m,u,w)` use the regular half-fraction

\[
pmuw=g,\qquad g\in\{-1,+1\}.
\]

For every world, free signs `(p,m,u)` enumerate all eight combinations and

\[
w=gpmu,\qquad r=-pu,\qquad v=pmw=gu.
\]

Thus each world contains `8 content profiles * 8 topologies = 64` exact
two-fact composition prompts. The fraction has resolution IV: content main
effects are aliased only with three-factor interactions. Balancing `g` across
worlds supplies a role-level foldover, while all primary topology estimands
remain exact within-world matched contrasts.

## 7. Direct components and attribution control

Each world contains:

| Family | Construction | Calls/world |
|---|---|---:|
| two-fact composition | 8 content profiles x `(o,q,a)` | 64 |
| property retrieval | `p`, property-option order, `(o,q,a)` | 32 |
| codebook lookup | 8 fraction-matched content profiles | 8 |
| single-TARGET composition | 8 content profiles x `(q,a)` | 32 |
| total | | **136** |

Property retrieval moves the same `Q` and `A` blocks around the two facts and
therefore tests whether topology alone breaks direct copying. Lookup tests every
content profile used by the composition fraction. Single-TARGET composition
removes the competing OTHER fact; its high performance is required to
distinguish competitor/context sensitivity from generic codebook chaining
failure.

No item is selected by correctness, margin, symbol identity, topology effect,
or agreement with V4/V5.

## 8. Registered estimands

Let `M` be the correct-answer logit minus the other registered-answer logit.
For world `z`, content profile `c`, and topology `(q,a)`, define the matched
TARGET-second advantage

\[
D_z(q,a)=\frac18\sum_c
  [M_{z,c,o=+1,q,a}-M_{z,c,o=-1,q,a}].
\]

The primary endpoint contrast is

\[
R_z=D_z(-1,-1)-D_z(+1,+1).
\]

Positive `R` means moving both blocks after the facts shifts the order effect
from TARGET-second toward TARGET-first. The endpoint sign-reversal hypothesis
requires both

\[
D(-1,-1)>0
\]

and

\[
D(+1,+1)<0.
\]

TASK- and answer-set-specific interactions are

\[
Q_z=\frac12[D_z(-1,-1)+D_z(-1,+1)]
    -\frac12[D_z(+1,-1)+D_z(+1,+1)],
\]

\[
A_z=\frac12[D_z(-1,-1)+D_z(+1,-1)]
    -\frac12[D_z(-1,+1)+D_z(+1,+1)].
\]

The residual joint placement interaction is reported from the complete
`o*q*a` factorial. `Q` and `A` identify separate text-block placement effects;
neither is inferred by comparing V4 and V5 across studies.

Secondary estimands compare two-fact and single-TARGET composition at the same
content and `(q,a)` levels. These separate competitor-dependent degradation
from topology-dependent generic chaining.

## 9. Inference and gates

World is the inferential and resampling unit. Use exactly 10,000 percentile
bootstrap resamples with seed `260806`. These are fixed-panel stability bounds,
not population confidence intervals over symbols, prompts, or models.

Discovery component admission requires:

1. retrieval, lookup, and single-TARGET composition each at least `0.95`
   overall;
2. each family at least `0.90` in every world and every registered marginal
   factor/topology level;
3. all eight fraction-matched lookup rows correct in every world;
4. no answer tie and a unique global argmax inside the valid pair for every
   row; and
5. all hashes, token identities, topology octets, answer ledgers, call counts,
   and response-site equality checks valid.

Component admission authorizes confirmation independent of the discovery
effect sign. Confirmation applies the same component gates unchanged.

The maximum topology result additionally requires, separately in discovery and
confirmation:

- `D(-1,-1)` bootstrap lower bound above zero and positive in at least 6/8
  worlds;
- `D(+1,+1)` bootstrap upper bound below zero and negative in at least 6/8
  worlds;
- `R` bootstrap lower bound above zero and positive in at least 6/8 worlds;
- V5-like second-minus-first accuracy at least `0.10`; and
- V4-like first-minus-second accuracy at least `0.10`.

`Q`, `A`, and the joint interaction are always reported with fixed-panel
intervals. No dominant block is selected unless its registered contrast
replicates with the same sign and a nonzero interval in confirmation.

## 10. Status hierarchy

Use the first applicable status:

1. `V6A_QUALIFICATION_ENGINEERING_INVALID`;
2. `V6A_QUALIFICATION_COMPONENT_FAIL`;
3. `V6A_ENGINEERING_INVALID`;
4. `V6A_DISCOVERY_COMPONENT_FAIL`;
5. `V6A_CONFIRMATION_COMPONENT_FAIL`;
6. `V6A_NO_REPLICATED_TOPOLOGY_EFFECT`;
7. `V6A_ORDER_EFFECT_WITHOUT_SIGN_REVERSAL`;
8. `V6A_SCAFFOLD_SENSITIVE_ORDER_REVERSAL_SUPPORTED`.

The final status may be accompanied by replicated TASK-placement,
answer-set-placement, or joint-interaction descriptors. These descriptors do
not replace the primary status.

## 11. Claim boundary and V6B authorization

Maximum success supports only:

> In Qwen2.5-7B-Instruct and the registered synthetic prompt family, moving
> fixed TASK and valid-answer blocks around a fixed fact block causally changes
> the sign of TARGET-order sensitivity, with the replicated factorial contrasts
> identifying the contribution of each block placement.

It does not establish an activation gap, latent knowledge, biology, a physical
law, a universal recency mechanism, or model-family generality.

V6B is authorized for design—not model execution—only after V6A components and
the relevant topology contrast replicate. V6B must use another token-disjoint
fit/localization/holdout bank, freeze a final `ANSWER:` response-site
intervention, include answer/null/generic-cue shams, and make no use of V5
localization or holdout artifacts.

## 한국어 요약

V6A는 activation patch 실험이 아니라 V4와 V5 사이에서 함께 이동했던
`TASK`와 `VALID OUTPUTS` 블록을 분리하는 행동 topology 식별 실험이다.
두 블록의 전/후 위치와 TARGET fact 순서를 완전 교차하고, content 요인은
resolution-IV half-fraction으로 균형화한다.

각 world는 two-fact composition 64개, retrieval 32개, lookup 8개,
single-TARGET composition 32개로 총 136 calls를 가진다. Discovery 8 worlds와
confirmation 8 worlds의 총 호출 수는 2,176이다. 두 role과 모든 world의
property/code character는 서로 완전히 다르며, tokenizer-only preflight로 모든
prompt context와 assistant continuation에서 단일 token임을 증명한다.

7B 모델은 먼저 composition이 0개인 384-call disposable component
qualification을 통과해야 한다. 따라서 모델 선택이 gap 방향이나 크기를 볼 수
없다. Discovery component가 통과하면 효과 방향과 무관하게 confirmation을
실행하여 optional stopping을 방지한다.

최대 성공은 고정 synthetic scaffold 안에서 TASK/answer-set 위치가 TARGET
순서 효과의 부호를 바꾼다는 행동적 인과 결론만 지지한다. Activation gap,
latent biological knowledge, biology, physical law는 지지하지 않는다. 이후
V6B는 완전히 새로운 token bank와 고정 `ANSWER:` response site에서 별도로
사전등록해야 한다.
