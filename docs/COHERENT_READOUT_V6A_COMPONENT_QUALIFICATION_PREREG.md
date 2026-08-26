# Coherent Readout V6A: component-only model qualification preregistration

Registration state: `FROZEN_BEFORE_ANY_QUALIFICATION_MODEL_FORWARD`  
Date: 2026-08-03  
Mode: disposable direct-component qualification with zero composition calls

## 1. Purpose and firewall

This phase asks only whether one prospectively fixed model can perform the two
direct components required by V6A: copying a labeled property and applying a
two-rule property-to-code lookup. It is not part of the V6A topology experiment
and cannot estimate the codebook-composition TARGET-order gap, the composition
topology interaction, an activation gap, or a biological effect. Its retrieval
panel can measure direct-copy accuracy across scaffolds, as disclosed below.

The candidate is fixed to:

- model: `Qwen/Qwen2.5-7B-Instruct`;
- revision: `a09a35458c702b33eeacc393d103063234e8bc28`;
- source parameter dtype and execution dtype: `bfloat16`;
- device: Apple MPS;
- attention implementation: PyTorch SDPA; and
- stored next-token logits: full vocabulary in little-endian float32.

The move from the earlier 1.5B model is justified only by its failed direct
retrieval/lookup gate. No composition margin, order-effect magnitude, or
order-effect sign is used to select the 7B model or this prompt wording.

Qualification entities, instance keys, prompts, property characters, and code
characters are disposable. They are disjoint from V2--V5 and from the reserved
V6A discovery/confirmation bank. Prompt wording and thresholds may not be
repaired after any qualification output is observed.

## 2. Fixed bank

The bank contains eight worlds and 48 prompts per world, for exactly 384 model
forwards:

| Family | Fixed factorial per world | Calls/world | Total |
|---|---|---:|---:|
| property retrieval | full `p * v * o * q * a` | 32 | 256 |
| codebook lookup | full `p * m * r * v` | 16 | 128 |
| composition | prohibited | 0 | 0 |

For retrieval, `q` moves the unchanged TASK block before/after the facts, `a`
moves the unchanged valid-output block before/after the facts, and `o` swaps
only TARGET/OTHER fact order. This topology crossing tests whether direct
copying remains valid under every scaffold that V6A will compare. It does not
combine retrieval with the codebook and therefore is not composition.

The 32 registered one-character symbols are globally unique across property and
code roles. The static fixture, builder, manifest, preregistration, runner,
analyzer, and their focused tests are all file-hash locked before execution.

## 3. Response site and tokenizer contract

Every call uses three chat messages: the frozen system instruction, the user
prompt, and assistant content `ANSWER:`. The chat is rendered with:

```text
add_generation_prompt = false
continue_final_message = true
enable_thinking = false
```

The rendered chat must end exactly at `ANSWER:`. The colon must be the final
attended token, and the registered answer must be exactly one contextual next
token. Generation is prohibited.

Tokenizer-only planning must establish all of the following before a model is
loaded:

1. every occurrence of every registered character occupies exactly one prompt
   token and shares no lexical token;
2. each character has one stable prompt token ID across its occurrences and
   one stable continuation token ID across its answer contexts;
3. all 32 characters occur in both the prompt-token and continuation-token
   registries;
4. each retrieval content skeleton contains the exact eight `(o,q,a)` vertices;
5. all members of a topology octet have equal token count, attention-mask
   shape, response-site index, and response-site token ID; and
6. all 384 rendered prompt IDs are unique.

Planning performs tokenizer calls but exactly zero model forwards. The complete
tokenized call plan and tokenization receipt are immutable inputs to execution.

## 4. One-shot execution

The phase order is:

1. build and validate the static fixture;
2. run the zero-forward tokenizer plan;
3. independently replay the frozen plan and all implementation/dependency
   hashes;
4. perform disk, MPS-headroom, and bfloat16-kernel preflight checks;
5. write a hash-complete immutable execution-attempt record;
6. load the model directly onto MPS using the frozen device map; and
7. execute the 384 prompts once in call-plan order.

Each forward is teacher-forced and retains only the final response-site logits
from the model while persisting the complete vocabulary row as float32. Raw
logits are written once in six 64-row shards. Each result row binds its cell,
prompt, answer IDs, response site, raw-logit location and hash, reconstructed
diagnostics, and call-plan hash.

Partial resume, artifact overwrite, prompt regeneration, model generation, and
selective reruns are prohibited. Once the immutable attempt exists, a failed or
partial execution requires a new experimental revision.

## 5. Frozen performance gates

`answer_correct` means the registered correct-answer logit is strictly greater
than the other registered-answer logit. The analyzer evaluates the following
groups separately within each family:

- the complete family;
- every family-by-world group;
- every family-by-registered-correct-answer group; and
- every level of every active registered factor:
  - retrieval: `p`, `v`, `o`, `q`, and `a`;
  - lookup: `p`, `m`, `r`, and `v`;
- every retrieval `q * a` scaffold group; and
- every retrieval `o * q * a` order-by-scaffold group.

Qualification passes only if all conditions hold:

1. retrieval accuracy is at least `0.98` overall;
2. lookup accuracy is at least `0.98` overall;
3. accuracy is at least `0.90` in every world, correct-answer, marginal
   factor-level, and registered retrieval joint scaffold group listed above;
4. every row has exactly one full-vocabulary maximum and that maximum token is
   one of its two registered answer tokens;
5. every registered-answer comparison is non-tied; and
6. every engineering, fixture, hash, schema, call-count, no-composition,
   tokenizer, response-site, raw-logit, record, and manifest replay check passes.

There is no rounding before threshold comparison. The exact minimum counts are:

- retrieval overall: `251/256`;
- retrieval per world: `29/32`;
- retrieval per correct property character: `15/16`;
- retrieval per marginal factor level: `116/128`;
- retrieval per `q * a` group: `58/64`;
- retrieval per `o * q * a` group: `29/32`;
- lookup overall: `126/128`;
- lookup per world: `15/16`; and
- lookup per marginal factor level: `58/64`; and
- lookup per correct code character: `8/8`.

Because every code character is globally unique and is correct in eight lookup
rows, the last requirement intentionally implies `128/128` lookup accuracy even
though the separately reported overall threshold is `0.98`. Single-factor
marginal gates and the two prospectively named retrieval joint gates are the
only grouped combinations. No row or world can be excluded based on
correctness, margin, symbol identity, or argmax behavior.

## 6. Independent analysis and statuses

The frozen analyzer must reconstruct every diagnostic from the immutable raw
float32 vocabulary rows, validate every row against the call plan, replay all
file and canonical hashes, and verify that the record set is exactly the 384
planned calls. It must reject any composition family, any nonzero composition
count, any generation call, or any additional model call.

The retrieval panel can reveal direct-copy accuracy differences across
`(o,q,a)`, but it cannot reveal the codebook-composition endpoint. To prevent
that direct-component information from changing the downstream study, the
complete V6A topology design—including exact block text, symbol allocation,
fraction, estimands, and thresholds—is frozen and file-hash bound in this
qualification's dependency lock before execution. The qualification analyzer
reports only the registered accuracy gates and does not estimate retrieval
margin or topology-effect contrasts.

Use exactly one terminal status:

1. `V6A_QUALIFICATION_ENGINEERING_INVALID`: execution or immutable replay is
   incomplete or invalid; this is not evidence of model inability;
2. `V6A_QUALIFICATION_COMPONENT_FAIL`: the execution is engineering-valid but
   one or more frozen performance gates fail; or
3. `V6A_QUALIFICATION_COMPONENT_PASS`: every engineering and performance gate
   passes.

Analysis writes no discovery or confirmation authorization as a side effect.
A PASS is only a prerequisite for separately freezing and executing the fresh
V6A topology-identification bank.

## 7. Claim boundary

A PASS supports only this statement:

> At the fixed assistant response site, the registered Qwen2.5-7B-Instruct
> revision satisfies the preregistered direct property-retrieval and codebook-
> lookup gates on the disposable synthetic qualification panel.

It does not establish composition, a topology effect, an activation gap,
latent knowledge, biology, a physical law, or generality beyond this fixed
model and prompt family.

## 한국어 요약

이 단계는 V6A 본 실험 전에 수행하는 **composition 0회의 일회용 직접 성분
qualification**이다. 고정된 Qwen2.5-7B-Instruct 7B revision이 property 복사와
2-rule lookup을 수행할 수 있는지만 본다. 8개 world에서 retrieval 256개와
lookup 128개, 총 384 forward를 실행한다.

모든 prompt는 assistant prefill `ANSWER:`의 마지막 colon에서 다음 한 token을
채점한다. Tokenizer-only plan이 32개 문자의 prompt/continuation 단일-token
계약, 정확한 topology octet, 동일 response site를 먼저 증명한다. 이후 full-
vocabulary float32 logits를 6개 immutable shard에 저장하고 독립 analyzer가
모든 진단값과 hash를 원자료에서 재구성한다.

두 family의 전체 accuracy는 각각 0.98 이상, 모든 world·정답 label·factor
level과 사전등록된 retrieval `q*a`, `o*q*a` group은 0.90 이상이어야 하며,
모든 row의 unique global argmax가 등록된 두 answer token 중 하나여야 한다.
Lookup은 code symbol별 8/8 조건 때문에 의도적으로 전체 128/128을 요구한다.
통과해도 직접 성분 능력만 지지한다.
Composition, topology effect, activation gap, 잠재지식, biology, physical law는
아직 결론낼 수 없다.
