# Coherent Readout V6A: component qualification V2 engineering revision

Registration state: `FROZEN_BEFORE_V2_LOAD_SMOKE_OR_MODEL_FORWARD`  
Date: 2026-08-03  
Execution revision: `v2_mps_allocator_warmup_bypass`

## 1. Why a separate revision is required

The V1 qualification plan froze 384 direct-component calls with zero
composition calls. Its execution attempt stopped during model loading, before
the first forward, because Transformers 5.2.0 attempted to allocate one
14.19-GiB float16 MPS caching-warmup buffer. macOS rejected that single buffer
with `RuntimeError: Invalid buffer size: 14.19 GiB`.

V1 is terminally sealed as `V6A_QUALIFICATION_ENGINEERING_INVALID`:

- V1 call-plan SHA-256:
  `bc1aacd6e101e4da06e50ba4fcc930d31ea7a49578775c6b9f152a0aed48866e`;
- V1 plan-manifest file SHA-256:
  `0be35c0d6635924658985b5ea47b6f02b09c4245e10ba00bd28735e00eedaaf9`;
- V1 attempt file SHA-256:
  `fb13237f77995932b3fb9fced04ddc1395d123a7e56f3592b1bd0657e911af3b`;
- V1 analysis file SHA-256:
  `34f1a622862b54c66e99882b014706ba117390153b358361953ab6293c60bec8`;
- V1 records, raw logits, and execution manifest: absent; and
- V1 model forwards: exactly zero.

The zero-forward fact comes from the contemporaneously observed traceback in
`from_pretrained` allocator warm-up, before a model object was returned. The
sealed V1 artifact set independently proves that no records, raw-logit shard,
or execution manifest was completed, but the V1 attempt schema did not persist
a per-forward journal. V2 therefore binds both the recorded load-time failure
provenance and the absence of every V1 scientific-output artifact; it does not
infer zero merely from an absent completed manifest.

V1 is not resumed or overwritten. V2 uses a separate result root, runner,
analyzer, tests, preregistration, plan, attempt, records, shards, and analysis.

## 2. Frozen scientific identity

V2 is an engineering-only execution revision. It must preserve V1 exactly for:

- model ID and revision;
- fixture, worlds, symbols, entities, keys, user prompts, system message, and
  assistant prefill;
- rendered input IDs, attention masks, answer token IDs, response-site IDs and
  indices, prompt IDs, cell IDs, and call order;
- 384 total calls: 256 property retrieval, 128 codebook lookup, and zero
  composition;
- all accuracy, world, answer-label, factor-level, `q*a`, `o*q*a`, unique-
  global-argmax, tie, tokenizer, hash, and replay gates; and
- all claim boundaries in the V1 qualification preregistration and frozen V6A
  topology design.

The V2 plan must independently reconstruct the fixture and tokenizer outputs,
then prove that its complete `cells + prompts` scientific registry has the same
canonical SHA-256 as V1. V1 has no model output, so reuse of its still-unseen
qualification bank does not condition V2 on behavior.

V2 also binds the exact V1 design, dependency lock, tokenization receipt,
attempt, and terminal analysis file hashes. The current model assets/config and
the shared builder, fixture, fixture manifest, V1 preregistration, frozen V6A
topology design, and builder test must match their V1 dependency bindings.

## 3. Sole authorized engineering change

The only allowed change is the model-loading policy:

```text
v1: direct MPS device_map + Transformers caching_allocator_warmup
v2: direct MPS device_map + scoped skip of caching_allocator_warmup when and
    only when every expanded target device is MPS
```

Transformers' warm-up is a performance allocation; it does not load a model
parameter, change a tensor value, alter inference math, or define a prompt. The
V2 runner temporarily replaces that one function only inside
`AutoModelForCausalLM.from_pretrained`, requires exactly one all-MPS bypass
call, and restores the original function in a `finally` block. Any non-MPS
device in the expanded map, zero bypass calls, multiple bypass calls, or failed
restoration is engineering-invalid.

All weights still load directly from the same hash-locked safetensors onto MPS
as bfloat16. Parameter and buffer device/dtype checks remain unchanged. SDPA,
`use_cache=false`, `logits_to_keep=1`, full-vocabulary float32 storage, and all
forward semantics remain unchanged.

## 4. Load-only smoke test

Before freezing the V2 call plan, one load-only smoke phase is authorized. It:

- hashes the full model and V2 implementation dependency set;
- verifies the sealed V1 files and absence of V1 scientific outputs by opaque
  file hashes/path existence only, without JSON-parsing the V1 prompt plan;
- invokes the V2 loader without parsing, rendering, tokenizing, or inspecting
  the contents of any fixture prompt; the dependency lock only streams the
  fixture file as opaque bytes for hashing;
- performs zero model forwards and zero generation;
- verifies model dimensions, parameter/buffer devices and dtypes, exactly one
  all-MPS warm-up bypass, and restoration of the original Transformers
  function; and
- writes no scientific result.

The smoke phase may report only loader success/failure and engineering metadata.
It cannot inspect logits, activations, margins, accuracy, topology, or an answer.
V2 planning/execution is allowed only if this smoke succeeds. The actual
qualification then loads a fresh model in a fresh process under the identical
frozen loader code.

## 5. V2 phase and status rules

The phase sequence is:

1. V2 static/mock tests;
2. load-only zero-forward smoke;
3. V2 zero-forward plan freeze and a dedicated, zero-write independent analyzer
   replay;
4. one immutable 384-forward qualification attempt; and
5. one frozen independent analysis.

Partial resume, repeated qualification execution, overwrite, or scientific
change is prohibited. The performance thresholds and terminal meanings remain:

- `V6A_QUALIFICATION_ENGINEERING_INVALID`: V2 execution or replay invalid;
- `V6A_QUALIFICATION_COMPONENT_FAIL`: valid execution misses a frozen gate;
- `V6A_QUALIFICATION_COMPONENT_PASS`: every frozen gate passes.

A PASS supports direct-component qualification only. It does not establish
composition, a behavioral topology effect, an activation gap, latent knowledge,
biology, a physical law, or model-family generality.

## 한국어 요약

V1은 첫 forward 전 model-loading 단계에서 Transformers의 14.19-GiB MPS
warm-up buffer 오류로 종료됐으며 `ENGINEERING_INVALID`로 봉인되었다. 따라서
동일 attempt를 재개하지 않고 완전히 분리된 V2 execution revision을 사용한다.

V2는 model, fixture, 384개 prompt, token IDs, response site, call order, 모든
gate와 claim boundary를 V1과 동일하게 유지한다. 유일한 변경은 모든 target
device가 MPS일 때만 Transformers의 성능용 allocator warm-up을 scoped하게
건너뛰고 즉시 원함수를 복원하는 것이다. 먼저 prompt·logit·activation을 전혀
parse·render·tokenize·inspect하지 않는 load-only/zero-forward smoke test로
loader를 검증한 후 V2 plan을 고정한다. 이때 fixture 파일은 dependency hash를
위한 불투명 byte stream으로만 읽힌다.
