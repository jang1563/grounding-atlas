# Coherent readout v2: non-biological syntax-selection preregistration

Status: **frozen development-only syntax selection; no biological model input has
been authorized**  
Freeze date: 2026-08-02

## Purpose and firewall

Version 1 failed because the model usually emitted `label`, `Label`, or a semantic
class token rather than the registered opaque answer token. Version 2 first selects
one global answer context using only direct, non-biological declarations. It does
not test biology, knowledge, calibration, latent state, an activation gap, or a
physical law.

The complete syntax bank is
`signal/syntax/coherent_readout_v2_syntax_bank.json`; its manifest is
`signal/syntax/coherent_readout_v2_syntax_bank.manifest.json`. It contains eight
lexical-pair clusters, both truth polarities per pair, and 16 items. The pair—not
the individual declaration—is the inferential unit.

Frozen SHA-256 values:

- builder: `637b8e363f4c63e3d81ac03b2fbe0ee08694947fb1966733740023a5d5660085`;
- fixture: `d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8`;
- fixture manifest: `b7b8467c2595af5bf5cfb414cb052289062e721300ea3415c23d22de16c22ae5`.

## Model roles and compute lock

Freeze both call plans before either model forward pass.

1. Software/artifact smoke only: `Qwen/Qwen2.5-0.5B-Instruct`, revision
   `7ae557604adf67be50417f59c2c2f167def9a775`, cached safetensors SHA-256
   `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
2. Syntax selection: `Qwen/Qwen2.5-1.5B-Instruct`, revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, cached safetensors SHA-256
   `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.

The tokenizer ID and revision equal the model ID and revision. Both plans freeze
Apple MPS as the execution backend, model dtype `float32`, model-config vocabulary
size, effective chat-template SHA-256, contextual answer-token IDs, Python and
dependency versions, and exact runner/analyzer hashes. Local cached files only,
`trust_remote_code=False`, evaluation mode, `use_cache=False`, and no generation or
logit processor are allowed.

An artifact/extraction failure in the 0.5B smoke run stops the sequence. A valid
0.5B behavioral failure cannot select, eliminate, rank, or veto any context. Only
the frozen 1.5B result selects.

## Frozen factorial and candidate registry

For every candidate and item, cross mapping-line order
`positive_first/negative_first` with opaque remapping
`positive_is_x/positive_is_y`. Each model therefore has exactly
`4 candidates x 8 pairs x 2 truths x 2 orders x 2 remappings = 256` records.

Every mapping line is exactly `label {opaque} means {class}`. Only its registered
order changes. `{declaration}` is copied byte-for-byte from the fixture. Every
rendered `\n` below is exactly one U+000A line-feed byte sequence in UTF-8; CRLF is
forbidden.

Candidate priority is frozen as follows.

1. `assistant_prefill_answer_colon`
   - system: `Complete the assistant answer with exactly X or Y and no other text.`
   - user: `Classify the record from its explicit declaration.\n{line_1}\n{line_2}\n{declaration}`
   - assistant final-message prefix: `Answer:`
   - render with `continue_final_message=True`; candidate suffixes are ` X` and
     ` Y`.
2. `system_user_exact`
   - system: `Follow the user's label mapping. Your entire response must be exactly one character: X or Y. Do not write any other text.`
   - user: `Classify the record from its explicit declaration.\n{line_1}\n{line_2}\n{declaration}\nReturn exactly X or Y.`
   - render with `add_generation_prompt=True`; candidate suffixes are `X` and `Y`.
3. `user_only_exact`
   - one user message: `Follow the label mapping. Answer with exactly one character: X or Y. Do not write any other text.\nClassify the record from its explicit declaration.\n{line_1}\n{line_2}\n{declaration}\nAnswer:`
   - render with `add_generation_prompt=True`; candidate suffixes are `X` and `Y`.
4. `user_label_colon_v1_control`
   - one user message: `Classify the record using exactly one label.\n{line_1}\n{line_2}\n{declaration}\nLabel:`
   - render with `add_generation_prompt=True`; candidate suffixes are `X` and `Y`.

No extra whitespace or terminal newline may be inserted. Tokenizer-only preflight
found stable single tokens `1599/809` for the prefill candidate and `55/56` for the
other candidates; the plan must independently recompute and freeze them.

The prefill candidate uses the exact system, user, and assistant messages above
with `tokenize=True`, `continue_final_message=True`,
`add_generation_prompt=False`, `enable_thinking=False`, `return_dict=True`, and
`return_attention_mask=True`. The other three use
their stated message lists with `tokenize=True`, `add_generation_prompt=True`,
`continue_final_message=False`, `enable_thinking=False`, `return_dict=True`, and
`return_attention_mask=True`. A tokenize-false
rendering must retokenize to exactly the same input IDs as the tokenize-true call.

## Raw measurement and artifact validity

Before loading weights, freeze the bank, manifest, this preregistration, candidate
registry, runner and analyzer, model/tokenizer/weight/backend/environment locks,
all 256 rendered prompts and tokenized inputs, record identities, and sidecar row
order. Execute exactly one raw next-token forward per record.

Write one exact JSONL record and one little-endian float32 full-vocabulary NPY row
per request. The analyzer must reconstruct the retained logits, the complete set of
maximum-logit token IDs, log-sum-exp, row and matrix hashes, and every derived
statistic. Any missing, duplicate, nonfinite, processed, out-of-vocabulary,
hash-inconsistent, or topologically invalid record invalidates the artifact. A
finite reconstructable maximum tie is valid raw behavior but sets both `native` and
`native_correct` to false. Token-ID ordering must never break it.

## Candidate metrics

Let `g` be the opaque token mapped to the declared class and `w` the other opaque
token. From the raw vocabulary logits define

\[
E=z_g-z_w, \qquad s=\tanh(E/2).
\]

`native` means the unique full-vocabulary argmax is either registered opaque token.
`native_correct` means it is exactly `g`. A nonlabel token or the wrong opaque token
fails native correctness; a semantic class-token echo cannot rescue it.

For each item, compute the existing four-form `O`, `R`, and `I` effects from the
truth-aligned `s`. Within lexical pair `j`, for each effect `A` define

\[
M_{A,j}=\tfrac12(A_{positive,j}+A_{negative,j}),\qquad
D_{A,j}=\tfrac12(A_{positive,j}-A_{negative,j}).
\]

The polarity-difference `D` is mandatory because pair averaging could otherwise
hide equal-and-opposite truth-polarity failures.

## Conjunctive eligibility gate

A candidate is eligible only if every condition passes:

1. exact schema, topology, token, sidecar, and provenance validity;
2. `native >=61/64` overall, `>=15/16` in every order-by-remapping form, and
   `>=7/8` in every form-by-truth-polarity cell;
3. `native_correct` meets those same three floors;
4. within every form, at least `15/16` records have `s>0` and median `s>=0.50`,
   corresponding to at least 3:1 conditional odds for the correct versus wrong
   opaque token;
5. all 16 items have four-form range `max(s)-min(s)<=0.20`; and
6. all six eight-pair vectors `M_O,D_O,M_R,D_R,M_I,D_I` pass strict equivalence
   within `+/-0.06`: their 90% Student-t interval lies strictly inside the margin,
   both one-sided t-TOST p-values are below `0.05`, both exact shifted sign-flip
   TOST p-values are below `0.05`, and every leave-one-pair-out mean lies strictly
   inside the margin.

Exact sign-flip conclusions are conditional on pair-effect sign symmetry. This is
a required and unverified assumption, and the result remains a fixed-bank
engineering result rather than population inference.

## Deterministic single-context selection

If multiple candidates are eligible, rank them lexicographically by:

1. highest minimum-form `native_correct` count;
2. highest total `native_correct`;
3. highest minimum form-by-polarity `native_correct` count;
4. highest minimum-form `native` count;
5. highest total `native`;
6. lowest maximum item range;
7. lowest maximum absolute mean among the six `M/D` estimands;
8. highest minimum-form median `s`; and
9. the frozen candidate priority above, then candidate ID.

Integer comparisons are exact. Floating comparisons use the exact values
reconstructed from the frozen float32 sidecar with no adjustable epsilon, weighted
score, or post-hoc rounding. Select one global context only; item-, pair-, truth-,
or form-specific syntax mixing is forbidden.

No eligible candidate produces `SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT`. It
does not permit threshold relaxation, item/form deletion, candidate addition,
candidate reordering, or a biological fallback. The selected analysis artifact and
its SHA-256 must be frozen before any biological model call. A later biological
failure cannot trigger fallback to a runner-up.

## Boundary for the later biological stage

Syntax success demonstrates only direct-declaration output-channel behavior. A
new, disjoint biological development plan is still required.

For any later input, keep three quantities separate:

\[
M_{channel}=\max(z_X,z_Y)-\max_{v\notin\{X,Y\}}z_v,
\]

\[
M_{correct}=z_g-\max_{v\ne g}z_v,
\]

and `E_choice=z_g-z_wrong`. Both `M_correct` and `E_choice` are reserved for an
analysis with an independent orthogonal gold label `g`. Without such truth, define
only the preregistered orientation contrast

\[
E_{oriented}=z_{positive}-z_{negative},
\]

and report it alongside `M_channel` and the native opaque choice when present. When
a nonlabel token is native, `E_oriented` is a hypothetical forced-choice or
conditional contrastive energy—not a probability or native model decision.

Future interventions must report native-channel transitions, changes in
`M_channel` and, with orthogonal truth, `M_correct`, paired changes in
`E_oriented` (or gold-aligned `E_choice`) within every frozen form, and
intervention-by-form effects. An energy shift
without native-channel and orthogonal-correctness rescue supports only bounded
interface sensitivity. Activation/integration-gap language additionally requires
independent fact access, native-output rescue, selective erasure/add-back/reverse
controls, bidirectional mediation, donor-level power, orthogonal labels, and
independent biological and model-family replication.

## Claim boundary / 해석 경계

**English.** This stage can select one model-specific answer syntax. It cannot
establish biological correctness, knowledge, natural use, latent representation,
an activation gap, calibration, or a physical law.

**한국어.** 이 단계는 해당 모델에 맞는 하나의 출력 문법만 선택할 수 있다.
생물학적 정확성, 지식, 자연적 사용, 잠재 표현, activation gap, calibration,
또는 물리 법칙을 입증하지 않는다. 16개 문항의 독립 단위는 8개 lexical pair이며,
1.5B에서 모든 후보가 실패하면 생물학 단계로 넘어가지 않는다.
