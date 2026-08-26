# Level-0 coherent binary readout: development preregistration

Status: **development-only; outcome-exposed fixture; no confirmatory or biological
inference**  
Freeze date: 2026-08-02

## Objective

This run asks one measurement question: can an open-weight causal language model
produce a sufficiently order- and opaque-remapping-invariant two-token readout for
later biological experiments? It does not test whether any cell label is correct and
does not test knowledge, natural use, an integration/gain gap, underactivation, or a
physical law.

The runner must obtain the `X` and `Y` logits from the same raw, unprocessed
next-token vector. Natural-language generation and separately elicited decimal
probabilities are forbidden.

## Development fixture and firewall

The only input is
`signal/single_cell/coherent_readout_dev_fixture.json`, built from the eight
donor-explicit base cells in the completed GSE96583 sentinel experiment. It contains
eight source cells crossed with two readouts (`lineage`, `cytotoxic_state`) and the
single `unmodified` input family: 16 input rows and 64 order-by-remapping forward
passes per model.

Frozen fixture SHA-256:
`047be03ee29691506710cf9483cfe20a45c5d28e859818db9f612632d8d05367`.

Frozen fixture-manifest SHA-256:
`3ef29d500f85883063ceb842532b1b865ea8bee1b28222b02daabbd05a2b477e`.

Every source cell has prior model-outcome exposure. It is permanently in the
`development_only_outcome_exposed` partition and cannot be promoted into
confirmation. The fixture contains no ground-truth cytotoxic-state label and no
orthogonal protein label. The deposited cell annotation is not an analysis target.

## Model sequence

Run the models independently; never pool their items or donor vectors.

1. Software/extraction smoke only:
   `Qwen/Qwen2.5-0.5B-Instruct`, revision
   `7ae557604adf67be50417f59c2c2f167def9a775`.
2. Development behavior check:
   `Qwen/Qwen2.5-1.5B-Instruct`, revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Both runs use local cached files only, `trust_remote_code=False`, evaluation mode,
`use_cache=False`, one unpadded sequence per forward pass, and model dtype
`float32`. The tokenizer revision equals the model revision. The model configuration,
not `len(tokenizer)`, freezes the padded output-vocabulary size. The 0.5B result is
never promoted from software smoke to claim-bearing evidence.

## Frozen request construction

For every fixture row, cross:

- biological-positive mapping line first versus biological-negative mapping line
  first; and
- biological-positive class mapped to `X` versus to `Y`.

Use the exact prompt renderer in `eval/coherent_binary_readout.py`. Apply the frozen
tokenizer chat template with `add_generation_prompt=True` and thinking disabled. The
rendered-text tokenization must exactly equal the direct tokenized chat-template
output. Resolve `X` and `Y` at the exact answer position and require one stable token
ID for each over the complete call plan.

Before any model forward pass, write and freeze an exact design containing the
fixture, fixture-manifest, this preregistration, runner, call-plan, chat-template,
model, tokenizer, dtype, vocabulary, donor, source-item, prompt, input-token, and
expected-record identities. The development margin status is
`candidate_unqualified`.

## Raw measurement and independent reconstruction

Write one JSONL row and one little-endian float32 full-vocabulary sidecar row per
planned request. The sidecar is mandatory. The independent analyzer must reconstruct
from it:

- `x_logit`, `y_logit`, and the biologically aligned logit difference;
- full-vocabulary argmax and argmax logit;
- full-vocabulary log-sum-exp;
- the two-token conditional score; and
- every row and matrix checksum.

Any missing, duplicate, unexpected, nonfinite, processed, out-of-vocabulary, hash-
inconsistent, or non-Cartesian record invalidates the execution artifact. Four form
prompts, execution-input hashes, and forward traces must be distinct for each concrete
item/readout.

## Candidate Level-0 decision

Apply gates separately to each readout × concrete input family. Thresholds may be
tightened before a future run but never relaxed beyond these values:

- global and group full-vocabulary format adherence at least `0.95`;
- per-donor/group format adherence at least `0.90`;
- exact conditional-score complement error at most `1e-6`;
- order, remapping, and interaction donor effects: 90% t interval strictly inside
  `(-0.06,+0.06)`, both exact shifted sign-flip TOST p-values below `0.05`, and all
  leave-one-donor-out means strictly inside the margin;
- at least 95% of items with four-form range at most `0.20`; and
- any item with absolute four-form mean at least `0.20` must have the same strict
  sign in all forms.

The exact sign-flip conclusions are conditional on donor-effect sign symmetry; this
assumption is required and unverified. With only eight development donors, a pass is
`DEVELOPMENT_LEVEL0_CANDIDATE_PASS_MARGIN_NOT_QUALIFIED`, not a confirmatory Level-0
pass.

## Phase-0 consequence

Only a candidate Level-0 pass may be converted into the authenticated donor matrix
for the `level0_only` joint-power simulator. Phase 0 must independently qualify the
candidate `0.06` margin and may select a Level-0 donor-count candidate in `12..20`.
That candidate is not the final `n_conf`: the final donor target must power the full
registered claim hierarchy. A format, coherence, topology, nuisance, guardrail,
margin, covariance, support, or power failure is a stop, not permission to change the
threshold after seeing outputs.

## Claim boundary / 해석 경계

**English.** A successful development run validates only this model-specific output
measurement interface. A failure falsifies the current readout implementation or its
suitability for that model; it does not falsify the underlying biology. Neither result
is evidence for knowledge, a causal integration/gain gap, relative underactivation,
or a physical law.

**한국어.** 개발 실행의 성공은 해당 모델에서 이 출력 측정 인터페이스가 동결된
조건을 만족한다는 뜻뿐이다. 실패는 현재 readout 구현 또는 해당 모델과의 적합성을
반증하지만 생물학 자체를 반증하지 않는다. 성공과 실패 어느 쪽도 잠재지식, 인과적
integration/gain gap, 상대적 underactivation, 또는 물리 법칙의 증거가 아니다.
