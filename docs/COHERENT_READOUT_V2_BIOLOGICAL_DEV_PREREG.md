# Coherent readout v2: conditional biological development preregistration

Status: **frozen before syntax outcomes; conditional development-only execution**  
Freeze date: 2026-08-02

## Purpose and claim firewall

This stage asks whether one answer envelope selected on the non-biological syntax
bank remains a native, form-stable output channel for ranked single-cell genes. It
is not a test of biological correctness because the fixture contains no independent
gold lineage or state label. It does not test latent knowledge, a causal activation
gap, calibration, mediation, or a physical law.

The complete four-candidate biological projection registry was frozen before any
syntax outcome was observed:

- builder SHA-256:
  `beb6d697151ecfca83a80a785b2eaf69458cccbe376b612c95503b082c12c10b`;
- registry SHA-256:
  `c1e5de573c58734b57d1c2884086cf948e5fdd1f0ab6cce2ae78ed747fc54304`;
- registry canonical SHA-256:
  `b009d16e420d993405661077741c4638cb5a701abeb8bcf4603741ee90ce62a3`;
- manifest SHA-256:
  `fd6c4f4a0408da01bc491df24ae7702b6ea5d490d59e7fd50a8d081465f81808`.

Its firewall is permanent development-only, no gold, no promotion to confirmation,
and no biology, knowledge, or activation inference from the projection itself.

## Conditional entry and no fallback

Biological planning is authorized only if all of the following are true:

1. the 0.5B smoke artifact is structurally valid;
2. the frozen 1.5B syntax analysis has status `SYNTAX_SELECTION_PASS`, selection
   authority is true, and exactly one selected projection is present;
3. the complete syntax result bytes, canonical result, selected projection, raw
   record content, full-vocabulary matrix, call plan, design, code, model, and
   environment hashes validate independently; and
4. the selected candidate occurs in the pre-outcome biological projection registry
   with the same source candidate definition and definition SHA-256.

The syntax result file SHA-256 and selected-projection SHA-256 must be frozen in the
biological design before tokenization or weight loading. Any missing entry,
tokenization mismatch, biological failure, or interrupted execution stops the
stage. It cannot activate a runner-up, mix candidates, edit a prompt, relax a
threshold, remove a donor/item/form, or selectively rerun a result.

An infrastructure interruption may be followed only by a complete re-execution of
the identical 64-row frozen plan. The runner must retain results in memory and write
final raw artifacts only after all 64 forwards complete; partial-result resumption
is forbidden.

## Frozen biological source and model

The biological fixture is
`signal/single_cell/coherent_readout_v2_bio_fixture.json`:

- builder SHA-256:
  `63b33ca4bc21efc3afbdaca45b7c16afba37b4f35f308155251dcd6ea923b8da`;
- fixture SHA-256:
  `d9c8256cc249f5f3b1b5ea07d99bdb80927b1c2b6b50bcb17540ae3ea0dd601a`;
- fixture canonical SHA-256:
  `0bfd809d44c6b7e18461d711098ae28155cbefde1c01e19ada728acad8709b77`;
- manifest SHA-256:
  `3d67bb639e3ff2c1e4adcd673c15b9108468c377006843d8599afb90fe56eeda`;
- exclusion-registry SHA-256:
  `12342bfd34d305b69c6396670be35e6d0256b6ddd7e117fd56b9910fd72780fa`;
- selected-cell-registry SHA-256:
  `67c581b17f9006a03de8036b33e4f0ee04471d0a5c4b8aa20378048c81ddb0bf`;
- frozen project Qwen-result scan SHA-256:
  `88123881ada8594095a9cba20c0ee5775b4c15dd3cf93ec71207e5b1dc661262`.

The scan supports only project-record outcome non-exposure at freeze. It makes no
claim about external data or pretraining exposure.

Execute only `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, cached
`model.safetensors` SHA-256
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
The tokenizer ID/revision must equal the model ID/revision. Use local files only,
`trust_remote_code=False`, Apple MPS, model dtype `float32`, evaluation/inference
mode, `use_cache=False`, no generation, and no logit processor. Verify actual
cached bytes before planning and execution and actual parameter and buffer
device/dtype after loading.

## Outcome-blind answer-envelope projection

The exact biological core, with no terminal newline or answer cue, is:

```text
Classify the cell using exactly one label.
{line_1}
{line_2}
Genes, highest expression rank first: {gene_sentence}
```

Every mapping line is exactly `label {opaque} means {class}`. The registry replaces
the syntax bank's direct-declaration task body with this biological core while
preserving the selected candidate's system/output instruction, terminal cue,
message roles, rendering flags, and answer suffixes. The phrase `from its explicit
declaration` is forbidden in biological messages. `Label:` and `Answer:` cues may
never be combined.

Use the one registry entry matching the selected candidate globally. Recompute the
contextual X/Y token IDs across all 64 biological prompts. They must be constant and
must exactly equal the selected syntax projection's X/Y IDs. Otherwise stop without
fallback.

All rendered newlines are one U+000A. `tokenize=False` rendering must retokenize
byte-for-token to the `tokenize=True`, `return_dict=True`,
`return_attention_mask=True` result. No extra whitespace or terminal newline may
be introduced.

## Exact topology and pre-forward freeze

The fixture supplies eight donors, two readouts (`lineage` and
`cytotoxic_state`), and one `unmodified` family. Cross each fixture row with the
four registered order-by-remapping forms:

\[
8\ donors\times2\ readouts\times1\ family\times2\ orders\times2\ remappings
=64\ calls.
\]

Use exact fixture-record order followed by this exact form order:

1. `positive_first / positive_is_x`;
2. `positive_first / positive_is_y`;
3. `negative_first / positive_is_x`;
4. `negative_first / positive_is_y`.

Before loading weights, freeze the validated syntax result/projection, biological
fixture and manifest, projection registry and manifest, this preregistration,
model/tokenizer/config/chat-template and local asset hashes, Python/platform and
dependency versions, runner/analyzer/imported-source hashes, all 64 exact messages,
rendered prompts, input IDs and masks, contextual answer IDs, record identities,
sidecar row order, call-plan SHA-256, and design SHA-256. The plan-only artifact
must state zero model calls.

Donor, barcode, and source-entity identifiers may be stored as provenance but may
not occur in model-visible messages.

## Raw measurement and artifact validity

Execute exactly one raw next-token forward per plan row. Store one exact-key JSONL
record and one corresponding row in a finite, C-contiguous, little-endian float32
NPY matrix of exact shape `(64, vocab_size)`, with `allow_pickle=False`.

The independent analyzer must reconstruct X/Y logits, full-vocabulary log-sum-exp,
row and matrix hashes, the complete set of maximum-logit token IDs, tie count, and
every derived metric exactly. A finite maximum tie is valid observed behavior but
is never native; token ordering or `argmax` may not resolve it. Any missing,
duplicate, nonfinite, processed, out-of-vocabulary, hash-inconsistent, topologically
invalid, or provenance-inconsistent record invalidates the artifact.

No raw record, design, analysis, or report may contain fields named or interpreted
as gold, correct/wrong token, `native_correct`, `M_correct`, or `E_choice`.

## No-gold measurements

For each form, orient the two opaque tokens using the fixture's registered positive
and negative classes:

\[
E_{oriented}=z_{positive}-z_{negative},\qquad
s_{oriented}=\tanh(E_{oriented}/2).
\]

Also report:

\[
M_{channel}=\max(z_X,z_Y)-\max_{v\notin\{X,Y\}}z_v,
\]

\[
G_{choice}=|z_X-z_Y|=|E_{oriented}|.
\]

Report two-token vocabulary mass descriptively. `native` is true only when the
unique full-vocabulary maximum is X or Y. Only then report native X/Y choice and
its positive/negative orientation; otherwise the native choice is null. When
native is false, `E_oriented`, `s_oriented`, and `G_choice` are hypothetical
forced-choice contrasts, not native decisions or probabilities.

Positive and negative are registered axes, not truth. No directional or magnitude
gate is permitted for `E_oriented`, `s_oriented`, `M_channel`, `G_choice`, or
two-token mass. In particular, there is no post-hoc positive `M_channel` margin.

## Conjunctive development gate

The numerical margins below remain candidate-unqualified engineering margins.
They do not provide a confirmatory sample size or qualify Phase 0.

### Native channel

Require all three:

- at least `61/64` native overall;
- at least `31/32` native within each readout; and
- at least 90% native within every donor-by-readout four-form cell, which is
  discretely `4/4` for all 16 cells and therefore forces `64/64` overall.

### Item guardrails

Within each donor/readout item, compute the four-form range of `s_oriented`.
All eight items in each readout must satisfy
`max(s_oriented)-min(s_oriented) <= 0.20`; equality passes.

If the absolute four-form mean is at least `0.20`, every form must have the same
strict sign as that mean. A zero or opposite-sign form fails this strong-item
guardrail. Weak items have no directional requirement.

### Form effects and equivalence

For each donor and readout, let `PF/NF` denote mapping-line order and `PX/PY`
denote which opaque token represents the positive orientation. Define:

\[
O=\tfrac12(s_{PF,PX}+s_{PF,PY}-s_{NF,PX}-s_{NF,PY}),
\]

\[
R=\tfrac12(s_{PF,PX}+s_{NF,PX}-s_{PF,PY}-s_{NF,PY}),
\]

\[
I=\tfrac12[(s_{PF,PX}-s_{PF,PY})-(s_{NF,PX}-s_{NF,PY})].
\]

For each readout separately, all three eight-donor vectors `O`, `R`, and `I` must
pass strict equivalence within `+/-0.06`:

- the 90% Student-t interval lies strictly inside the margin;
- both one-sided Student-t TOST p-values are below `0.05`;
- both exact shifted sign-flip TOST p-values are below `0.05`; and
- every leave-one-donor-out mean lies strictly inside the margin.

Equality at an equivalence boundary fails. Exact sign-flip conclusions are
conditional on the required and unverified donor-effect sign-symmetry assumption.
The inference scope is the fixed eight-donor engineering fixture.

## Ordered statuses

Apply status precedence exactly:

1. `BIO_V2_ARTIFACT_INVALID` for an invalid prerequisite or artifact;
2. `BIO_V2_NATIVE_CHANNEL_FAIL` if artifact validity passes but any native-channel
   condition fails;
3. `BIO_V2_FORM_STABILITY_FAIL` if the native channel passes but any item range,
   strong-sign, or readout-specific O/R/I equivalence condition fails; or
4. `BIO_V2_DEVELOPMENT_PASS_LEVEL0_ONLY_NO_GOLD` only if every condition passes.

Only the fourth status may mark the six readout-by-effect components as
`level0_only_no_gold`. It cannot produce `n_conf`, qualify Phase 0, authorize
confirmation, trigger syntax fallback, or support biology, knowledge, activation,
mediation, or physics claims.

## Claim boundary / 해석 경계

**English.** This is a development-only output-channel and form-invariance result
on eight fixed donors without independent gold labels. It does not establish
biological correctness, latent knowledge, a causal activation gap, or a physical
law.

**한국어.** 이는 독립적인 gold label이 없는 고정 8명 donor 개발 세트에서 출력
채널과 형식 불변성만 평가한다. 생물학적 정확성, 잠재지식, 인과적 activation
gap, 또는 물리 법칙을 입증하지 않는다.
