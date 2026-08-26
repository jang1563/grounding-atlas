# Coherent binary readout: development Level-0 result

Status: **STOP — both frozen Qwen runs are `DEVELOPMENT_READOUT_FORMAT_INVALID`**  
Execution date: 2026-08-02  
Scope: outcome-exposed readout engineering only; no biological, knowledge,
activation-gap, calibration, or confirmatory inference

## Registered result

Both model plans were frozen before either forward pass. Each plan crossed eight
GSE96583 development donors, two readouts, one input family, two mapping-line
orders, and two opaque-label remappings: 64 raw float32 forward passes per model.
The independent analyzer reconstructed every retained logit, full-vocabulary
argmax, log-sum-exp, row hash, and matrix hash from the mandatory sidecar.
Execution used Apple MPS (`mps:0`), PyTorch 2.10.0, Transformers 5.3.0, and
model execution dtype float32. The cached safetensors-file SHA-256 values were
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
(0.5B) and
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
(1.5B).

| model | preregistered role | native `X/Y` argmax | cytotoxic-state `O / R / I` | lineage `O / R / I` | item range pass | decision |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | software smoke only | 3/64 (4.7%) | -1.170 / -0.627 / -0.198 | -1.198 / +0.067 / +0.136 | 0/8 in both readouts | format invalid; all nuisance gates fail |
| Qwen2.5-1.5B-Instruct | development behavior | 6/64 (9.4%) | -0.876 / -0.939 / -0.300 | -0.331 / +0.348 / -0.256 | 0/8 in both readouts | format invalid; all nuisance gates fail |

The registered nuisance-equivalence margin was `±0.06`. The values above are
donor-first diagnostic means on the `[-2,+2]` factorial-effect scale. They are not
biological effects. Every 90% interval crossed or lay outside the equivalence
region, every exact shifted sign-flip TOST conjunction failed, and every group
failed its item-range guardrail. Extraction itself was exact: the maximum two-token
complement residual was zero for both models.

## Failure localization

This was not a near-threshold power failure. The model usually did not use the
registered opaque label as its native first token:

- 0.5B greedy tokens: `label` 42/64, `Label` 19/64, `X` 3/64;
- 1.5B greedy tokens: `label` 48/64, `NK` 10/64, `Y` 6/64.

Thus the retained `X/Y` logits were often scores for non-native alternatives while
the model began an instruction echo or emitted a semantic class token. The two
permitted tokens still occurred among high-logit candidates in many rows, but their
conditional contrast changed drastically with mapping-line order and opaque
remapping. Increasing scale from 0.5B to 1.5B did not repair the output channel.

The bounded discovery is therefore an **output-channel and instruction-binding
failure shared by two sizes of one Qwen model family**. It does not show absent cell
biology or absent marker knowledge. It does show that this prompt/readout cannot be
used as a native model decision and cannot support a biological or activation-gap
analysis.

## Registered consequence

The 1.5B result did not pass Level 0, so the Level-0-to-power bridge and Phase-0
simulation were not run. Thresholds were not relaxed, failed forms were not dropped,
and the four scores were not averaged into a rescued biological endpoint. The 0.5B
run remains quarantined as software smoke regardless of its result. Confirmatory
execution remains `NO-GO` because no qualifying same-cell orthogonal
lineage-by-intracellular-state public cohort exists.

## Successor decision

Version 1 remains immutable. A successor must be a new, explicitly labeled
development preregistration and should separate two estimands:

1. **Native answer compliance.** Select an answer prefix/template only on a
   non-biological, label-known syntax bank; then freeze it before any cell outcome.
   Retain the full-vocabulary native-answer gate.
2. **Choice-conditioned energy.** If `X/Y` logits are used when another token is
   greedy, name the quantity a conditional contrastive energy—not a probability or
   native decision. For causal token or activation interventions, estimate paired
   intervention effects within each frozen prompt form and test the
   intervention-by-form interactions. Large prompt-form main effects cannot be
   relabeled as biology.

A conditional-energy effect could support a bounded interface-sensitivity result.
An activation/integration-gap claim would still require native-output rescue,
orthogonal labels, intact-cue patching, selective erasure/reverse controls,
bidirectional mediation, adequate donor power, and independent replication.

## Artifacts

- 0.5B: [design](qwen2.5-0.5b-instruct/design.json),
  [plan manifest](qwen2.5-0.5b-instruct/plan_manifest.json),
  [run manifest](qwen2.5-0.5b-instruct/run_manifest.json),
  [raw records](qwen2.5-0.5b-instruct/raw.jsonl),
  [full-vocabulary sidecar](qwen2.5-0.5b-instruct/full_vocab_logits.npy), and
  [registered report](qwen2.5-0.5b-instruct/result.md).
- 1.5B: [design](qwen2.5-1.5b-instruct/design.json),
  [plan manifest](qwen2.5-1.5b-instruct/plan_manifest.json),
  [run manifest](qwen2.5-1.5b-instruct/run_manifest.json),
  [raw records](qwen2.5-1.5b-instruct/raw.jsonl),
  [full-vocabulary sidecar](qwen2.5-1.5b-instruct/full_vocab_logits.npy), and
  [registered report](qwen2.5-1.5b-instruct/result.md).

Frozen call-plan SHA-256 values are
`bb678962349e0a462c120fffd29caa29151fbfc64b66017481903d0541a03de9`
(0.5B) and
`0263c0b9be1e025e7836fea077e7559f9914d961b3f7d80ad9f992641561553b`
(1.5B). Full-logit matrix SHA-256 values are
`a4d0656fce150d3b8dc814d5ae766562b1b7bf27c7229ad0617d0a37bcff3f20`
and `d64b8870fd608ba124b56a65a9a3dc0875e15ffd01ff6e14e21670d3d68cd158`,
respectively.

## 한국어 요약

두 모델 모두 파일·해시·sidecar 재구성에는 성공했지만 Level 0에서 중단됐다.
허용된 `X/Y`가 실제 첫 토큰 argmax인 비율은 0.5B에서 3/64, 1.5B에서
6/64뿐이었다. 대부분 `label`, `Label`, 또는 `NK`를 먼저 출력했고,
order/remapping 효과는 `±0.06` 허용 범위를 크게 넘었다. 이는 생물학 지식의
부재가 아니라 현재 prompt와 출력 채널의 부적합을 보여준다. 따라서 power
simulation과 생물학/activation-gap 분석은 실행하지 않았다. 다음 버전은 비생물학적
syntax bank에서 answer context를 먼저 동결하고, native answer와 조건부 contrastive
energy를 명확히 분리해야 한다.
