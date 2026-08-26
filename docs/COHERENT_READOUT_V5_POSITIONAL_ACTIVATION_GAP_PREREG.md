# Coherent readout V5: causal TARGET-order/context activation-gap preregistration

Freeze state: `FROZEN_BEFORE_ANY_MODEL_FORWARD`  
Prospective date: 2026-08-03  
Execution authorization: activation-gap stage explicitly approved by the user;
all later phases remain conditional on the preregistered gates below.  
Mode: synthetic, development-only, causal intervention  
Primary model: `Qwen/Qwen2.5-1.5B-Instruct`  
Primary site: the `resid_post` state at the TARGET fact's property token  
Candidate decoder blocks: `[12, 16, 20, 24]`

## 1. Question and boundary

V4 established a behavioral asymmetry but did not establish an activation gap:
direct arbitrary-code lookup failed, and the observed composition deficit was
concentrated when the TARGET fact appeared second. V5 therefore asks a new,
prospectively identified question:

> In fresh symbolic worlds with independently admitted component behavior, is
> the loss in correct-answer margin caused by placing the TARGET fact second
> selectively mediated by a TARGET-property-token order/context coordinate?

The operational activation gap is not a hidden store of biological knowledge.
It requires the paired margin deficit

\[
G = E_w\{M_{w,\mathrm{first}}-M_{w,\mathrm{second}}\},
\]

where `M` is the correct-code logit minus the other valid-code logit for the
same semantic item in the fixed eight-pair intervention panel and, at layer
`l`, an independently replicated activation-coordinate separation

\[
Z_l=E_w\{(h_{w,\mathrm{second}}-h_{w,\mathrm{first}})^T u_{o,l}\}>0.
\]

The direction `u_o` is fitted only in fit worlds; `Z` is evaluated cross-fit in
localization and holdout worlds. The causal test changes only this coordinate
of the TARGET-property-token residual state. The corresponding margin gap
across all 16 factorial semantic pairs is retained as a diagnostic but is not
the denominator for intervention-effect ratios.

Even a positive result supports only a model-, prompt-, token-, and layer-local
causal TARGET-order/context-state mediation claim in a synthetic task. Moving
TARGET between the two registered orders jointly changes its absolute token
position and whether OTHER FACT is already in its prefix. The design therefore
does not identify a pure absolute-position effect separately from prefix-context
interference. It does not establish latent biological knowledge, a biological
activation gap, a physical law, a universal variable-binding circuit, or
model-family generality.

## 2. Fresh worlds and prospective split

The fixture contains 32 prompt-unique worlds. World is the inferential and
resampling unit. Roles are assigned before model execution and never reused:

| Role | Worlds | Baseline calls | Purpose |
|---|---:|---:|---|
| fit | 8 | 448 | fit axes and pass fit admission |
| localization | 8 | 448 | reproduce the gap and choose one layer |
| holdout | 16 | 896 | one-shot confirmation |
| total | 32 | 1,792 | |

Every prompt includes a registered instance key and world-specific entities.
No rendered user prompt may be duplicated. Property and answer symbols are
uppercase single-character labels selected prospectively; tokenizer preflight
must prove that every registered composition TARGET-property occurrence and
every next-answer label is exactly one contextual token.

The 8 property-symbol pairs and 4 code-symbol pairs recur across roles, but
each of their 32 pairings occurs in exactly one world and therefore one role.
The split is combinatorially disjoint, not token-disjoint: holdout tests unseen
property-pair/code-pair combinations, not unseen individual symbols.

The static bank ledger is frozen before tokenizer preflight:

| Artifact | SHA-256 |
|---|---|
| `signal/syntax/build_coherent_readout_v5_positional_activation_bank.py` | `9356b383daf0eb4bdf501eb201a2834b4acb41c05638fa76c4cbece6d9948150` |
| `signal/syntax/coherent_readout_v5_positional_activation_bank.json` | `defa5ed2c0ab1f0f6c7ac7cf5eaa4abe453daac682676b9d783770ffae6da903` |
| fixture canonical JSON | `0ac414c77ea6a84f4003b42ca9ca2a356d3a61f3c0e92abfb5a0aa4868abb75c` |
| `signal/syntax/coherent_readout_v5_positional_activation_bank.manifest.json` | `01bd21c52e307b62a073f57f0a3e9086096894e7bb975d6ceda3d0f8930729cb` |
| `eval/model_hooks.py` | `62495bd77adc40d7fd5e5643df334eb98aba363f5b81b4b7925314e877bad0c4` |

Runner, analyzer, focused-test, tokenizer-receipt, and environment hashes are
bound dynamically by the final zero-forward `design.json`, avoiding a
mutual-hash cycle.

Each world contains exactly:

- 8 direct property-retrieval prompts;
- 16 direct codebook-lookup prompts; and
- 32 composed prompts forming the complete `2^5` factorial below.

The codebook, valid-output line, and explicit task instruction precede the two
fact lines. This makes mapping, output labels, and task intent causally
available at the intervention site. The two order mates have the same codebook,
facts, queried entity, semantic answer, and valid outputs; only TARGET/OTHER
fact order changes.

## 3. Composition factorial and fixed intervention panel

The five signs are:

- `p`: TARGET property, first (`-1`) versus second (`+1`) property symbol;
- `m`: identity (`-1`) versus swapped (`+1`) property-to-code mapping;
- `r`: first versus reversed codebook rule order;
- `v`: first versus reversed valid-output order; and
- `o`: TARGET fact first (`-1`) versus second (`+1`).

With code-0 signed `-1` and code-1 signed `+1`, the correct-answer sign is the
preregistered interaction `a = -p*m`; its activation axis is equivalently the
Walsh `pm` axis because a global sign does not change coordinate replacement.
All 32 cells per world are used for behavior and axis fitting. Intervention
uses a fixed eight-pair regular fraction with independent base columns `p,m,r`
and

\[
v=p m r.
\]

For each selected semantic cell, both `o=-1` and `o=+1` are registered as an
exact matched pair. Selection is intention-to-treat: no cell, prompt, source,
world, layer, or direction may be selected by observed correctness or effect.

## 4. Model and numerical lock

The final pre-forward design binds exact hashes for the fixture builder,
fixture, fixture manifest, preregistration, runner, analyzer, hook helper,
focused tests, model revision, cached model files, tokenizer files, and chat
template. Any mismatch stops before a forward call.

- model: `Qwen/Qwen2.5-1.5B-Instruct`;
- model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- decoder blocks: 28; residual width: 1,536;
- tested blocks: exactly `[12,16,20,24]`;
- hook: output of `model.model.layers[layer]` (`resid_post`);
- site: exact token matched by the fixture's TARGET-property character span
  and fast-tokenizer offset mapping;
- batch size 1, evaluation mode, inference mode, `use_cache=false`;
- float32 model execution on MPS with SDPA attention;
- no generation, demonstrations, prompt repair, or adaptive dose search; and
- patch strength exactly 1.0 (coordinate replacement).

The tokenizer-only `plan` phase must prove prompt uniqueness, registered span
identity, one-token properties and answer labels, source/recipient shape
compatibility, no truncation, call-plan completeness, and zero model-forward
calls. Failure is `V5_ENGINEERING_INVALID`.

## 5. Fit-only directions

Let `h_l(w,x)` be the unpatched TARGET-property-token residual vector for fit
world `w`, layer `l`, and factorial cell `x`. For Walsh sign `chi_j`, define

\[
\beta_{j,l}=\frac1{8}\sum_w\frac1{32}\sum_x\chi_j(x)h_l(w,x).
\]

The genuine named coefficients are `p`, `m`, `r`, `v`, `o`, and native answer
`pm`. The raw TARGET-order/context coefficient is `beta_o`. It is residualized by CPU
float64 SVD against `{beta_p,beta_m,beta_r,beta_v,beta_pm}`. The raw answer
coefficient `beta_pm` is residualized against
`{beta_p,beta_m,beta_r,beta_v,beta_o}`. The relative SVD rank tolerance is
`1e-6`. Each residual is normalized to a unit direction.

One deterministic null direction is constructed from the eight within-world
`beta_o` vectors using the first balanced nonconstant column of canonical
Sylvester `H_8`, with worlds in increasing frozen world-ID order. It is
residualized against all six pooled genuine named coefficients and normalized.
No null sign or axis may be redrawn.

For every candidate layer, the fit basis must satisfy all of:

- all coefficients and residuals finite;
- residual/raw norm ratio at least `0.10` for order/context, answer, and null axes;
- unit norm within `1e-5`;
- maximum pairwise absolute cosine among the three axes at most `0.35`;
- sign-aligned even/odd-world order/context-axis cosine at least `0.25`; and
- minimum sign-aligned leave-one-world-out order/context-axis cosine with the
  full-fit direction at least `0.25`.

Every layer must pass because the entire basis is frozen before localization.
A failure is `V5_FIT_BASIS_INVALID`; the grid may not be reduced post hoc.

## 6. Behavioral admission

Behavior is scored from full next-token logits. For each prompt, the predicted
task answer is the higher-logit member of that prompt's two registered valid
labels. Ties are failures. Full-vocabulary logits and global-maximum
diagnostics remain stored for independent replay.

Fit admission requires:

1. retrieval and lookup accuracy each at least `0.95` overall;
2. retrieval and lookup accuracy at least `0.90` in every world and at each
   marginal level (`-1` and `+1`) of every factor present in that component
   family;
3. target-first composition accuracy at least `0.95`;
4. target-second composition error at least `0.25`;
5. target-first minus target-second accuracy at least `0.10`;
6. a world-bootstrap 95% lower finite-panel stability bound for `G` greater
   than zero; and
7. every registered direct prerequisite for the fixed intervention panel
   correct.

Localization and holdout baseline phases repeat the same criteria within their
own disjoint worlds. In addition, both the natural margin gap `G` and cross-fit
activation-coordinate separation `Z` must have world-bootstrap 95% lower
bounds above zero and be positive in at least 6/8 localization worlds and 12/16
holdout worlds. No baseline failure may be reinterpreted as evidence for or
against activation mediation.

The bootstrap uses 10,000 resamples of registered worlds, percentile intervals,
and the fixed seed `260803`; prompts within a world are never resampled
independently. Because property and code pairs recur across worlds, these are
finite-panel stability intervals used as preregistered gates, not population
confidence intervals over unseen symbol vocabularies.

## 7. Exact interventions and controls

For a matched first/second state pair, define the order/context scalar dose

\[
d=(h_F-h_S)^Tu_o.
\]

Exact order/context-coordinate rescue is `h'_S=h_S+d u_o`, and reverse damage
is `h'_F=h_F-d u_o`. The registered fit-only center cancels algebraically in
`d` but is stored for replay.

For every fixed semantic pair and tested layer, localization runs six calls:

1. `positional_rescue`: `h'_S=h_S+d u_o`;
2. `positional_damage`: `h'_F=h_F-d u_o`;
3. `answer_rescue_sham`: `h'_S=h_S+d u_A`;
4. `answer_damage_sham`: `h'_F=h_F-d u_A`;
5. `null_rescue_sham`: `h'_S=h_S+d u_N`; and
6. `null_damage_sham`: `h'_F=h_F-d u_N`.

The answer/null shams are additive dose-matched controls, not coordinate copies
using their own generally smaller source/recipient projections. Every primary
and sham row must store `d`, expected and observed perturbation L2 norms, and
pass `abs(observed-|d|) <= 1e-5*max(1,|d|)`. This prevents circular specificity
from unequal intervention strength. The source and recipient always have the
same native answer, so generic source-answer copying predicts no change. Once
per world and layer, an exact self-state identity patch must reproduce the
unpatched logits within `1e-4` for both valid labels and preserve the global
argmax set.

Call counts are frozen:

| Phase | Formula | Calls |
|---|---:|---:|
| fit baseline | `8*56` | 448 |
| localization baseline | `8*56` | 448 |
| localization patch | `8*8*4*6 + 8*4` | 1,568 |
| holdout baseline | `16*56` | 896 |
| holdout patch | `16*8*1*6 + 16*1` | 784 |
| total | | **4,144** |

No omitted, repeated, unregistered, or additional forward is permitted.

## 8. Causal estimands and localization gate

For each semantic pair, let `M_F` and `M_S` be unpatched correct-oriented
margins for TARGET-first and TARGET-second prompts. Define `g=M_F-M_S`.
For the order/context patches (legacy condition prefix `positional_`),

\[
R=M_{S\leftarrow F}^{(o)}-M_S,\qquad
D=M_F-M_{F\leftarrow S}^{(o)},\qquad C=(R+D)/2.
\]

`R` is sufficiency-like rescue; `D` is necessity-like degradation. For answer
and null shams, define non-cancelling control magnitudes

\[
K_A=E\{(|R_A|+|D_A|)/2\},\qquad
K_N=E\{(|R_N|+|D_N|)/2\}.
\]

Phase-level ratios use means across fixed pairs and worlds, for example
`E[R]/E[g]`; ratios are never computed per row.

A localization layer passes only if its cross-fit `Z` bootstrap lower bound is
above zero, `Z` is positive in at least 6/8 worlds, and all of:

- `E[R]/G >= 0.30` and `E[D]/G >= 0.30`;
- world-bootstrap lower bounds for `E[R]`, `E[D]`, and `E[C]` exceed zero;
- both rescue and damage are positive in at least 6/8 worlds;
- `(E[C]-K_A)/G >= 0.20` and `(E[C]-K_N)/G >= 0.20`; and
- bootstrap lower bounds for both signed specificity contrasts exceed zero.

All identity checks must pass. If multiple layers pass, the shallower block is
selected. If none passes, the experiment stops before holdout patching with
`V5_NO_PREREGISTERED_CAUSAL_LAYER`.

## 9. Holdout confirmation

The selected layer, all axes, signs, thresholds, prompts, pairs, and analysis
code are hash-locked before any holdout baseline is read. Holdout patching is
authorized only after independent holdout component/gap admission.

The final causal criteria are the localization criteria applied unchanged at
the selected layer, except `Z`, rescue, and damage must each be positive in at
least 12/16 holdout worlds. Every `Z`, effect, and specificity bootstrap lower
bound named in Sections 6 and 8 must remain greater than zero. Localization and
holdout estimates are reported separately; they are never pooled for the
confirmatory status.

## 10. Artifact and phase discipline

Each phase writes immutable attempt, JSONL record, float32 full-vocabulary logit
shard, activation sidecar where applicable, and execution-manifest artifacts.
Sidecars include shape, dtype, row-order, and byte hashes. The analyzer
independently reconstructs expected rows, diagnostics, margins, activation
indices, call counts, and every artifact hash before issuing the next immutable
authorization.

Authorized sequence:

1. `plan` (zero forwards);
2. `fit-baseline` -> fit analysis/basis and localization-baseline entry;
3. `localization-baseline` -> localization-patch entry;
4. `localization-patch` -> selected-layer lock and holdout-baseline entry;
5. `holdout-baseline` -> holdout-patch entry; and
6. `holdout-patch` -> final analysis.

The runner rejects missing, extra, stale, overwritten, self-inconsistent, or
out-of-order artifacts. Phase-local and cumulative forward counts are reported.

## 11. Terminal status hierarchy

The analyzer returns exactly one terminal status at the first applicable tier:

1. `V5_ENGINEERING_INVALID`;
2. `V5_FIT_COMPONENT_ADMISSION_FAIL`;
3. `V5_FIT_BASIS_INVALID`;
4. `V5_LOCALIZATION_COMPONENT_ADMISSION_FAIL`;
5. `V5_LOCALIZATION_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED`;
6. `V5_LOCALIZATION_ENGINEERING_INVALID`;
7. `V5_NO_PREREGISTERED_CAUSAL_LAYER`;
8. `V5_HOLDOUT_COMPONENT_ADMISSION_FAIL`;
9. `V5_HOLDOUT_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED`;
10. `V5_FINAL_ENGINEERING_INVALID`;
11. `V5_NO_REPLICATED_CAUSAL_GAP_CLOSURE`;
12. `V5_NONSPECIFIC_CAUSAL_GAP_CLOSURE`;
13. `V5_CAUSAL_GAP_CLOSURE_NATURAL_USE_NOT_ESTABLISHED`; or
14. `V5_CAUSAL_TARGET_ORDER_CONTEXT_ACTIVATION_GAP_SUPPORTED`.

The success status requires admitted components, replicated natural margin and
order/context-coordinate gaps, bidirectional selective mediation, engineering
validity, and all preregistered holdout criteria. A weaker status may describe
rescue without necessity-like damage, but cannot be called natural causal
mediation.

## 12. Verifiability and prohibited adaptation

The following are prohibited after the first forward: changing prompt wording,
world roles, factor coding, panel cells, token site, layer grid, axes, null
signs, thresholds, bootstrap seed, patch strength, call count, status logic, or
claim language; filtering by correctness or effect; unregistered reruns;
combining discovery and holdout; or substituting label probability for the
registered two-code margin.

Raw logits, activations, exact prompts, token IDs, offset maps, source graph,
runner/analyzer hashes, environment lock, and analysis calculations are retained
so a third party can reproduce both the computation and the stopping decision.
