# GSE96583 held-out GNLY/NKG7/CCL5 factorial result

`8` SLE donors, one held-out cell per donor, `480` unique model calls, exact parse rate `100.0%`.

The primary interface is `ab_pa`. Negative effects indicate NK-directed output leverage. `a` is absolute target-mask movement, `h` is reference-prevalence-balanced control-mask movement, and `r=a-h` is their matched contrast.

## Primary decision

- Dual-scale anchor gate: **FAIL** (exact IUT p=1)
- Endpoint: **anchor_gate_failed** (rJ equivalence=False, aJ equivalence=False; rJ material=False, aJ material=False)
- Strong sparse-GNLY gate: **FAIL**
- Isolated target-erasure-specific wording: **not allowed**

| scale/vector | donor mean [95% CI] | material < -0.03 | equivalent ±0.03 |
|---|---:|---:|---:|
| `r:A_GNLY` | -0.0125 [-0.0421, +0.0171] | FAIL | FAIL |
| `a:A_GNLY` | -0.0125 [-0.0421, +0.0171] | FAIL | FAIL |
| `r:T_full_triple` | +0.0750 [-0.1023, +0.2523] | FAIL | FAIL |
| `a:T_full_triple` | +0.0750 [-0.1023, +0.2523] | FAIL | FAIL |
| `r:J_increment_NKG7_CCL5_after_GNLY` | +0.0875 [-0.0881, +0.2631] | FAIL | FAIL |
| `a:J_increment_NKG7_CCL5_after_GNLY` | +0.0875 [-0.1194, +0.2944] | FAIL | FAIL |
| `r:Q_GNLY_on_NKG7_CCL5_background` | -0.0125 [-0.0421, +0.0171] | FAIL | FAIL |
| `a:Q_GNLY_on_NKG7_CCL5_background` | -0.0125 [-0.0421, +0.0171] | FAIL | FAIL |

## Canonical seven-subset surface

| target subset | r mean [95% CI] | a mean [95% CI] | h mean [95% CI] |
|---|---:|---:|---:|
| `GNLY` | -0.0125 [-0.0421, +0.0171] | -0.0125 [-0.0421, +0.0171] | +0.0000 [-0.0447, +0.0447] |
| `NKG7` | +0.0875 [-0.1194, +0.2944] | +0.0750 [-0.1023, +0.2523] | -0.0125 [-0.0421, +0.0171] |
| `CCL5` | +0.0000 [+0.0000, +0.0000] | -0.0125 [-0.0421, +0.0171] | -0.0125 [-0.0421, +0.0171] |
| `GNLY+NKG7` | -0.0125 [-0.0421, +0.0171] | -0.0125 [-0.0421, +0.0171] | +0.0000 [+0.0000, +0.0000] |
| `GNLY+CCL5` | -0.0125 [-0.0421, +0.0171] | -0.0125 [-0.0421, +0.0171] | +0.0000 [+0.0000, +0.0000] |
| `NKG7+CCL5` | +0.0875 [-0.0881, +0.2631] | +0.0875 [-0.0881, +0.2631] | +0.0000 [+0.0000, +0.0000] |
| `GNLY+NKG7+CCL5` | +0.0750 [-0.1023, +0.2523] | +0.0750 [-0.1023, +0.2523] | +0.0000 [+0.0000, +0.0000] |

## Conditional localization

Conditional Shapley localization was not tested because the dual-scale distributed endpoint did not pass.

## Prompt-surface analysis

| form | anchor | endpoint |
|---|---:|---|
| `ab_pa` | FAIL | `anchor_gate_failed` |
| `ab_pb` | FAIL | `anchor_gate_failed` |
| `ba_pa` | FAIL | `anchor_gate_failed` |
| `ba_pb` | FAIL | `anchor_gate_failed` |

Prompt robust: **FAIL**. This requires every form to preserve the canonical anchor and endpoint and all twelve registered interaction intervals (six vectors × two prompt factors) to lie within ±0.03.

## Sensitivity and inference

The secondary ±0.05 analysis classified the canonical endpoint as `anchor_gate_failed`. It cannot rescue the ±0.03 primary decision.
Student-t7 intervals use eight unweighted donor effects. Exact Rademacher p-values enumerate all 2^8 sign assignments and are exact only under donor-effect sign symmetry. Exact binomial sign tests are also reported, with ties counted against the direction.

## Provenance and boundary

- Raw checkpoint SHA-256: `b625d8cf8f65e15863d19f8eb3b8300c8d0f84be6709eaac7edd94339dfede33`
- Request-plan SHA-256: `db5af60a5142c6f79cbac7abdba85520077efd75fab732db0bfda80f6843a20e`
- Runner SHA-256: `a7a35904dbf0b8c519ed58b4844fbe2a772035478f526f7f496d23bbe1315e23`
- Runtime/dependency manifest SHA-256: `2e5bc6771c95bdfa04c012a9527e2d750cd89b4ca4ed9a685c851840c7c24928`

held-out-cell text-interface dependence in one model and one eight-donor SLE control cohort; not a biological perturbation, pathway mechanism, annotation-truth validation, latent-knowledge proof, hidden-state activation-gap test, mathematical invariant, or physical law.
