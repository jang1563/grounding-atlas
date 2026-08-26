# GSE96583 donor-aware expression-context module result

Eight SLE donors, `56` cells, `1120` unique calls, parse rate `100.0%`.

Positive effects push P(CD8) upward relative to a matched neutral
deletion; negative effects push it toward NK. Intervals and tests use
eight unweighted donor effects.

| context | module | donor effect [95% CI] | t p | exact p | signs | pass |
|---|---|---:|---:|---:|---:|---:|
| T_plus_cytotoxic | TCR/CD8 | +0.114 [+0.093, +0.136] | 2.2374e-06 | 0.0039062 | 8/8 | true |
| T_plus_cytotoxic | cytotoxic effector | -0.065 [-0.127, -0.002] | 0.021995 | 0.019531 | 7/8 | true |
| NK_receptor_plus_cytotoxic | NK receptor/identity | -0.053 [-0.086, -0.019] | 0.0036748 | 0.015625 | 6/8 | false |
| NK_receptor_plus_cytotoxic | cytotoxic effector | -0.116 [-0.192, -0.040] | 0.0042844 | 0.0039062 | 8/8 | true |

- Gate A (`T_plus_cytotoxic`) passed: **true**, exact IUT p=`0.019531`
- Gate B (`NK_receptor_plus_cytotoxic`) raw component gate: **false**; hierarchical confirmatory pass: **false**
- All prompt robustness checks passed: **false**
- All unmasked-minus-neutral sham intervals lie within ±0.03: **false**

## Interpretation boundary

causal sensitivity to rendered marker-name deletion in one model and one eight-donor SLE control cohort; not annotation truth, a gene perturbation, pathway causality, a hidden-state activation route, latent-knowledge proof, or a physical law.

Deposited labels are descriptive only. Expression support selected the paired modules; the experiment does not estimate T-versus-receptor effects on the same cells.
