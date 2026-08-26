# Single-cell prompt-role reversal: graded signal plus polarity bias

The same cells and named/anonymous renderings were rerun after reversing
the two answer classes. Reported P(B) was mapped back to P(A)=1-P(B), so
biological-class stability and broader prompt-role following can be compared directly.

| task | forward correct shift A / B | reversed correct shift A / B | named AUROC F→R | order-averaged grounding M | biological-class B_class | prompt-role P_role | aligned delta rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| CD8+ T vs NK | -0.073 / +0.487 | +0.327 / +0.018 | 0.826→0.836 | +0.190 [+0.162, +0.218] | -0.126 [-0.181, -0.070] | -0.434 [-0.478, -0.390] | +0.472 |
| CD14+ vs CD16+ monocyte | +0.060 / +0.221 | +0.595 / -0.051 | 0.763→0.906 | +0.206 [+0.180, +0.231] | +0.242 [+0.191, +0.294] | -0.404 [-0.445, -0.361] | +0.358 |

`M` is the mean correct-oriented named-minus-anonymous shift over both
classes and orientations. `B_class` is positive when biological class A receives
the larger update across orientations and negative when class B does.
`P_role` is positive when the first prompt role is
favored and negative when the second role is favored. Confidence intervals use
a class-stratified paired bootstrap over the same cell IDs.

The anonymous arm reveals the source of the role effect. The modal probability assigned to whichever class was queried first was CD8+ T vs NK: 0.85 (99.0%) forward and 0.85 (99.5%) reversed; CD14+ vs CD16+ monocyte: 0.85 (97.5%) forward and 0.85 (100.0%) reversed. In this Haiku pilot, gene symbols therefore act mainly by correcting a high, input-insensitive default for the queried-first class; the second class has much more headroom for a correct semantic update.

Across both tasks, `M` is positive and `P_role` is negative with intervals excluding zero, while the aligned per-cell delta correlation remains positive. The shared result is a graded, item-specific semantic correction combined with a larger second-role correction. Biological-class asymmetry is task-specific rather than shared.

The orientation-interaction guardrail is CD8+ T vs NK: +0.034 [-0.009, +0.078]; CD14+ vs CD16+ monocyte: -0.132 [-0.173, -0.090]. The interval includes zero for CD8/NK but not for the monocyte task, so an additive biological-class plus prompt-role description is insufficient for the latter. Neither result is an equivalence test.

The swap changes both list position and which class is queried as the reported
probability. It therefore identifies a bundled prompt-role/polarity effect, not
pure first-versus-second list-position causality. A pure position test must cross
list order (A/B versus B/A) with queried target (P(A) versus P(B)).

The forward artifact is historical while the reverse run is new. The dated model
identifier reduces but does not eliminate model-drift risk; a claim-bearing run
should rerun all prompt forms contemporaneously and interleave their call order.

This output-level intervention does not establish hidden-state causality,
pre-existing latent knowledge, or training exposure.
