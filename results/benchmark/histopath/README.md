# Histopathology (VLM arm): the encoding axis and the frontier caveat

GroundBench task `histo/pcam_tumor`: an H&E patch from PatchCamelyon (400 balanced
96x96 tiles) -> tumor vs normal, fed to frontier vision-language models. This note
records the three-arm picture and is explicit about what is measurable on closed models.

## The three arms

| arm | what it is | AUROC |
|---|---|---|
| ceiling (pathology FM CONCH, reference) | a dedicated histopathology foundation model | ~0.90 |
| encode: open-VLM hidden-state probe | linear probe on Qwen2.5-VL-7B patch activations (`eval/activation_arm_histo.py`) | 0.827 |
| encode: cheap color head | logistic regression on per-channel color statistics (`eval/head_baseline.py`) | 0.730 |
| verbalize: frontier VLM output | P(tumor) asked directly | opus 0.436, sonnet 0.559, gpt-4o 0.570 |

## What it shows

The tumor signal is plainly extractable from the patch: an open VLM encodes it at
0.827 (above the cheap color ceiling, approaching the dedicated pathology FM), and even
a dumb color classifier reaches 0.730. Yet asked directly "estimate the probability this
patch contains tumor", every frontier model answers at chance (CIs spanning 0.5, ECE
~0.32, opus in fact slightly anti-correlated). This is the largest expression gap
measured anywhere in the atlas, and it is robust at the frontier.

It is also the one web-rich task the frontier does NOT ground. Tumor morphology is
heavily documented and imaged online, so by the web-exposure law the models "should"
verbalize it, and capability did close the other web-rich gaps (MSA 0.95-0.99,
variant-text 0.87-0.97, single-cell-name 0.75-0.98). It does not close this one. So
web-exposure is necessary but not sufficient: for medical images a second gate (refusal
or hedging on a diagnostic claim, or a vision-to-verbalization expression gap) blocks
the output even when the information is encoded and even at the frontier.

## The honest caveat (what cannot be measured here)

Frontier models are closed-weight, so we cannot probe THEIR hidden states. The
"encoded but not verbalized" claim for the frontier is by analogy: an open VLM of the
same family encodes tumor at 0.827 while verbalizing it at 0.463, and on this bench a
cheap color head extracts it at 0.730 while the frontier verbalizes at chance. Both show
the signal is present in the image; we infer the frontier encodes it too but cannot
prove it without activation access. A frontier-VLM activation probe is therefore not on
the roadmap (infeasible); strengthening the open-VLM probe is.
