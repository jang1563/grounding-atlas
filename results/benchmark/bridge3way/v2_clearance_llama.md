# Experiment 2 v2: clearance (weakest-specialist endpoint) + Llama-3.1-8B (cross-architecture)

Extends [v1_herg.md](v1_herg.md). Same 3 arms, same shared folds, same parity contract + per-item paired
test (`eval/analyze_bridge3way.py`). orchestrate is LLM-independent (a head on the ChemBERTa embedding),
so it is reused unchanged; only bridge + LoRA re-run per endpoint/model. Date 2026-06-27.

## Axis 1: clearance on Qwen3-8B (the LLM-win candidate)
clearance is the endpoint where the cheap specialist is WEAKEST (Morgan cold-ceiling 0.746, selectivity
0.242), so it is where an LLM placement had the best shot at a win. It did not win.

| arm (shared 800-item test) | WITHIN | TRANSFER (5 -> clearance) |
|---|---|---|
| orchestrate (head) | 0.607 | 0.537 |
| bridge (-> LLM) | 0.615 (bypass 0.624) | 0.556 (bypass 0.545) |
| in-weight LoRA | 0.575 (base 0.545) | 0.464 (base 0.545) |
| cheap Morgan specialist | **0.746** | floor 0.545 |

- WITHIN: the three LLM placements are statistically INDISTINGUISHABLE (bridge - bypass -0.009 CI
  [-0.030, +0.011]; bridge - orchestrate +0.008 CI [-0.040, +0.055], both include 0) and ALL sit well
  below the cheap Morgan specialist (0.746). On the endpoint most favorable to an LLM win, no LLM arm
  wins and placement does not matter.
- TRANSFER: orchestrate 0.537 / bridge 0.556 / bypass 0.545, all at the Morgan floor 0.545, no paired CI
  excludes 0. No transfer (same as hERG).

## Axis 2: hERG on Llama-3.1-8B-Instruct (cross-architecture)
Does the "frozen LLM is dead weight in the read" finding hold in the field-standard bio-bridge substrate?

| arm (shared 793-item hERG test) | WITHIN | TRANSFER (5 -> hERG) |
|---|---|---|
| orchestrate (head, LLM-independent) | 0.893 | 0.435 |
| bridge (-> Llama) | 0.812 (bypass 0.865) | 0.507 (bypass 0.437) |
| in-weight LoRA (Llama) | 0.728 (base 0.515) | 0.456 (base 0.515) |

- WITHIN: orchestrate (0.893) > bridge (0.812) > LoRA (0.728), and **the Llama bridge again loses to its
  own bypass head, formally: bridge - bypass = -0.051, CI [-0.102, -0.012], EXCLUDES 0** (and the head
  beats the bridge: bridge - orchestrate -0.078, CI [-0.174, -0.020], excludes 0). The "frozen LLM is
  dead weight in the read" finding REPLICATES cross-architecture with a CI that excludes 0 (it was -0.041
  CI[-0.077,-0.016] on Qwen).
- TRANSFER: all at or near the floor (orchestrate 0.435 / bridge 0.507 / bypass 0.437); the bridge 0.507
  is right at the Morgan floor (0.510), not a real transfer. No placement transfers in Llama either.

## v2 verdict
"route, don't train" is robust along both stress axes: on the WEAKEST-specialist endpoint (clearance) no
LLM arm beats the cheap specialist and the placements are indistinguishable; and on a DIFFERENT
architecture (Llama, the bridge-standard substrate) the in-language bridge still loses to a bare head on
the same projection (the frozen LLM is dead weight) and nothing transfers across properties. The result
generalizes beyond the single v1 cell.

## The full 2x2 matrix (endpoint x architecture), WITHIN-property
The 4th cell (clearance-Llama) ran: orchestrate 0.607 (LLM-independent) ~ bridge 0.598 (bypass 0.615) >
LoRA 0.507 (no lift, base 0.513) - all below the cheap specialist 0.746, same as clearance-Qwen.

| within-property AUROC | orchestrate (head) | bridge (-> LLM) | bypass (head) | LoRA | cheap specialist |
|---|---|---|---|---|---|
| hERG  / Qwen   | 0.893 | 0.83-0.85 | 0.87 | 0.73 | 0.895 |
| hERG  / Llama  | 0.893 | 0.81 | 0.87 | 0.73 | 0.895 |
| clear / Qwen   | 0.61  | 0.62 | 0.62 | 0.58 | 0.746 |
| clear / Llama  | 0.61  | 0.60 | 0.62 | 0.51 | 0.746 |

In every cell: (i) no LLM placement beats the cheap specialist; (ii) the bridge never beats its own
bypass head (significantly so on hERG, CI excludes 0 in both architectures; indistinguishable on the
noisier clearance); (iii) LoRA is the weakest; (iv) nothing transfers across properties (a separate row,
all at the cross-property floor). "Route, don't train" holds across 2 endpoints x 2 architectures.

Remaining (extra-dimension, not run; the verdict is robust without them): MoLFormer-XL substrate (needs
trust_remote_code), the additive-at-peak injection (inject at the per-model selectivity-peak layer,
hERG-Qwen L35, vs layer-0; unlikely to overturn a head that does not use the LLM at all), and the
dedicated ChemBERTa-MLM pretraining-naive control (already largely self-resolved: ChemBERTa-MTR transfers
below the plain-Morgan floor).
