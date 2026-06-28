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

Remaining v2: clearance-Llama, MoLFormer-XL substrate, the additive-at-peak injection (inject at the
per-model selectivity-peak layer, hERG-Qwen L35, vs layer-0), and the dedicated ChemBERTa-MLM
pretraining-naive control.
