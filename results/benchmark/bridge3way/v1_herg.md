# Experiment 2 (3-way calibrated LLM x SFM bridge) - v1 result (hERG headline)

Per [docs/BRIDGE_3WAY_PREREG.md](../../../docs/BRIDGE_3WAY_PREREG.md). v1 scope: Qwen3-8B, one endpoint
(hERG), one fold (`lpo_herg_clearance`), ChemBERTa-77M-MTR embedding, layer-0 soft-prompt (PRIMARY), no
bridge/LoRA sweep. All three arms scored on the SAME 793-item shared held-out test split (the parity
contract). Date 2026-06-27.

## The 3-way, on the shared 793 hERG test items

| arm | WITHIN-property | TRANSFER (pooled 5 -> hERG) |
|---|---|---|
| **B orchestrate** (trained head on the frozen embedding, LLM untouched) | **0.893** | 0.435 |
| **A bridge** (embedding -> soft-prompt -> frozen LLM verbalizes) | 0.853  (bypass 0.873) | 0.485  (bypass 0.474) |
| **C in-weight LoRA** (finetune the LLM on SMILES text) | 0.731  (base 0.479) | 0.446  (base 0.479) |
| reference | cheap Morgan specialist 0.895 | cross-property floor (Morgan) 0.510 |

(bridge param count 8.6M ~ LoRA-r16 scale; the bypass = the SAME projection -> a fixed linear read-out,
no transformer, the equal-budget head control.)

## Verdict: "route, don't train" confirmed, fairly (the pre-committed "prior holds" outcome)

**H1 (within-property ordering): orchestrate > bridge > LoRA, and the bridge never beats its own
LLM-bypass.**
- The trained head on the frozen embedding (orchestrate, 0.893) is the best placement, essentially at
  the cheap Morgan specialist (0.895).
- The learned bridge (0.853) is WORSE than orchestrate by -0.04, and crucially **worse than its own
  LLM-bypass head (0.873)**: routing the embedding through the frozen LLM in-language adds nothing over a
  bare head on the same projection. The frozen LLM is dead weight in the read. H1b (bridge > orchestrate
  AND > bypass) does NOT fire.
- In-weight LoRA (0.731) lifts the LLM's output a lot from base (0.479 -> 0.731) but lands well below the
  head. Training the model's weights is the weakest placement.

**H2 (held-out-property transfer): no arm transfers; the read is property-specific.**
- A read trained on 5 ADMET properties, applied to held-out hERG, sits at or below chance for ALL three
  placements (orchestrate 0.435, bridge 0.485, LoRA 0.446), at/under the cross-property Morgan floor
  (0.510). None excludes the floor. This is the pre-registered "transfer premise fails" negative:
  reading the SFM is a property-specific skill, not a general one, regardless of placement. H2b
  (bridge transfers better) does NOT fire.

**Net:** the closed-weight-friendly placement (a thin head on the open SFM, LLM untouched) wins
within-property; the in-language bridge and in-weight LoRA do not earn their extra machinery; and the
read does not generalize across properties for any of them. This is the fair test the prereg set up to
be able to overturn "route, don't train", and it came down on confirm.

## Honest caveats (v1)
- **Paired CIs not yet computed.** The arms dumped AUROCs but not per-item scores, so the prereg's
  `paired_cluster_boot` (AUROC_A - AUROC_B CI) and `score_arm` (AURC / temperature-scaled ECE) are a
  cheap v2 re-run (add a per-item dump to `bridge_arm.py` / `ws3_lora.py`). The point estimates already
  decide the direction (bridge < orchestrate AND < bypass; all transfers at/below the floor), but the
  formal H1/H2 thresholds ride on the paired test.
- **MTR pretraining leakage (the #1 deferred caveat).** ChemBERTa-MTR is pretrained on ~200 computed
  descriptors, so even the failed transfer is on a property-pretrained substrate; the ChemBERTa-MLM /
  Morgan pretraining-naive control is v2.
- **Single model / endpoint / fold.** Llama-3.1-8B, the additive-at-peak injection, the clearance
  held-out endpoint, folds 2-3, and MoLFormer are the v2 matrix.
